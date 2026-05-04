// isoquant_kv_runtime.cpp — Metal-cpp dispatch for IsoQuant fused KV decode
//
// Implements the host-side pipeline management: buffer creation, kernel dispatch,
// command buffer encoding.  All GPU work is submitted via Metal-cpp (no Obj-C++).
//
// Build: requires Metal-cpp headers (metal-cpp/) and linking against Metal.framework.
//   xcrun -sdk macosx metal -c isoquant_kv_kernels.metal -o kernels.air
//   xcrun -sdk macosx metallib kernels.air -o isoquant_kv_kernels.metallib
//   clang++ -std=c++20 -framework Metal -framework Foundation \
//           -I metal-cpp/ isoquant_kv_runtime.cpp -c -o runtime.o

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "isoquant_kv_runtime.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>

namespace isoquant {

static bool use_dimparallel_value_kernel(uint32_t num_heads, uint32_t seq_len, uint32_t head_dim) {
    (void)head_dim;
    const uint32_t total_rows = num_heads * seq_len;
    // Empirical split from the native benchmark:
    // - dim-parallel wins on very short rows and large aggregate workloads
    // - word-striped stays better in the middle regime
    return seq_len <= 256 || total_rows >= 4096;
}

static ValueAccumStrategy resolve_value_accum_strategy(
    ValueAccumStrategy requested, uint32_t num_heads, uint32_t seq_len, uint32_t head_dim
) {
    if (requested == ValueAccumStrategy::Auto) {
        return use_dimparallel_value_kernel(num_heads, seq_len, head_dim)
            ? ValueAccumStrategy::DimParallel
            : ValueAccumStrategy::WordStriped;
    }
    return requested;
}

// ============================================================
// Impl (pimpl)
// ============================================================

struct IsoQuantKVRuntime::Impl {
    MTL::Device*               device   = nullptr;
    MTL::CommandQueue*         queue    = nullptr;
    MTL::Library*              library  = nullptr;

    // Pipeline states for each kernel
    MTL::ComputePipelineState* pso_fused_qk_dot     = nullptr;
    MTL::ComputePipelineState* pso_fused_decode     = nullptr;
    MTL::ComputePipelineState* pso_softmax          = nullptr;
    MTL::ComputePipelineState* pso_value_accum_word = nullptr;
    MTL::ComputePipelineState* pso_value_accum_dim  = nullptr;
    MTL::ComputePipelineState* pso_inverse_rot      = nullptr;
    MTL::ComputePipelineState* pso_pack_indices     = nullptr;

    bool ready = false;
    ValueAccumStrategy value_accum_strategy = ValueAccumStrategy::Auto;

    MTL::ComputePipelineState* make_pso(const char* name) {
        NS::Error* error = nullptr;
        auto fn = library->newFunction(
            NS::String::string(name, NS::UTF8StringEncoding)
        );
        if (!fn) {
            throw std::runtime_error(
                std::string("Metal function not found: ") + name
            );
        }
        auto pso = device->newComputePipelineState(fn, &error);
        fn->release();
        if (!pso) {
            std::string msg = "Failed to create PSO for ";
            msg += name;
            if (error) {
                msg += ": ";
                msg += error->localizedDescription()->utf8String();
            }
            throw std::runtime_error(msg);
        }
        return pso;
    }

    void release_pso(MTL::ComputePipelineState*& pso) {
        if (pso) { pso->release(); pso = nullptr; }
    }

    ~Impl() {
        release_pso(pso_fused_qk_dot);
        release_pso(pso_fused_decode);
        release_pso(pso_softmax);
        release_pso(pso_value_accum_word);
        release_pso(pso_value_accum_dim);
        release_pso(pso_inverse_rot);
        release_pso(pso_pack_indices);
        if (library) library->release();
        if (queue)   queue->release();
        // device is not owned by us
    }
};

// ============================================================
// Constructor / destructor / move
// ============================================================

IsoQuantKVRuntime::IsoQuantKVRuntime(MTL::Device* device)
    : impl_(std::make_unique<Impl>())
{
    impl_->device = device;
    impl_->queue  = device->newCommandQueue();
    if (!impl_->queue) {
        throw std::runtime_error("Failed to create Metal command queue");
    }
}

IsoQuantKVRuntime::~IsoQuantKVRuntime() = default;

IsoQuantKVRuntime::IsoQuantKVRuntime(IsoQuantKVRuntime&&) noexcept = default;
IsoQuantKVRuntime& IsoQuantKVRuntime::operator=(IsoQuantKVRuntime&&) noexcept = default;

// ============================================================
// Pipeline loading
// ============================================================

bool IsoQuantKVRuntime::load_pipeline(const std::string& metallib_path) {
    NS::Error* error = nullptr;
    auto url = NS::URL::fileURLWithPath(
        NS::String::string(metallib_path.c_str(), NS::UTF8StringEncoding)
    );
    impl_->library = impl_->device->newLibrary(url, &error);
    if (!impl_->library) {
        return false;
    }

    try {
        impl_->pso_fused_qk_dot = impl_->make_pso("fused_qk_dot_3bit");
        impl_->pso_fused_decode = impl_->make_pso("fused_decode_attention_3bit");
        impl_->pso_softmax      = impl_->make_pso("softmax_1d");
        impl_->pso_value_accum_word = impl_->make_pso("fused_value_accum_3bit_wordstriped");
        impl_->pso_value_accum_dim  = impl_->make_pso("fused_value_accum_3bit_dimparallel");
        impl_->pso_inverse_rot  = impl_->make_pso("inverse_rotation_wht_so4");
        impl_->pso_pack_indices = impl_->make_pso("pack_indices_3bit");

        // HIGH 6: Validate threadgroup memory for D=512 kernel
        if (impl_->pso_inverse_rot) {
            size_t max_tg_mem = impl_->pso_inverse_rot->maxTotalThreadgroupMemory();
            size_t required_tg_mem = 512 * sizeof(float);  // vec[512] in threadgroup
            if (required_tg_mem > max_tg_mem) {
                throw std::runtime_error(
                    "Kernel inverse_rotation_wht_so4 requires " +
                    std::to_string(required_tg_mem) + " bytes threadgroup memory, but device limit is " +
                    std::to_string(max_tg_mem) + " bytes (D=512 may not fit on this GPU)"
                );
            }
        }
        impl_->ready = true;
    } catch (...) {
        impl_->ready = false;
        return false;
    }
    return true;
}

bool IsoQuantKVRuntime::is_ready() const {
    return impl_ && impl_->ready;
}

void IsoQuantKVRuntime::set_value_accum_strategy(ValueAccumStrategy strategy) {
    impl_->value_accum_strategy = strategy;
}

ValueAccumStrategy IsoQuantKVRuntime::value_accum_strategy() const {
    return impl_->value_accum_strategy;
}

ValueAccumStrategy IsoQuantKVRuntime::resolved_value_accum_strategy(const IsoQuantKVCache& cache) const {
    return resolve_value_accum_strategy(
        impl_->value_accum_strategy, cache.num_heads, cache.values.seq_len, cache.head_dim
    );
}

const char* IsoQuantKVRuntime::value_accum_strategy_name(ValueAccumStrategy strategy) {
    switch (strategy) {
        case ValueAccumStrategy::Auto:
            return "auto";
        case ValueAccumStrategy::WordStriped:
            return "word";
        case ValueAccumStrategy::DimParallel:
            return "dim";
    }
    return "unknown";
}

// ============================================================
// Buffer helpers
// ============================================================

static MTL::Buffer* make_buffer(MTL::Device* dev, size_t bytes, const void* data = nullptr) {
    auto buf = dev->newBuffer(
        bytes,
        MTL::ResourceStorageModeShared
    );
    if (!buf) {
        throw std::runtime_error("Failed to allocate Metal buffer");
    }
    if (data) {
        std::memcpy(buf->contents(), data, bytes);
    }
    return buf;
}

// ============================================================
// Cache management
// ============================================================

IsoQuantKVCache IsoQuantKVRuntime::create_cache(
    uint32_t num_heads, uint32_t head_dim, uint32_t max_seq_len
) const {
    IsoQuantKVCache cache;
    cache.num_heads = num_heads;
    cache.head_dim  = head_dim;
    cache.max_seq   = max_seq_len;

    uint32_t pw = cache.packed_words_per_dim();

    // Pre-allocate packed storage for max sequence length
    size_t packed_bytes = (size_t)num_heads * max_seq_len * pw * sizeof(uint32_t);
    size_t norms_bytes  = (size_t)num_heads * max_seq_len * sizeof(float);

    cache.keys.packed_indices   = make_buffer(impl_->device, packed_bytes);
    cache.keys.norms            = make_buffer(impl_->device, norms_bytes);
    cache.keys.capacity         = max_seq_len;
    cache.keys.seq_len          = 0;

    cache.values.packed_indices = make_buffer(impl_->device, packed_bytes);
    cache.values.norms          = make_buffer(impl_->device, norms_bytes);
    cache.values.capacity       = max_seq_len;
    cache.values.seq_len        = 0;

    return cache;
}

void IsoQuantKVRuntime::set_centroids(
    IsoQuantKVCache& cache,
    const float* centroids,
    uint32_t num_levels
) const {
    cache.num_levels = num_levels;
    cache.centroids = make_buffer(
        impl_->device,
        num_levels * sizeof(float),
        centroids
    );
}

void IsoQuantKVRuntime::set_so4_blocks(
    IsoQuantKVCache& cache,
    const float* blocks,
    bool use_hadamard
) const {
    uint32_t num_blocks = cache.head_dim / 4;
    size_t bytes = (size_t)cache.num_heads * num_blocks * 16 * sizeof(float);
    cache.so4_blocks = make_buffer(impl_->device, bytes, blocks);
    cache.use_hadamard = use_hadamard;
}

void IsoQuantKVRuntime::release_cache(IsoQuantKVCache& cache) const {
    auto release = [](MTL::Buffer*& b) { if (b) { b->release(); b = nullptr; } };
    release(cache.keys.packed_indices);
    release(cache.keys.norms);
    release(cache.values.packed_indices);
    release(cache.values.norms);
    release(cache.centroids);
    release(cache.so4_blocks);
}

// ============================================================
// Pack + append
// ============================================================

MTL::Buffer* IsoQuantKVRuntime::pack_indices(
    const uint8_t* indices,
    uint32_t total_values
) const {
    uint32_t packed_count = (total_values + 7) / 8;

    auto src_buf = make_buffer(impl_->device, total_values, indices);
    auto dst_buf = make_buffer(impl_->device, packed_count * sizeof(uint32_t));

    auto cmd = impl_->queue->commandBuffer();
    auto enc = cmd->computeCommandEncoder();

    enc->setComputePipelineState(impl_->pso_pack_indices);
    enc->setBuffer(src_buf, 0, 0);
    enc->setBuffer(dst_buf, 0, 1);
    enc->setBytes(&total_values, sizeof(uint32_t), 2);

    MTL::Size grid(packed_count, 1, 1);
    uint32_t tg_raw = std::min(packed_count, 256u);
    uint32_t tg_size = (tg_raw > 0) ? (1u << (31 - __builtin_clz(tg_raw))) : 1u;
    MTL::Size group(tg_size, 1, 1);
    enc->dispatchThreads(grid, group);

    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    if (cmd->status() != MTL::CommandBufferStatusCompleted) {
        NS::Error* err = cmd->error();
        std::string msg = "metal command buffer failed: status=" + std::to_string(cmd->status());
        if (err) msg += ", error=" + std::string(err->localizedDescription()->utf8String());
        throw std::runtime_error(msg);
    }

    src_buf->release();
    return dst_buf;
}

void IsoQuantKVRuntime::append_compressed(
    IsoQuantKVCache& cache,
    const uint8_t* k_indices,
    const float* k_norms,
    const uint8_t* v_indices,
    const float* v_norms,
    uint32_t chunk_len
) const {
    uint32_t H  = cache.num_heads;
    uint32_t D  = cache.head_dim;
    uint32_t pw = cache.packed_words_per_dim();
    uint32_t T  = cache.keys.seq_len;

    if (T + chunk_len > cache.keys.capacity) {
        throw std::runtime_error(
            "isoquant cache overflow: T=" + std::to_string(T) +
            ", chunk_len=" + std::to_string(chunk_len) +
            ", capacity=" + std::to_string(cache.keys.capacity)
        );
    }

    // Pack indices on GPU
    uint32_t total_vals = H * chunk_len * D;
    auto k_packed_chunk = pack_indices(k_indices, total_vals);
    auto v_packed_chunk = pack_indices(v_indices, total_vals);

    // Copy packed data into cache buffers at offset T
    size_t packed_row_bytes = (size_t)pw * sizeof(uint32_t);
    size_t norm_row_bytes   = sizeof(float);

    auto k_dst = static_cast<uint8_t*>(cache.keys.packed_indices->contents());
    auto v_dst = static_cast<uint8_t*>(cache.values.packed_indices->contents());
    auto kn_dst = static_cast<float*>(cache.keys.norms->contents());
    auto vn_dst = static_cast<float*>(cache.values.norms->contents());

    auto k_src = static_cast<const uint8_t*>(k_packed_chunk->contents());
    auto v_src = static_cast<const uint8_t*>(v_packed_chunk->contents());

    for (uint32_t h = 0; h < H; ++h) {
        size_t dst_offset = (h * cache.keys.capacity + T) * packed_row_bytes;
        size_t src_offset = h * chunk_len * packed_row_bytes;
        std::memcpy(k_dst + dst_offset, k_src + src_offset, chunk_len * packed_row_bytes);
        std::memcpy(v_dst + dst_offset, v_src + src_offset, chunk_len * packed_row_bytes);

        size_t norm_dst = h * cache.keys.capacity + T;
        size_t norm_src = h * chunk_len;
        std::memcpy(kn_dst + norm_dst, k_norms + norm_src, chunk_len * norm_row_bytes);
        std::memcpy(vn_dst + norm_dst, v_norms + norm_src, chunk_len * norm_row_bytes);
    }

    cache.keys.seq_len   += chunk_len;
    cache.values.seq_len += chunk_len;

    k_packed_chunk->release();
    v_packed_chunk->release();
}

// ============================================================
// Kernel dispatch
// ============================================================

void IsoQuantKVRuntime::dispatch_fused_qk_dot(
    const IsoQuantKVCache& cache,
    MTL::Buffer* query_buf,
    MTL::Buffer* scores_buf,
    uint32_t threads_per_tg
) const {
    uint32_t T = cache.keys.seq_len;
    uint32_t H = cache.num_heads;
    uint32_t D = cache.head_dim;
    uint32_t stride = cache.keys.capacity;

    auto cmd = impl_->queue->commandBuffer();
    auto enc = cmd->computeCommandEncoder();

    enc->setComputePipelineState(impl_->pso_fused_qk_dot);
    enc->setBuffer(cache.keys.packed_indices, 0, 0);
    enc->setBuffer(cache.centroids, 0, 1);
    enc->setBuffer(cache.keys.norms, 0, 2);
    enc->setBuffer(query_buf, 0, 3);
    enc->setBuffer(scores_buf, 0, 4);
    enc->setBytes(&D, sizeof(uint32_t), 5);
    enc->setBytes(&T, sizeof(uint32_t), 6);
    enc->setBytes(&stride, sizeof(uint32_t), 7);
    enc->setBytes(&H, sizeof(uint32_t), 8);

    // Grid: (seq_len, num_heads) — one threadgroup per (token, head) pair
    MTL::Size grid(T, H, 1);
    MTL::Size group(threads_per_tg, 1, 1);
    enc->dispatchThreadgroups(grid, group);

    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    if (cmd->status() != MTL::CommandBufferStatusCompleted) {
        NS::Error* err = cmd->error();
        std::string msg = "metal command buffer failed: status=" + std::to_string(cmd->status());
        if (err) msg += ", error=" + std::string(err->localizedDescription()->utf8String());
        throw std::runtime_error(msg);
    }
}

void IsoQuantKVRuntime::dispatch_softmax(
    MTL::Buffer* scores_buf,
    uint32_t seq_len,
    uint32_t num_heads,
    uint32_t threads_per_tg
) const {
    auto cmd = impl_->queue->commandBuffer();
    auto enc = cmd->computeCommandEncoder();

    enc->setComputePipelineState(impl_->pso_softmax);
    enc->setBuffer(scores_buf, 0, 0);
    enc->setBytes(&seq_len, sizeof(uint32_t), 1);
    enc->setBytes(&num_heads, sizeof(uint32_t), 2);

    MTL::Size grid(num_heads, 1, 1);
    MTL::Size group(threads_per_tg, 1, 1);
    enc->dispatchThreadgroups(grid, group);

    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    if (cmd->status() != MTL::CommandBufferStatusCompleted) {
        NS::Error* err = cmd->error();
        std::string msg = "metal command buffer failed: status=" + std::to_string(cmd->status());
        if (err) msg += ", error=" + std::string(err->localizedDescription()->utf8String());
        throw std::runtime_error(msg);
    }
}

void IsoQuantKVRuntime::dispatch_fused_value_accum(
    const IsoQuantKVCache& cache,
    MTL::Buffer* attn_weights_buf,
    MTL::Buffer* output_buf,
    uint32_t threads_per_tg
) const {
    uint32_t T = cache.values.seq_len;
    uint32_t H = cache.num_heads;
    uint32_t D = cache.head_dim;
    uint32_t stride = cache.values.capacity;

    auto cmd = impl_->queue->commandBuffer();
    auto enc = cmd->computeCommandEncoder();

    const ValueAccumStrategy strategy = resolve_value_accum_strategy(
        impl_->value_accum_strategy, H, T, D
    );
    const bool use_dim = strategy == ValueAccumStrategy::DimParallel;
    enc->setComputePipelineState(
        use_dim ? impl_->pso_value_accum_dim : impl_->pso_value_accum_word
    );
    enc->setBuffer(cache.values.packed_indices, 0, 0);
    enc->setBuffer(cache.centroids, 0, 1);
    enc->setBuffer(cache.values.norms, 0, 2);
    enc->setBuffer(attn_weights_buf, 0, 3);
    enc->setBuffer(output_buf, 0, 4);
    enc->setBytes(&D, sizeof(uint32_t), 5);
    enc->setBytes(&T, sizeof(uint32_t), 6);
    enc->setBytes(&stride, sizeof(uint32_t), 7);
    enc->setBytes(&H, sizeof(uint32_t), 8);

    uint32_t value_threads = use_dim ? std::min<uint32_t>(threads_per_tg, D) : threads_per_tg;
    MTL::Size grid(H, 1, 1);
    MTL::Size group(value_threads, 1, 1);
    enc->dispatchThreadgroups(grid, group);

    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    if (cmd->status() != MTL::CommandBufferStatusCompleted) {
        NS::Error* err = cmd->error();
        std::string msg = "metal command buffer failed: status=" + std::to_string(cmd->status());
        if (err) msg += ", error=" + std::string(err->localizedDescription()->utf8String());
        throw std::runtime_error(msg);
    }
}

double IsoQuantKVRuntime::profile_fused_value_accum_gpu_ms(
    const IsoQuantKVCache& cache,
    MTL::Buffer* attn_weights_buf,
    MTL::Buffer* output_buf,
    uint32_t threads_per_tg
) const {
    uint32_t T = cache.values.seq_len;
    uint32_t H = cache.num_heads;
    uint32_t D = cache.head_dim;
    uint32_t stride = cache.values.capacity;

    auto cmd = impl_->queue->commandBuffer();
    auto enc = cmd->computeCommandEncoder();

    const ValueAccumStrategy strategy = resolve_value_accum_strategy(
        impl_->value_accum_strategy, H, T, D
    );
    const bool use_dim = strategy == ValueAccumStrategy::DimParallel;

    enc->setComputePipelineState(
        use_dim ? impl_->pso_value_accum_dim : impl_->pso_value_accum_word
    );
    enc->setBuffer(cache.values.packed_indices, 0, 0);
    enc->setBuffer(cache.centroids, 0, 1);
    enc->setBuffer(cache.values.norms, 0, 2);
    enc->setBuffer(attn_weights_buf, 0, 3);
    enc->setBuffer(output_buf, 0, 4);
    enc->setBytes(&D, sizeof(uint32_t), 5);
    enc->setBytes(&T, sizeof(uint32_t), 6);
    enc->setBytes(&stride, sizeof(uint32_t), 7);
    enc->setBytes(&H, sizeof(uint32_t), 8);

    uint32_t value_threads = use_dim ? std::min<uint32_t>(threads_per_tg, D) : threads_per_tg;
    enc->dispatchThreadgroups(MTL::Size(H, 1, 1), MTL::Size(value_threads, 1, 1));
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    if (cmd->status() != MTL::CommandBufferStatusCompleted) {
        NS::Error* err = cmd->error();
        std::string msg = "metal command buffer failed: status=" + std::to_string(cmd->status());
        if (err) msg += ", error=" + std::string(err->localizedDescription()->utf8String());
        throw std::runtime_error(msg);
    }

    const double start = cmd->kernelStartTime();
    const double end = cmd->kernelEndTime();
    if (end > start) {
        return (end - start) * 1000.0;
    }
    return -1.0;
}

void IsoQuantKVRuntime::dispatch_inverse_rotation(
    const IsoQuantKVCache& cache,
    MTL::Buffer* data_buf,
    uint32_t threads_per_tg
) const {
    uint32_t H = cache.num_heads;
    uint32_t D = cache.head_dim;
    uint32_t use_had = cache.use_hadamard ? 1u : 0u;

    auto cmd = impl_->queue->commandBuffer();
    auto enc = cmd->computeCommandEncoder();

    enc->setComputePipelineState(impl_->pso_inverse_rot);
    enc->setBuffer(data_buf, 0, 0);
    enc->setBuffer(cache.so4_blocks, 0, 1);
    enc->setBytes(&D, sizeof(uint32_t), 2);
    enc->setBytes(&H, sizeof(uint32_t), 3);
    enc->setBytes(&use_had, sizeof(uint32_t), 4);

    MTL::Size grid(H, 1, 1);
    MTL::Size group(threads_per_tg, 1, 1);
    enc->dispatchThreadgroups(grid, group);

    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    if (cmd->status() != MTL::CommandBufferStatusCompleted) {
        NS::Error* err = cmd->error();
        std::string msg = "metal command buffer failed: status=" + std::to_string(cmd->status());
        if (err) msg += ", error=" + std::string(err->localizedDescription()->utf8String());
        throw std::runtime_error(msg);
    }
}

// ============================================================
// Fused attention — full pipeline
// ============================================================

void IsoQuantKVRuntime::fused_attention_gpu(
    const IsoQuantKVCache& cache,
    MTL::Buffer* query_buf,
    float scale,
    MTL::Buffer* output_buf,
    uint32_t threads_per_tg
) const {
    (void)scale;
    uint32_t T = cache.keys.seq_len;
    uint32_t H = cache.num_heads;
    uint32_t D = cache.head_dim;
    uint32_t k_stride = cache.keys.capacity;
    uint32_t v_stride = cache.values.capacity;
    uint32_t decode_threads = (D <= 128) ? 128u : ((D <= 256) ? 256u : 512u);
    uint32_t inverse_threads = std::min<uint32_t>(threads_per_tg, D);

    // --- Single command buffer for the full pipeline ---
    auto cmd = impl_->queue->commandBuffer();
    auto enc = cmd->computeCommandEncoder();

    // Kernel C': fused QK + softmax + weighted V accumulation
    enc->setComputePipelineState(impl_->pso_fused_decode);
    enc->setBuffer(cache.keys.packed_indices, 0, 0);
    enc->setBuffer(cache.values.packed_indices, 0, 1);
    enc->setBuffer(cache.centroids, 0, 2);
    enc->setBuffer(cache.keys.norms, 0, 3);
    enc->setBuffer(cache.values.norms, 0, 4);
    enc->setBuffer(query_buf, 0, 5);
    enc->setBuffer(output_buf, 0, 6);
    enc->setBytes(&D, sizeof(uint32_t), 7);
    enc->setBytes(&T, sizeof(uint32_t), 8);
    enc->setBytes(&k_stride, sizeof(uint32_t), 9);
    enc->setBytes(&v_stride, sizeof(uint32_t), 10);
    enc->setBytes(&H, sizeof(uint32_t), 11);
    enc->dispatchThreadgroups(MTL::Size(H, 1, 1), MTL::Size(decode_threads, 1, 1));
    enc->memoryBarrier(MTL::BarrierScopeBuffers);

    // Kernel D: inverse rotation (in-place on output)
    uint32_t use_had = cache.use_hadamard ? 1u : 0u;
    enc->setComputePipelineState(impl_->pso_inverse_rot);
    enc->setBuffer(output_buf, 0, 0);
    enc->setBuffer(cache.so4_blocks, 0, 1);
    enc->setBytes(&D, sizeof(uint32_t), 2);
    enc->setBytes(&H, sizeof(uint32_t), 3);
    enc->setBytes(&use_had, sizeof(uint32_t), 4);
    enc->dispatchThreadgroups(MTL::Size(H, 1, 1), MTL::Size(inverse_threads, 1, 1));

    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    if (cmd->status() != MTL::CommandBufferStatusCompleted) {
        NS::Error* err = cmd->error();
        std::string msg = "metal command buffer failed: status=" + std::to_string(cmd->status());
        if (err) msg += ", error=" + std::string(err->localizedDescription()->utf8String());
        throw std::runtime_error(msg);
    }
}

void IsoQuantKVRuntime::fused_attention(
    const IsoQuantKVCache& cache,
    const float* query,
    float scale,
    float* output,
    uint32_t threads_per_tg
) const {
    uint32_t H = cache.num_heads;
    uint32_t D = cache.head_dim;
    size_t vec_bytes = (size_t)H * D * sizeof(float);

    // Copy query to GPU
    auto q_buf = make_buffer(impl_->device, vec_bytes, query);

    // Allocate output on GPU
    auto out_buf = make_buffer(impl_->device, vec_bytes);

    // Apply attention scale in a scratch buffer (do not mutate caller's data)
    // score = (q * scale) · k * norm
    // We bake scale into the query to avoid an extra pass.
    auto q_scaled_buf = make_buffer(impl_->device, vec_bytes);
    auto q_ptr = static_cast<float*>(q_scaled_buf->contents());
    std::memcpy(q_ptr, query, vec_bytes);
    for (size_t i = 0; i < (size_t)H * D; ++i) {
        q_ptr[i] *= scale;
    }

    fused_attention_gpu(cache, q_scaled_buf, scale, out_buf, threads_per_tg);

    // Copy result back to host
    std::memcpy(output, out_buf->contents(), vec_bytes);
    q_scaled_buf->release();

    q_buf->release();
    out_buf->release();
}

