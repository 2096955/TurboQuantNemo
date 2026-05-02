// isoquant_kv_runtime.hpp — C++ interface for IsoQuant fused KV decode on Metal
//
// Host-side API for the 4-kernel pipeline:
//   1. pack_indices_3bit  — compress uint8 indices to 3-bit packed uint32_t
//   2. fused_qk_dot       — Q·Kᵀ directly from packed K (no K materialisation)
//   3. softmax            — numerically stable in-place softmax
//   4. fused_value_accum  — weighted V sum in rotated space (no V materialisation)
//   5. inverse_rotation   — WHT + SO(4), applied once on aggregated output
//
// Usage:
//   IsoQuantKVRuntime runtime(device);
//   runtime.load_pipeline("isoquant_kv_kernels.metallib");
//
//   // Per-layer setup
//   IsoQuantKVCache cache(num_heads, head_dim, max_seq_len);
//   cache.set_centroids(centroids_data, num_levels);
//   cache.set_so4_blocks(block_data);
//
//   // Write compressed KV
//   cache.append_compressed(k_indices, k_norms, v_indices, v_norms, count);
//
//   // Fused decode (4 kernels, 1 command buffer)
//   runtime.fused_attention(cache, query, scale, output);

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Forward-declare Metal-cpp types to avoid pulling the header into every TU.
// The .cpp includes the real headers.
namespace MTL {
class Device;
class CommandQueue;
class ComputePipelineState;
class Buffer;
class Library;
}  // namespace MTL

namespace isoquant {

enum class ValueAccumStrategy {
    Auto = 0,
    WordStriped = 1,
    DimParallel = 2,
};

// ============================================================
// IsoQuantKVCache — per-layer compressed KV storage
// ============================================================

struct CompressedKV {
    // Packed 3-bit indices: [num_heads, seq_len, packed_words_per_dim]
    // packed_words_per_dim = head_dim / 8
    MTL::Buffer* packed_indices = nullptr;

    // Per-token norms: [num_heads, seq_len]
    MTL::Buffer* norms = nullptr;

    uint32_t seq_len   = 0;
    uint32_t capacity  = 0;  // max tokens before realloc
};

struct IsoQuantKVCache {
    uint32_t num_heads  = 0;
    uint32_t head_dim   = 0;
    uint32_t max_seq    = 0;

    CompressedKV keys;
    CompressedKV values;

    // Lloyd-Max centroids: [num_levels] (8 for 3-bit)
    MTL::Buffer* centroids = nullptr;
    uint32_t num_levels = 8;

    // SO(4) block matrices (pre-transposed for inverse): [num_heads, head_dim/4, 4, 4]
    MTL::Buffer* so4_blocks = nullptr;

    // Configuration
    bool use_hadamard = true;
    uint32_t bit_width = 3;

    // Derived
    uint32_t packed_words_per_dim() const { return head_dim / 8; }
    uint32_t current_seq_len() const { return keys.seq_len; }
};

// ============================================================
// IsoQuantKVRuntime — Metal pipeline manager
// ============================================================

class IsoQuantKVRuntime {
public:
    explicit IsoQuantKVRuntime(MTL::Device* device);
    ~IsoQuantKVRuntime();

    // Non-copyable, movable
    IsoQuantKVRuntime(const IsoQuantKVRuntime&) = delete;
    IsoQuantKVRuntime& operator=(const IsoQuantKVRuntime&) = delete;
    IsoQuantKVRuntime(IsoQuantKVRuntime&&) noexcept;
    IsoQuantKVRuntime& operator=(IsoQuantKVRuntime&&) noexcept;

    // Load compiled Metal library (.metallib) and create pipeline states
    bool load_pipeline(const std::string& metallib_path);

    // Check if all pipelines are ready
    bool is_ready() const;

    // Control how Kernel C is selected.
    // Auto uses an empirical heuristic tuned from the native benchmark:
    // short rows and very large aggregate workloads prefer dim-parallel;
    // the middle regime stays on the original word-striped kernel.
    void set_value_accum_strategy(ValueAccumStrategy strategy);
    ValueAccumStrategy value_accum_strategy() const;
    ValueAccumStrategy resolved_value_accum_strategy(const IsoQuantKVCache& cache) const;
    static const char* value_accum_strategy_name(ValueAccumStrategy strategy);

    // ----------------------------------------------------------
    // Buffer management
    // ----------------------------------------------------------

    // Create a cache for one layer
    IsoQuantKVCache create_cache(
        uint32_t num_heads,
        uint32_t head_dim,
        uint32_t max_seq_len
    ) const;

    // Set centroids (copies to GPU buffer)
    void set_centroids(
        IsoQuantKVCache& cache,
        const float* centroids,
        uint32_t num_levels
    ) const;

    // Set SO(4) block matrices (pre-transposed, copies to GPU)
    void set_so4_blocks(
        IsoQuantKVCache& cache,
        const float* blocks,  // [num_heads, head_dim/4, 4, 4] row-major
        bool use_hadamard
    ) const;

    // Pack uint8 indices into 3-bit packed uint32_t on GPU
    // indices: [num_heads * seq_chunk * head_dim] uint8
    // Returns GPU buffer of packed uint32_t
    MTL::Buffer* pack_indices(
        const uint8_t* indices,
        uint32_t total_values
    ) const;

    // Append compressed tokens to cache
    // k_indices, v_indices: [num_heads, chunk_len, head_dim] uint8
    // k_norms, v_norms:     [num_heads, chunk_len] float
    void append_compressed(
        IsoQuantKVCache& cache,
        const uint8_t* k_indices,
        const float* k_norms,
        const uint8_t* v_indices,
        const float* v_norms,
        uint32_t chunk_len
    ) const;

    // ----------------------------------------------------------
    // Fused attention (the main entry point)
    // ----------------------------------------------------------

    // Execute the full 4-kernel pipeline:
    //   1. fused_qk_dot      — scores = Q·Kᵀ from packed K
    //   2. softmax            — in-place softmax on scores
    //   3. fused_value_accum  — output_rot = Σ attn[t] * V[t] in rotated space
    //   4. inverse_rotation   — output = Π⁻¹(output_rot)
    //
    // query:  [num_heads, head_dim] float (already in rotated space)
    // output: [num_heads, head_dim] float (in original space after inverse rotation)
    // scale:  1/sqrt(head_dim)
    //
    // All work is submitted in a single command buffer.
    void fused_attention(
        const IsoQuantKVCache& cache,
        const float* query,         // host pointer, copied to GPU
        float scale,
        float* output,              // host pointer, results copied back
        uint32_t threads_per_tg = 32
    ) const;

    // Lower-level: same as above but with pre-allocated GPU buffers
    // (avoids host↔GPU copies per call — use for sustained decode)
    void fused_attention_gpu(
        const IsoQuantKVCache& cache,
        MTL::Buffer* query_buf,     // [num_heads, head_dim] float, GPU
        float scale,
        MTL::Buffer* output_buf,    // [num_heads, head_dim] float, GPU
        uint32_t threads_per_tg = 32
    ) const;

    // ----------------------------------------------------------
    // Individual kernel dispatch (for profiling / testing)
    // ----------------------------------------------------------

    void dispatch_fused_qk_dot(
        const IsoQuantKVCache& cache,
        MTL::Buffer* query_buf,
        MTL::Buffer* scores_buf,
        uint32_t threads_per_tg = 32
    ) const;

    void dispatch_softmax(
        MTL::Buffer* scores_buf,
        uint32_t seq_len,
        uint32_t num_heads,
        uint32_t threads_per_tg = 32
    ) const;

    void dispatch_fused_value_accum(
        const IsoQuantKVCache& cache,
        MTL::Buffer* attn_weights_buf,
        MTL::Buffer* output_buf,
        uint32_t threads_per_tg = 128
    ) const;

    // Profile Kernel C using Metal command-buffer kernel timestamps.
    // Returns milliseconds on success, or a negative value if timing is unavailable.
    double profile_fused_value_accum_gpu_ms(
        const IsoQuantKVCache& cache,
        MTL::Buffer* attn_weights_buf,
        MTL::Buffer* output_buf,
        uint32_t threads_per_tg = 128
    ) const;

    void dispatch_inverse_rotation(
        const IsoQuantKVCache& cache,
        MTL::Buffer* data_buf,  // in-place
        uint32_t threads_per_tg = 32
    ) const;

    // ----------------------------------------------------------
    // Cleanup
    // ----------------------------------------------------------

    void release_cache(IsoQuantKVCache& cache) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace isoquant
