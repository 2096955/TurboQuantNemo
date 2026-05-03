// isoquant_kv_kernels.metal — Fused IsoQuant KV decode pipeline for Apple Silicon
//
// Four kernels that replace the dense Python path:
//   Kernel A: fused_qk_dot_3bit     — packed 3-bit K decode + Q·Kᵀ, no K materialisation
//   Kernel B: softmax_1d            — numerically stable softmax (max-subtract + exp + normalise)
//   Kernel C: fused_value_accum_3bit — packed 3-bit V decode + weighted sum in rotated space
//   Kernel D: inverse_rotation_wht_so4 — WHT butterfly + SO(4) block rotation, applied once
//
// Design:
//   - 1 threadgroup = 1 token (Kernels A/C) or 1 head (Kernel D)
//   - 32 threads per threadgroup (tunable via THREADS_PER_TG)
//   - simd_sum for reductions — no shared-memory reduce needed
//   - query cached in threadgroup memory
//   - centroids in constant memory (8 floats for 3-bit = 64 bytes, fits constant cache)
//   - 3-bit packing: 8 values per 24 bits in a uint32_t (bits 0..23, bits 24..31 unused)
//
// Memory layout (all row-major):
//   K_packed: [num_heads, cache_stride, packed_words_per_dim]  — uint32_t
//   V_packed: [num_heads, cache_stride, packed_words_per_dim]  — uint32_t
//   norms:    [num_heads, cache_stride]                        — float
//   centroids:[num_levels]                                — float (constant, 8 for 3-bit)
//   q:        [num_heads, head_dim]                       — float
//   scores:   [num_heads, seq_len]                        — float
//
// Attribution: kernel specs from bottleneck critique; implementation by Claude.

#include <metal_stdlib>
using namespace metal;

// ============================================================
// Constants
// ============================================================

constant uint VALUES_PER_WORD = 8;       // 8 × 3-bit values packed in 24 bits
constant uint BITS_PER_VALUE  = 3;
constant uint VALUE_MASK      = 0x7;     // (1 << 3) - 1

// ============================================================
// Helpers: 3-bit decode
// ============================================================

// Decode lower 4 values (bits 0..11) from a packed uint32_t word
inline float4 decode_3bit_lo(uint word, constant float* centroids) {
    return float4(
        centroids[(word >>  0) & VALUE_MASK],
        centroids[(word >>  3) & VALUE_MASK],
        centroids[(word >>  6) & VALUE_MASK],
        centroids[(word >>  9) & VALUE_MASK]
    );
}

// Decode upper 4 values (bits 12..23) from a packed uint32_t word
inline float4 decode_3bit_hi(uint word, constant float* centroids) {
    return float4(
        centroids[(word >> 12) & VALUE_MASK],
        centroids[(word >> 15) & VALUE_MASK],
        centroids[(word >> 18) & VALUE_MASK],
        centroids[(word >> 21) & VALUE_MASK]
    );
}

// ============================================================
// Kernel A: Fused Q·Kᵀ from 3-bit packed K
// ============================================================
//
// Each threadgroup computes the attention score for ONE (head, token) pair.
// Grid:  (seq_len, num_heads, 1)
// Group: (threads_per_tg, 1, 1)
//
// Replaces: unpack → dequant → inverse_rotate → matmul(q, K^T)
// With:     fused decode + dot in rotated space (skip inverse rotation entirely)

kernel void fused_qk_dot_3bit(
    device const uint*    K_packed       [[buffer(0)]],   // [H, T, packed_words]
    constant float*       centroids      [[buffer(1)]],   // [8]
    device const float*   norms          [[buffer(2)]],   // [H, T]
    device const float*   q              [[buffer(3)]],   // [H, D]
    device float*         scores         [[buffer(4)]],   // [H, T]
    constant uint&        head_dim       [[buffer(5)]],
    constant uint&        seq_len        [[buffer(6)]],
    constant uint&        cache_stride   [[buffer(7)]],
    constant uint&        num_heads      [[buffer(8)]],

    uint3 tid3      [[thread_position_in_threadgroup]],
    uint3 tg_size3  [[threads_per_threadgroup]],
    uint3 gid3      [[threadgroup_position_in_grid]],     // (token_idx, head_idx)
    uint  simd_lane [[thread_index_in_simdgroup]]
) {
    uint tid = tid3.x;
    uint tg_size = tg_size3.x;
    uint token_idx = gid3.x;
    uint head_idx  = gid3.y;
    uint packed_words = head_dim / VALUES_PER_WORD;

    // --- Load query into threadgroup memory (coalesced, one-time cost) ---
    threadgroup float q_local[512];  // max head_dim = 512
    device const float* q_head = q + head_idx * head_dim;
    for (uint i = tid; i < head_dim; i += tg_size) {
        q_local[i] = q_head[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Pointer to this token's packed K for this head ---
    uint kv_offset = (head_idx * cache_stride + token_idx) * packed_words;
    device const uint* packed = K_packed + kv_offset;

    // --- Striped dot product accumulation ---
    float acc = 0.0f;

    for (uint w = tid; w < packed_words; w += tg_size) {
        uint word = packed[w];
        uint base = w * VALUES_PER_WORD;

        float4 k_lo = decode_3bit_lo(word, centroids);
        float4 k_hi = decode_3bit_hi(word, centroids);

        float4 q_lo = float4(q_local[base + 0], q_local[base + 1],
                             q_local[base + 2], q_local[base + 3]);
        float4 q_hi = float4(q_local[base + 4], q_local[base + 5],
                             q_local[base + 6], q_local[base + 7]);

        acc += dot(k_lo, q_lo);
        acc += dot(k_hi, q_hi);
    }

    // --- SIMD reduction (no shared-memory needed) ---
    float sum = simd_sum(acc);

    // --- Write score (lane 0 only) ---
    if (simd_lane == 0) {
        uint norm_idx = head_idx * cache_stride + token_idx;
        uint score_idx = head_idx * seq_len + token_idx;
        scores[score_idx] = sum * norms[norm_idx];
    }
}

// ============================================================
// Kernel B: Numerically stable softmax
// ============================================================
//
// Each threadgroup processes ONE head's scores.
// Grid:  (num_heads, 1, 1)
// Group: (threads_per_tg, 1, 1)
//
// Three passes: (1) find max, (2) exp + partial sum, (3) normalise.

kernel void softmax_1d(
    device float*        scores     [[buffer(0)]],   // [H, T] — in-place
    constant uint&       seq_len    [[buffer(1)]],
    constant uint&       num_heads  [[buffer(2)]],

    uint3 tid3      [[thread_position_in_threadgroup]],
    uint3 tg_size3  [[threads_per_threadgroup]],
    uint3 gid3      [[threadgroup_position_in_grid]],
    uint  simd_lane [[thread_index_in_simdgroup]]
) {
    uint tid = tid3.x;
    uint tg_size = tg_size3.x;
    uint gid = gid3.x;
    device float* row = scores + gid * seq_len;

    // --- Pass 1: find max ---
    float local_max = -INFINITY;
    for (uint i = tid; i < seq_len; i += tg_size) {
        local_max = max(local_max, row[i]);
    }
    float global_max = simd_max(local_max);

    // --- Pass 2: exp(x - max) + partial sum ---
    float local_sum = 0.0f;
    for (uint i = tid; i < seq_len; i += tg_size) {
        float v = exp(row[i] - global_max);
        row[i] = v;
        local_sum += v;
    }
    float global_sum = simd_sum(local_sum);

    // --- Pass 3: normalise ---
    float inv_sum = 1.0f / global_sum;
    for (uint i = tid; i < seq_len; i += tg_size) {
        row[i] *= inv_sum;
    }
}

// ============================================================
// Kernel C: Fused weighted value accumulation from 3-bit packed V
// ============================================================
//
// Each threadgroup computes ONE head's output vector in rotated space.
// Grid:  (num_heads, 1, 1)
// Group: (threads_per_tg, 1, 1)
//
// Accumulates: output[d] = Σ_t  attn[t] * norm[t] * centroid[V_packed[t][d]]
//
// The output remains in rotated space — inverse rotation is Kernel D.

kernel void fused_value_accum_3bit_wordstriped(
    device const uint*    V_packed       [[buffer(0)]],   // [H, T, packed_words]
    constant float*       centroids      [[buffer(1)]],   // [8]
    device const float*   norms          [[buffer(2)]],   // [H, T]
    device const float*   attn_weights   [[buffer(3)]],   // [H, T]
    device float*         output         [[buffer(4)]],   // [H, D]
    constant uint&        head_dim       [[buffer(5)]],
    constant uint&        seq_len        [[buffer(6)]],
    constant uint&        cache_stride   [[buffer(7)]],
    constant uint&        num_heads      [[buffer(8)]],

    uint3 tid3      [[thread_position_in_threadgroup]],
    uint3 tg_size3  [[threads_per_threadgroup]],
    uint3 gid3      [[threadgroup_position_in_grid]],     // head index
    uint  simd_lane [[thread_index_in_simdgroup]]
) {
    uint tid = tid3.x;
    uint tg_size = tg_size3.x;
    uint head_idx = gid3.x;
    uint packed_words = head_dim / VALUES_PER_WORD;

    // Packed-word striping: good when T is mid-sized and decode locality matters more
    // than saturating every lane in the threadgroup.
    threadgroup float accum[512];
    for (uint i = tid; i < head_dim; i += tg_size) {
        accum[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = 0; t < seq_len; ++t) {
        uint norm_idx = head_idx * cache_stride + t;
        float w = attn_weights[head_idx * seq_len + t] * norms[norm_idx];
        if (w == 0.0f) continue;

        device const uint* packed = V_packed + (head_idx * cache_stride + t) * packed_words;

        for (uint pw = tid; pw < packed_words; pw += tg_size) {
            uint word = packed[pw];
            uint base = pw * VALUES_PER_WORD;

            float4 v_lo = decode_3bit_lo(word, centroids);
            float4 v_hi = decode_3bit_hi(word, centroids);

            accum[base + 0] += w * v_lo[0];
            accum[base + 1] += w * v_lo[1];
            accum[base + 2] += w * v_lo[2];
            accum[base + 3] += w * v_lo[3];
            accum[base + 4] += w * v_hi[0];
            accum[base + 5] += w * v_hi[1];
            accum[base + 6] += w * v_hi[2];
            accum[base + 7] += w * v_hi[3];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* out_head = output + head_idx * head_dim;
    for (uint i = tid; i < head_dim; i += tg_size) {
        out_head[i] = accum[i];
    }
}

kernel void fused_value_accum_3bit_dimparallel(
    device const uint*    V_packed       [[buffer(0)]],   // [H, T, packed_words]
    constant float*       centroids      [[buffer(1)]],   // [8]
    device const float*   norms          [[buffer(2)]],   // [H, T]
    device const float*   attn_weights   [[buffer(3)]],   // [H, T]
    device float*         output         [[buffer(4)]],   // [H, D]
    constant uint&        head_dim       [[buffer(5)]],
    constant uint&        seq_len        [[buffer(6)]],
    constant uint&        cache_stride   [[buffer(7)]],
    constant uint&        num_heads      [[buffer(8)]],

    uint3 tid3      [[thread_position_in_threadgroup]],
    uint3 tg_size3  [[threads_per_threadgroup]],
    uint3 gid3      [[threadgroup_position_in_grid]],     // head index
    uint  simd_lane [[thread_index_in_simdgroup]]
) {
    uint tid = tid3.x;
    uint tg_size = tg_size3.x;
    uint head_idx = gid3.x;
    uint packed_words = head_dim / VALUES_PER_WORD;

    // Map threads to output dimensions directly so all lanes stay busy.
    // This avoids the old packed-word striping, which only used D/8 lanes.
    device float* out_head = output + head_idx * head_dim;
    for (uint d = tid; d < head_dim; d += tg_size) {
        uint word_idx = d / VALUES_PER_WORD;
        uint bit_pos = (d % VALUES_PER_WORD) * BITS_PER_VALUE;
        float sum = 0.0f;

        for (uint t = 0; t < seq_len; ++t) {
            uint norm_idx = head_idx * cache_stride + t;
            float w = attn_weights[head_idx * seq_len + t] * norms[norm_idx];
            if (w == 0.0f) continue;

            device const uint* packed = V_packed + (head_idx * cache_stride + t) * packed_words;
            uint word = packed[word_idx];
            uint idx = (word >> bit_pos) & VALUE_MASK;
            sum += w * centroids[idx];
        }

        out_head[d] = sum;
    }
}

// ============================================================
// Kernel C': Fully fused decode attention
// ============================================================
//
// One threadgroup processes one head:
//   1. compute max logit over T from packed K
//   2. recompute logits, apply softmax normalisation on the fly
//   3. accumulate weighted packed V directly into output_rot
//
// This removes score materialisation and the separate softmax dispatch.

kernel void fused_decode_attention_3bit(
    device const uint*    K_packed       [[buffer(0)]],
    device const uint*    V_packed       [[buffer(1)]],
    constant float*       centroids      [[buffer(2)]],
    device const float*   k_norms        [[buffer(3)]],
    device const float*   v_norms        [[buffer(4)]],
    device const float*   q              [[buffer(5)]],   // [H, D], pre-scaled
    device float*         output         [[buffer(6)]],   // [H, D], rotated space
    constant uint&        head_dim       [[buffer(7)]],
    constant uint&        seq_len        [[buffer(8)]],
    constant uint&        k_cache_stride [[buffer(9)]],
    constant uint&        v_cache_stride [[buffer(10)]],
    constant uint&        num_heads      [[buffer(11)]],

    uint3 tid3      [[thread_position_in_threadgroup]],
    uint3 tg_size3  [[threads_per_threadgroup]],
    uint3 gid3      [[threadgroup_position_in_grid]]
) {
    uint tid = tid3.x;
    uint tg_size = tg_size3.x;
    uint head_idx = gid3.x;
    uint packed_words = head_dim / VALUES_PER_WORD;

    threadgroup float q_local[512];
    threadgroup float reduce_buf[512];
    threadgroup float max_logit;
    threadgroup float denom;
    threadgroup float weight_shared;

    float out_accum = 0.0f;

    if (tid < head_dim) {
        q_local[tid] = q[head_idx * head_dim + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        max_logit = -INFINITY;
        denom = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 1: find max logit
    for (uint t = 0; t < seq_len; ++t) {
        if (tid < tg_size) {
            reduce_buf[tid] = 0.0f;
        }
        if (tid < head_dim) {
            uint word_idx = tid / VALUES_PER_WORD;
            uint bit_pos = (tid % VALUES_PER_WORD) * BITS_PER_VALUE;
            device const uint* packed = K_packed + (head_idx * k_cache_stride + t) * packed_words;
            uint word = packed[word_idx];
            uint idx = (word >> bit_pos) & VALUE_MASK;
            float k_val = centroids[idx] * k_norms[head_idx * k_cache_stride + t];
            reduce_buf[tid] = q_local[tid] * k_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = tg_size >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce_buf[tid] += reduce_buf[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            max_logit = max(max_logit, reduce_buf[0]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 2: softmax-normalised weighted V accumulation
    for (uint t = 0; t < seq_len; ++t) {
        if (tid < tg_size) {
            reduce_buf[tid] = 0.0f;
        }
        if (tid < head_dim) {
            uint word_idx = tid / VALUES_PER_WORD;
            uint bit_pos = (tid % VALUES_PER_WORD) * BITS_PER_VALUE;
            device const uint* k_packed = K_packed + (head_idx * k_cache_stride + t) * packed_words;
            uint word = k_packed[word_idx];
            uint idx = (word >> bit_pos) & VALUE_MASK;
            float k_val = centroids[idx] * k_norms[head_idx * k_cache_stride + t];
            reduce_buf[tid] = q_local[tid] * k_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = tg_size >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce_buf[tid] += reduce_buf[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            weight_shared = exp(reduce_buf[0] - max_logit);
            denom += weight_shared;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < head_dim) {
            uint word_idx = tid / VALUES_PER_WORD;
            uint bit_pos = (tid % VALUES_PER_WORD) * BITS_PER_VALUE;
            device const uint* v_packed = V_packed + (head_idx * v_cache_stride + t) * packed_words;
            uint word = v_packed[word_idx];
            uint idx = (word >> bit_pos) & VALUE_MASK;
            float v_val = centroids[idx] * v_norms[head_idx * v_cache_stride + t];
            out_accum += weight_shared * v_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < head_dim) {
        output[head_idx * head_dim + tid] = denom > 0.0f ? (out_accum / denom) : 0.0f;
    }
}

// ============================================================
// Kernel D: Inverse rotation — WHT butterfly + SO(4) block pass
// ============================================================
//
// Applied ONCE per head on the aggregated output vector.
// This is O(d log d + d) vs the old O(T × d²).
//
// Each threadgroup processes ONE head.
// Grid:  (num_heads, 1, 1)
// Group: (threads_per_tg, 1, 1)
//
// Two phases:
//   Phase 1: In-place Walsh-Hadamard Transform (butterfly, log₂(d) stages)
//   Phase 2: 4×4 SO(4) block rotations (d/4 independent blocks)
//
// The WHT undoes the global pre-mixing; SO(4) blocks undo the isoclinic rotation.
// Combined: Π⁻¹ = (SO(4) blocks) ∘ (WHT) applied to the output vector.

kernel void inverse_rotation_wht_so4(
    device float*         data           [[buffer(0)]],   // [H, D] — in-place
    constant float*       so4_blocks     [[buffer(1)]],   // [H, D/4, 4, 4] — transposed block matrices
    constant uint&        head_dim       [[buffer(2)]],
    constant uint&        num_heads      [[buffer(3)]],
    constant uint&        use_hadamard   [[buffer(4)]],   // 1 = apply WHT, 0 = skip

    uint3 tid3      [[thread_position_in_threadgroup]],
    uint3 tg_size3  [[threads_per_threadgroup]],
    uint3 gid3      [[threadgroup_position_in_grid]],     // head index
    uint  simd_lane [[thread_index_in_simdgroup]]
) {
    uint tid = tid3.x;
    uint tg_size = tg_size3.x;
    uint gid = gid3.x;
    // Load vector into threadgroup memory for in-place transforms
    threadgroup float vec[512];
    device float* head_data = data + gid * head_dim;

    for (uint i = tid; i < head_dim; i += tg_size) {
        vec[i] = head_data[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Phase 1: Inverse WHT (unnormalised butterfly) ---
    if (use_hadamard) {
        // WHT is self-inverse up to normalisation: H⁻¹ = H / d
        // We apply the butterfly stages, then scale by 1/√d at the end.
        for (uint stride = 1; stride < head_dim; stride <<= 1) {
            for (uint i = tid; i < head_dim / 2; i += tg_size) {
                // Bit-reverse pair selection for butterfly
                uint block = i / stride;
                uint offset = i % stride;
                uint idx_a = block * (stride * 2) + offset;
                uint idx_b = idx_a + stride;

                float a = vec[idx_a];
                float b = vec[idx_b];
                vec[idx_a] = a + b;
                vec[idx_b] = a - b;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Normalise: 1/√d (WHT is orthogonal with this scaling)
        float inv_sqrt_d = rsqrt(float(head_dim));
        for (uint i = tid; i < head_dim; i += tg_size) {
            vec[i] *= inv_sqrt_d;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Phase 2: SO(4) block rotations ---
    // Each block is a 4×4 orthogonal matrix applied to 4 consecutive elements.
    // d/4 independent blocks per head, fully parallel.
    uint num_blocks = head_dim / 4;

    // SO(4) block matrices: [H, num_blocks, 4, 4]
    // For inverse: we use the transposed blocks (passed pre-transposed by host)
    constant float* head_blocks = so4_blocks + gid * num_blocks * 16;

    for (uint b = tid; b < num_blocks; b += tg_size) {
        uint base = b * 4;
        constant float* M = head_blocks + b * 16;  // 4×4 row-major

        float x0 = vec[base + 0];
        float x1 = vec[base + 1];
        float x2 = vec[base + 2];
        float x3 = vec[base + 3];

        vec[base + 0] = M[ 0]*x0 + M[ 1]*x1 + M[ 2]*x2 + M[ 3]*x3;
        vec[base + 1] = M[ 4]*x0 + M[ 5]*x1 + M[ 6]*x2 + M[ 7]*x3;
        vec[base + 2] = M[ 8]*x0 + M[ 9]*x1 + M[10]*x2 + M[11]*x3;
        vec[base + 3] = M[12]*x0 + M[13]*x1 + M[14]*x2 + M[15]*x3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Write back to global memory ---
    for (uint i = tid; i < head_dim; i += tg_size) {
        head_data[i] = vec[i];
    }
}

// ============================================================
// Utility: pack_indices_3bit
// ============================================================
//
// Packs 8-bit index array into 3-bit packed uint32_t.
// Each output word holds 8 values in bits 0..23.
//
// Grid:  total_packed_words
// Group: 1 (simple, could be optimised)

kernel void pack_indices_3bit(
    device const uchar*  indices    [[buffer(0)]],   // [total_values]
    device uint*          packed    [[buffer(1)]],   // [total_values / 8]
    constant uint&        total_values [[buffer(2)]],

    uint gid [[thread_position_in_grid]]
) {
    uint base = gid * VALUES_PER_WORD;
    if (base >= total_values) return;

    uint word = 0;
    for (uint i = 0; i < VALUES_PER_WORD && (base + i) < total_values; ++i) {
        word |= (uint(indices[base + i]) & VALUE_MASK) << (i * BITS_PER_VALUE);
    }
    packed[gid] = word;
}
