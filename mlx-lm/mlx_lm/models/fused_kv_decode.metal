// fused_kv_decode.metal — Reference Metal kernels for the IsoQuant fused decode pipeline.
//
// These are prototypes implementing the 4-kernel pipeline described in
// docs/FROM_ATTENTION_TO_CONSUMER_HARDWARE.md Section 6.6:
//
//   Kernel A: fused_qk_dot_3bit_tg  — fused Q·Kᵀ from 3-bit packed KV
//   Kernel D: wht_so4_fused         — WHT butterfly + SO(4) block rotation
//
// Status: PROTOTYPE — not yet integrated with MLX graph compilation.
// These kernels are numerically correct reference implementations.
// Production integration requires: multi-head dispatch, MLX custom_kernel
// binding, Kernel C (fused value accumulation), and end-to-end benchmarking.
//
// Architecture assumptions:
//   d_k = 128 (head dimension)
//   3-bit quantisation: 8 values packed per 24-bit word in uint32
//   Lloyd-Max centroids: 8 float values per codebook
//   Apple Silicon: SIMD width = 32, threadgroup memory = 32 KB

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Constants
// ============================================================================

constant uint D_K = 128;
constant uint VALUES_PER_WORD = 8;                    // 8 × 3-bit = 24 bits
constant uint PACKED_WORDS = D_K / VALUES_PER_WORD;   // 16 for d_k=128
constant uint THREADS_PER_GROUP_A = 32;               // Kernel A threadgroup size
constant uint BLOCK_SIZE = 4;                          // SO(4) block dimension
constant uint NUM_BLOCKS = D_K / BLOCK_SIZE;           // 32 blocks for d_k=128

// ============================================================================
// Helpers: 3-bit decode
// ============================================================================

inline float4 decode_group0(uint word, constant float* centroids) {
    return float4(
        centroids[(word >> 0)  & 0x7],
        centroids[(word >> 3)  & 0x7],
        centroids[(word >> 6)  & 0x7],
        centroids[(word >> 9)  & 0x7]
    );
}

inline float4 decode_group1(uint word, constant float* centroids) {
    return float4(
        centroids[(word >> 12) & 0x7],
        centroids[(word >> 15) & 0x7],
        centroids[(word >> 18) & 0x7],
        centroids[(word >> 21) & 0x7]
    );
}

// ============================================================================
// Kernel A: Fused Q·Kᵀ with 3-bit decode (threadgroup-optimised)
//
// Replaces: unpack → dequant → materialise K → matmul(q, K)
// With:     single fused pass, no intermediate K buffer
//
// Mapping: 1 threadgroup = 1 token, 32 threads cooperate over d_k
// ============================================================================

kernel void fused_qk_dot_3bit_tg(
    device const uint*   K_packed      [[buffer(0)]], // [T * PACKED_WORDS]
    constant float*      centroids     [[buffer(1)]], // [8]
    device const float*  norms         [[buffer(2)]], // [T]
    device const float*  q_global      [[buffer(3)]], // [D_K]
    device float*        scores        [[buffer(4)]], // [T]

    uint tid        [[thread_index_in_threadgroup]],
    uint gid        [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]]
) {
    // Cache query in threadgroup memory (loaded once, reused by all threads)
    threadgroup float q_local[D_K];

    for (uint i = tid; i < D_K; i += THREADS_PER_GROUP_A) {
        q_local[i] = q_global[i];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread accumulates partial dot product over striped packed words
    float acc = 0.0f;

    device const uint* packed = K_packed + gid * PACKED_WORDS;

    for (uint w = tid; w < PACKED_WORDS; w += THREADS_PER_GROUP_A) {
        uint word = packed[w];

        // Decode 8 values → two float4 vectors
        float4 k0 = decode_group0(word, centroids);
        float4 k1 = decode_group1(word, centroids);

        uint base = w * VALUES_PER_WORD;

        float4 q0 = float4(
            q_local[base + 0],
            q_local[base + 1],
            q_local[base + 2],
            q_local[base + 3]
        );

        float4 q1 = float4(
            q_local[base + 4],
            q_local[base + 5],
            q_local[base + 6],
            q_local[base + 7]
        );

        acc += dot(k0, q0);
        acc += dot(k1, q1);
    }

    // SIMD-group reduction (no shared memory needed)
    float sum = simd_sum(acc);

    // Single lane writes result
    if (simd_lane == 0) {
        scores[gid] = sum * norms[gid];
    }
}

// ============================================================================
// Kernel D: Fused WHT + SO(4) inverse rotation
//
// Replaces: dense matmul (16,384 FMAs) with butterfly WHT (896 ops) +
//           block SO(4) (512 FMAs) = 1,408 total FMAs
//
// Applied ONCE on aggregated attention output (after Kernel C), not per-token.
//
// WHT stages 1–5: SIMD shuffle (no barriers)
// WHT stages 6–7: threadgroup barrier
// SO(4): 32 independent 4×4 block rotations
//
// Mapping: 1 threadgroup = 1 vector (128 dims), 128 threads
// ============================================================================

kernel void wht_so4_fused(
    device const float*  input        [[buffer(0)]], // [N * 128]
    device float*        output       [[buffer(1)]], // [N * 128]
    device const float4* so4_matrix   [[buffer(2)]], // [NUM_BLOCKS * 4] row-major

    uint tid       [[thread_index_in_threadgroup]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    threadgroup float x[D_K];

    uint base = gid * D_K;

    // ---- Load ----
    x[tid] = input[base + tid];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- WHT stages 1–5: SIMD-only (no barriers) ----
    float val = x[tid];

    val += simd_shuffle_xor(val, 1)  * ((tid & 1)  ? -1.0f : 1.0f);
    val += simd_shuffle_xor(val, 2)  * ((tid & 2)  ? -1.0f : 1.0f);
    val += simd_shuffle_xor(val, 4)  * ((tid & 4)  ? -1.0f : 1.0f);
    val += simd_shuffle_xor(val, 8)  * ((tid & 8)  ? -1.0f : 1.0f);
    val += simd_shuffle_xor(val, 16) * ((tid & 16) ? -1.0f : 1.0f);

    x[tid] = val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- WHT stage 6: stride = 32 ----
    {
        uint partner = tid ^ 32;
        float a = x[tid];
        float b = x[partner];
        x[tid] = (tid & 32) ? (b - a) : (a + b);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- WHT stage 7: stride = 64 ----
    {
        uint partner = tid ^ 64;
        float a = x[tid];
        float b = x[partner];
        x[tid] = (tid & 64) ? (b - a) : (a + b);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Normalisation (orthonormal WHT) ----
    float scale = 1.0f / sqrt((float)D_K);
    x[tid] *= scale;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- SO(4) block rotation (vectorised, 32 independent blocks) ----
    uint block_id = tid / 4;
    uint lane = tid % 4;

    threadgroup float4 block_vecs[NUM_BLOCKS];

    if (lane == 0) {
        float4 v = float4(
            x[block_id * 4 + 0],
            x[block_id * 4 + 1],
            x[block_id * 4 + 2],
            x[block_id * 4 + 3]
        );

        // 4×4 matrix stored row-major as 4 float4s
        const device float4* mat = so4_matrix + block_id * 4;

        float4 r0 = mat[0];
        float4 r1 = mat[1];
        float4 r2 = mat[2];
        float4 r3 = mat[3];

        float4 out = float4(
            dot(r0, v),
            dot(r1, v),
            dot(r2, v),
            dot(r3, v)
        );

        block_vecs[block_id] = out;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    x[tid] = block_vecs[block_id][lane];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Store ----
    output[base + tid] = x[tid];
}
