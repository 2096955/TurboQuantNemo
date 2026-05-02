// isoquant_kv_reference.cpp — CPU reference implementation for correctness testing
//
// Scalar C++ implementations of every kernel in the fused pipeline.
// These are intentionally simple and unoptimised — they exist solely to
// validate the Metal kernels produce bit-identical (or within-epsilon) results.
//
// Usage:
//   #include "isoquant_kv_reference.hpp"
//   auto scores = ref::fused_qk_dot(packed_k, centroids, norms, query, H, T, D);
//   ref::softmax_inplace(scores.data(), T, H);
//   auto output_rot = ref::fused_value_accum(packed_v, centroids, v_norms,
//                                            scores.data(), H, T, D);
//   ref::inverse_rotation_wht_so4(output_rot.data(), so4_blocks, D, H, use_had);

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace isoquant { namespace ref {

// ============================================================
// 3-bit decode
// ============================================================

static inline float decode_one(uint32_t word, int pos, const float* centroids) {
    return centroids[(word >> (pos * 3)) & 0x7];
}

static void unpack_3bit(
    const uint32_t* packed,
    const float* centroids,
    float* out,
    uint32_t dim
) {
    uint32_t pw = dim / 8;
    for (uint32_t w = 0; w < pw; ++w) {
        uint32_t word = packed[w];
        for (int i = 0; i < 8; ++i) {
            out[w * 8 + i] = decode_one(word, i, centroids);
        }
    }
}

// ============================================================
// Kernel A reference: fused Q·Kᵀ
// ============================================================

std::vector<float> fused_qk_dot(
    const uint32_t* K_packed,   // [H, T, packed_words]
    const float* centroids,     // [num_levels]
    const float* norms,         // [H, T]
    const float* query,         // [H, D]
    uint32_t H,
    uint32_t T,
    uint32_t D
) {
    uint32_t pw = D / 8;
    std::vector<float> scores(H * T);
    std::vector<float> k_vec(D);

    for (uint32_t h = 0; h < H; ++h) {
        const float* q = query + h * D;
        for (uint32_t t = 0; t < T; ++t) {
            const uint32_t* packed = K_packed + (h * T + t) * pw;
            unpack_3bit(packed, centroids, k_vec.data(), D);

            float dot = 0.0f;
            for (uint32_t d = 0; d < D; ++d) {
                dot += q[d] * k_vec[d];
            }

            scores[h * T + t] = dot * norms[h * T + t];
        }
    }
    return scores;
}

// ============================================================
// Kernel B reference: softmax (in-place)
// ============================================================

void softmax_inplace(float* scores, uint32_t T, uint32_t H) {
    for (uint32_t h = 0; h < H; ++h) {
        float* row = scores + h * T;

        // Find max
        float max_val = -std::numeric_limits<float>::infinity();
        for (uint32_t t = 0; t < T; ++t) {
            if (row[t] > max_val) max_val = row[t];
        }

        // Exp + sum
        float sum = 0.0f;
        for (uint32_t t = 0; t < T; ++t) {
            row[t] = std::exp(row[t] - max_val);
            sum += row[t];
        }

        // Normalise
        float inv_sum = 1.0f / sum;
        for (uint32_t t = 0; t < T; ++t) {
            row[t] *= inv_sum;
        }
    }
}

// ============================================================
// Kernel C reference: fused value accumulation
// ============================================================

std::vector<float> fused_value_accum(
    const uint32_t* V_packed,       // [H, T, packed_words]
    const float* centroids,         // [num_levels]
    const float* norms,             // [H, T]
    const float* attn_weights,      // [H, T]
    uint32_t H,
    uint32_t T,
    uint32_t D
) {
    uint32_t pw = D / 8;
    std::vector<float> output(H * D, 0.0f);
    std::vector<float> v_vec(D);

    for (uint32_t h = 0; h < H; ++h) {
        float* out = output.data() + h * D;

        for (uint32_t t = 0; t < T; ++t) {
            float w = attn_weights[h * T + t] * norms[h * T + t];
            if (w == 0.0f) continue;

            const uint32_t* packed = V_packed + (h * T + t) * pw;
            unpack_3bit(packed, centroids, v_vec.data(), D);

            for (uint32_t d = 0; d < D; ++d) {
                out[d] += w * v_vec[d];
            }
        }
    }
    return output;
}

// ============================================================
// Kernel D reference: inverse rotation (WHT + SO(4))
// ============================================================

// In-place Walsh-Hadamard Transform (unnormalised)
static void wht_inplace(float* vec, uint32_t dim) {
    for (uint32_t stride = 1; stride < dim; stride <<= 1) {
        for (uint32_t i = 0; i < dim / 2; ++i) {
            uint32_t block = i / stride;
            uint32_t offset = i % stride;
            uint32_t idx_a = block * (stride * 2) + offset;
            uint32_t idx_b = idx_a + stride;

            float a = vec[idx_a];
            float b = vec[idx_b];
            vec[idx_a] = a + b;
            vec[idx_b] = a - b;
        }
    }

    // Normalise for orthogonality
    float inv_sqrt = 1.0f / std::sqrt((float)dim);
    for (uint32_t i = 0; i < dim; ++i) {
        vec[i] *= inv_sqrt;
    }
}

// Apply 4×4 block rotation to 4 consecutive elements
static void so4_block_inplace(float* vec, const float* M) {
    float x0 = vec[0], x1 = vec[1], x2 = vec[2], x3 = vec[3];
    vec[0] = M[ 0]*x0 + M[ 1]*x1 + M[ 2]*x2 + M[ 3]*x3;
    vec[1] = M[ 4]*x0 + M[ 5]*x1 + M[ 6]*x2 + M[ 7]*x3;
    vec[2] = M[ 8]*x0 + M[ 9]*x1 + M[10]*x2 + M[11]*x3;
    vec[3] = M[12]*x0 + M[13]*x1 + M[14]*x2 + M[15]*x3;
}

void inverse_rotation_wht_so4(
    float* data,                 // [H, D] — in-place
    const float* so4_blocks,     // [H, D/4, 4, 4] row-major (pre-transposed)
    uint32_t D,
    uint32_t H,
    bool use_hadamard
) {
    uint32_t num_blocks = D / 4;

    for (uint32_t h = 0; h < H; ++h) {
        float* vec = data + h * D;

        // Phase 1: WHT
        if (use_hadamard) {
            wht_inplace(vec, D);
        }

        // Phase 2: SO(4) blocks
        const float* head_blocks = so4_blocks + h * num_blocks * 16;
        for (uint32_t b = 0; b < num_blocks; ++b) {
            so4_block_inplace(vec + b * 4, head_blocks + b * 16);
        }
    }
}

// ============================================================
// Pack indices (CPU reference)
// ============================================================

std::vector<uint32_t> pack_indices_3bit(
    const uint8_t* indices,
    uint32_t total_values
) {
    uint32_t packed_count = (total_values + 7) / 8;
    std::vector<uint32_t> packed(packed_count, 0);

    for (uint32_t i = 0; i < total_values; ++i) {
        uint32_t word_idx = i / 8;
        uint32_t bit_pos  = (i % 8) * 3;
        packed[word_idx] |= ((uint32_t)(indices[i] & 0x7)) << bit_pos;
    }
    return packed;
}

// ============================================================
// End-to-end reference pipeline
// ============================================================

std::vector<float> fused_attention_reference(
    const uint32_t* K_packed,
    const uint32_t* V_packed,
    const float* centroids,
    const float* k_norms,
    const float* v_norms,
    const float* query,          // [H, D] — already in rotated space
    float scale,
    const float* so4_blocks,     // [H, D/4, 4, 4]
    uint32_t H,
    uint32_t T,
    uint32_t D,
    bool use_hadamard
) {
    // Scale query
    std::vector<float> q_scaled(H * D);
    for (uint32_t i = 0; i < H * D; ++i) {
        q_scaled[i] = query[i] * scale;
    }

    // Kernel A
    auto scores = fused_qk_dot(K_packed, centroids, k_norms, q_scaled.data(), H, T, D);

    // Kernel B
    softmax_inplace(scores.data(), T, H);

    // Kernel C
    auto output = fused_value_accum(V_packed, centroids, v_norms, scores.data(), H, T, D);

    // Kernel D
    inverse_rotation_wht_so4(output.data(), so4_blocks, D, H, use_hadamard);

    return output;
}

}}  // namespace isoquant::ref
