// isoquant_kv_reference.hpp — CPU reference interface for correctness testing
#pragma once

#include <cstdint>
#include <vector>

namespace isoquant { namespace ref {

// Kernel A: fused Q·Kᵀ from packed 3-bit K
std::vector<float> fused_qk_dot(
    const uint32_t* K_packed,   // [H, T, packed_words]
    const float* centroids,     // [num_levels]
    const float* norms,         // [H, T]
    const float* query,         // [H, D]
    uint32_t H, uint32_t T, uint32_t D
);

// Kernel B: in-place softmax
void softmax_inplace(float* scores, uint32_t T, uint32_t H);

// Kernel C: fused weighted V sum in rotated space
std::vector<float> fused_value_accum(
    const uint32_t* V_packed,
    const float* centroids,
    const float* norms,
    const float* attn_weights,
    uint32_t H, uint32_t T, uint32_t D
);

// Kernel D: inverse rotation (WHT + SO(4), in-place)
void inverse_rotation_wht_so4(
    float* data,                 // [H, D]
    const float* so4_blocks,     // [H, D/4, 4, 4]
    uint32_t D, uint32_t H, bool use_hadamard
);

// Pack uint8 indices to 3-bit packed uint32_t
std::vector<uint32_t> pack_indices_3bit(const uint8_t* indices, uint32_t total_values);

// End-to-end reference pipeline
std::vector<float> fused_attention_reference(
    const uint32_t* K_packed,
    const uint32_t* V_packed,
    const float* centroids,
    const float* k_norms,
    const float* v_norms,
    const float* query,
    float scale,
    const float* so4_blocks,
    uint32_t H, uint32_t T, uint32_t D,
    bool use_hadamard
);

}}  // namespace isoquant::ref
