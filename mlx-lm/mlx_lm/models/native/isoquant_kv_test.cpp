// isoquant_kv_test.cpp — Correctness test: Metal kernels vs CPU reference
//
// Generates random compressed KV data, runs both Metal and CPU paths,
// compares element-wise.  Exits 0 on pass, 1 on failure.
//
// Build via CMake (see CMakeLists.txt), then run:
//   ./isoquant_test

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "isoquant_kv_runtime.hpp"
#include "isoquant_kv_reference.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// ============================================================
// Test configuration
// ============================================================

static constexpr uint32_t NUM_HEADS  = 8;
static constexpr uint32_t HEAD_DIM   = 128;
static constexpr uint32_t SEQ_LEN    = 64;
static constexpr uint32_t NUM_LEVELS = 8;       // 3-bit
static const float        SCALE      = 1.0f / std::sqrt(float(HEAD_DIM));
static constexpr float    EPSILON    = 1e-4f;   // max absolute error

// ============================================================
// Helpers
// ============================================================

static bool approx_eq(float a, float b, float eps = EPSILON) {
    return std::fabs(a - b) <= eps;
}

static float max_abs_error(const float* a, const float* b, size_t n) {
    float max_err = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float err = std::fabs(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

static MTL::Buffer* make_shared_buffer(MTL::Device* device, size_t bytes, const void* data = nullptr) {
    auto buf = device->newBuffer(bytes, MTL::ResourceStorageModeShared);
    if (!buf) return nullptr;
    if (data) std::memcpy(buf->contents(), data, bytes);
    return buf;
}

// Generate random Lloyd-Max centroids (sorted, typical shape)
static std::vector<float> make_centroids(std::mt19937& rng) {
    std::vector<float> c(NUM_LEVELS);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : c) v = dist(rng);
    std::sort(c.begin(), c.end());
    return c;
}

// Generate random uint8 indices in [0, NUM_LEVELS)
static std::vector<uint8_t> make_indices(std::mt19937& rng, size_t count) {
    std::vector<uint8_t> idx(count);
    std::uniform_int_distribution<int> dist(0, NUM_LEVELS - 1);
    for (auto& v : idx) v = static_cast<uint8_t>(dist(rng));
    return idx;
}

// Generate random norms > 0
static std::vector<float> make_norms(std::mt19937& rng, size_t count) {
    std::vector<float> n(count);
    std::uniform_real_distribution<float> dist(0.1f, 5.0f);
    for (auto& v : n) v = dist(rng);
    return n;
}

// Generate random query vector
static std::vector<float> make_query(std::mt19937& rng) {
    std::vector<float> q(NUM_HEADS * HEAD_DIM);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : q) v = dist(rng);
    return q;
}

// Generate random SO(4) block matrices (orthogonal-ish for testing)
static std::vector<float> make_so4_blocks(std::mt19937& rng) {
    uint32_t num_blocks = HEAD_DIM / 4;
    size_t total = NUM_HEADS * num_blocks * 16;
    std::vector<float> blocks(total);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Generate random 4x4 matrices (not truly orthogonal,
    // but sufficient for testing the pipeline)
    for (size_t i = 0; i < total; i += 16) {
        // Simple: use identity + small perturbation
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                blocks[i + r * 4 + c] = (r == c) ? 1.0f : dist(rng) * 0.1f;
    }
    return blocks;
}

// ============================================================
// Tests
// ============================================================

static bool test_pack_indices() {
    printf("Test: pack_indices_3bit... ");

    std::mt19937 rng(42);
    uint32_t total = NUM_HEADS * SEQ_LEN * HEAD_DIM;
    auto indices = make_indices(rng, total);

    // CPU reference
    auto packed_ref = isoquant::ref::pack_indices_3bit(indices.data(), total);

    // Verify unpacking matches
    for (uint32_t i = 0; i < total; ++i) {
        uint32_t word_idx = i / 8;
        uint32_t bit_pos  = (i % 8) * 3;
        uint8_t recovered = (packed_ref[word_idx] >> bit_pos) & 0x7;
        if (recovered != indices[i]) {
            printf("FAIL at index %u: expected %u, got %u\n", i, indices[i], recovered);
            return false;
        }
    }

    printf("PASS (%u values)\n", total);
    return true;
}

static bool test_fused_qk_dot() {
    printf("Test: fused_qk_dot (CPU reference)... ");

    std::mt19937 rng(123);
    auto centroids = make_centroids(rng);
    auto k_indices = make_indices(rng, NUM_HEADS * SEQ_LEN * HEAD_DIM);
    auto k_norms   = make_norms(rng, NUM_HEADS * SEQ_LEN);
    auto query     = make_query(rng);

    // Pack indices
    auto packed = isoquant::ref::pack_indices_3bit(
        k_indices.data(), NUM_HEADS * SEQ_LEN * HEAD_DIM
    );

    // Run reference
    auto scores = isoquant::ref::fused_qk_dot(
        packed.data(), centroids.data(), k_norms.data(),
        query.data(), NUM_HEADS, SEQ_LEN, HEAD_DIM
    );

    // Manual verification for first score
    float manual_dot = 0.0f;
    for (uint32_t d = 0; d < HEAD_DIM; ++d) {
        float k_val = centroids[k_indices[d]];
        manual_dot += query[d] * k_val;
    }
    manual_dot *= k_norms[0];

    if (!approx_eq(scores[0], manual_dot, 1e-3f)) {
        printf("FAIL: scores[0]=%f, manual=%f\n", scores[0], manual_dot);
        return false;
    }

    printf("PASS (H=%u, T=%u, D=%u)\n", NUM_HEADS, SEQ_LEN, HEAD_DIM);
    return true;
}

static bool test_softmax() {
    printf("Test: softmax (CPU reference)... ");

    std::mt19937 rng(456);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> scores(NUM_HEADS * SEQ_LEN);
    for (auto& v : scores) v = dist(rng);

    isoquant::ref::softmax_inplace(scores.data(), SEQ_LEN, NUM_HEADS);

    // Check sums to 1 for each head
    for (uint32_t h = 0; h < NUM_HEADS; ++h) {
        float sum = 0.0f;
        for (uint32_t t = 0; t < SEQ_LEN; ++t) {
            float v = scores[h * SEQ_LEN + t];
            if (v < 0.0f || v > 1.0f) {
                printf("FAIL: softmax out of range at [%u,%u]: %f\n", h, t, v);
                return false;
            }
            sum += v;
        }
        if (!approx_eq(sum, 1.0f, 1e-5f)) {
            printf("FAIL: head %u sum=%f (expected 1.0)\n", h, sum);
            return false;
        }
    }

    printf("PASS\n");
    return true;
}

static bool test_end_to_end_reference() {
    printf("Test: end-to-end reference pipeline... ");

    std::mt19937 rng(789);
    auto centroids  = make_centroids(rng);
    auto k_indices  = make_indices(rng, NUM_HEADS * SEQ_LEN * HEAD_DIM);
    auto v_indices  = make_indices(rng, NUM_HEADS * SEQ_LEN * HEAD_DIM);
    auto k_norms    = make_norms(rng, NUM_HEADS * SEQ_LEN);
    auto v_norms    = make_norms(rng, NUM_HEADS * SEQ_LEN);
    auto query      = make_query(rng);
    auto so4_blocks = make_so4_blocks(rng);

    auto k_packed = isoquant::ref::pack_indices_3bit(
        k_indices.data(), NUM_HEADS * SEQ_LEN * HEAD_DIM
    );
    auto v_packed = isoquant::ref::pack_indices_3bit(
        v_indices.data(), NUM_HEADS * SEQ_LEN * HEAD_DIM
    );

    auto output = isoquant::ref::fused_attention_reference(
        k_packed.data(), v_packed.data(),
        centroids.data(), k_norms.data(), v_norms.data(),
        query.data(), SCALE, so4_blocks.data(),
        NUM_HEADS, SEQ_LEN, HEAD_DIM,
        /*use_hadamard=*/true
    );

    // Basic sanity: output should be finite and non-zero
    bool all_finite = true;
    bool any_nonzero = false;
    for (size_t i = 0; i < output.size(); ++i) {
        if (!std::isfinite(output[i])) { all_finite = false; break; }
        if (output[i] != 0.0f) any_nonzero = true;
    }

    if (!all_finite) {
        printf("FAIL: non-finite values in output\n");
        return false;
    }
    if (!any_nonzero) {
        printf("FAIL: all-zero output\n");
        return false;
    }

    printf("PASS (output size=%zu, non-zero, finite)\n", output.size());
    return true;
}

static bool test_metal_vs_reference() {
    printf("Test: Metal kernels vs CPU reference... ");

    // Get Metal device
    auto device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        printf("SKIP (no Metal device)\n");
        return true;
    }

    isoquant::IsoQuantKVRuntime runtime(device);
    if (!runtime.load_pipeline("isoquant_kv_kernels.metallib")) {
        printf("SKIP (metallib not found — run cmake first)\n");
        device->release();
        return true;
    }

    std::mt19937 rng(1337);
    auto centroids  = make_centroids(rng);
    auto k_indices  = make_indices(rng, NUM_HEADS * SEQ_LEN * HEAD_DIM);
    auto v_indices  = make_indices(rng, NUM_HEADS * SEQ_LEN * HEAD_DIM);
    auto k_norms    = make_norms(rng, NUM_HEADS * SEQ_LEN);
    auto v_norms    = make_norms(rng, NUM_HEADS * SEQ_LEN);
    auto query      = make_query(rng);
    auto so4_blocks = make_so4_blocks(rng);

    // CPU reference
    auto k_packed = isoquant::ref::pack_indices_3bit(
        k_indices.data(), NUM_HEADS * SEQ_LEN * HEAD_DIM
    );
    auto v_packed = isoquant::ref::pack_indices_3bit(
        v_indices.data(), NUM_HEADS * SEQ_LEN * HEAD_DIM
    );

    auto ref_output = isoquant::ref::fused_attention_reference(
        k_packed.data(), v_packed.data(),
        centroids.data(), k_norms.data(), v_norms.data(),
        query.data(), SCALE, so4_blocks.data(),
        NUM_HEADS, SEQ_LEN, HEAD_DIM,
        /*use_hadamard=*/true
    );

    // Metal path
    auto cache = runtime.create_cache(NUM_HEADS, HEAD_DIM, SEQ_LEN * 2);
    runtime.set_centroids(cache, centroids.data(), NUM_LEVELS);
    runtime.set_so4_blocks(cache, so4_blocks.data(), /*use_hadamard=*/true);
    runtime.append_compressed(
        cache,
        k_indices.data(), k_norms.data(),
        v_indices.data(), v_norms.data(),
        SEQ_LEN
    );

    std::vector<float> metal_output(NUM_HEADS * HEAD_DIM);
    runtime.fused_attention(
        cache, query.data(), SCALE,
        metal_output.data()
    );

    // Compare
    float err = max_abs_error(
        ref_output.data(), metal_output.data(), NUM_HEADS * HEAD_DIM
    );

    runtime.release_cache(cache);
    device->release();

    if (err > EPSILON) {
        printf("FAIL: max abs error = %e (threshold %e)\n", err, EPSILON);
        auto q_scaled = query;
        for (float& v : q_scaled) v *= SCALE;

        auto scores_ref = isoquant::ref::fused_qk_dot(
            k_packed.data(), centroids.data(), k_norms.data(),
            q_scaled.data(), NUM_HEADS, SEQ_LEN, HEAD_DIM
        );

        auto q_buf = make_shared_buffer(device, q_scaled.size() * sizeof(float), q_scaled.data());
        auto scores_buf = make_shared_buffer(device, scores_ref.size() * sizeof(float));
        auto output_rot_buf = make_shared_buffer(device, metal_output.size() * sizeof(float));

        if (q_buf && scores_buf && output_rot_buf) {
            runtime.dispatch_fused_qk_dot(cache, q_buf, scores_buf);
            auto* scores_gpu = static_cast<float*>(scores_buf->contents());
            float qk_err = max_abs_error(scores_ref.data(), scores_gpu, scores_ref.size());
            printf("  stage A (QK) max abs error = %e\n", qk_err);

            auto scores_softmax_ref = scores_ref;
            isoquant::ref::softmax_inplace(scores_softmax_ref.data(), SEQ_LEN, NUM_HEADS);
            runtime.dispatch_softmax(scores_buf, SEQ_LEN, NUM_HEADS);
            float softmax_err = max_abs_error(
                scores_softmax_ref.data(), scores_gpu, scores_softmax_ref.size()
            );
            printf("  stage B (softmax) max abs error = %e\n", softmax_err);

            auto output_rot_ref = isoquant::ref::fused_value_accum(
                v_packed.data(), centroids.data(), v_norms.data(),
                scores_softmax_ref.data(), NUM_HEADS, SEQ_LEN, HEAD_DIM
            );
            runtime.dispatch_fused_value_accum(cache, scores_buf, output_rot_buf);
            auto* output_rot_gpu = static_cast<float*>(output_rot_buf->contents());
            float value_err = max_abs_error(
                output_rot_ref.data(), output_rot_gpu, output_rot_ref.size()
            );
            printf("  stage C (value accum) max abs error = %e\n", value_err);

            runtime.dispatch_inverse_rotation(cache, output_rot_buf);
            float final_seq_err = max_abs_error(
                ref_output.data(), output_rot_gpu, ref_output.size()
            );
            printf("  stage D / sequential runtime max abs error = %e\n", final_seq_err);
        }

        if (output_rot_buf) output_rot_buf->release();
        if (scores_buf) scores_buf->release();
        if (q_buf) q_buf->release();

        // Print first few mismatches
        for (size_t i = 0; i < ref_output.size() && i < 10; ++i) {
            if (!approx_eq(ref_output[i], metal_output[i])) {
                printf("  [%zu] ref=%f metal=%f diff=%e\n",
                       i, ref_output[i], metal_output[i],
                       std::fabs(ref_output[i] - metal_output[i]));
            }
        }
        return false;
    }

    printf("PASS (max error=%e, threshold=%e)\n", err, EPSILON);
    return true;
}

// ============================================================
// Main
// ============================================================

int main() {
    printf("=== IsoQuant Native KV Decode — Correctness Tests ===\n\n");

    int failures = 0;

    if (!test_pack_indices())        ++failures;
    if (!test_fused_qk_dot())        ++failures;
    if (!test_softmax())             ++failures;
    if (!test_end_to_end_reference()) ++failures;
    if (!test_metal_vs_reference())  ++failures;

    printf("\n--- %d test(s) failed ---\n", failures);
    return failures > 0 ? 1 : 0;
}
