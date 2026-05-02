#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "isoquant_kv_runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

struct Config {
    uint32_t num_heads;
    uint32_t head_dim;
    uint32_t seq_len;
    const char* label;
};

struct Timings {
    double fused_host_ms = 0.0;
    double fused_gpu_ms = 0.0;
    double qk_ms = 0.0;
    double softmax_ms = 0.0;
    double value_ms = 0.0;
    double value_gpu_ms = -1.0;
    double inverse_ms = 0.0;
};

struct Result {
    Config config;
    Timings timings;
    isoquant::ValueAccumStrategy resolved_strategy = isoquant::ValueAccumStrategy::Auto;
};

struct Options {
    std::string json_out;
    std::string csv_out;
    int warmup_iters = 8;
    int bench_iters = 100;
    isoquant::ValueAccumStrategy strategy = isoquant::ValueAccumStrategy::Auto;
};

MTL::Buffer* make_shared_buffer(MTL::Device* device, size_t bytes, const void* data = nullptr) {
    auto* buf = device->newBuffer(bytes, MTL::ResourceStorageModeShared);
    if (!buf) {
        return nullptr;
    }
    if (data) {
        std::memcpy(buf->contents(), data, bytes);
    }
    return buf;
}

std::vector<float> make_centroids(std::mt19937& rng) {
    std::vector<float> c(8);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : c) {
        v = dist(rng);
    }
    std::sort(c.begin(), c.end());
    return c;
}

std::vector<uint8_t> make_indices(std::mt19937& rng, size_t count) {
    std::vector<uint8_t> idx(count);
    std::uniform_int_distribution<int> dist(0, 7);
    for (auto& v : idx) {
        v = static_cast<uint8_t>(dist(rng));
    }
    return idx;
}

std::vector<float> make_norms(std::mt19937& rng, size_t count) {
    std::vector<float> values(count);
    std::uniform_real_distribution<float> dist(0.1f, 5.0f);
    for (auto& v : values) {
        v = dist(rng);
    }
    return values;
}

std::vector<float> make_query(std::mt19937& rng, uint32_t num_heads, uint32_t head_dim) {
    std::vector<float> q(static_cast<size_t>(num_heads) * head_dim);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : q) {
        v = dist(rng);
    }
    return q;
}

std::vector<float> make_so4_blocks(std::mt19937& rng, uint32_t num_heads, uint32_t head_dim) {
    const uint32_t num_blocks = head_dim / 4;
    std::vector<float> blocks(static_cast<size_t>(num_heads) * num_blocks * 16);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < blocks.size(); i += 16) {
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                blocks[i + r * 4 + c] = (r == c) ? 1.0f : dist(rng) * 0.1f;
            }
        }
    }
    return blocks;
}

double elapsed_ms(const clock_type::time_point& start, const clock_type::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

isoquant::ValueAccumStrategy parse_strategy(const std::string& value) {
    if (value == "auto") return isoquant::ValueAccumStrategy::Auto;
    if (value == "word") return isoquant::ValueAccumStrategy::WordStriped;
    if (value == "dim") return isoquant::ValueAccumStrategy::DimParallel;
    throw std::runtime_error("Unknown --value-kernel strategy: " + value);
}

void print_help(const char* argv0) {
    std::printf(
        "Usage: %s [--value-kernel auto|word|dim] [--warmup-iters N] [--bench-iters N] "
        "[--json-out PATH] [--csv-out PATH]\n",
        argv0
    );
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + flag);
            }
            return argv[++i];
        };

        if (arg == "--help") {
            print_help(argv[0]);
            std::exit(0);
        } else if (arg == "--json-out") {
            options.json_out = require_value("--json-out");
        } else if (arg == "--csv-out") {
            options.csv_out = require_value("--csv-out");
        } else if (arg == "--warmup-iters") {
            options.warmup_iters = std::stoi(require_value("--warmup-iters"));
        } else if (arg == "--bench-iters") {
            options.bench_iters = std::stoi(require_value("--bench-iters"));
        } else if (arg == "--value-kernel") {
            options.strategy = parse_strategy(require_value("--value-kernel"));
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    return options;
}

Result run_config(
    isoquant::IsoQuantKVRuntime& runtime,
    MTL::Device* device,
    const Config& cfg,
    int warmup_iters,
    int bench_iters
) {
    std::mt19937 rng(cfg.num_heads * 1000003u + cfg.seq_len * 101u + cfg.head_dim);

    const size_t total_values = static_cast<size_t>(cfg.num_heads) * cfg.seq_len * cfg.head_dim;
    const size_t norm_count = static_cast<size_t>(cfg.num_heads) * cfg.seq_len;
    const size_t vec_count = static_cast<size_t>(cfg.num_heads) * cfg.head_dim;

    auto centroids = make_centroids(rng);
    auto k_indices = make_indices(rng, total_values);
    auto v_indices = make_indices(rng, total_values);
    auto k_norms = make_norms(rng, norm_count);
    auto v_norms = make_norms(rng, norm_count);
    auto query = make_query(rng, cfg.num_heads, cfg.head_dim);
    auto so4_blocks = make_so4_blocks(rng, cfg.num_heads, cfg.head_dim);

    auto cache = runtime.create_cache(cfg.num_heads, cfg.head_dim, cfg.seq_len * 2);
    runtime.set_centroids(cache, centroids.data(), 8);
    runtime.set_so4_blocks(cache, so4_blocks.data(), true);
    runtime.append_compressed(
        cache,
        k_indices.data(), k_norms.data(),
        v_indices.data(), v_norms.data(),
        cfg.seq_len
    );

    const float scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));
    std::vector<float> host_output(vec_count);

    auto q_scaled = query;
    for (float& v : q_scaled) {
        v *= scale;
    }

    auto* q_buf = make_shared_buffer(device, vec_count * sizeof(float), q_scaled.data());
    auto* scores_buf = make_shared_buffer(device, norm_count * sizeof(float));
    auto* out_buf = make_shared_buffer(device, vec_count * sizeof(float));

    if (!q_buf || !scores_buf || !out_buf) {
        throw std::runtime_error("Failed to allocate benchmark buffers");
    }

    for (int i = 0; i < warmup_iters; ++i) {
        runtime.fused_attention(cache, query.data(), scale, host_output.data());
        runtime.fused_attention_gpu(cache, q_buf, scale, out_buf);
        runtime.dispatch_fused_qk_dot(cache, q_buf, scores_buf);
        runtime.dispatch_softmax(scores_buf, cfg.seq_len, cfg.num_heads);
        runtime.dispatch_fused_value_accum(cache, scores_buf, out_buf);
        runtime.dispatch_inverse_rotation(cache, out_buf);
    }

    Result result;
    result.config = cfg;
    result.resolved_strategy = runtime.resolved_value_accum_strategy(cache);

    auto start = clock_type::now();
    for (int i = 0; i < bench_iters; ++i) {
        runtime.fused_attention(cache, query.data(), scale, host_output.data());
    }
    result.timings.fused_host_ms = elapsed_ms(start, clock_type::now()) / bench_iters;

    start = clock_type::now();
    for (int i = 0; i < bench_iters; ++i) {
        runtime.fused_attention_gpu(cache, q_buf, scale, out_buf);
    }
    result.timings.fused_gpu_ms = elapsed_ms(start, clock_type::now()) / bench_iters;

    double qk_total = 0.0;
    double softmax_total = 0.0;
    double value_total = 0.0;
    double inverse_total = 0.0;
    double value_gpu_total = 0.0;
    int value_gpu_samples = 0;

    for (int i = 0; i < bench_iters; ++i) {
        auto t0 = clock_type::now();
        runtime.dispatch_fused_qk_dot(cache, q_buf, scores_buf);
        auto t1 = clock_type::now();
        runtime.dispatch_softmax(scores_buf, cfg.seq_len, cfg.num_heads);
        auto t2 = clock_type::now();
        runtime.dispatch_fused_value_accum(cache, scores_buf, out_buf);
        auto t3 = clock_type::now();
        runtime.dispatch_inverse_rotation(cache, out_buf);
        auto t4 = clock_type::now();

        qk_total += elapsed_ms(t0, t1);
        softmax_total += elapsed_ms(t1, t2);
        value_total += elapsed_ms(t2, t3);
        inverse_total += elapsed_ms(t3, t4);
    }

    for (int i = 0; i < bench_iters; ++i) {
        runtime.dispatch_fused_qk_dot(cache, q_buf, scores_buf);
        runtime.dispatch_softmax(scores_buf, cfg.seq_len, cfg.num_heads);
        const double gpu_ms = runtime.profile_fused_value_accum_gpu_ms(cache, scores_buf, out_buf);
        if (gpu_ms >= 0.0) {
            value_gpu_total += gpu_ms;
            ++value_gpu_samples;
        }
    }

    result.timings.qk_ms = qk_total / bench_iters;
    result.timings.softmax_ms = softmax_total / bench_iters;
    result.timings.value_ms = value_total / bench_iters;
    result.timings.inverse_ms = inverse_total / bench_iters;
    result.timings.value_gpu_ms = value_gpu_samples > 0
        ? value_gpu_total / value_gpu_samples
        : -1.0;

    out_buf->release();
    scores_buf->release();
    q_buf->release();
    runtime.release_cache(cache);
    return result;
}

void write_csv(const std::string& path, const std::vector<Result>& results) {
    std::ofstream out(path);
    out << "config,num_heads,head_dim,seq_len,value_strategy,fused_host_ms,fused_gpu_ms,"
           "kernel_a_ms,kernel_b_ms,kernel_c_ms,kernel_c_gpu_ms,kernel_d_ms,stage_sum_ms\n";
    for (const auto& result : results) {
        const auto& t = result.timings;
        const double stage_sum = t.qk_ms + t.softmax_ms + t.value_ms + t.inverse_ms;
        out << result.config.label << ','
            << result.config.num_heads << ','
            << result.config.head_dim << ','
            << result.config.seq_len << ','
            << isoquant::IsoQuantKVRuntime::value_accum_strategy_name(result.resolved_strategy) << ','
            << t.fused_host_ms << ','
            << t.fused_gpu_ms << ','
            << t.qk_ms << ','
            << t.softmax_ms << ','
            << t.value_ms << ','
            << t.value_gpu_ms << ','
            << t.inverse_ms << ','
            << stage_sum << '\n';
    }
}

void write_json(
    const std::string& path,
    const std::vector<Result>& results,
    const Options& options,
    const char* device_name
) {
    std::ofstream out(path);
    out << "{\n";
    out << "  \"device\": \"" << device_name << "\",\n";
    out << "  \"value_kernel_requested\": \""
        << isoquant::IsoQuantKVRuntime::value_accum_strategy_name(options.strategy) << "\",\n";
    out << "  \"warmup_iters\": " << options.warmup_iters << ",\n";
    out << "  \"bench_iters\": " << options.bench_iters << ",\n";
    out << "  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        const auto& t = result.timings;
        const double stage_sum = t.qk_ms + t.softmax_ms + t.value_ms + t.inverse_ms;
        out << "    {\n";
        out << "      \"config\": \"" << result.config.label << "\",\n";
        out << "      \"num_heads\": " << result.config.num_heads << ",\n";
        out << "      \"head_dim\": " << result.config.head_dim << ",\n";
        out << "      \"seq_len\": " << result.config.seq_len << ",\n";
        out << "      \"value_strategy\": \""
            << isoquant::IsoQuantKVRuntime::value_accum_strategy_name(result.resolved_strategy) << "\",\n";
        out << "      \"fused_host_ms\": " << t.fused_host_ms << ",\n";
        out << "      \"fused_gpu_ms\": " << t.fused_gpu_ms << ",\n";
        out << "      \"kernel_a_ms\": " << t.qk_ms << ",\n";
        out << "      \"kernel_b_ms\": " << t.softmax_ms << ",\n";
        out << "      \"kernel_c_ms\": " << t.value_ms << ",\n";
        out << "      \"kernel_c_gpu_ms\": " << t.value_gpu_ms << ",\n";
        out << "      \"kernel_d_ms\": " << t.inverse_ms << ",\n";
        out << "      \"stage_sum_ms\": " << stage_sum << "\n";
        out << "    }" << (i + 1 == results.size() ? '\n' : ',') ;
    }
    out << "  ]\n";
    out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
    const Options options = parse_args(argc, argv);

    auto* pool = NS::AutoreleasePool::alloc()->init();
    auto* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::fprintf(stderr, "No Metal device visible in this process.\n");
        pool->release();
        return 1;
    }

    isoquant::IsoQuantKVRuntime runtime(device);
    runtime.set_value_accum_strategy(options.strategy);
    if (!runtime.load_pipeline("isoquant_kv_kernels.metallib")) {
        std::fprintf(stderr, "Failed to load isoquant_kv_kernels.metallib\n");
        device->release();
        pool->release();
        return 1;
    }

    const Config configs[] = {
        {4, 128, 128,  "H=4 T=128 D=128"},
        {4, 128, 512,  "H=4 T=512 D=128"},
        {8, 128, 2048, "H=8 T=2048 D=128"},
    };

    std::vector<Result> results;
    results.reserve(sizeof(configs) / sizeof(configs[0]));

    std::printf(
        "%-16s %-8s %12s %12s %10s %10s %10s %12s %10s %12s\n",
        "config", "kernelC", "fused_host", "fused_gpu", "kernel_A", "kernel_B",
        "kernel_C", "kernel_C_gpu", "kernel_D", "stage_sum"
    );

    for (const auto& cfg : configs) {
        results.push_back(run_config(runtime, device, cfg, options.warmup_iters, options.bench_iters));
        const auto& result = results.back();
        const auto& t = result.timings;
        const double stage_sum = t.qk_ms + t.softmax_ms + t.value_ms + t.inverse_ms;
        std::printf(
            "%-16s %-8s %10.3f ms %10.3f ms %8.3f ms %8.3f ms %8.3f ms %10.3f ms %8.3f ms %10.3f ms\n",
            result.config.label,
            isoquant::IsoQuantKVRuntime::value_accum_strategy_name(result.resolved_strategy),
            t.fused_host_ms,
            t.fused_gpu_ms,
            t.qk_ms,
            t.softmax_ms,
            t.value_ms,
            t.value_gpu_ms,
            t.inverse_ms,
            stage_sum
        );
    }

    if (!options.csv_out.empty()) {
        write_csv(options.csv_out, results);
    }
    if (!options.json_out.empty()) {
        write_json(options.json_out, results, options, device->name()->utf8String());
    }

    device->release();
    pool->release();
    return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "benchmark failed: %s\n", e.what());
        return 1;
    }
}
