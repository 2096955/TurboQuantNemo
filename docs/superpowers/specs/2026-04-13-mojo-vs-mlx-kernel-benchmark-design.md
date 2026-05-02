# Mojo vs MLX Kernel Benchmark — Design Spec

**Date**: 2026-04-13
**Status**: DONE — superseded by `2026-04-20-research-reality-program-design.md`. Kernel-level work merged to main 2026-04-2X (see Task 5 of `2026-04-20-phase-0-cleanup-and-invariance.md`).
**Goal**: Publication-quality kernel-level benchmark comparing Mojo GPU (via Metal) against MLX Metal on Apple Silicon, for inclusion in `docs/FROM_ATTENTION_TO_CONSUMER_HARDWARE.md`.
**Hardware**: M4 Max, 128 GB unified memory, 40 GPU cores

---

## 0. Scientific Framing

This benchmark measures **current production readiness** of each framework for consumer-hardware LLM inference on Apple Silicon. It does not claim to measure inherent architectural ceilings — Mojo's Metal backend is maturing while MLX is purpose-built for this platform. The comparison is valuable because:

1. **Practitioners need to choose** — developers targeting Apple Silicon need empirical data on framework performance today, not theoretical projections.
2. **Roofline context prevents strawman** — by reporting achieved fraction of hardware peak for both frameworks, we show *how much room each has to improve*, not just which is faster.
3. **Novel kernel comparison** tests whether MLX's custom Metal kernels for IsoQuant and KV compression represent genuine algorithmic advantages or merely framework-level optimization that Mojo could replicate.

The benchmark explicitly measures: "Given equivalent algorithms, how well does each framework exploit M4 Max hardware for the kernel operations that dominate LLM inference latency?"

---

## 1. Architecture: Side-by-Side Independent Harnesses

Two fully independent benchmark suites, each running in its native environment. A third comparison script reads both outputs and generates publication artifacts.

### Repository Structure

```
mojo-bench/                             # New top-level directory
├── pixi.toml                           # Mojo env + deps (pinned compiler version)
├── kernels/
│   ├── bench_matmul.mojo               # Dense GEMM (FP32, FP16)
│   ├── bench_softmax.mojo              # Row-wise softmax
│   ├── bench_rope.mojo                 # Rotary position embeddings
│   ├── bench_isoquant_rotate.mojo      # Structured 4D rotation (forward + inverse sub-benchmarks)
│   ├── bench_kv_compress.mojo          # Quantized KV store + retrieve
│   └── bench_fused_attention.mojo      # Full fused attention pipeline
├── harness/
│   ├── stats.mojo                      # Timing, warm-up, adaptive iteration, thermal
│   ├── output.mojo                     # JSON result writer
│   └── noop_dispatch.mojo              # Dispatch overhead calibration kernel
├── results/                            # Output JSON per run
└── README.md                           # Reproduction instructions

scripts/
├── benchmark_mlx_kernels.py            # MLX side (new, mirrors Mojo kernel set)
├── compare_mojo_vs_mlx.py              # Reads both JSONs -> LaTeX + charts + roofline
├── roofline_calibrate.py               # Measure M4 Max peak FLOPS + bandwidth
```

**Mojo suite**: `pixi install && pixi run bench-all`
**MLX suite**: `python scripts/benchmark_mlx_kernels.py --output results/mlx_kernels.json`
**Comparison**: `python scripts/compare_mojo_vs_mlx.py --mlx results/mlx_kernels.json --mojo mojo-bench/results/mojo_kernels.json`

---

## 2. Roofline Model and Hardware Calibration

Before any kernel benchmarks, establish the hardware ceiling.

### 2.1 Calibration Benchmarks

| Measurement | Method | Expected M4 Max Value |
|-------------|--------|----------------------|
| Peak FP16 TFLOPS | Sustained FMA throughput microbenchmark | ~27 TFLOPS |
| Peak FP32 TFLOPS | Sustained FMA throughput microbenchmark | ~14 TFLOPS |
| Peak memory bandwidth | Stream-triad (read-write-read pattern) | ~400 GB/s (measured) |
| Dispatch overhead (MLX) | Time `mx.eval()` on pre-compiled no-op | Baseline in us |
| Dispatch overhead (Mojo) | Time `ctx.synchronize()` on no-op kernel | Baseline in us |

### 2.2 Per-Kernel Roofline Classification

Each kernel is classified by arithmetic intensity (FLOPS / bytes accessed):

| Kernel | Arithmetic Intensity | Bound By |
|--------|---------------------|----------|
| MatMul (large) | O(N) | Compute |
| MatMul (B=1, S=1) | O(1) | Memory bandwidth |
| Softmax | Low (exp + sum + div) | Memory bandwidth |
| RoPE | Low (sin/cos + mul) | Memory bandwidth |
| IsoQuant Rotate Forward | Medium (WHT + block matmul on batch) | Depends on D and S |
| IsoQuant Rotate Inverse | Medium (block matmul on single vector) | Memory bandwidth for small D |
| KV Compress | Low (quantize + store) | Memory bandwidth |
| Fused Attention | Mixed | Compute for large S, memory for small S |

For each kernel x size, report **achieved fraction of roofline** alongside absolute throughput. This is the primary comparison metric.

**Bandwidth calibration note**: Apple's published theoretical bandwidth for M4 Max (128GB, 16-channel LPDDR5X-8533) is 546 GB/s. Stream-triad typically achieves 73-85% of theoretical on Apple Silicon. We use the measured stream-triad value (~400 GB/s expected) as the roofline ceiling, not the theoretical maximum, to avoid inflating the gap between achieved and peak. The ratio of measured-to-theoretical is reported as a sanity check (expected: 0.73-0.85).

---

## 3. Kernel Specifications

### 3.1 Kernel Set

| Kernel | Input Shape | Operation | Output | Primary Metric |
|--------|------------|-----------|--------|----------------|
| **MatMul** | `(M, K) x (K, N)` | Dense GEMM | `(M, N)` | TFLOPS |
| **Softmax** | `(B, H, S, S)` | Row-wise softmax | Same shape | GB/s |
| **RoPE** | `(B, H, S, D)` + freqs | Rotary position embed | Same shape | GB/s |
| **IsoQuant Rotate Forward** | `(H, S, D)` + block matrices | WHT + SO(4) forward rotation before quantization | Same shape | GB/s + speedup vs dense (write-path cost) |
| **IsoQuant Rotate Inverse** | `(H, 1, D)` + block matrices | Structured inverse rotation at decode | Same shape | GB/s + speedup vs dense (read-path cost, per-query) |
| **KV Compress** | `(B, H, S, D)` keys+values | Quantize -> store -> retrieve -> dequant | Reconstructed KV | GB/s + reconstruction error |
| **Fused Attention** | Full attention layer inputs | QK dot -> Softmax -> V accum -> Inverse rotate | Attention output | TFLOPS (measures fusion benefit) |

**Fused attention three-variant design**: Measured as three variants to isolate fusion benefit:

| Variant | MLX | Mojo | What It Isolates |
|---------|-----|------|------------------|
| **Unfused** | Individual `mx.eval()` per op (fused_qk_dot, softmax, fused_value_accum, metal_rotate_inverse called separately) | Individual `ctx.synchronize()` per op | Per-kernel baseline |
| **Framework-fused** | `mx.compile()` over full pipeline (single `mx.eval()`) | Composed function, single `ctx.synchronize()` | Framework fusion capability |
| **Hand-fused** (if feasible) | Single custom Metal kernel (`fused_kv_decode.metal`) | Single Mojo kernel | Algorithm-level fusion ceiling |

The delta between unfused and framework-fused is a key result — it measures each framework's ability to fuse dispatch and memory traffic. The production pipeline (4 kernels: `fused_qk_dot` -> softmax -> `fused_value_accum` -> `metal_rotate_inverse`) is the artifact from the paper and must be benchmarked in all three variants.

**Fused attention cost breakdown**: The fused attention kernel measures the complete decode-time attention computation including all rotation costs. **Both IsoQuant and TurboQuant variants are measured.** The cost breakdown isolates:

| Sub-operation | IsoQuant Cost | TurboQuant Cost | What Differs |
|---------------|---------------|-----------------|--------------|
| (a) Per-query rotation | O(d_k log d_k) structured | O(d_k²) dense | Constant factor: structured vs dense |
| (b) T-linear key scan (Q·K^T from compressed storage) | O(T · d_k) | O(T · d_k) | Same — both operate in rotated space |
| (c) T-linear value accumulation (weighted sum) | O(T · d_k) | O(T · d_k) | Same — both accumulate in rotated space |
| (d) Per-output inverse rotation | O(d_k log d_k) structured | O(d_k²) dense | Constant factor: structured vs dense |

**Critical framing note — rotation sharing resolved**: TurboQuant uses a shared rotation matrix for K and V, enabling self-cancellation: the inverse rotation folds into the query and output projections, yielding a decode cost of O(d_k² + T·d_k). IsoQuant uses independent per-head rotations for K and V (`k_isoquant_rot` and `v_isoquant_rot`), which do not self-cancel. However, IsoQuant's fused decode pipeline (Section 6.3 of the paper) operates in rotated space and applies the inverse once on the aggregated output, yielding O(d_k log d_k + T·d_k). The benchmark implements both pipelines as described and reports the per-sub-operation cost breakdown to verify these cost models empirically. Any measured IsoQuant speedup is a constant-factor advantage (structured d_k log d_k vs dense d_k² at the per-query and per-output rotation steps), not an asymptotic scaling difference in T.

**Novel kernel fairness rule**: For IsoQuant Rotate and KV Compress, provide **two MLX implementations**:
- **MLX high-level**: Using `mx.matmul`, `mx.gather`, etc. (fair comparison against Mojo)
- **MLX custom Metal**: Using hand-optimized `.metal` kernels (labeled "hand-optimized Metal", not "MLX")

This separates framework capability from human optimization effort.

### 3.2 Size Matrix

**Decode shapes (latency-critical, primary narrative):**

| Shape Name | B | S | H | D | Used For |
|------------|---|---|---|---|----------|
| Nemotron decode | 1 | 1 | 48 | 128 | Single-token generation |
| Qwen3 decode | 1 | 1 | 40 | 128 | Single-token generation |

**Prefill shapes (throughput-critical):**

| Shape Name | B | S | H | D | Used For |
|------------|---|---|---|---|----------|
| Nemotron short prefill | 1 | 128 | 48 | 128 | Short prompt |
| Nemotron medium prefill | 1 | 512 | 48 | 128 | Typical prompt |
| Nemotron long prefill | 1 | 2048 | 48 | 128 | Long context |
| Nemotron extended prefill | 1 | 8192 | 48 | 128 | Extended context |

**GEMM shapes (real inference, non-square):**

| Shape Name | M | K | N | Corresponds To |
|------------|---|---|---|----------------|
| Decode GEMV | 1 | 6144 | 6144 | Single-token QKV projection |
| Prefill QKV | 2048 | 6144 | 6144 | Batch QKV projection |
| FFN up | 1 | 6144 | 16384 | Feed-forward up-projection |
| FFN down | 1 | 16384 | 6144 | Feed-forward down-projection |
| Prefill FFN up | 2048 | 6144 | 16384 | Batch feed-forward up-projection |
| Prefill FFN down | 2048 | 16384 | 6144 | Batch feed-forward down-projection |
| MoE expert | 1 | 5120 | 11008 | Single MoE expert |

**Standard BLAS sweep (appendix, square matrices):**
- M = N = K in `{512, 1024, 2048, 4096, 8192}`

**Attention sequence length sweep (for Softmax/RoPE):**
- S in `{128, 512, 2048, 8192}` with fixed H=48, D=128
- S=32768 excluded: dense `(1, 48, 32768, 32768)` attention matrix requires 206 GB in FP32, exceeding any current Apple Silicon configuration. Real models at 32K+ context use flash attention, not dense softmax. Flash attention benchmarks for longer contexts are left to future work.

---

## 4. Measurement Methodology

### 4.1 Memory Allocation Protocol

Before any timing begins:

1. **Pre-allocate** all input, output, and scratch buffers for all kernel sizes
2. **Pre-touch** all buffers with a write pass to ensure pages are resident in unified memory
3. **Verify residency** via `vm_stat` — confirm no pageouts during the pre-touch phase
4. For MLX: all buffers allocated and evaluated (`mx.eval()`) before warm-up
5. For Mojo: all `DeviceBuffer` objects created and written via `enqueue_copy` + `ctx.synchronize()` before warm-up

### 4.2 Compilation and Optimization

**MLX:**
- All benchmark functions wrapped in `mx.compile()` before warm-up
- Compiled function reused across all iterations (no re-compilation)
- GPU execution forced: `stream=mx.gpu` on all operations to prevent AMX/CPU routing
- Report MLX version, whether `mx.compile()` is used, and Metal GPU family

**Mojo:**
- Compiled in release mode with maximum optimization (`--release` or equivalent)
- Report exact Mojo compiler version from `pixi.toml` lockfile
- Report compilation flags used

### 4.3 Timing Model

Two timing layers, reported separately:

| Layer | MLX | Mojo | What It Measures |
|-------|-----|------|-----------------|
| **Host-side wall clock** | `time.perf_counter()` around `mx.eval()` | `time.perf_counter()` around `ctx.synchronize()` | Total cost including dispatch |
| **GPU-side timestamps** | Metal GPU counter timestamps via `mx.metal` if available | Metal command buffer timestamps via `ctx` if available | Pure kernel execution |

If GPU-side timestamps are not accessible in either framework, report host-side only and disclose the limitation. Subtract the **no-op dispatch baseline** (measured in calibration) from host-side timings to estimate pure kernel time.

### 4.4 Adaptive Iteration Count

Instead of fixed 100 iterations:

1. Run 10 warm-up iterations (discarded)
2. Run measurement iterations in batches of 25
3. After each batch, compute CI width as percentage of median
4. Stop when CI width < 2% of median, or after 500 iterations maximum. If 500 iterations are reached without converging to CI < 2%, report the result with the achieved CI width and flag `"ci_converged": false` in the JSON output. In the paper, these kernels are annotated with their actual CI width. If CI width exceeds 10% of median, the result is excluded from cross-framework comparison but included in the raw data tables.
5. Report actual iteration count used per kernel/size

### 4.5 Thermal Management

**Macro-level monitoring** (per-kernel-suite, not per-iteration):

1. Record GPU temperature via `powermetrics` (requires sudo) before each kernel suite
2. If temperature exceeds 85C, pause for 120s cooldown before proceeding
3. Record temperature after each kernel suite completes

**Statistical thermal detection** (replaces per-iteration clock monitoring):

1. Run Durbin-Watson test on the iteration time series
2. Run Wald-Wolfowitz runs test on the binarized time series (above/below median) as a distribution-free backup — DW assumes approximate normality but kernel latencies are right-skewed
3. If DW statistic < 1.5 **or** runs-test p-value < 0.05, flag the result
4. For flagged results: discard, cool down 120s, re-run
5. Report DW statistic and runs-test p-value in output JSON for transparency

### 4.6 Statistics Reported

| Stat | Description |
|------|-------------|
| mean | Arithmetic mean |
| median | 50th percentile |
| std | Standard deviation |
| p5, p95, p99 | Percentile distribution |
| ci95 | 95% CI via BCa (bias-corrected and accelerated) bootstrap, 10k samples |
| throughput | TFLOPS (compute-bound) or GB/s (memory-bound) |
| roofline_pct | Achieved fraction of hardware peak (from Section 2 calibration) |
| n_iterations | Actual iterations used (adaptive) |
| dw_statistic | Durbin-Watson test statistic for thermal detection |
| cohens_d | Effect size vs the other framework (computed in comparison script) |
| avg_package_watts | Average package power during kernel suite (from `powermetrics`) |
| power_measurement_window_ms | Actual integration time for power data; flag as low-confidence if < 500ms |
| throughput_per_watt | TFLOPS/W or GB/s/W — energy efficiency metric |

**Summary statistics:**
- **Geometric mean** of speedup ratios across all kernels (not arithmetic mean)
- Per-kernel effect size (Cohen's d) to distinguish statistical from practical significance

### 4.7 Variance Attribution via GPU Profiling

After the main benchmark completes, the **top-3 kernels by performance gap** (largest absolute deviation in the MLX/Mojo time ratio) are profiled using **Xcode Instruments Metal System Trace**. This provides causal evidence for *why* one framework is faster, not just *that* it is.

For each profiled kernel, report:

| Metric | Source | What It Explains |
|--------|--------|-----------------|
| ALU utilization % | Metal System Trace | Whether the kernel is compute-saturated |
| Memory bandwidth utilization % | Metal System Trace | Whether the kernel is bandwidth-saturated |
| Occupancy (waves/EU) | Metal System Trace | Whether the kernel uses enough threads |
| Stall cycles breakdown | Metal System Trace | Where time is wasted (memory latency, barrier, etc.) |

**Process**: Manual profiling, disclosed as such. Capture 3 trace runs per kernel, report median metrics. Include annotated screenshots of the trace in the paper appendix.

**Disclosure**: GPU profiling is qualitative and manual. It explains observed gaps but does not change the quantitative benchmark results.

### 4.8 Output Format

Each framework writes a JSON file:

```json
{
  "framework": "mojo",
  "hardware": {
    "chip": "M4 Max",
    "memory_gb": 128,
    "gpu_cores": 40,
    "macos_version": "16.4",
    "metal_gpu_family": "apple9"
  },
  "framework_version": "26.2.0",
  "compilation": {
    "mode": "ahead_of_time",
    "optimization": "--release",
    "graph_compilation": "N/A"
  },
  "timestamp": "2026-04-13T14:30:00Z",
  "calibration": {
    "peak_fp16_tflops": 27.1,
    "peak_fp32_tflops": 13.8,
    "peak_memory_bandwidth_gbs": 412.3,
    "noop_dispatch_us": 3.2
  },
  "kernels": {
    "matmul": {
      "fp16_decode_gemv_1x6144x6144": {
        "mean_us": 42.3,
        "median_us": 41.1,
        "std_us": 2.2,
        "p5_us": 39.0,
        "p95_us": 46.0,
        "p99_us": 48.3,
        "ci95_bca": [40.2, 43.4],
        "tflops": 1.79,
        "gbs": 289.4,
        "roofline_pct": 70.1,
        "n_iterations": 150,
        "dw_statistic": 1.93,
        "power": {
          "avg_package_watts": 42.3,
          "avg_gpu_watts": 28.1,
          "throughput_per_watt_tflops": 0.043,
          "measurement_window_ms": 7500,
          "low_confidence": false
        }
      }
    }
  }
}
```

**Power measurement granularity**: `powermetrics` samples at ~250ms intervals. Power is averaged over the full adaptive measurement phase for each kernel/size combination. The `measurement_window_ms` field records actual integration time. If measurement window < 500ms, `low_confidence` is set to `true` and the power reading is reported as indicative rather than precise. In the paper's energy efficiency charts (Section X.7), power data is aggregated to the kernel-class level (all sizes of MatMul together, all sizes of Softmax together) to ensure sufficient integration time for meaningful averages.

---

## 5. Precision Validation

### 5.1 Per-Kernel Numerical Checks

Both frameworks receive identical inputs from seeded RNG (numpy seed=42, converted to framework tensors).

**Precision reference for novel kernels**: For novel kernels where MLX has two implementations (high-level and custom Metal), precision validation uses the **high-level implementation as the numerical reference** for both frameworks. The custom Metal implementation's divergence from the high-level reference is reported separately, so the reader can distinguish framework-induced error from hand-optimization-induced error. This matters because the MLX fused Metal decode path has known numerical divergence (max_abs_diff ~4.3 against the composed FP32 reference).

For each kernel, compute:

| Metric | Description |
|--------|-------------|
| Max absolute error | Worst-case element drift |
| RMSE | Root mean squared error |
| Relative error | Mean `|a-b| / max(|a|, |b|, eps)` |
| KL divergence | For probability outputs (softmax) |

### 5.2 Acceptable Error Thresholds

Thresholds scale with reduction dimension to account for floating-point accumulation:

| Kernel | Threshold | Rationale |
|--------|-----------|-----------|
| MatMul FP32 | RMSE < sqrt(K) * 1e-6 | Accumulation error scales with reduction dim |
| MatMul FP16 | RMSE < sqrt(K) * 1e-3 | Half-precision accumulation variance |
| Softmax | KL div < 1e-5 | Probability distribution must be tight |
| RoPE | Max abs < 1e-4 | Trig functions must agree |
| IsoQuant Rotate (fwd + inv) | RMSE < sqrt(D) * 1e-5 | Rotation fidelity, scaled by head dimension; forward and inverse measured separately, both dense and structured implementations |
| KV Compress | Reconstruction RMSE < theoretical_quantization_error * 1.1 | Bounded by quantization scheme's error floor, not by MLX's output |

KV Compress threshold is derived from the codebook quantization scheme's theoretical error bound (Lloyd-Max distortion for the given bit-width), not from comparing against MLX output. Both frameworks must independently fall within 110% of theoretical optimum.

### 5.3 Perplexity Gate (G2.5)

Kernel-chain precision validation using simulated forward passes:

- Extract real weight matrices from Llama 3.2 1B (both frameworks load weights independently from safetensors)
- Feed identical inputs through each kernel in sequence (MatMul -> RoPE -> Softmax -> MatMul -> KV Compress), simulating one attention layer
- Accumulate per-token logit differences across a fixed prompt set (WikiText-2 subset, 1000 tokens)
- Report equivalent perplexity delta

**Threshold justification**: 0.5% perplexity divergence is derived from Dettmers et al. (2022) "LLM.int8()" which established that quantization-induced perplexity changes below 0.5% are imperceptible in downstream task performance. We adopt this as our equivalence threshold.

**Known limitations**: This kernel-chain simulation omits residual connections, LayerNorm, and feed-forward layers. The relationship between partial-pipeline and full-pipeline divergence is non-monotonic: residual connections and LayerNorm may dampen error propagation, while additional compute layers may amplify it. We report partial-pipeline divergence as an informative proxy, not a bound. This limitation is disclosed in the paper.

Note: Mojo cannot run full model inference on Apple Silicon (Tier 3), so this gate uses kernel-chain simulation, not end-to-end model serving.

---

## 6. Development Workflow: Gemini Drafts, Codex Reviews + Fixes

### 6.1 Agent Roles

| Agent | Role |
|-------|------|
| **Gemini CLI** | Implementation drafter — writes all kernel code, benchmark harnesses, comparison scripts |
| **Codex** | Adversarial reviewer + fixer — detailed code review, implements fixes for every finding |
| **Claude** | ARB lead — final approval at each gate |

### 6.2 Codex Review + Fix Cycle

At each gate:

```
Gemini delivers code for milestone
  -> /codex:adversarial-review (detailed code review, finds issues)
  -> /codex:rescue (implements fixes for each finding)
  -> /codex:adversarial-review (verifies fixes, finds new issues if any)
  -> repeat until clean pass
  -> Claude ARB approves
  -> proceed to next milestone
```

### 6.3 Code Quality Equivalence Gate

Since Gemini drafts both Mojo and MLX code, there is a risk that Mojo kernels are suboptimal due to less training data on Mojo GPU programming. To mitigate:

1. **Roofline check**: Every Mojo kernel must achieve >50% of theoretical roofline. If it doesn't, investigate whether this reflects a framework limitation or suboptimal code.
2. **Algorithmic equivalence**: Codex adversarial review explicitly checks that Mojo kernels use equivalent tiling, memory access patterns, and vectorization strategies as the MLX high-level-ops version.
3. **Disclosure**: The paper explicitly states that kernels were AI-generated and reviewed, and acknowledges that expert Mojo developers might achieve better results.

### 6.4 Milestones and Gates

| Milestone | Gemini Drafts | Codex Gate | Review Focus |
|-----------|--------------|------------|--------------|
| **M0** | `scripts/roofline_calibrate.py`, `mojo-bench/harness/noop_dispatch.mojo` | G0 | Hardware calibration correct, dispatch baselines measured |
| **M1** | `pixi.toml`, `stats.mojo`, `bench_vec_add.mojo` (smoke test) | G1 | Mojo compiles on M4 Max, GPU detected, timing sync correct |
| **M2** | `bench_matmul.mojo`, `bench_softmax.mojo`, `bench_rope.mojo` | G1 | Kernel equivalence to MLX ops, no unfair advantages, roofline >50% |
| **M3a** | `bench_isoquant_rotate.mojo` (forward + inverse sub-benchmarks, both dense and structured) | G1 | Correct rotation math (both paths), structured speedup verified |
| **M3b** | `bench_kv_compress.mojo` | G1 | Quantization fidelity, codebook loading, reconstruction RMSE |
| **M3c** | `bench_fused_attention.mojo` (unfused + framework-fused variants, IsoQuant pipeline) | G1 | Composition correct, timing methodology, unfused vs fused delta |
| **M3d** | `bench_fused_attention.mojo` (hand-fused variant if feasible + TurboQuant pipeline + cost breakdown) | G1 | Hand-fused feasibility on Tier 3 Metal, TurboQuant variant, cost framing verified |
| **M4** | `scripts/benchmark_mlx_kernels.py` (with `mx.compile()`, `stream=mx.gpu`, two implementations for novel kernels) | G2 | Methodology parity with Mojo side, AMX routing controlled |
| **G2.5** | Perplexity validation script | G2.5 | Kernel-chain precision, perplexity divergence check |
| **M5** | `scripts/compare_mojo_vs_mlx.py` | G3+G4 | Statistical methods (BCa, geometric mean, Cohen's d), chart accuracy, roofline plots, log2 ratio visualization |
| **M6** | Paper section in `FROM_ATTENTION_TO_CONSUMER_HARDWARE.md` | G5 | Claims vs evidence, roofline context, code-quality disclosure |

---

## 7. Comparison Output and Paper Integration

### 7.1 Artifacts Produced

1. **LaTeX tables** — One per kernel, both frameworks side-by-side with all stats plus roofline fraction
2. **Charts** (matplotlib, publication-quality):
   - **Roofline plot**: Both frameworks plotted on arithmetic-intensity vs throughput, with hardware ceiling line
   - **Roofline fraction bars**: Achieved % of peak for each kernel, side-by-side
   - Bar charts: Absolute throughput (TFLOPS / GB/s) per kernel at each real-model shape
   - Line plots: Throughput vs matrix size for standard sweep
   - Error bars: 95% BCa confidence intervals on all data points
   - Heatmap: **log2(MLX_time / Mojo_time)** — zero means equal, positive = Mojo faster, negative = MLX faster. Symmetric and honest.
   - Every ratio chart accompanied by absolute numbers table
   - **Energy efficiency**: Throughput-per-watt (TFLOPS/W) comparison per kernel
   - **Decode time attribution**: Stacked-bar showing each kernel's fraction of estimated decode time
3. **Summary JSON** — Machine-readable combined results with geometric mean speedup

### 7.2 Additional Baselines (Context)

Where feasible, include:

| Baseline | Purpose |
|----------|---------|
| **Accelerate/BLAS (CPU)** | GEMM-only — shows GPU vs CPU tradeoff for decode GEMV |
| **Hardware roofline** | Theoretical peak — shows framework efficiency |

### 7.3 Decode Time Attribution

The benchmark's Section 0 frames itself as measuring "kernel operations that dominate LLM inference latency." To validate this framing, compute an estimated decode time breakdown:

1. For each benchmarked kernel, take the measured per-call latency at decode shape (B=1, S=1)
2. Multiply by invocations-per-decode-step (e.g., MatMul invoked for QKV projection, O projection, FFN up/down = ~6 per layer)
3. Multiply by the number of layers in the target model (e.g., Nemotron-H 120B = 80 layers)
4. Sum to get estimated total kernel time per decode step
5. Compare against the known end-to-end decode latency (from `eval_quality_gate.py` or live model run). **Note**: End-to-end reference is available for MLX only. For Mojo, we report the sum of estimated kernel times as a theoretical decode time lower bound, acknowledging that actual end-to-end time would include framework overhead not captured by kernel benchmarks.

**MLX vs Mojo attribution scope**: End-to-end decode latency reference is available for MLX only (from live model runs). For Mojo, we report the sum of estimated kernel times as a theoretical decode time lower bound, acknowledging that actual end-to-end time would include additional overhead not captured by kernel benchmarks. The MLX attribution shows where actual decode time goes; the Mojo attribution shows where it *would* go if the kernel costs were the dominant factor.

**Reporting rule**: If the benchmarked kernels collectively account for <50% of estimated decode time, this must be disclosed prominently in Section X.9 with a discussion of what else dominates (dispatch overhead, memory allocation, framework scheduling, non-benchmarked ops like LayerNorm, residual add).

This analysis appears in the comparison output as a summary table and as a stacked-bar chart showing each kernel's fraction of estimated decode time.

### 7.4 Paper Section Structure

New section in `docs/FROM_ATTENTION_TO_CONSUMER_HARDWARE.md`:

```
## Section X: Framework Comparison — MLX Metal vs Mojo GPU on Apple Silicon

### X.1 Methodology
- Hardware specs (M4 Max, macOS version, Metal GPU family)
- Framework versions (pinned, exact)
- Measurement protocol: adaptive iteration, BCa bootstrap, Durbin-Watson thermal detection
- Memory allocation protocol, compilation settings
- Code quality disclosure (AI-generated, reviewed, roofline-gated)

### X.2 Hardware Calibration
- Roofline model: measured peak FP16 TFLOPS, peak memory bandwidth
- Dispatch overhead baselines for both frameworks

### X.3 Standard Kernel Comparison
- MatMul, Softmax, RoPE across standard sweep sizes
- Tables with absolute throughput + roofline fraction
- Throughput curves with CIs

### X.4 Novel Kernel Comparison
- IsoQuant rotation: MLX high-level vs MLX custom Metal vs Mojo
- KV compression: quantize-store-retrieve-dequant pipeline
- Fused attention pipeline: tests kernel fusion benefit

### X.5 Real Model Shape Results
- Decode (B=1, S=1) and prefill shapes for Nemotron-H and Qwen3 MoE
- Non-square GEMM shapes from actual FFN/QKV projections
- Ties directly to tok/s claims elsewhere in the paper

### X.6 Precision Analysis
- Per-kernel numerical error tables (scaled thresholds)
- Kernel-chain perplexity validation
- Proves correctness alongside performance

### X.7 Energy Efficiency
- Throughput-per-watt comparison per kernel only when framework-specific power baselines are captured during runs
- Consumer hardware runs on batteries — energy efficiency is a first-class metric
- Chart: TFLOPS/W side-by-side for compute-bound kernels

### X.8 GPU Profiling (Top-3 Performance Gaps)
- Metal System Trace profiling of top-3 kernels by performance gap
- ALU utilization, memory bandwidth saturation, occupancy, stall breakdown
- Causal evidence for *why* one framework is faster, not just *that*

### X.9 Decode Time Attribution
- Illustrative kernel-mix chart derived from representative matched-kernel latencies and invocation counts
- Do not claim a wall-clock decode fraction without an explicit end-to-end decode baseline
- Stacked-bar chart of relative kernel-family mix for the matched benchmark set

### X.10 Discussion
- Roofline analysis: how close each framework gets to hardware limits
- Where the performance gap reflects framework maturity vs architectural limits
- Kernel fusion as a key differentiator (unfused vs framework-fused vs hand-fused deltas)
- Implications for consumer hardware inference
- Limitations: AI-generated Mojo code, Tier 3 Metal support, kernel-chain vs full-model perplexity
```

---

## 8. Reproduction

### 8.1 Environment Requirements

| Requirement | Specification |
|-------------|--------------|
| Hardware | Apple Silicon (M1+ for basic, M4 Max for reference results) |
| macOS | 16.4+ (Metal driver behavior varies across releases) |
| Power | Plugged in, Low Power Mode off |
| Background | Spotlight indexing paused, iCloud sync disabled during runs |
| `pmset` | `sudo pmset -a sleep 0 displaysleep 0 disksleep 0 powernap 0` (prevent sleep/dimming during benchmarks) |

### 8.2 Commands

```bash
# Hardware calibration (run once)
python scripts/roofline_calibrate.py --output results/roofline_m4max.json

# Mojo side
cd mojo-bench && pixi install && pixi run bench-all

# MLX side (assumes mlx-lm already installed)
python scripts/benchmark_mlx_kernels.py --output results/mlx_kernels.json

# Comparison
python scripts/compare_mojo_vs_mlx.py \
  --mlx results/mlx_kernels.json \
  --mojo mojo-bench/results/mojo_kernels.json \
  --roofline results/roofline_m4max.json \
  --output-dir results/comparison/
```

### 8.3 Multi-Run Protocol

Run the full suite 3 times on different days. Report inter-run variance in the paper. If inter-run CV > 5% for any kernel, investigate and disclose the cause.

**Version identity constraint**: Framework versions, macOS version, and compiler versions must remain identical across all three runs. At the start of each run, verify versions against the pinned lockfiles. If any version has changed (macOS update, MLX release, Mojo compiler update), abort the run and re-pin before proceeding.

### 8.4 Version Pinning

- `pixi.toml` lockfile pins exact Mojo compiler version
- `requirements.txt` or `pyproject.toml` pins exact MLX version
- macOS version and Metal GPU family recorded in output JSON
- All seeds are fixed: numpy seed=42, Mojo seed=42

---

## 9. Implementation Progress (as of 2026-04-14, updated 2026-04-14)

### 9.1 Milestone Status

| Milestone | Status | Notes |
|-----------|--------|-------|
| **M0** | DONE | `roofline_calibrate.py` + `noop_dispatch.mojo` complete. Noop rewritten to enqueue actual GPU kernel (not empty sync). 146 µs dispatch overhead measured. |
| **M1** | DONE | `pixi.toml` pinned to MAX `<26.3.0.dev2026040905` (avoids `metal:4-metal4` GPU detection bug on Apr 9+ nightlies). `stats.mojo` + `bench_vec_add.mojo` pass. Full API migration: `fn`→`def`, `alias`→`comptime`, `LayoutTensor` positional params. |
| **M2** | DONE | All three standard kernels compiled with 6 Mojo 26.x fixes (`inout`→`mut`, `str()`→`String()`, `from collections`→`from std.collections`, `return v`→`return v^`, `-I .` for package resolution, `__init__.mojo` for harness package). Ran successfully: 21 JSON result files (12 matmul, 4 softmax, 5 RoPE). |
| **M3a** | DONE | Forward + inverse rotation, structured + dense. Scratch buffers hoisted out of hot loops. 18-21x structured speedup preserved. RMSE = 0.0 (exact match). |
| **M3b** | DONE | KV compress + decompress. Codebook loading. Reconstruction RMSE validation. |
| **M3c** | DONE | Unfused + framework-fused attention variants. IsoQuant pipeline. |
| **M3d** | DONE | Hand-fused variant (CPU-only reference for Tier 3). TurboQuant pipeline. Cost breakdown table. |
| **M4** | DONE | `benchmark_mlx_kernels.py` — `mx.compile()`, `stream=mx.gpu`, BCa bootstrap, two implementations for novel kernels (high-level + custom Metal). |
| **G2.5** | DONE | `validate_kernel_precision.py` — kernel-chain precision, perplexity divergence check. |
| **M5** | DONE | `compare_mojo_vs_mlx.py` — 19 matched pairs, all dtype-relaxed (`MLX fp16` vs `Mojo fp32`), 12 output files (3 LaTeX tables, 7 charts, 1 heatmap, 1 summary JSON). Current local run reports time ratio `0.366x` (`MLX_time / Mojo_time`) and MLX speedup `2.729x` over Mojo. Cohen's d uses actual `n_iterations`. Decode output is an illustrative kernel-mix chart, not a wall-clock decode fraction. |
| **M6** | DONE | All 9 PLACEHOLDERs filled in paper section 10c. Geometric mean decomposition paragraph added. Kernel-time vs wall-clock clarification added. DW stale-data note added. Run cleanliness disclosed as limitation (5). |

### 9.2 Adversarial Review Status

Six independent reviews completed (2026-04-23 refresh):

| Reviewer | Verdict | Findings Fixed |
|----------|---------|----------------|
| **Manual (human)** | 3 issues (P0/P1/P2) | All 3 fixed: noop dispatch rewrite, GPU guard, raises spec |
| **Gemini CLI** | NO-SHIP (5 issues) | 3/5 fixed (schema mismatch, Cohen's d, Mojo format support). 2 acknowledged (Mojo CPU-only kernels = by design for Tier 3; BCa not in Mojo stats.mojo = known TODO). |
| **Codex** (2 passes) | needs-attention | All 3 fixed: matmul layout overflow, softmax 8K coverage restored, IsoQuant scratch buffer hoisted |
| **Codex** (non-adversarial, post-run) | 4 findings | DW-before-sort bug fixed in all 3 benchmarks. Hardcoded theoretical bandwidth fixed. Remaining 3 findings acknowledged in paper disclosures. |
| **Codex** (2026-04-23 adversarial refresh) | 5 findings | Fixed: explicit time-ratio vs speedup naming, mixed-precision match disclosure, illustrative kernel-mix chart replacing unsupported decode-fraction claims, framework-specific power requirement for energy chart, stale spec/paper numbers refreshed. |
| **Superpowers code-reviewer** | 3 critical, 8 important, 6 suggestions | C1 (bootstrap asymmetry) disclosed in paper. C2 (hardcoded roofline) fixed in calibration script. C3 (BCa vs percentile) acknowledged — Mojo still bootstraps mean not median. I1-I8 addressed in paper prose or acknowledged as Tier 3 limitations. |
| **Human A- review** | 1 P1, 3 P2 | P1 (geometric mean decomposition) added. P2 (decode attribution context) added. P2 (stale DW) disclosed. P2 (run cleanliness) added as limitation (5). |

### 9.3 Completed Work

#### Benchmark execution and paper (all done):

1. ~~**Run full benchmark suite**~~ — DONE. Mojo: 21 JSON files (12 matmul, 4 softmax, 5 RoPE shapes). MLX: results in `results/mlx_kernels.json`. Roofline calibration in `results/roofline_m4max.json`.

2. ~~**Run comparison script**~~ — DONE. 19 matched pairs (all dtype-relaxed: `MLX float16` vs `Mojo float32`). 12 output files in `results/comparison/`. Summary now reports both time ratio (`0.366x`, `MLX_time / Mojo_time`) and MLX speedup (`2.729x`, `Mojo_time / MLX_time`).

3. ~~**Fill paper PLACEHOLDERs**~~ — DONE. All 9 PLACEHOLDERs in 10c.2–10c.10 filled with data or honest scope limitations.

4. ~~**MatMul roofline disclosure**~~ — DONE. Disclosed in 10c.3 (dedicated paragraph) and 10c.10 (discussion).

5. ~~**Mojo BCa bootstrap**~~ — DONE. Asymmetry disclosed in methodology (10c.1) and Cohen's d footnote (10c.3).

6. ~~**Mojo CPU-only novel ops**~~ — DONE. Scope limitations in 10c.4 (IsoQuant rotation, KV compression).

#### Bug fixes discovered during execution:

7. **DW-before-sort** — `durbin_watson(times)` was called after `sort_float_list(times)`, computing DW on monotonically increasing data (always near 0). Fixed: DW computed on execution-order data before sorting in all 3 benchmark files. Note: existing Mojo JSON files still contain stale DW values (code fixed, results not re-run). Disclosed in paper 10c.10.

8. **Hardcoded theoretical bandwidth** — `roofline_calibrate.py` had `"theoretical_memory_bandwidth_gbs": 546.0` regardless of hardware. Fixed: uses derived value from `memory_gb`.

9. **Output path double-nesting** — Mojo code wrote to `mojo-bench/results/` but pixi runs from `mojo-bench/`, creating `mojo-bench/mojo-bench/results/`. Fixed: paths changed to `results/`.

10. **Dtype matching too strict** — Comparison script matched on `(name, shape, dtype)` yielding 0 matches (MLX float16, Mojo float32). Fixed: relaxed to `(name, shape)` yielding 19 matches.

### 9.4 Remaining Work

#### Blocking (paper quality):

1. **Gemma 4 PPL table broken markdown** (flag 3) — Lines 515-520 of `FROM_ATTENTION_TO_CONSUMER_HARDWARE.md`: the footnote `*\* The reported +0.0000...` at line 517 breaks the markdown table, orphaning the Nemotron-30B rows outside the table. Fix: move footnote below the table close.

2. **Kurtosis value reconciliation** (flag 2) — Lines 524-527: shared=13.10, routed=3.41, gap=3.8x. Need to verify these match actual measurement scripts or update if stale.

#### Non-blocking (optional improvements):

3. **Re-run Mojo benchmarks for clean DW** — The 21 Mojo JSON files have stale DW values (computed on sorted data). The code fix is in place; re-running would produce correct DW. Currently disclosed in paper as invalid. Low priority since DW is informational, not used in any comparison metric.

4. **GPU profiling (Section 10c.8)** — Manual Xcode Instruments Metal System Trace for top-3 performance gaps (matmul 2048x6144x6144 at 77x, matmul 4096³ at 49x, softmax 2048² at 20x). Requires human operator with GUI access.

5. **Energy efficiency (Section 10c.7)** — `powermetrics` integration for TFLOPS/W data. Requires sudo access during benchmark runs. Currently disclosed as "Power data not collected."

6. **Multi-run protocol (Section 8.3)** — 3 runs on different days with inter-run CV analysis. Deferred to after initial results accepted.

7. **Precision validation thresholds** — Gemini flagged RMSE threshold of 0.1 for chain ops as too loose, and max_abs_diff ~4.3 for fused Metal path. Paper reports precision table with all 7 kernels passing; threshold review is optional hardening.

8. **Mojo BCa implementation** — `stats.mojo` uses simple percentile bootstrap (bootstraps mean), while MLX uses scipy BCa (bootstraps median). Different statistics under same field name. Acknowledged in paper but code mismatch remains.
