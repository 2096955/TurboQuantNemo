# Mojo vs MLX Kernel Benchmark — Design Spec

**Date**: 2026-04-13
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
| Peak memory bandwidth | Stream-triad (read-write-read pattern) | ~400 GB/s |
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

**Critical framing note**: Both IsoQuant and TurboQuant accumulate in rotated space and rotate the output once (self-cancellation). The asymptotic structure is O(d_k² + T·d_k) for TurboQuant and O(d_k log d_k + T·d_k) for IsoQuant — the difference is in the per-query rotation constant, not in T-scaling. The benchmark must frame any speedup as a constant-factor advantage (structured vs dense rotation at d_k=128) unless measured data at multiple T values demonstrates otherwise. If K and V use independent rotations (no self-cancellation), this must be stated explicitly as it changes the cost model to O(T·d_k²) for TurboQuant.

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
- S in `{128, 512, 2048, 8192, 32768}` with fixed H=48, D=128

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
4. Stop when CI width < 2% of median, or after 500 iterations maximum (raised from 200 to handle sub-10us kernels)
5. Report actual iteration count used per kernel/size

This handles the timer-resolution problem for sub-100us kernels (more iterations) while avoiding waste on stable, slower kernels.

### 4.5 Thermal Management

**Macro-level monitoring** (per-kernel-suite, not per-iteration):

1. Record GPU temperature via `powermetrics` (requires sudo) before each kernel suite
2. If temperature exceeds 85C, pause for 120s cooldown before proceeding
3. Record temperature after each kernel suite completes

**Statistical thermal detection** (replaces per-iteration clock monitoring):

1. Run Durbin-Watson test on the iteration time series
2. If DW statistic < 1.5 (positive autocorrelation, indicating thermal ramp), flag the result
3. For flagged results: discard, cool down 120s, re-run
4. Report DW statistic in output JSON for transparency

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
| avg_gpu_watts | Average GPU power during kernel suite (from `powermetrics`) |
| throughput_per_watt | TFLOPS/W or GB/s/W — energy efficiency metric |

**Summary statistics:**
- **Geometric mean** of speedup ratios across all kernels (not arithmetic mean)
- Per-kernel effect size (Cohen's d) to distinguish statistical from practical significance

### 4.7 Variance Attribution via GPU Profiling

After the main benchmark completes, the **top-3 kernels by performance gap** (largest MLX/Mojo speedup ratio) are profiled using **Xcode Instruments Metal System Trace**. This provides causal evidence for *why* one framework is faster, not just *that* it is.

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
          "throughput_per_watt_tflops": 0.043
        }
      }
    }
  }
}
```

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
| **M3** | `bench_isoquant_rotate.mojo` (forward + inverse sub-benchmarks), `bench_kv_compress.mojo`, `bench_fused_attention.mojo` (unfused, framework-fused, hand-fused variants; both IsoQuant and TurboQuant pipelines; cost breakdown of rotation vs T-linear ops) | G1 | Correct rotation math (both paths), quantization fidelity, three-variant fusion pipeline, TurboQuant variant measured, cost framing verified |
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
5. Compare against the known end-to-end decode latency (from `eval_quality_gate.py` or live model run)

**Reporting rule**: If the benchmarked kernels collectively account for <50% of estimated decode time, this must be disclosed prominently in Section X.7 with a discussion of what else dominates (dispatch overhead, memory allocation, framework scheduling, non-benchmarked ops like LayerNorm, residual add).

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
- Throughput-per-watt comparison per kernel (from `powermetrics` during runs)
- Consumer hardware runs on batteries — energy efficiency is a first-class metric
- Chart: TFLOPS/W side-by-side for compute-bound kernels

### X.8 GPU Profiling (Top-3 Performance Gaps)
- Metal System Trace profiling of top-3 kernels by performance gap
- ALU utilization, memory bandwidth saturation, occupancy, stall breakdown
- Causal evidence for *why* one framework is faster, not just *that*

### X.9 Decode Time Attribution
- Estimated fraction of decode time attributable to each benchmarked kernel
- If benchmarked kernels account for <50% of decode time, disclose prominently
- Stacked-bar chart of kernel contributions to total decode latency

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

### 8.4 Version Pinning

- `pixi.toml` lockfile pins exact Mojo compiler version
- `requirements.txt` or `pyproject.toml` pins exact MLX version
- macOS version and Metal GPU family recorded in output JSON
- All seeds are fixed: numpy seed=42, Mojo seed=42
