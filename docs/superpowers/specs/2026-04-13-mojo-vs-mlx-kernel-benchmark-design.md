# Mojo vs MLX Kernel Benchmark — Design Spec

**Date**: 2026-04-13
**Goal**: Publication-quality kernel-level benchmark comparing Mojo GPU (via Metal) against MLX Metal on Apple Silicon, for inclusion in `docs/FROM_ATTENTION_TO_CONSUMER_HARDWARE.md`.
**Hardware**: M4 Max, 128 GB unified memory, 40 GPU cores

---

## 1. Architecture: Side-by-Side Independent Harnesses

Two fully independent benchmark suites, each running in its native environment. A third comparison script reads both outputs and generates publication artifacts.

### Repository Structure

```
mojo-bench/                             # New top-level directory
├── pixi.toml                           # Mojo env + deps
├── kernels/
│   ├── bench_matmul.mojo               # Dense GEMM (FP32, FP16)
│   ├── bench_softmax.mojo              # Row-wise softmax
│   ├── bench_rope.mojo                 # Rotary position embeddings
│   ├── bench_isoquant_rotate.mojo      # Structured 4D rotation
│   └── bench_kv_compress.mojo          # Quantized KV store + retrieve
├── harness/
│   ├── stats.mojo                      # Timing, warm-up, thermal detection
│   └── output.mojo                     # JSON result writer
├── results/                            # Output JSON per run
└── README.md                           # Reproduction instructions

scripts/
├── benchmark_mlx_kernels.py            # MLX side (new, mirrors Mojo kernel set)
├── compare_mojo_vs_mlx.py              # Reads both JSONs -> LaTeX + charts
```

**Mojo suite**: `pixi install && pixi run bench-all`
**MLX suite**: `python scripts/benchmark_mlx_kernels.py --output results/mlx_kernels.json`
**Comparison**: `python scripts/compare_mojo_vs_mlx.py --mlx results/mlx_kernels.json --mojo mojo-bench/results/mojo_kernels.json`

---

## 2. Kernel Specifications

### 2.1 Kernel Set

| Kernel | Input Shape | Operation | Output | Primary Metric |
|--------|------------|-----------|--------|----------------|
| **MatMul** | `(M, K) x (K, N)` | Dense GEMM | `(M, N)` | TFLOPS |
| **Softmax** | `(B, H, S, S)` | Row-wise softmax | Same shape | GB/s |
| **RoPE** | `(B, H, S, D)` + freqs | Rotary position embed | Same shape | GB/s |
| **IsoQuant Rotate** | `(H, S, D)` + block matrices | Structured 4D rotation | Same shape | GB/s + speedup vs dense |
| **KV Compress** | `(B, H, S, D)` keys+values | Quantize -> store -> retrieve -> dequant | Reconstructed KV | GB/s + reconstruction error |

### 2.2 Size Matrix

**Real model shapes (primary narrative):**

| Model | H (heads) | D (head_dim) | Intermediate | Hidden |
|-------|-----------|-------------|--------------|--------|
| Nemotron-H 120B | 48 | 128 | 16384 | 6144 |
| Qwen3 MoE | 40 | 128 | 11008 | 5120 |

**Standard BLAS sweep (appendix):**
- M, N, K in `{512, 1024, 2048, 4096, 8192}`

All kernel x size combinations are run for both frameworks.

---

## 3. Measurement Methodology

### 3.1 Statistical Protocol

Per kernel/size combination:

1. **Thermal baseline** -- Read GPU junction temp via `powermetrics` before starting
2. **Warm-up phase** -- 10 iterations, results discarded
3. **Measurement phase** -- 100 timed iterations
4. **Thermal rejection** -- If any iteration's GPU clock drops >10% from baseline, flag it; if >20% of iterations are flagged, reject the entire run, cool down 60s, retry
5. **Synchronization** -- MLX: `mx.eval()` after each op. Mojo: `ctx.synchronize()` after each kernel dispatch

### 3.2 Statistics Reported

| Stat | Description |
|------|-------------|
| mean | Arithmetic mean of 100 runs |
| median | 50th percentile |
| std | Standard deviation |
| p5, p95, p99 | Percentile distribution |
| CI_95 | 95% confidence interval (bootstrap, 10k samples) |
| throughput | TFLOPS (compute-bound) or GB/s (memory-bound) |
| thermal_rejected | Count of thermally-rejected iterations |

### 3.3 Output Format

Each framework writes a JSON file:

```json
{
  "framework": "mojo",
  "hardware": {
    "chip": "M4 Max",
    "memory_gb": 128,
    "gpu_cores": 40
  },
  "mojo_version": "26.x",
  "timestamp": "2026-04-13T...",
  "kernels": {
    "matmul": {
      "fp16_2048x2048x2048": {
        "mean_us": 142.3,
        "median_us": 140.1,
        "std_us": 5.2,
        "p5_us": 135.0,
        "p95_us": 151.0,
        "p99_us": 158.3,
        "ci95": [139.2, 145.4],
        "tflops": 12.1,
        "thermal_rejected": 0,
        "n_iterations": 100
      }
    }
  }
}
```

---

## 4. Precision Validation

### 4.1 Per-Kernel Numerical Checks

Both frameworks receive identical inputs (seeded RNG). For each kernel, compute:

| Metric | Description |
|--------|-------------|
| Max absolute error | Worst-case element drift |
| RMSE | Root mean squared error |
| Relative error | Mean `|a-b| / max(|a|, |b|)` |
| KL divergence | For probability outputs (softmax) |

### 4.2 Acceptable Error Thresholds

| Kernel | Threshold | Rationale |
|--------|-----------|-----------|
| MatMul FP32 | RMSE < 1e-6 | IEEE 754 compliance |
| MatMul FP16 | RMSE < 1e-3 | Half-precision accumulation variance |
| Softmax | KL div < 1e-5 | Probability distribution must be tight |
| RoPE | Max abs < 1e-4 | Trig functions must agree |
| IsoQuant Rotate | RMSE < 1e-4 | Rotation fidelity |
| KV Compress | Reconstruction RMSE matches MLX | Quantization error identical given same codebook |

### 4.3 Perplexity Gate (G2.5)

End-to-end quality validation using simulated forward passes:

- Extract real weight matrices and KV tensors from a small model (Llama 3.2 1B via MLX)
- Feed identical inputs through each kernel in both frameworks in sequence (MatMul -> RoPE -> Softmax -> KV Compress), simulating one attention layer
- Accumulate per-token log-likelihood differences across a fixed prompt set (WikiText-2 subset, 1000 tokens)
- Report equivalent perplexity delta: if accumulated logit drift exceeds 0.5% perplexity divergence, flag as precision issue
- Note: Mojo cannot run full model inference on Apple Silicon (Tier 3), so this gate uses kernel-chain simulation, not end-to-end model serving
- This proves outputs are numerically equivalent across the full kernel pipeline, not just individually correct

---

## 5. Development Workflow: Gemini Drafts, Codex Reviews + Fixes

### 5.1 Agent Roles

| Agent | Role |
|-------|------|
| **Gemini CLI** | Implementation drafter -- writes all kernel code, benchmark harnesses, comparison scripts |
| **Codex** | Adversarial reviewer + fixer -- reviews code in detail, implements fixes for every finding |
| **Claude** | ARB lead -- final approval at each gate |

### 5.2 Codex Review + Fix Cycle

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

### 5.3 Milestones and Gates

| Milestone | Gemini Drafts | Codex Gate | Review Focus |
|-----------|--------------|------------|--------------|
| **M1** | `pixi.toml`, `stats.mojo`, `bench_vec_add.mojo` (smoke test) | G1 | Mojo compiles on M4 Max, GPU detected, timing sync correct |
| **M2** | `bench_matmul.mojo`, `bench_softmax.mojo`, `bench_rope.mojo` | G1 | Kernel equivalence to MLX ops, no unfair advantages |
| **M3** | `bench_isoquant_rotate.mojo`, `bench_kv_compress.mojo` | G1 | Correct rotation math, quantization fidelity |
| **M4** | `scripts/benchmark_mlx_kernels.py` | G2 | Methodology parity with Mojo side |
| **G2.5** | Perplexity validation script | G2.5 | End-to-end precision, perplexity divergence check |
| **M5** | `scripts/compare_mojo_vs_mlx.py` | G3+G4 | Statistical methods, chart accuracy, no misleading visuals |
| **M6** | Paper section in `FROM_ATTENTION_TO_CONSUMER_HARDWARE.md` | G5 | Claims vs evidence, methodology description, result interpretation |

---

## 6. Comparison Output and Paper Integration

### 6.1 Artifacts Produced

1. **LaTeX tables** -- One per kernel, both frameworks side-by-side with all stats
2. **Charts** (matplotlib, publication-quality):
   - Bar charts: MLX vs Mojo per kernel at each real-model shape
   - Line plots: Throughput vs matrix size for standard sweep
   - Error bars: 95% CI on all data points
   - Heatmap: Speedup ratio (MLX/Mojo) across all kernel x size combinations
3. **Summary JSON** -- Machine-readable combined results

### 6.2 Paper Section Structure

New section in `docs/FROM_ATTENTION_TO_CONSUMER_HARDWARE.md`:

```
## Section X: Framework Comparison -- MLX Metal vs Mojo GPU on Apple Silicon

### X.1 Methodology
- Hardware specs, framework versions, measurement protocol, thermal rejection

### X.2 Standard Kernel Comparison
- MatMul, Softmax, RoPE across standard sweep sizes
- Table + throughput curves

### X.3 Novel Kernel Comparison
- IsoQuant rotation: structured 4D vs Mojo equivalent
- KV compression: quantize-store-retrieve-dequant pipeline
- Custom Metal kernels vs generic Mojo GPU path

### X.4 Real Model Shape Results
- Nemotron-H and Qwen3 MoE dimensions
- Ties directly to tok/s claims elsewhere in the paper

### X.5 Precision Analysis
- Per-kernel numerical error tables
- Perplexity comparison on Llama 3.2 1B
- Proves correctness alongside performance

### X.6 Discussion
- Why MLX's Metal-native path wins on Apple Silicon
- Where Mojo's cross-platform abstraction costs performance
- Implications for consumer hardware inference
```

---

## 7. Reproduction

A reviewer can reproduce the full benchmark with:

```bash
# Mojo side
cd mojo-bench && pixi install && pixi run bench-all

# MLX side (assumes mlx-lm already installed)
python scripts/benchmark_mlx_kernels.py --output results/mlx_kernels.json

# Comparison
python scripts/compare_mojo_vs_mlx.py \
  --mlx results/mlx_kernels.json \
  --mojo mojo-bench/results/mojo_kernels.json \
  --output-dir results/comparison/
```

All random inputs are seeded. Results include hardware fingerprint and framework versions. Thermal rejection is logged for transparency.
