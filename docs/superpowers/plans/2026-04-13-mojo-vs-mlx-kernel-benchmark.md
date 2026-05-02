# Mojo vs MLX Kernel Benchmark — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a publication-quality kernel-level benchmark comparing Mojo GPU (via Metal) against MLX Metal on M4 Max, with roofline analysis, statistical rigor, energy metrics, GPU profiling, and decode time attribution.

**Architecture:** Two independent benchmark suites (`mojo-bench/` for Mojo, `scripts/benchmark_mlx_kernels.py` for MLX) writing JSON results, with a comparison script (`scripts/compare_mojo_vs_mlx.py`) that reads both and generates LaTeX tables, matplotlib charts, and roofline plots. A calibration script (`scripts/roofline_calibrate.py`) establishes hardware ceilings first.

**Tech Stack:** Mojo (via pixi), MLX (Python), numpy, scipy (BCa bootstrap, Durbin-Watson), matplotlib, Metal System Trace (manual profiling)

**Agent Workflow:** Gemini CLI drafts each milestone → Codex adversarial-review → Codex rescue (fix) → re-review → Claude ARB approves → next milestone.

**Spec:** `docs/superpowers/specs/2026-04-13-mojo-vs-mlx-kernel-benchmark-design.md` (v3, commit `7c1b2c3`)

---

## File Structure

### New Files

```
mojo-bench/                                    # Mojo benchmark suite
├── pixi.toml                                  # Mojo env, pinned compiler, bench-all task
├── kernels/
│   ├── bench_matmul.mojo                      # Dense GEMM (FP16, FP32)
│   ├── bench_softmax.mojo                     # Row-wise softmax
│   ├── bench_rope.mojo                        # Rotary position embeddings
│   ├── bench_isoquant_rotate.mojo             # Forward + inverse sub-benchmarks
│   ├── bench_kv_compress.mojo                 # Quantized KV store + retrieve
│   └── bench_fused_attention.mojo             # 3-variant: unfused, framework-fused, hand-fused
├── harness/
│   ├── stats.mojo                             # Timing, warmup, adaptive iteration, thermal, power
│   ├── output.mojo                            # JSON writer with framework-agnostic schema
│   └── noop_dispatch.mojo                     # Dispatch overhead calibration kernel
├── results/                                   # Output JSON per run
└── README.md                                  # Reproduction instructions

scripts/
├── roofline_calibrate.py                      # Hardware ceiling measurement
├── benchmark_mlx_kernels.py                   # MLX side (mirrors Mojo kernel set)
└── compare_mojo_vs_mlx.py                     # Reads both JSONs → LaTeX, charts, roofline
```

### Existing Files Referenced (read-only unless specified)

```
mlx-lm/mlx_lm/models/mlx_isoquant.py          # IsoQuant rotation: structured_rotate_forward/inverse, build_isoquant_rotation_components
mlx-lm/mlx_lm/models/isoquant_metal_kernels.py # Metal kernels: metal_rotate_forward/inverse
mlx-lm/mlx_lm/models/fused_kv_decode_kernels.py # Fused pipeline: fused_qk_dot, fused_value_accum, fused_inverse_rotate, fully_fused_attention
mlx-lm/mlx_lm/models/mlx_turboquant.py        # TurboQuantCompressor, codebooks
mlx-lm/mlx_lm/models/base.py                  # scaled_dot_product_attention
mlx-lm/scripts/benchmark_fused_attention.py    # Reference: existing timing patterns
mlx-lm/scripts/benchmark_isoquant_rotation.py  # Reference: existing rotation benchmark
```

---

## Task 1: Hardware Roofline Calibration Script (M0)

**Files:**
- Create: `scripts/roofline_calibrate.py`
- Create: `results/` (directory)

This script measures peak FP16 TFLOPS, peak FP32 TFLOPS, peak memory bandwidth, and dispatch overhead for both MLX and (later) Mojo.

- [ ] **Step 1: Create results directory and script skeleton**

```bash
mkdir -p results
```

```python
# scripts/roofline_calibrate.py
"""Measure M4 Max hardware ceilings for roofline analysis.

Measures:
  - Peak FP16 TFLOPS (sustained FMA throughput)
  - Peak FP32 TFLOPS (sustained FMA throughput)
  - Peak memory bandwidth (stream-triad pattern)
  - MLX dispatch overhead (no-op mx.eval)
  - Power baseline (avg package + GPU watts via powermetrics)

Usage:
    python scripts/roofline_calibrate.py --output results/roofline_m4max.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time

import mlx.core as mx
import numpy as np


def measure_peak_flops(dtype: mx.Dtype, warmup: int = 10, iters: int = 50) -> float:
    """Sustained FMA throughput via large square matmul.

    Uses 4096x4096 matrices — large enough to saturate GPU compute units.
    Returns TFLOPS.
    """
    N = 4096
    a = mx.random.normal((N, N)).astype(dtype)
    b = mx.random.normal((N, N)).astype(dtype)
    mx.eval(a, b)

    # Warmup
    for _ in range(warmup):
        c = mx.matmul(a, b)
        mx.eval(c)

    # Measure
    t0 = time.perf_counter()
    for _ in range(iters):
        c = mx.matmul(a, b)
        mx.eval(c)
    elapsed = (time.perf_counter() - t0) / iters

    flops = 2 * N * N * N  # 2*N^3 for matmul
    tflops = flops / elapsed / 1e12
    return tflops


def measure_peak_bandwidth(warmup: int = 10, iters: int = 50) -> float:
    """Stream-triad pattern: a = b + scalar * c.

    Measures sustained memory bandwidth in GB/s.
    Uses large arrays to avoid cache effects.
    """
    N = 64 * 1024 * 1024  # 64M elements = 256 MB per array (float32)
    a = mx.zeros((N,), dtype=mx.float32)
    b = mx.random.normal((N,), dtype=mx.float32)
    c = mx.random.normal((N,), dtype=mx.float32)
    scalar = mx.array(3.0, dtype=mx.float32)
    mx.eval(a, b, c, scalar)

    for _ in range(warmup):
        a = b + scalar * c
        mx.eval(a)

    t0 = time.perf_counter()
    for _ in range(iters):
        a = b + scalar * c
        mx.eval(a)
    elapsed = (time.perf_counter() - t0) / iters

    # 2 reads (b, c) + 1 write (a) = 3 * N * 4 bytes
    bytes_moved = 3 * N * 4
    gbs = bytes_moved / elapsed / 1e9
    return gbs


def measure_dispatch_overhead(warmup: int = 100, iters: int = 500) -> float:
    """Time mx.eval() on a pre-compiled no-op to measure dispatch latency.

    Returns overhead in microseconds.
    """
    x = mx.array(0.0)
    mx.eval(x)

    for _ in range(warmup):
        y = x + 0.0
        mx.eval(y)

    t0 = time.perf_counter()
    for _ in range(iters):
        y = x + 0.0
        mx.eval(y)
    elapsed = (time.perf_counter() - t0) / iters

    return elapsed * 1e6  # microseconds


def read_powermetrics(duration_ms: int = 2000) -> dict:
    """Read average power from powermetrics (requires sudo).

    Returns dict with avg_package_watts and avg_gpu_watts, or None values
    if powermetrics is unavailable or not run with sudo.
    """
    try:
        result = subprocess.run(
            ["sudo", "powermetrics", "--samplers", "gpu_power,cpu_power",
             "-i", str(duration_ms), "-n", "1", "--format", "plist"],
            capture_output=True, text=True, timeout=duration_ms / 1000 + 5,
        )
        if result.returncode != 0:
            return {"avg_package_watts": None, "avg_gpu_watts": None}

        # Parse plist for power values (simplified — real impl uses plistlib)
        import plistlib
        data = plistlib.loads(result.stdout.encode())
        pkg_power = data.get("processor", {}).get("package_watts", None)
        gpu_power = data.get("gpu", {}).get("gpu_power", None)
        return {"avg_package_watts": pkg_power, "avg_gpu_watts": gpu_power}
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return {"avg_package_watts": None, "avg_gpu_watts": None}


def get_hardware_info() -> dict:
    """Collect hardware metadata."""
    import platform
    gpu_family = "unknown"
    try:
        # MLX exposes metal device info
        info = mx.metal.device_info()
        gpu_family = info.get("gpu_family", "unknown")
    except Exception:
        pass

    return {
        "chip": platform.processor() or "Apple Silicon",
        "memory_gb": round(mx.metal.device_info().get("memory_size", 0) / (1024**3)),
        "gpu_cores": mx.metal.device_info().get("max_threads_per_threadgroup", 0),
        "macos_version": platform.mac_ver()[0],
        "metal_gpu_family": gpu_family,
    }


def main():
    parser = argparse.ArgumentParser(description="Roofline hardware calibration")
    parser.add_argument("--output", type=str, default="results/roofline_m4max.json")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--skip-power", action="store_true",
                        help="Skip powermetrics (requires sudo)")
    args = parser.parse_args()

    print("=== Roofline Calibration ===")

    print("Measuring peak FP16 TFLOPS...")
    fp16_tflops = measure_peak_flops(mx.float16, args.warmup, args.iters)
    print(f"  Peak FP16: {fp16_tflops:.2f} TFLOPS")

    print("Measuring peak FP32 TFLOPS...")
    fp32_tflops = measure_peak_flops(mx.float32, args.warmup, args.iters)
    print(f"  Peak FP32: {fp32_tflops:.2f} TFLOPS")

    print("Measuring peak memory bandwidth...")
    bw_gbs = measure_peak_bandwidth(args.warmup, args.iters)
    theoretical_bw = 546.0  # M4 Max 128GB: 16-channel LPDDR5X-8533
    bw_ratio = bw_gbs / theoretical_bw
    print(f"  Peak bandwidth: {bw_gbs:.1f} GB/s (measured)")
    print(f"  Theoretical: {theoretical_bw:.0f} GB/s, ratio: {bw_ratio:.2f} (expected 0.73-0.85)")

    print("Measuring dispatch overhead...")
    dispatch_us = measure_dispatch_overhead()
    print(f"  Dispatch overhead: {dispatch_us:.2f} us")

    power = {"avg_package_watts": None, "avg_gpu_watts": None}
    if not args.skip_power:
        print("Measuring baseline power (2s sample)...")
        power = read_powermetrics()
        if power["avg_package_watts"]:
            print(f"  Package: {power['avg_package_watts']:.1f} W, GPU: {power['avg_gpu_watts']:.1f} W")
        else:
            print("  Power metrics unavailable (run with sudo for power data)")

    result = {
        "hardware": get_hardware_info(),
        "calibration": {
            "peak_fp16_tflops": round(fp16_tflops, 2),
            "peak_fp32_tflops": round(fp32_tflops, 2),
            "peak_memory_bandwidth_gbs": round(bw_gbs, 1),
            "theoretical_memory_bandwidth_gbs": 546.0,
            "bandwidth_efficiency_ratio": round(bw_gbs / 546.0, 3),
            "noop_dispatch_us": round(dispatch_us, 2),
        },
        "power_baseline": power,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run calibration script (smoke test)**

Run: `cd /Users/anthonylui/QwenCoderLocal && python scripts/roofline_calibrate.py --output results/roofline_m4max.json --skip-power --iters 5 --warmup 2`

Expected: JSON written to `results/roofline_m4max.json` with non-zero values for all calibration fields. FP16 should be ~20-27 TFLOPS, FP32 ~10-14 TFLOPS, bandwidth ~300-450 GB/s on M4 Max.

- [ ] **Step 3: Commit M0 calibration script**

```bash
git add scripts/roofline_calibrate.py
git commit -m "feat(benchmark): add roofline hardware calibration script (M0)"
```

---

## Task 2: Mojo Environment and Harness Setup (M1)

**Files:**
- Create: `mojo-bench/pixi.toml`
- Create: `mojo-bench/harness/stats.mojo`
- Create: `mojo-bench/harness/output.mojo`
- Create: `mojo-bench/harness/noop_dispatch.mojo`
- Create: `mojo-bench/kernels/bench_vec_add.mojo` (smoke test)
- Create: `mojo-bench/README.md`

**Delegation:** Dispatch to Gemini CLI:
```bash
delegate-to-gemini "Create Mojo benchmark environment in mojo-bench/. See spec at docs/superpowers/specs/2026-04-13-mojo-vs-mlx-kernel-benchmark-design.md sections 1, 4.1-4.6, 4.8. Create: pixi.toml (pin Mojo compiler, add bench-all task), harness/stats.mojo (timing with warmup, adaptive iteration count stopping at CI<2% median or 500 iters max, Durbin-Watson thermal detection on time series, BCa bootstrap with 10k samples), harness/output.mojo (JSON writer with framework-agnostic compilation schema per spec section 4.8), harness/noop_dispatch.mojo (no-op kernel for dispatch overhead measurement), and kernels/bench_vec_add.mojo (simple GPU vector add smoke test). All GPU work must use ctx.synchronize() for timing. Include power metric placeholders in JSON output."
```

- [ ] **Step 1: Create `mojo-bench/pixi.toml`**

```toml
[project]
name = "mojo-bench"
description = "Mojo GPU kernel benchmarks for Apple Silicon"
channels = ["https://conda.modular.com/max-nightly", "conda-forge"]
platforms = ["osx-arm64"]

[dependencies]
max = ">=26.2"

[tasks]
bench-all = "mojo run kernels/bench_vec_add.mojo"
bench-smoke = "mojo run kernels/bench_vec_add.mojo"
```

- [ ] **Step 2: Create `mojo-bench/harness/noop_dispatch.mojo`**

Minimal GPU kernel that does nothing, used to measure dispatch overhead:

```mojo
from gpu.host import DeviceContext
from time import perf_counter_ns

fn measure_noop_dispatch(warmup: Int = 100, iters: Int = 500) -> Float64:
    """Measure GPU dispatch overhead in microseconds."""
    var ctx = DeviceContext()

    # Warmup
    for _ in range(warmup):
        ctx.synchronize()

    # Measure
    var start = perf_counter_ns()
    for _ in range(iters):
        ctx.synchronize()
    var elapsed = perf_counter_ns() - start

    return Float64(elapsed) / Float64(iters) / 1000.0  # ns -> us


fn main():
    var overhead_us = measure_noop_dispatch()
    print("Dispatch overhead:", overhead_us, "us")
```

- [ ] **Step 3: Create `mojo-bench/harness/stats.mojo`**

Benchmark harness with adaptive iteration, BCa bootstrap, Durbin-Watson, and power metrics:

```mojo
from collections import List
from math import sqrt, log
from time import perf_counter_ns

alias MAX_ITERS = 500
alias BATCH_SIZE = 25
alias WARMUP_ITERS = 10
alias BOOTSTRAP_SAMPLES = 10_000
alias CI_TARGET_PCT = 2.0  # Stop when CI width < 2% of median
alias DW_THRESHOLD = 1.5   # Flag if Durbin-Watson < 1.5


@value
struct BenchResult:
    var mean_us: Float64
    var median_us: Float64
    var std_us: Float64
    var p5_us: Float64
    var p95_us: Float64
    var p99_us: Float64
    var ci95_lo: Float64
    var ci95_hi: Float64
    var n_iterations: Int
    var dw_statistic: Float64
    var timings_us: List[Float64]


fn sort_float_list(inout arr: List[Float64]):
    """Simple insertion sort for benchmark result lists."""
    for i in range(1, len(arr)):
        var key = arr[i]
        var j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


fn percentile(sorted_arr: List[Float64], pct: Float64) -> Float64:
    """Get percentile from sorted array."""
    var idx = Int(Float64(len(sorted_arr) - 1) * pct / 100.0)
    return sorted_arr[idx]


fn compute_mean(arr: List[Float64]) -> Float64:
    var s: Float64 = 0.0
    for i in range(len(arr)):
        s += arr[i]
    return s / Float64(len(arr))


fn compute_std(arr: List[Float64], mean: Float64) -> Float64:
    var s: Float64 = 0.0
    for i in range(len(arr)):
        var d = arr[i] - mean
        s += d * d
    return sqrt(s / Float64(len(arr) - 1)) if len(arr) > 1 else 0.0


fn durbin_watson(timings: List[Float64]) -> Float64:
    """Compute Durbin-Watson statistic for serial autocorrelation detection."""
    if len(timings) < 3:
        return 2.0  # No autocorrelation assumed
    var sum_sq_diff: Float64 = 0.0
    var sum_sq_resid: Float64 = 0.0
    var mean = compute_mean(timings)
    for i in range(len(timings)):
        var resid = timings[i] - mean
        sum_sq_resid += resid * resid
        if i > 0:
            var diff = (timings[i] - mean) - (timings[i-1] - mean)
            sum_sq_diff += diff * diff
    return sum_sq_diff / sum_sq_resid if sum_sq_resid > 0 else 2.0


fn bca_bootstrap_ci(timings: List[Float64], seed: Int = 42) -> Tuple[Float64, Float64]:
    """BCa bootstrap 95% CI with 10k samples.

    Simplified implementation — production version should use proper
    BCa bias-correction and acceleration. This implements percentile
    bootstrap as a baseline; Codex review should verify BCa correctness.
    """
    var n = len(timings)
    if n < 5:
        var lo = timings[0] if n > 0 else 0.0
        var hi = timings[n - 1] if n > 0 else 0.0
        return (lo, hi)

    var boot_means = List[Float64]()
    # Simple LCG for reproducibility (no stdlib rand in Mojo yet)
    var rng_state = seed
    for _ in range(BOOTSTRAP_SAMPLES):
        var s: Float64 = 0.0
        for _ in range(n):
            rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
            var idx = rng_state % n
            s += timings[idx]
        boot_means.append(s / Float64(n))

    sort_float_list(boot_means)
    var lo_idx = Int(Float64(BOOTSTRAP_SAMPLES) * 0.025)
    var hi_idx = Int(Float64(BOOTSTRAP_SAMPLES) * 0.975)
    return (boot_means[lo_idx], boot_means[hi_idx])
```

- [ ] **Step 4: Create `mojo-bench/harness/output.mojo`**

JSON output writer using the framework-agnostic schema from spec section 4.8:

```mojo
from collections import Dict
from pathlib import Path

fn write_kernel_result_json(
    framework: String,
    framework_version: String,
    hardware: Dict[String, String],
    compilation: Dict[String, String],
    calibration: Dict[String, Float64],
    kernel_name: String,
    shape_name: String,
    result: BenchResult,
    tflops: Float64,
    gbs: Float64,
    roofline_pct: Float64,
    power: Dict[String, Float64],
    output_path: String,
):
    """Write benchmark result to JSON file.

    Uses framework-agnostic 'compilation' field per spec:
    {
      "compilation": {
        "mode": "ahead_of_time",
        "optimization": "--release",
        "graph_compilation": "N/A"
      }
    }
    """
    # Implementation: build JSON string manually (Mojo lacks serde)
    # Gemini will implement full JSON serialization here
    pass
```

- [ ] **Step 5: Create `mojo-bench/kernels/bench_vec_add.mojo` (smoke test)**

```mojo
"""Smoke test: GPU vector addition to verify Mojo Metal backend works."""
from gpu.host import DeviceContext
from gpu import thread_idx, block_idx, block_dim
from layout import LayoutTensor, Layout
from memory import UnsafePointer
from time import perf_counter_ns


fn gpu_vec_add_kernel[
    layout: Layout
](a: LayoutTensor[mut=False, dtype=DType.float32, layout=layout],
  b: LayoutTensor[mut=False, dtype=DType.float32, layout=layout],
  c: LayoutTensor[mut=True, dtype=DType.float32, layout=layout]):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    if tid < a.shape[0]:
        c[tid] = a[tid] + b[tid]


fn main():
    var N = 1024
    var ctx = DeviceContext()

    print("Mojo GPU smoke test: vector add (N=", N, ")")

    # Allocate and fill host buffers
    var host_a = UnsafePointer[Float32].alloc(N)
    var host_b = UnsafePointer[Float32].alloc(N)
    var host_c = UnsafePointer[Float32].alloc(N)

    for i in range(N):
        host_a[i] = Float32(i)
        host_b[i] = Float32(i * 2)

    # Copy to device, launch kernel, copy back
    var dev_a = ctx.enqueue_create_buffer[DType.float32](N)
    var dev_b = ctx.enqueue_create_buffer[DType.float32](N)
    var dev_c = ctx.enqueue_create_buffer[DType.float32](N)

    ctx.enqueue_copy(dev_a, host_a, N)
    ctx.enqueue_copy(dev_b, host_b, N)

    var threads_per_block = 256
    var blocks = (N + threads_per_block - 1) // threads_per_block

    ctx.enqueue_function[gpu_vec_add_kernel[Layout.row_major(N)]](
        dev_a.as_tensor(), dev_b.as_tensor(), dev_c.as_tensor(),
        grid_dim=(blocks,), block_dim=(threads_per_block,),
    )

    ctx.enqueue_copy(host_c, dev_c, N)
    ctx.synchronize()

    # Verify
    var max_err: Float64 = 0.0
    for i in range(N):
        var expected = Float32(i + i * 2)
        var err = abs(Float64(host_c[i]) - Float64(expected))
        if err > max_err:
            max_err = err

    print("Max error:", max_err)
    if max_err < 1e-6:
        print("PASS")
    else:
        print("FAIL")

    host_a.free()
    host_b.free()
    host_c.free()
```

- [ ] **Step 6: Create `mojo-bench/README.md`**

```markdown
# Mojo GPU Kernel Benchmarks

Kernel-level benchmarks for Mojo GPU on Apple Silicon (Metal backend).
Part of the Mojo vs MLX framework comparison in `docs/FROM_ATTENTION_TO_CONSUMER_HARDWARE.md`.

## Setup

```bash
pixi install
```

## Run

```bash
pixi run bench-smoke    # Vector add smoke test
pixi run bench-all      # Full benchmark suite
```

## Results

Output JSON files are written to `results/`.
Compare with MLX results using `scripts/compare_mojo_vs_mlx.py`.

See `docs/superpowers/specs/2026-04-13-mojo-vs-mlx-kernel-benchmark-design.md` for methodology.
```

- [ ] **Step 7: Install Mojo environment and run smoke test**

```bash
cd mojo-bench && pixi install
pixi run bench-smoke
```

Expected: "PASS" with max error < 1e-6. If this fails, Mojo GPU/Metal is not working on this machine — stop and investigate before proceeding.

- [ ] **Step 8: Gate G1 — Codex adversarial review of M1**

```
/codex:adversarial-review --background
```

Review focus: Mojo compiles on M4 Max, GPU detected, timing sync correct, stats.mojo BCa implementation, output.mojo schema matches spec section 4.8.

- [ ] **Step 9: Fix Codex findings and commit M1**

```bash
git add mojo-bench/
git commit -m "feat(benchmark): add Mojo environment and benchmark harness (M1)"
```

---

## Task 3: Standard Mojo Kernels — MatMul, Softmax, RoPE (M2)

**Files:**
- Create: `mojo-bench/kernels/bench_matmul.mojo`
- Create: `mojo-bench/kernels/bench_softmax.mojo`
- Create: `mojo-bench/kernels/bench_rope.mojo`

**Delegation:** Dispatch to Gemini CLI:
```bash
delegate-to-gemini "Implement 3 standard Mojo GPU kernels in mojo-bench/kernels/. Spec: docs/superpowers/specs/2026-04-13-mojo-vs-mlx-kernel-benchmark-design.md sections 3.1, 3.2.

1. bench_matmul.mojo — Dense GEMM (FP16 and FP32). Tiled implementation using threadgroup memory. Size matrix from spec section 3.2: Decode GEMV (1x6144x6144), Prefill QKV (2048x6144x6144), FFN up/down (1x6144x16384, 1x16384x6144), Prefill FFN up/down (2048x6144x16384, 2048x16384x6144), MoE expert (1x5120x11008), plus standard square sweep {512,1024,2048,4096,8192}. Report TFLOPS as primary metric.

2. bench_softmax.mojo — Row-wise softmax over (B,H,S,S) tensors. Numerically stable (max-subtract). Sizes: S in {128,512,2048,8192,32768} with H=48,D=128. Report GB/s.

3. bench_rope.mojo — Rotary position embeddings on (B,H,S,D) tensors. Pre-compute sin/cos frequencies. Same sequence sweep. Report GB/s.

Each kernel must: use harness/stats.mojo for adaptive timing, write JSON via harness/output.mojo, pre-allocate and pre-touch all buffers before warmup per spec section 4.1, use ctx.synchronize() for timing, seed=42 for input data."
```

- [ ] **Step 1: Implement `bench_matmul.mojo`**

The kernel must implement tiled GEMM using threadgroup shared memory. Input shapes from spec section 3.2 (decode GEMVs, prefill QKV, FFN up/down, prefill FFN, MoE expert, square sweep). Reports TFLOPS = 2*M*N*K / elapsed / 1e12.

Key implementation requirements:
- Pre-allocate all matrices for all sizes before any timing
- Use `ctx.synchronize()` after each `mx.eval` equivalent
- Adaptive iteration count from `harness/stats.mojo`
- JSON output per spec 4.8 schema

- [ ] **Step 2: Implement `bench_softmax.mojo`**

Row-wise softmax: `out[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))`. Numerically stable max-subtract pattern. Input shape `(1, 48, S, S)` for S in `{128, 512, 2048, 8192, 32768}`. Reports GB/s = bytes_read_and_written / elapsed / 1e9.

- [ ] **Step 3: Implement `bench_rope.mojo`**

RoPE: apply rotary embeddings using pre-computed sin/cos frequency table. Input shape `(1, 48, S, 128)`. Same sequence sweep. Reports GB/s.

- [ ] **Step 4: Run all three kernels**

```bash
cd mojo-bench
mojo run kernels/bench_matmul.mojo
mojo run kernels/bench_softmax.mojo
mojo run kernels/bench_rope.mojo
```

Expected: JSON files in `mojo-bench/results/` with non-zero TFLOPS/GB/s values. Verify numerical correctness against numpy reference (max abs error within spec thresholds from section 5.2).

- [ ] **Step 5: Roofline check — verify >50% of peak**

For each kernel/size, compute `roofline_pct = measured / peak * 100`. If any kernel achieves <50% of roofline, investigate whether this is framework limitation or suboptimal code (spec section 6.3).

- [ ] **Step 6: Gate G1 — Codex adversarial review of M2**

```
/codex:adversarial-review --background
```

Review focus: Kernel equivalence to MLX ops (same algorithms), no unfair advantages (e.g., fused ops in one but not other), roofline >50%, tiling/vectorization quality, memory access patterns.

- [ ] **Step 7: Fix Codex findings and commit M2**

```bash
git add mojo-bench/kernels/bench_matmul.mojo mojo-bench/kernels/bench_softmax.mojo mojo-bench/kernels/bench_rope.mojo
git commit -m "feat(benchmark): add standard Mojo kernels — MatMul, Softmax, RoPE (M2)"
```

---

## Task 4a: IsoQuant Rotation Kernel (M3a)

**Files:**
- Create: `mojo-bench/kernels/bench_isoquant_rotate.mojo`

**Risk:** Low — self-contained math, no external dependencies beyond harness.

**Delegation:** Dispatch to Gemini CLI:
```bash
delegate-to-gemini "Implement bench_isoquant_rotate.mojo with TWO sub-benchmarks. Spec: docs/superpowers/specs/2026-04-13-mojo-vs-mlx-kernel-benchmark-design.md section 3.1. Reference: mlx-lm/mlx_lm/models/mlx_isoquant.py (structured_rotate_forward/inverse, build_isoquant_rotation_components).

Forward: WHT + SO(4) forward rotation on batch (H,S,D) — write-path cost.
Inverse: Structured inverse rotation on single vector (H,1,D) — read-path cost.
Both dense and structured implementations for each. Measure speedup of structured vs dense. Report GB/s.
Math: isoclinic SO(4) via quaternion pairs (q_L * v * conj(q_R)) on 4D blocks."
```

- [ ] **Step 1: Implement `bench_isoquant_rotate.mojo` with forward and inverse sub-benchmarks**

Forward rotation: WHT (Walsh-Hadamard Transform) + SO(4) block rotation on batch of vectors `(H, S, D)`. Uses 4x4 block matrices from isoclinic quaternion decomposition.

Inverse rotation: Same block matrices, transpose applied to single output vector `(H, 1, D)`.

Both must have a dense matmul reference implementation alongside the structured (block-diagonal) version.

Key math from `mlx_isoquant.py:275-290`:
- Forward: `x_rot = structured_rotate_forward(x, block_matrices_t, use_hadamard)` — applies WHT then 4x4 block matmuls
- Inverse: `x_out = structured_rotate_inverse(x_rot, block_matrices, use_hadamard)` — applies 4x4 block matmuls then inverse WHT

- [ ] **Step 2: Run and verify rotation kernel**

```bash
cd mojo-bench && mojo run kernels/bench_isoquant_rotate.mojo
```

Expected: Structured speedup >1x vs dense at D=128. Numerical RMSE < sqrt(D) * 1e-5 for both forward and inverse.

- [ ] **Step 3: Gate G1 — Codex adversarial review of M3a**

```
/codex:adversarial-review --background
```

Review focus: Correct rotation math, structured vs dense equivalence, both forward and inverse paths tested.

- [ ] **Step 4: Fix Codex findings and commit M3a**

```bash
git add mojo-bench/kernels/bench_isoquant_rotate.mojo
git commit -m "feat(benchmark): add IsoQuant rotation kernel — forward + inverse (M3a)"
```

---

## Task 4b: KV Compression Kernel (M3b)

**Files:**
- Create: `mojo-bench/kernels/bench_kv_compress.mojo`

**Risk:** Low — depends on M3a patterns but is self-contained.

- [ ] **Step 1: Implement `bench_kv_compress.mojo` with codebook VQ**

Pipeline: quantize input vectors to 3-bit indices via nearest-centroid lookup → pack indices → unpack → dequantize via centroid gather + norm rescaling.

Load codebook from `mlx-lm/mlx_lm/models/turboquant_codebooks/dim_128_3bit.npz`. Convert centroids to Mojo tensors.

Report: GB/s (bytes processed / time) and reconstruction RMSE (compare dequantized vs original).

- [ ] **Step 2: Run and verify KV compression**

```bash
cd mojo-bench && mojo run kernels/bench_kv_compress.mojo
```

Expected: Reconstruction RMSE < theoretical_quantization_error * 1.1. GB/s reported for all shapes.

- [ ] **Step 3: Codex review and commit M3b**

```bash
git add mojo-bench/kernels/bench_kv_compress.mojo
git commit -m "feat(benchmark): add KV compression kernel (M3b)"
```

---

## Task 4c: Fused Attention — Unfused + Framework-Fused (M3c)

**Files:**
- Create: `mojo-bench/kernels/bench_fused_attention.mojo` (initial: unfused + framework-fused only)

**Risk:** Medium — composition of prior kernels, timing methodology matters.

- [ ] **Step 1: Implement unfused and framework-fused variants (IsoQuant pipeline)**

Both variants implement: QK dot product → softmax → V accumulation → inverse rotation.

**Unfused**: 4 separate kernel launches with `ctx.synchronize()` between each.
**Framework-fused**: Single composed function, one `ctx.synchronize()` at the end.

The delta between these two is a key result — it measures Mojo's ability to fuse dispatch and memory traffic.

**K/V rotation sharing (resolved)**: TurboQuant uses a shared rotation for K and V (self-cancellation). IsoQuant uses independent per-head rotations (`k_isoquant_rot`, `v_isoquant_rot`) but its fused pipeline operates in rotated space and applies inverse once on the aggregated output.

- [ ] **Step 2: Run and verify fused attention (unfused + framework-fused)**

```bash
cd mojo-bench && mojo run kernels/bench_fused_attention.mojo
```

Expected: Framework-fused should be faster than unfused. Both should produce numerically equivalent outputs.

- [ ] **Step 3: Codex review and commit M3c**

```bash
git add mojo-bench/kernels/bench_fused_attention.mojo
git commit -m "feat(benchmark): add fused attention — unfused + framework-fused (M3c)"
```

---

## Task 4d: Fused Attention — Hand-Fused + TurboQuant + Cost Breakdown (M3d)

**Files:**
- Modify: `mojo-bench/kernels/bench_fused_attention.mojo` (add hand-fused variant + TurboQuant)

**Risk:** High — hand-fused single kernel may be infeasible on Tier 3 Metal. If infeasible, M3a-M3c still proceed.

- [ ] **Step 1: Attempt hand-fused single-kernel variant**

Single GPU kernel dispatch implementing online softmax (no intermediate materialisation of attention weights). If Mojo's Tier 3 Metal backend cannot express this, document the limitation and skip — the unfused and framework-fused variants from M3c remain valid.

- [ ] **Step 2: Add TurboQuant pipeline variant**

Implement the same three variants (unfused/framework-fused/hand-fused) using TurboQuant's dense rotation instead of IsoQuant's structured rotation. The cost breakdown must isolate per-sub-operation timing:
- (a) Per-query rotation cost (structured O(d_k log d_k) for IsoQuant vs dense O(d_k²) for TurboQuant)
- (b) T-linear key scan (same for both — operates in rotated space)
- (c) T-linear value accumulation (same for both — accumulates in rotated space)
- (d) Per-output inverse rotation (same structured vs dense split as (a))

Any measured IsoQuant speedup is a constant-factor advantage at the rotation steps, not an asymptotic scaling difference in T.

- [ ] **Step 3: Gate G1 — Codex adversarial review of M3d**

```
/codex:adversarial-review --background
```

Review focus: Hand-fused feasibility, TurboQuant variant correctness, cost framing matches spec (constant-factor, not asymptotic).

- [ ] **Step 4: Fix Codex findings and commit M3d**

```bash
git add mojo-bench/kernels/bench_fused_attention.mojo
git commit -m "feat(benchmark): add hand-fused + TurboQuant variants + cost breakdown (M3d)"
```

---

## Task 5: MLX Benchmark Script (M4)

**Files:**
- Create: `scripts/benchmark_mlx_kernels.py`

**Delegation:** Dispatch to Gemini CLI:
```bash
delegate-to-gemini "Create scripts/benchmark_mlx_kernels.py mirroring the Mojo kernel set. Spec: docs/superpowers/specs/2026-04-13-mojo-vs-mlx-kernel-benchmark-design.md sections 3, 4. Reference existing benchmarks: mlx-lm/scripts/benchmark_fused_attention.py, mlx-lm/scripts/benchmark_isoquant_rotation.py.

Requirements:
- All operations use stream=mx.gpu to prevent AMX routing
- All benchmark functions wrapped in mx.compile() before warmup
- Adaptive iteration count (CI<2% median or 500 max)
- BCa bootstrap (10k samples) via scipy.stats.bootstrap
- Durbin-Watson via statsmodels.stats.stattools.durbin_watson
- powermetrics integration for energy metrics (avg_package_watts, avg_gpu_watts, throughput_per_watt)
- Framework-agnostic JSON output per spec section 4.8

Kernels to benchmark (same shapes as Mojo side):
1. MatMul — mx.matmul, all shapes from spec 3.2 including Prefill FFN up/down
2. Softmax — mx.softmax, sequence sweep
3. RoPE — standard apply_rotary_pos_emb pattern
4. IsoQuant Rotate Forward — two implementations: (a) dense mx.matmul, (b) structured_rotate_forward from mlx_isoquant.py
5. IsoQuant Rotate Inverse — two implementations: (a) dense mx.matmul, (b) structured_rotate_inverse from mlx_isoquant.py
6. KV Compress — two implementations: (a) high-level TurboQuantCompressor ops, (b) custom Metal path if available
7. Fused Attention — THREE variants × TWO pipelines:
   Variants: a) Unfused (individual mx.eval() per op), b) Framework-fused (mx.compile(), single mx.eval()), c) Hand-fused (fully_fused_attention single Metal kernel)
   Pipelines: IsoQuant (structured rotation) AND TurboQuant (dense rotation, self-cancellation)
   Cost breakdown must isolate: per-query rotation, T-linear key scan, T-linear value accum, per-output inverse rotation
   Critical: frame any IsoQuant speedup as constant-factor (d_k log d_k vs d_k²), not asymptotic scaling, unless multi-T data proves otherwise

For IsoQuant and KV Compress, provide TWO MLX implementations (high-level ops vs custom Metal). Label the custom Metal version as 'hand-optimized Metal', not 'MLX'.

JSON output with compilation field:
{
  'compilation': {
    'mode': 'jit',
    'optimization': 'mx.compile()',
    'graph_compilation': 'enabled'
  }
}"
```

- [ ] **Step 1: Create `scripts/benchmark_mlx_kernels.py` skeleton**

```python
#!/usr/bin/env python3
"""MLX Metal kernel benchmarks — mirrors the Mojo kernel set.

Measures all kernels from the Mojo vs MLX benchmark spec with:
- mx.compile() wrapping all benchmark functions
- stream=mx.gpu on all operations (prevent AMX routing)
- Adaptive iteration count (CI < 2% median or 500 max)
- BCa bootstrap (10k samples)
- Durbin-Watson thermal detection
- powermetrics energy integration

Usage:
    python scripts/benchmark_mlx_kernels.py --output results/mlx_kernels.json
    python scripts/benchmark_mlx_kernels.py --kernel matmul --output results/mlx_matmul.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np
from scipy import stats as scipy_stats

# Import IsoQuant and fused attention components
from mlx_lm.models.mlx_isoquant import (
    build_isoquant_rotation_components,
    structured_rotate_forward,
    structured_rotate_inverse,
)
from mlx_lm.models.fused_kv_decode_kernels import (
    fused_qk_dot,
    fused_value_accum,
    fused_inverse_rotate,
    fully_fused_attention,
    pack_indices_3bit,
)
from mlx_lm.models.mlx_turboquant import TurboQuantCompressor, get_default_codebook_dir


@dataclass
class BenchResult:
    mean_us: float = 0.0
    median_us: float = 0.0
    std_us: float = 0.0
    p5_us: float = 0.0
    p95_us: float = 0.0
    p99_us: float = 0.0
    ci95_bca: tuple[float, float] = (0.0, 0.0)
    ci_converged: bool = True  # False if max_iters reached without CI < 2%
    ci_width_pct: float = 0.0  # Actual CI width as % of median
    n_iterations: int = 0
    dw_statistic: float = 2.0
    runs_test_p: float = 1.0  # Wald-Wolfowitz runs test p-value
    timings_us: list[float] = field(default_factory=list)


MAX_ITERS = 500
BATCH_SIZE = 25
WARMUP_ITERS = 10


def adaptive_bench(fn, warmup: int = WARMUP_ITERS, max_iters: int = MAX_ITERS) -> BenchResult:
    """Run benchmark with adaptive iteration count.

    Stops when 95% CI width < 2% of median, or after max_iters.

    Failure mode: If max_iters reached without CI convergence, result is
    reported with ci_converged=False and actual CI width. If CI width > 10%
    of median, the result is excluded from cross-framework comparison but
    included in raw data tables.

    Thermal detection: Uses both Durbin-Watson (parametric) and
    Wald-Wolfowitz runs test (non-parametric, distribution-free backup).
    If DW < 1.5 OR runs-test p-value < 0.05, flags for cooldown and retry.
    """
    # Warmup
    for _ in range(warmup):
        fn()
        mx.eval()  # Ensure GPU work completes

    timings: list[float] = []

    while len(timings) < max_iters:
        for _ in range(BATCH_SIZE):
            t0 = time.perf_counter()
            fn()
            mx.eval()  # Synchronize
            elapsed = time.perf_counter() - t0
            timings.append(elapsed * 1e6)  # Convert to us

            if len(timings) >= max_iters:
                break

        # Check CI convergence
        if len(timings) >= 2 * BATCH_SIZE:
            arr = np.array(timings)
            median = float(np.median(arr))
            if median > 0:
                try:
                    ci = scipy_stats.bootstrap(
                        (arr,), np.median, n_resamples=1000,  # Quick check
                        confidence_level=0.95, method="BCa",
                    )
                    ci_width = ci.confidence_interval.high - ci.confidence_interval.low
                    if ci_width / median * 100 < 2.0:
                        break
                except Exception:
                    pass

    arr = np.array(timings)
    sorted_arr = np.sort(arr)

    # Full BCa bootstrap for final CI
    try:
        ci = scipy_stats.bootstrap(
            (arr,), np.median, n_resamples=10_000,
            confidence_level=0.95, method="BCa",
        )
        ci95 = (float(ci.confidence_interval.low), float(ci.confidence_interval.high))
    except Exception:
        ci95 = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    # Durbin-Watson (parametric)
    try:
        from statsmodels.stats.stattools import durbin_watson
        dw = float(durbin_watson(arr - np.mean(arr)))
    except ImportError:
        residuals = arr - np.mean(arr)
        dw = float(np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2))

    # Wald-Wolfowitz runs test (non-parametric backup for skewed distributions)
    median_val = np.median(arr)
    binary = (arr > median_val).astype(int)
    runs = 1 + int(np.sum(np.abs(np.diff(binary))))
    n1 = int(np.sum(binary))
    n0 = len(binary) - n1
    if n0 > 0 and n1 > 0:
        expected_runs = 1 + 2 * n0 * n1 / (n0 + n1)
        var_runs = 2 * n0 * n1 * (2 * n0 * n1 - n0 - n1) / ((n0 + n1)**2 * (n0 + n1 - 1))
        z_runs = (runs - expected_runs) / (var_runs**0.5) if var_runs > 0 else 0.0
        from scipy.stats import norm
        runs_p = float(2 * norm.sf(abs(z_runs)))  # two-tailed
    else:
        runs_p = 1.0

    return BenchResult(
        mean_us=float(np.mean(arr)),
        median_us=float(np.median(arr)),
        std_us=float(np.std(arr, ddof=1)),
        p5_us=float(np.percentile(arr, 5)),
        p95_us=float(np.percentile(arr, 95)),
        p99_us=float(np.percentile(arr, 99)),
        ci95_bca=ci95,
        n_iterations=len(timings),
        dw_statistic=dw,
        timings_us=timings,
    )


# ... (Gemini implements all kernel benchmarks here, following the delegation prompt)
```

- [ ] **Step 2: Implement MatMul, Softmax, RoPE benchmarks (standard kernels)**

Each uses `mx.compile()` wrapping, `stream=mx.gpu`, and `adaptive_bench()`. Same shapes as Mojo side. MatMul reports TFLOPS, Softmax and RoPE report GB/s.

- [ ] **Step 3: Implement IsoQuant Rotate Forward + Inverse benchmarks (two implementations each)**

For each direction (forward, inverse), implement:
- **MLX high-level**: `mx.matmul(x, R_T)` (dense matrix multiply)
- **MLX custom Metal**: `structured_rotate_forward()` / `structured_rotate_inverse()` from `mlx_isoquant.py`
- **MLX Metal kernel**: `metal_rotate_forward()` / `metal_rotate_inverse()` from `isoquant_metal_kernels.py`

Inputs: Forward gets `(H, S, D)` for S in prefill sweep, Inverse gets `(H, 1, D)` for decode.

- [ ] **Step 4: Implement KV Compress benchmark (two implementations)**

- **MLX high-level**: `TurboQuantCompressor.compress()` + `decompress()`
- **MLX custom Metal**: Direct use of fused Metal path if available

Report: GB/s + reconstruction RMSE.

- [ ] **Step 5: Implement Fused Attention benchmark (three variants)**

- **Unfused**: Individual `mx.eval()` after each of `fused_qk_dot()`, `mx.softmax()`, `fused_value_accum()`, `fused_inverse_rotate()`
- **Framework-fused**: Wrap the full pipeline in `mx.compile()`, single `mx.eval()`
- **Hand-fused**: Call `fully_fused_attention()` — single Metal kernel dispatch

- [ ] **Step 6: Add JSON output and powermetrics integration**

JSON output uses framework-agnostic `compilation` field:
```python
"compilation": {
    "mode": "jit",
    "optimization": "mx.compile()",
    "graph_compilation": "enabled"
}
```

Power metrics collected via `powermetrics` subprocess during each kernel suite. Include `measurement_window_ms` (actual integration time) and `low_confidence: true` if window < 500ms. In the paper's energy charts, aggregate to kernel-class level for meaningful averages (powermetrics samples at ~250ms intervals, so sub-second kernel runs have limited power precision).

- [ ] **Step 7: Run MLX benchmark suite**

```bash
cd /Users/anthonylui/QwenCoderLocal
python scripts/benchmark_mlx_kernels.py --output results/mlx_kernels.json
```

Expected: JSON with all kernel results. Verify numerical correctness against numpy references.

- [ ] **Step 8: Gate G2 — Codex adversarial review of M4**

```
/codex:adversarial-review --background
```

Review focus: Methodology parity with Mojo side, AMX routing controlled (stream=mx.gpu everywhere), mx.compile() applied before warmup, two implementations for novel kernels, three-variant fused attention, JSON schema matches spec.

- [ ] **Step 9: Fix Codex findings and commit M4**

```bash
git add scripts/benchmark_mlx_kernels.py
git commit -m "feat(benchmark): add MLX kernel benchmark script (M4)"
```

---

## Task 6: Perplexity Validation Script (G2.5)

**Files:**
- Create: `scripts/validate_kernel_precision.py`

- [ ] **Step 1: Create precision validation script**

```python
#!/usr/bin/env python3
"""Kernel-chain precision validation with perplexity gate.

Simulates one attention layer: MatMul → RoPE → Softmax → MatMul → KV Compress.
Feeds identical inputs through both MLX and numpy reference, measures divergence.
Reports equivalent perplexity delta.

Gate: 0.5% perplexity divergence threshold (Dettmers et al. 2022).

Precision reference: For novel kernels with two MLX implementations (high-level
and custom Metal), the HIGH-LEVEL implementation is the numerical reference for
both frameworks. The custom Metal implementation's divergence from the high-level
reference is reported separately (known divergence: max_abs_diff ~4.3 for fused
Metal decode path vs composed FP32 reference).

Perplexity proxy: The relationship between partial-pipeline and full-pipeline
divergence is non-monotonic — we report it as an informative proxy, not a bound.
"""
from __future__ import annotations

import argparse
import json

import mlx.core as mx
import numpy as np

from mlx_lm.models.mlx_isoquant import (
    build_isoquant_rotation_components,
    structured_rotate_forward,
    structured_rotate_inverse,
)
from mlx_lm.models.mlx_turboquant import TurboQuantCompressor, get_default_codebook_dir


def compute_per_kernel_error(
    kernel_name: str,
    mlx_output: mx.array,
    ref_output: np.ndarray,
) -> dict:
    """Compute error metrics between MLX and numpy reference."""
    mlx_np = np.array(mlx_output)
    diff = mlx_np - ref_output
    return {
        "kernel": kernel_name,
        "max_abs_error": float(np.max(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "relative_error": float(np.mean(
            np.abs(diff) / np.maximum(np.abs(ref_output), 1e-8)
        )),
    }


def kernel_chain_perplexity(
    num_tokens: int = 1000,
    num_heads: int = 8,
    head_dim: int = 128,
    seed: int = 42,
) -> dict:
    """Run kernel chain and compute perplexity delta.

    Simulates: MatMul(QKV) → RoPE → Softmax → MatMul(attn) → KV Compress
    Both in MLX and numpy reference.

    Returns dict with per-kernel errors and overall perplexity delta.
    """
    rng = np.random.default_rng(seed)

    # ... (Gemini implements full kernel chain simulation)
    # Key: accumulate logit differences across tokens
    # Report: perplexity delta vs 0.5% threshold

    return {
        "per_kernel_errors": [],
        "perplexity_delta_pct": 0.0,
        "threshold_pct": 0.5,
        "pass": True,
        "known_limitations": [
            "Omits residual connections, LayerNorm, and feed-forward layers",
            "Partial-pipeline divergence is a proxy, not a bound (residual/LN may dampen or amplify)",
            "Uses kernel-chain simulation, not end-to-end model inference",
            "Novel kernel precision uses high-level MLX as reference, not custom Metal",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Kernel precision validation")
    parser.add_argument("--output", type=str, default="results/precision_validation.json")
    parser.add_argument("--tokens", type=int, default=1000)
    args = parser.parse_args()

    print("=== Kernel Precision Validation (Gate G2.5) ===")

    result = kernel_chain_perplexity(num_tokens=args.tokens)

    for err in result["per_kernel_errors"]:
        status = "PASS" if err.get("within_threshold", True) else "FAIL"
        print(f"  {err['kernel']}: RMSE={err['rmse']:.6f} [{status}]")

    ppl_status = "PASS" if result["pass"] else "FAIL"
    print(f"\n  Perplexity delta: {result['perplexity_delta_pct']:.3f}% "
          f"(threshold: {result['threshold_pct']}%) [{ppl_status}]")

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run precision validation**

```bash
python scripts/validate_kernel_precision.py --output results/precision_validation.json
```

Expected: All per-kernel errors within thresholds from spec section 5.2. Perplexity delta < 0.5%.

- [ ] **Step 3: Commit G2.5**

```bash
git add scripts/validate_kernel_precision.py
git commit -m "feat(benchmark): add kernel precision validation with perplexity gate (G2.5)"
```

---

## Task 7: Comparison Script — Statistics, Charts, Roofline (M5)

**Files:**
- Create: `scripts/compare_mojo_vs_mlx.py`

**Delegation:** Dispatch to Gemini CLI:
```bash
delegate-to-gemini "Create scripts/compare_mojo_vs_mlx.py. Spec sections 4.6, 7.1-7.3. Reads mojo and mlx JSON results + roofline calibration, produces:

1. LaTeX tables — per kernel, both frameworks side-by-side with all stats + roofline fraction
2. matplotlib charts (publication quality, 300 DPI):
   a) Roofline plot: both frameworks on arithmetic-intensity vs throughput with hardware ceiling line
   b) Roofline fraction bars: achieved % of peak per kernel, side-by-side
   c) Bar charts: absolute throughput per kernel at each model shape
   d) Line plots: throughput vs matrix size for standard sweep
   e) Error bars: 95% BCa CIs on all data points
   f) Heatmap: log2(MLX_time/Mojo_time) — zero=equal, positive=Mojo faster, negative=MLX faster
   g) Energy efficiency: TFLOPS/W comparison per kernel
   h) Decode time attribution: stacked-bar showing each kernel fraction of decode time
   Every ratio chart accompanied by absolute numbers table.
3. Summary JSON with geometric mean speedup (not arithmetic mean)
4. Per-kernel Cohen's d effect size
5. Decode time attribution analysis: per-kernel latency × invocations-per-decode-step × layers → estimated total. If <50% of decode time, flag prominently.

Accept --mlx, --mojo, --roofline, --output-dir args."
```

- [x] **Step 1: Create `scripts/compare_mojo_vs_mlx.py` skeleton**

```python
#!/usr/bin/env python3
"""Compare Mojo vs MLX kernel benchmark results.

Reads JSON outputs from both frameworks + roofline calibration.
Produces LaTeX tables, publication-quality charts, and summary JSON.

Usage:
    python scripts/compare_mojo_vs_mlx.py \
      --mlx results/mlx_kernels.json \
      --mojo mojo-bench/results/mojo_kernels.json \
      --roofline results/roofline_m4max.json \
      --output-dir results/comparison/
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.size": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "serif",
})


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def cohens_d(group1: list[float], group2: list[float]) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return float((m1 - m2) / pooled_std) if pooled_std > 0 else 0.0


def geometric_mean_ratio(values: list[float]) -> float:
    """Geometric mean of positive ratios."""
    log_values = [np.log(v) for v in values if v > 0]
    return float(np.exp(np.mean(log_values))) if log_values else 1.0


def generate_roofline_plot(mlx: dict, mojo: dict, calibration: dict, output_dir: str):
    """Roofline plot: arithmetic intensity vs throughput with hardware ceiling."""
    # ... Gemini implements: plot both frameworks, hardware ceiling line,
    # label each kernel point, include legend
    pass


def generate_heatmap(mlx: dict, mojo: dict, output_dir: str):
    """Heatmap of log2(MLX_time / Mojo_time).
    Zero = equal, positive = Mojo faster, negative = MLX faster.
    """
    pass


def generate_energy_chart(mlx: dict, mojo: dict, output_dir: str):
    """TFLOPS/W comparison per kernel."""
    pass


def decode_time_attribution(mlx: dict, mojo: dict, output_dir: str) -> dict:
    """Build an illustrative kernel-mix chart for the matched benchmark set."""
    # Invocations per decode step per layer (for a standard transformer):
    invocations = {
        "matmul": 6,    # QKV proj (3) + O proj (1) + FFN up (1) + FFN down (1)
        "softmax": 1,   # Attention softmax
        "rope": 1,      # Applied to Q and K
        "isoquant_rotate_inverse": 1,  # Per-query inverse rotation
        "kv_compress": 1,  # KV cache quantize per token
        "fused_attention": 1,  # The whole attention block (if fused)
    }
    # Note: without an explicit end-to-end decode baseline and representative
    # per-model shapes, this remains an illustrative kernel mix only.
    # ... Gemini implements full computation
    return {"kind": "illustrative_kernel_mix", "has_end_to_end_decode_reference": False}


def generate_latex_tables(mlx: dict, mojo: dict, output_dir: str):
    """One LaTeX table per kernel, both frameworks side-by-side."""
    pass


def main():
    parser = argparse.ArgumentParser(description="Compare Mojo vs MLX benchmark results")
    parser.add_argument("--mlx", required=True, help="MLX results JSON")
    parser.add_argument("--mojo", required=True, help="Mojo results JSON")
    parser.add_argument("--roofline", required=True, help="Roofline calibration JSON")
    parser.add_argument("--output-dir", default="results/comparison/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    mlx_data = load_results(args.mlx)
    mojo_data = load_results(args.mojo)
    roofline = load_results(args.roofline)

    # Generate all outputs
    generate_roofline_plot(mlx_data, mojo_data, roofline["calibration"], args.output_dir)
    generate_heatmap(mlx_data, mojo_data, args.output_dir)
    generate_energy_chart(mlx_data, mojo_data, args.output_dir)
    decode_attr = decode_time_attribution(mlx_data, mojo_data, args.output_dir)
    generate_latex_tables(mlx_data, mojo_data, args.output_dir)

    # Summary with geometric mean
    # ... Gemini implements

    print(f"Comparison artifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Implement all chart generators and LaTeX tables**

Gemini fills in all `pass` stubs with publication-quality matplotlib code, LaTeX table generators, and the summary JSON with geometric mean speedup.

- [x] **Step 3: Run comparison on test data**

```bash
python scripts/compare_mojo_vs_mlx.py \
  --mlx results/mlx_kernels.json \
  --mojo mojo-bench/results \
  --roofline results/roofline_m4max.json \
  --output-dir results/comparison/
```

Expected: `results/comparison/` contains `.png` charts, `.tex` tables, and `summary.json`.  
Observed: comparison run succeeded with 19 matched kernel-shape pairs and wrote all expected artifacts.

- [x] **Step 4: Gate G3+G4 — Codex adversarial review of M5**

Observed: Codex adversarial review refresh completed on 2026-04-23. Fixes landed for ratio semantics, mixed-precision disclosure, illustrative kernel-mix output (replacing unsupported decode-fraction claims), framework-specific power requirements for the energy chart, and stale paper/spec numbers.

```
/codex:adversarial-review --background
```

Review focus: Statistical methods correct (BCa, geometric mean, Cohen's d), chart accuracy, roofline plots correct, log2 ratio heatmap symmetric, energy efficiency computed correctly, decode time attribution math correct and <50% disclosure rule implemented.

- [ ] **Step 5: Fix Codex findings and commit M5**

```bash
git add scripts/compare_mojo_vs_mlx.py
git commit -m "feat(benchmark): add comparison script with charts, LaTeX, roofline (M5)"
```

---

## Task 8: GPU Profiling Protocol (M5.5)

**Files:**
- Create: `docs/gpu-profiling-protocol.md` (procedure, not code)

This is manual work using Xcode Instruments. No code to write, but the procedure must be documented.

- [ ] **Step 1: Document GPU profiling protocol**

After the main benchmark completes, identify the top-3 kernels by performance gap (largest absolute deviation in the MLX/Mojo time ratio).

For each, profile using Xcode Instruments → Metal System Trace:
1. Open the benchmark executable in Instruments
2. Select "Metal System Trace" template
3. Run for 10 iterations of the specific kernel
4. Capture 3 trace runs, report median metrics:
   - ALU utilization %
   - Memory bandwidth utilization %
   - Occupancy (waves/EU)
   - Stall cycles breakdown

- [ ] **Step 2: Commit protocol document**

```bash
git add docs/gpu-profiling-protocol.md
git commit -m "docs: add GPU profiling protocol for variance attribution (M5.5)"
```

---

## Task 9: Paper Section Integration (M6)

**Files:**
- Modify: `docs/FROM_ATTENTION_TO_CONSUMER_HARDWARE.md` (add new section)

- [ ] **Step 1: Add paper section skeleton**

Add Section X per spec section 7.4:
- X.1 Methodology
- X.2 Hardware Calibration
- X.3 Standard Kernel Comparison
- X.4 Novel Kernel Comparison
- X.5 Real Model Shape Results
- X.6 Precision Analysis
- X.7 Energy Efficiency
- X.8 GPU Profiling (Top-3 Performance Gaps)
- X.9 Decode Time Attribution
- X.10 Discussion

Each subsection contains placeholder for results from `results/comparison/` — actual numbers filled in after benchmark runs.

- [ ] **Step 2: Add code quality disclosure**

In X.1 Methodology, include:

> All kernel implementations were AI-generated using Gemini CLI, adversarially reviewed by Codex, and approved by a human ARB lead. Expert Mojo developers may achieve better results. Every Mojo kernel was validated against a >50% roofline utilization gate. Full source code and reproduction instructions are available at `mojo-bench/README.md`.

- [ ] **Step 3: Gate G5 — Codex adversarial review of M6**

```
/codex:adversarial-review --background
```

Review focus: Claims vs evidence, roofline context present, code-quality disclosure included, energy efficiency section present, decode time attribution with <50% disclosure rule, limitations stated.

- [ ] **Step 4: Fix Codex findings and commit M6**

```bash
git add docs/FROM_ATTENTION_TO_CONSUMER_HARDWARE.md
git commit -m "docs: add Mojo vs MLX framework comparison section (M6)"
```

---

## Task 10: Update `pixi.toml` with Full Benchmark Suite (Finalize)

**Files:**
- Modify: `mojo-bench/pixi.toml`

- [ ] **Step 1: Update bench-all task to run all kernels**

```toml
[tasks]
bench-smoke = "mojo run kernels/bench_vec_add.mojo"
bench-matmul = "mojo run kernels/bench_matmul.mojo"
bench-softmax = "mojo run kernels/bench_softmax.mojo"
bench-rope = "mojo run kernels/bench_rope.mojo"
bench-isoquant = "mojo run kernels/bench_isoquant_rotate.mojo"
bench-kv-compress = "mojo run kernels/bench_kv_compress.mojo"
bench-fused-attn = "mojo run kernels/bench_fused_attention.mojo"
bench-all = """
mojo run kernels/bench_matmul.mojo && \
mojo run kernels/bench_softmax.mojo && \
mojo run kernels/bench_rope.mojo && \
mojo run kernels/bench_isoquant_rotate.mojo && \
mojo run kernels/bench_kv_compress.mojo && \
mojo run kernels/bench_fused_attention.mojo
"""
```

- [ ] **Step 2: Full end-to-end run**

```bash
# Calibrate
python scripts/roofline_calibrate.py --output results/roofline_m4max.json

# Mojo suite
cd mojo-bench && pixi run bench-all

# MLX suite
cd /Users/anthonylui/QwenCoderLocal
python scripts/benchmark_mlx_kernels.py --output results/mlx_kernels.json

# Precision validation
python scripts/validate_kernel_precision.py --output results/precision_validation.json

# Comparison
python scripts/compare_mojo_vs_mlx.py \
  --mlx results/mlx_kernels.json \
  --mojo mojo-bench/results/mojo_kernels.json \
  --roofline results/roofline_m4max.json \
  --output-dir results/comparison/
```

- [ ] **Step 3: Verify multi-run protocol with version identity**

Run the full suite 3 times on different days. At the start of each run, verify framework versions, macOS version, and compiler versions against the pinned lockfiles. If any version has changed (macOS update, MLX release, Mojo compiler update), abort the run and re-pin before proceeding.

Check inter-run CV < 5% for all kernels. If any kernel CV > 5%, investigate and document the cause.

- [ ] **Step 4: Final commit**

```bash
git add mojo-bench/pixi.toml results/
git commit -m "feat(benchmark): finalize Mojo vs MLX benchmark suite with full results"
```

---

## Execution Summary

| Milestone | Task | What | Gate | Risk |
|-----------|------|------|------|------|
| M0 | Task 1 | Roofline calibration | G0 | Low |
| M1 | Task 2 | Mojo environment + harness | G1 (Codex) | Low |
| M2 | Task 3 | Standard Mojo kernels | G1 (Codex) | Low |
| M3a | Task 4a | IsoQuant rotation (fwd + inv) | G1 (Codex) | Low |
| M3b | Task 4b | KV compression | G1 (Codex) | Low |
| M3c | Task 4c | Fused attention (unfused + framework-fused) | G1 (Codex) | Medium |
| M3d | Task 4d | Hand-fused + TurboQuant + cost breakdown | G1 (Codex) | **High** — may require fallback |
| M4 | Task 5 | MLX benchmark script | G2 (Codex) | Low |
| G2.5 | Task 6 | Precision validation | G2.5 | Low |
| M5 | Task 7 | Comparison script | G3+G4 (Codex) | Low |
| M5.5 | Task 8 | GPU profiling protocol | — | Manual |
| M6 | Task 9 | Paper section | G5 (Codex) | Low |
| — | Task 10 | Finalize + multi-run | — | — |

**M3 split rationale**: M3a-M3c can proceed even if M3d (hand-fused Mojo kernel) proves infeasible on Tier 3 Metal. Each sub-milestone is independently reviewable.

**Agent workflow at each gate:**
1. Gemini CLI drafts the code
2. `/codex:adversarial-review` finds issues
3. `/codex:rescue` fixes each finding
4. `/codex:adversarial-review` verifies fixes
5. Repeat until clean pass
6. Claude ARB approves
