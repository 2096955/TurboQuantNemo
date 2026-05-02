# Write-Path Profiler Usage

## Purpose

`scripts/profile_metal_counters.py` is the current low-overhead attribution tool
for the IsoQuant decode gap. It measures read-path and write-path components
separately, then predicts the ten-IsoQuant-layer contribution as:

```text
predicted_10layer_ms = read_serial_sum + 2 * active_write_serial_sum
```

The `2x` write multiplier accounts for K and V cache writes per decode step.

## Current Safety Status

As of `artifacts/metal-counters/system_state_20260502.md`, focused NPT8 pytest
collection aborts after a clean reboot and leaves a Python process in `UEs`.
Do not use new Metal timings as release-grade evidence until that abort is
isolated.

Safe without touching Metal:

```bash
python -m py_compile scripts/profile_metal_counters.py
python scripts/profile_metal_counters.py --help
```

Metal-dependent; run only after a clean system state:

```bash
python scripts/profile_metal_counters.py \
  --model /Users/anthonylui/Models/Qwen3.6-35B-A3B-nvfp4 \
  --output artifacts/metal-counters/profile_with_write.json \
  --prefill 4096 8192 \
  --skip-e2e \
  --skip-traces
```

## Modes To Compare

Baseline:

```bash
python scripts/profile_metal_counters.py \
  --model /Users/anthonylui/Models/Qwen3.6-35B-A3B-nvfp4 \
  --output artifacts/metal-counters/profile_with_write_baseline.json \
  --prefill 4096 8192 --skip-e2e --skip-traces
```

Fused encode:

```bash
python scripts/profile_metal_counters.py \
  --model /Users/anthonylui/Models/Qwen3.6-35B-A3B-nvfp4 \
  --output artifacts/metal-counters/profile_with_write_fused_encode.json \
  --prefill 4096 8192 --skip-e2e --skip-traces \
  --fused-encode
```

Preallocated cache:

```bash
python scripts/profile_metal_counters.py \
  --model /Users/anthonylui/Models/Qwen3.6-35B-A3B-nvfp4 \
  --output artifacts/metal-counters/profile_with_write_prealloc.json \
  --prefill 4096 8192 --skip-e2e --skip-traces \
  --prealloc
```

Combined:

```bash
python scripts/profile_metal_counters.py \
  --model /Users/anthonylui/Models/Qwen3.6-35B-A3B-nvfp4 \
  --output artifacts/metal-counters/profile_with_write_fused_prealloc.json \
  --prefill 4096 8192 --skip-e2e --skip-traces \
  --fused-encode --prealloc
```

## Decision Use

- If `compress_python` dominates and `compress_fused_metal` is materially
  faster, fused encode is the next optimization lever.
- If `cache_concat` dominates and `cache_prealloc` is materially faster,
  preallocation is the next optimization lever.
- If read + 2x active write still under-predicts the measured E2E gap, the
  residual is likely MLX/Python dispatch, graph construction, allocation, or
  synchronization overhead.

Do not promote a mode from this synthetic profiler alone. Promotion requires
paired E2E repeats plus PPL/quality parity.
