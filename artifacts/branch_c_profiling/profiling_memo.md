# Profiling Memo: IsoQuant NPT=8 Fused Kernel Decomposition

**Date:** 2026-04-27
**Model:** Qwen3.6-35B-A3B-nvfp4 (D=256, NPT=8)
**Context Lengths:** 4K, 8K (both tiled path, `_NPT8_TILED_T_THRESHOLD=512`)
**Status:** Awaiting clean rerun. The prior committed profile data is invalidated.

## Background

Phase 4/5 established a 1.91x latency gap between `nvfp4+isoquant` and
`nvfp4+default` at 8K context (50.3 vs 96.4 tok/s). PPL is effectively
lossless (0.024% divergence at 32K), so Branch C is a performance problem,
not a quality problem.

This profiling pass is intended to decompose the fused NPT=8 decode step into
six components:

| Component | What it measures |
|-----------|------------------|
| `query_rotation` | Forward SO(4)+WHT rotation of the query vector |
| `metal_kernel` | Tiled NPT=8 Metal dispatch (QK + softmax + V per tile) |
| `fa2_merge` | Python-side FlashAttention-style tile merge |
| `inverse_rotation` | Post-merge inverse SO(4)+WHT rotation |
| `compress_batch` | Decode key/value `_compress_batch` work |
| `pack_indices_3bit` | Decode 3-bit index packing into byte storage |

## Invalidated Data

The initial profile artifact from commit `8b04290` is not decision-grade and
must not be used for bottleneck ranking. The canonical
`artifacts/branch_c_profiling/npt8_profile.json` file has been replaced with
an explicit invalidation marker until the corrected profiler is rerun.

The invalidated run had these defects:

1. `patch_instrumentation()` was called inside the T-loop without a clean
   save/restore boundary, so the second context length wrapped already-wrapped
   functions.
2. Phase B and Phase C reused the same IsoQuant cache. The instrumented run
   therefore measured a cache that had already been extended by the unpatched
   run.
3. The kernel wrapper patched `_get_tiled_kernel()` rather than the cached
   `_tiled_kernel_cache["npt8_tiled"]` object, so kernel timing could be
   bypassed after cache hits.
4. The label `compression_and_packing` was wrong: only `pack_indices_3bit` was
   wrapped, while `_compress_batch` was not separately timed.
5. Timings were collected during cache prefill/finalize and warmup, then
   divided by decode-step count. That contaminated decode attribution with
   setup work.
6. `--roofline` was required but not read or enforced.
7. `--capture-traces` started and stopped capture without running a workload.
8. The memo made ALU/bandwidth claims without Metal counter data.

The visible symptom was a negative 8K residual: component sums exceeded the
instrumented wall-clock step time by 18.2 ms.

## Corrected Script

`scripts/profile_npt8_metal.py` now:

- Creates fresh caches for default, unpatched IsoQuant, and instrumented
  IsoQuant phases.
- Patches instrumentation only for the measured decode phase and restores
  originals after each context length.
- Clears component samples after warmup so prefill/finalize/warmup do not
  contaminate decode attribution.
- Patches the cached NPT=8 tiled kernel object directly.
- Separately records `compress_batch` and `pack_indices_3bit`.
- Reads the roofline JSON, aborts if BW efficiency is below 0.5, and warns if
  it is below the expected 0.73-0.85 range.
- Runs actual decode steps inside optional `mx.metal.start_capture()` /
  `mx.metal.stop_capture()`.
- Emits `call_count`, `missing_components`, `tiled_kernel_observed`,
  `residual_ms`, and `residual_pct` fields for sanity checking.

## Roofline Note

The current `roofline.json` records BW efficiency of 0.502. This barely clears
the hard script gate (0.5) but misses the plan target range (0.73-0.85). The
machine should be rebooted and roofline should be rerun before treating any new
profile as final.

## Rerun Gate

After reboot:

```bash
python3 scripts/roofline_calibrate.py \
  --output artifacts/branch_c_profiling/roofline.json \
  --iters 500

python3 scripts/profile_npt8_metal.py \
  --model /Users/anthonylui/Models/Qwen3.6-35B-A3B-nvfp4 \
  --output artifacts/branch_c_profiling/npt8_profile.json \
  --roofline artifacts/branch_c_profiling/roofline.json \
  --decode-steps 100
```

Treat the rerun as valid only if:

- `tiled_kernel_observed` is true for both 4K and 8K.
- `missing_components` is empty or any missing component is explicitly
  explained.
- `residual_ms` is non-negative and `residual_pct` is below 20%.
- 8K instrumentation overhead is small enough to preserve component ordering.
- Metal System Trace is collected before making ALU/bandwidth claims.

Until those gates pass, no Phase 6 or representation-redesign recommendation
is justified by this profiling work.
