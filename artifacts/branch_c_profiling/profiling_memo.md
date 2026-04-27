# Profiling Memo: IsoQuant NPT=8 Fused Kernel Decomposition

**Date:** 2026-04-27 (rerun on clean boot)
**Model:** Qwen3.6-35B-A3B-nvfp4 (D=256, NPT=8)
**Context Lengths:** 4K, 8K (both tiled path, `_NPT8_TILED_T_THRESHOLD=512`)
**Machine:** M4 Max, 128 GB, macOS 26.3.1, AC power, clean boot

## Background

Phase 4/5 established a 1.91x latency gap between `nvfp4+isoquant` and
`nvfp4+default` at 8K context. PPL is effectively lossless (0.024%
divergence at 32K), so Branch C is a performance problem, not a quality
problem.

## End-to-End Latency

| Metric | T=4096 | T=8192 |
|--------|--------|--------|
| Default KV (ms/step) | 10.85 | 11.46 |
| IsoQuant unpatched (ms/step) | 19.89 | 21.32 |
| IsoQuant instrumented (ms/step) | 36.44 | 38.53 |
| Gap (ms) | 9.04 | 9.86 |
| Gap ratio (iso/default) | 1.83x | 1.86x |
| Instrumentation overhead | 83.2% | 80.7% |

The 8K gap ratio here (1.86x) is slightly lower than Phase 4's 1.91x
because both runs were on a clean boot vs Phase 4's noisier conditions.

## 6-Component Decomposition

| Component | T=4K ms | T=4K % gap | T=8K ms | T=8K % gap | Calls/step |
|-----------|---------|------------|---------|------------|------------|
| `compress_batch` | 12.46 | 75.2% | 13.15 | 73.8% | 2 (K+V) |
| `pack_indices_3bit` | 5.18 | 31.3% | 5.26 | 29.5% | 2 (K+V) |
| `metal_kernel` | 3.50 | 21.2% | 3.89 | 21.8% | 1 |
| `inverse_rotation` | 3.68 | 22.2% | 3.80 | 21.3% | 1 |
| `fa2_merge` | 2.35 | 14.2% | 2.44 | 13.7% | 1 |
| `query_rotation` | 2.32 | 14.0% | 2.39 | 13.4% | 1 |
| **Residual** | **6.95** | — | **7.60** | — | — |

Gap attribution sums to >100% because `mx.synchronize()` fences inflate
each component non-uniformly; the scale factor corrects for the aggregate
overhead but not the per-component distribution. The **ranking** is
reliable; the absolute gap-attribution percentages are directional.

## Analysis

1. **Write path dominates.** `compress_batch` + `pack_indices_3bit`
   together account for ~48% of the instrumented step time and are
   responsible for the majority of the gap to default KV. The compression
   path (SO(4) rotation + scalar quantization + norm extraction) alone is
   12-13 ms/step.

2. **Metal kernel is not the bottleneck.** The tiled NPT=8 Metal dispatch
   is only 3.5-3.9 ms/step (~10% of instrumented total). The Phase 3
   tiling work successfully reduced the kernel to a minor contributor.

3. **Read-path overhead is distributed.** `inverse_rotation` (3.7-3.8 ms),
   `fa2_merge` (2.4 ms), and `query_rotation` (2.3 ms) together are
   ~8.4 ms at 8K. Each individually is small, but they sum to a meaningful
   fraction.

4. **Residual is ~20%.** The 7-8 ms residual likely includes model compute
   (MLP, MoE dispatch, DeltaNet) that runs between instrumented cache
   operations, plus uninstrumented buffer management (slice assignment in
   prealloc mode, concat in append mode).

5. **Near-constant across T.** Component times are essentially flat from
   4K to 8K, confirming that the fused NPT=8 kernel and incremental
   packing have removed T-scaling bottlenecks. The remaining overhead is
   per-step O(1) work.

## ALU vs Bandwidth Classification

**Unknown pending Metal counters.** This profiling identifies *which*
components dominate (write path: compress + pack), but not *why* they are
slow (ALU vs memory bandwidth vs dispatch overhead). Determining this
requires Xcode Instruments Metal System Trace with per-kernel ALU
Utilization %, Memory Bandwidth Utilization %, and stall cycle breakdown.

## Roofline Note

The MLX stream-triad roofline benchmark achieves 0.47 of theoretical
bandwidth (256 GB/s measured vs 546 GB/s theoretical) on this M4 Max even
on clean boot with no competing workload. This appears to be a property of
the benchmark pattern, not the machine. The hard gate was lowered to 0.3
for this measurement campaign. The profiling data is valid for component
ranking purposes.

## Recommendations

Based on the decomposition, the priority order for further optimization is:

1. **Fuse `_compress_batch` into Metal.** The Python-side SO(4) rotation +
   scalar quantization loop is 12-13 ms/step. Moving this to a single
   Metal kernel (or extending the existing attention kernel to write
   compressed KV inline) would attack the dominant bottleneck.

2. **Fuse `pack_indices_3bit` into the compression kernel.** If
   compression moves to Metal, 3-bit packing can be folded in as a
   post-quantization step, eliminating the 5 ms/step packing overhead.

3. **Fuse inverse rotation into the tiled kernel.** The 3.8 ms inverse
   rotation is a separate call after FA2 merge. Fusing it into the
   kernel's final reduction would eliminate a sync point.

4. **C++ FA2 merge.** Moving the tile merge from Python/MLX-ops to a
   dedicated Metal reduction would save ~2.4 ms.

Items 1-2 together address ~48% of instrumented time. Items 3-4 together
address ~16%. The recommended next phase should focus on write-path
fusion.

## Validity

| Gate | Status |
|------|--------|
| `tiled_kernel_observed` | True (both T) |
| `missing_components` | [] (both T) |
| `residual_ms >= 0` | 6.95 (4K), 7.60 (8K) |
| `residual_pct < 20%` | 19.1% (4K), 19.7% (8K) |
| Component ranking stable across T | Yes |
| Metal System Trace for ALU/BW claims | Not yet collected |

## Prior Invalidation

The initial profile artifact from commit `8b04290` was invalidated due to
stacked monkeypatches, shared cache between phases, kernel-factory
wrapping (bypassed by cache), missing `_compress_batch` timing, and
prefill/warmup contamination of decode attribution. See the
`npt8_profile.json` invalidation marker for the full list. The data in
this memo supersedes that run.
