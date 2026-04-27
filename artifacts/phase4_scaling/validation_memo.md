# Phase 4 Scaling Validation Memo

**Date:** 2026-04-27
**Machine:** M4 Max, 128 GB, macOS 26.3.1
**Model:** Qwen3.6-35B-A3B-nvfp4 with `ISOQUANT_BITS=3`, `ISOQUANT_USE_NPT8_FUSED=1`

## Benchmark Matrix

12 planned runs completed: 4 context lengths x 3 repeats. Each run measures a
2x2 cell matrix (`{baseline-4bit, nvfp4} x {default, isoquant}`). The 8K
canonical decision point is stable; longer contexts show high repeat variance,
so the 16K and 32K values should be treated as directional rather than final
paired-comparison estimates.

| T | iso tok/s | iso CV | default tok/s | default CV | gap (iso/default ms) |
|-------|-----------|--------|---------------|------------|----------------------|
| 4096 | 47.7 | 6.6% | 90.7 | 14.5% | 1.90x |
| 8192 | 50.3 | 0.9% | 96.4 | 1.1% | 1.91x |
| 16384 | 26.7 | 62.0% | 51.8 | 29.8% | 1.94x |
| 32768 | 34.2 | 23.1% | 61.9 | 39.3% | 1.81x |

## Speedup vs Phase 0 Baseline

The fused NPT=8 kernel materially improves IsoQuant decode throughput over
the pre-fusion Phase 0 baseline: 2.03x at 4K and 1.91x at 8K. The 16K and 32K
P4/P0 ratios are noisier because both Phase 0 and Phase 4 long-context runs
show substantial repeat variance. The reliable conclusion is that fusion
removed a large constant factor, not that the exact long-context speedup curve
has been cleanly measured.

## Brent Bound

The Brent-bound prediction `S(T) = T / (B + log2(T/B))` with `B=32` predicts
saturation at `T/B ~ 160` (`T ~ 5120`). The measured data confirms the broad
engineering direction: post-fusion IsoQuant is much faster than Phase 0, and
the gap to default KV remains in the 1.8-1.9x range at the stable 8K decision
point. Because Phase 0 had known swap-contaminated variance and Phase 4
long-context repeats are noisy, this run should not be used as a quantitative
validation of the Brent-bound curve.

## PPL Regression at 32K

| Metric | Value |
|--------|-------|
| PPL (default KV) | 5.6250 |
| PPL (isoquant KV) | 5.6236 |
| Divergence | 0.024% |
| Threshold | 5.0% |
| Gate | **PASS** |

IsoQuant KV cache is essentially lossless at 32K context. The 0.024% PPL
divergence is well inside the 5% gate and small enough to treat as measurement
noise for this validation pass.

## Summary

Phase 4 validates the important product-facing point: IsoQuant preserves
quality at 32K and the fused NPT=8 path roughly doubles stable 8K throughput
versus pre-fusion IsoQuant. It does not close the performance gap to default
KV: the canonical 8K ratio remains 1.91x slower per decode step. That residual
gap, not quality loss, is the primary remaining engineering problem.
