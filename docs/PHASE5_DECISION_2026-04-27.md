# Phase 5 Decision Gate — IsoQuant Decode Performance

**Date:** 2026-04-27
**Canonical plan:** `docs/superpowers/plans/2026-04-24-isoquant-decode-performance.md`

## Measured Gap

| T | nvfp4+iso tok/s | nvfp4+default tok/s | ms/step ratio (iso/default) |
|-------|-----------------|---------------------|-----------------------------|
| 4096 | 47.7 | 90.7 | 1.90x |
| 8192 | 50.3 | 96.4 | **1.91x** |
| 16384 | 26.7 | 51.8 | 1.94x |
| 32768 | 34.2 | 61.9 | 1.81x |

At the canonical evaluation point (`T=8192`), IsoQuant measures 19.9 ms/step
versus 10.4 ms/step for default KV. The gap is **1.91x**.

## PPL Quality Gate

Divergence at 32K: **0.024%** (threshold: 5%). **PASS.** IsoQuant is
effectively lossless in this validation.

## Decision Thresholds

| Branch | Gap range | Action |
|--------|-----------|--------|
| A | <= 1.3x | Write paper, ship as-is |
| B | 1.3-1.7x | Paper still strong, flag gap as future work |
| C | > 1.7x | Investigate representation (4-dim block VQ study) |

## Selected Branch: C

The stable 8K decision point is 1.91x, which is above the 1.7x Branch C
threshold. The longer-context points are noisier, but they do not overturn
the 8K decision: the mean gap remains above 1.7x at 16K and 32K as well.

## Interpretation

The fused NPT=8 kernel delivered the intended large constant-factor
improvement over pre-fusion IsoQuant. However, it did not reduce IsoQuant to
parity with default KV. Since the 32K PPL result passes cleanly, the blocker is
performance rather than compression quality.

The next phase should treat the residual gap as a kernel/representation
problem. Profiling should come before changing the representation, because the
current data does not distinguish ALU pressure from memory traffic or merge
overhead.

## Recommended Next Steps (Branch C)

1. **Profile the fused kernel** — Collect Metal counters to identify whether
   the bottleneck is ALU work, memory bandwidth, or tile-merge overhead.

2. **4-dim block VQ study** — Evaluate whether block vector quantization can
   reduce per-step dequantization and rotation overhead without damaging PPL.

3. **Deferred dequantization** — Investigate computing as much of attention as
   possible in the compressed domain, dequantizing only when necessary.

4. **Asymmetric bit allocation** — Profile whether keys and values have
   different sensitivity, then test non-uniform bit schedules if supported by
   the kernel path.

## Caveats

- Phase 0 baseline had high variance from swap pressure, so P4-vs-P0 speedup
  ratios should be interpreted cautiously.
- Phase 4 repeat variance is low at 8K (0.9% CV for `nvfp4+isoquant`, 1.1%
  for `nvfp4+default`) but high at 16K and 32K. The Branch C decision depends
  on the stable 8K gate, not on overreading the long-context means.
- All measurements are on one M4 Max 128 GB host. Different Apple Silicon GPU
  configurations may shift the absolute ratios.
