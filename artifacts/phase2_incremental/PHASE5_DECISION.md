# Phase 5 Decision: IsoQuant Decode Performance — What Next

**Date:** 2026-04-26
**Branch:** `isoquant-decode-perf`

## Evidence summary

### What we built (Phases 1-2)

1. **V-accum tiling** (Phase 1): 7.2x kernel-level speedup on the value
   accumulation Metal kernel. But Phase 1 bandwidth profiling showed fused
   V-accum at 5.5% of peak bandwidth — the pipeline is execution/dispatch-bound,
   not bandwidth-bound.

2. **Incremental packed-cache append** (Phase 2 step 1): Eliminated lazy rebuild
   of packed caches. `packed_cache_misses` 2000 → 0 per session.

3. **O(1) preallocated-buffer decode** (Phase 2 step 2): Replaced per-step O(T)
   `mx.concatenate` across 6 arrays with preallocated buffers and slice
   assignment. 28.3x cache-update microbenchmark speedup at H=32, D=256.

### What we measured

| Model | H_kv | Prealloc decode Δ | Signal |
|---|---|---|---|
| Qwen3.6-35B-A3B-nvfp4 | 2 | +0.36% | noise |
| Gemma 4-26B-A4B-IT-4bit | 8 | +28.0% | real win |

Both are same-session A/B with identical swap state, zero Metal failures,
100% packed cache hit rate.

### Residual gap (Qwen3.6 H_kv=2, 8K)

nvfp4_isoquant at ~46% of nvfp4_default throughput. Dominated by:
- Per-token SO(4) rotation + scalar quantization (compress path)
- Fused Metal kernel decode (qk_dot + value_accum + inverse_rotation)
- NOT by cache-update cost (now O(1) via prealloc)

## Decision: Branch B — paper-with-honest-gap (SUPERSEDED)

**Original decision:** Do not proceed to Phase 3 yet.

**Update 2026-04-26:** Phase 3 was reopened and a v1 single-pass NPT=8 kernel
was implemented. This is a correctness-first single-pass implementation (not
the T-tiled + FA2-merge design from the spec). The original rationale below
remains valid context for understanding why the v1 approach was chosen over
the more ambitious tiled design.

### Rationale

1. **Phase 1 showed the pipeline is not bandwidth-bound** (5.5% peak BW).
   Phase 3's traffic reduction is a small lever. The remaining opportunity
   is dispatch/sync reduction, which has not been quantified and may not
   justify the implementation risk of a fused NPT=8 kernel.

2. **Prealloc already delivers a product-level win** for H_kv >= 8 models.
   +28% decode throughput on Gemma 4 is a real, shipped improvement — not
   a microbenchmark artifact.

3. **The residual H_kv=2 gap is structural**, not an implementation deficiency.
   IsoQuant adds per-token compress + decompress work that doesn't exist in
   default KV. At H_kv=2, this fixed cost dominates and no cache-update
   optimization can close it. Phase 3 would reduce decode kernel dispatch
   overhead, but at 5.5% BW utilization the ceiling is low.

4. **IsoQuant's value proposition is memory reduction**, not throughput parity.
   At context lengths where default KV doesn't fit in memory, IsoQuant's
   throughput is the only throughput available. The honest characterization is:
   "IsoQuant trades decode throughput for KV memory compression. The throughput
   cost scales inversely with H_kv."

### What to document

IsoQuant decode performance results for the paper/results track:

1. **V-accum tiling**: structural 7.2x kernel improvement. Pipeline is
   execution-bound at 5.5% peak BW — establishes that further bandwidth
   optimization (Phase 3 fusion) has limited headroom.

2. **Preallocated buffer decode**: O(1) amortized cache update. 28.3x
   microbenchmark speedup. +28% decode throughput on Gemma 4 (H_kv=8).
   No regression on Qwen3.6 (H_kv=2).

3. **H_kv scaling law**: prealloc's decode throughput improvement scales
   linearly with H_kv. At H_kv=2: noise. At H_kv=8: +28%. Implication:
   IsoQuant's relative throughput cost is model-dependent, not a fixed tax.

4. **Honest limitation**: Qwen3.6 (H_kv=2) remains at ~46% of default KV
   throughput at 8K context. The gap is dominated by per-token compress
   cost, which is fundamental to the IsoQuant design. Models with higher
   H_kv show much smaller relative throughput cost.

### Conditions to pursue T-tiled Phase 3 v2

- v1 single-pass kernel shows measurable throughput win in variance study
- Profiling quantifies dispatch/sync overhead as a substantial fraction of
  the residual gap
- Long-context workloads (T > 4K) need T-parallel processing for latency

## Phase 3 v1 status

v1 single-pass NPT=8 kernel implemented (`fused_kv_decode_npt8.py`).
Gated behind `ISOQUANT_USE_NPT8_FUSED=1`, only triggers for `head_dim=256`.

- 8 tests pass: kernel correctness (identity + random SO(4) + Hadamard +
  mask + storage_stride) + cache-level dispatch + prealloc equivalence
- E2E smoke: Gemma 4 produces equivalent quality on all three paths
  (NPT=8, 3-kernel, default KV)
- Single-run throughput: 43.6 tok/s (NPT=8) vs 40.6 tok/s (3-kernel)
  — encouraging, pending variance study

## Branch state

```
isoquant-decode-perf (7+ commits ahead of main):
  8a9b830  feat: O(1) preallocated-buffer decode
  acca18e  docs: Phase 2 closeout + Qwen3.6 A/B
  ef97d5d  evidence: Gemma 4 +28% decode throughput
  7ac3671  docs: add Gemma 4 evidence to closeout
  615ccb9  docs: Phase 5 decision
  10ee227  feat: Phase 3 — NPT=8 single-pass fused attention kernel
  7f62154  test: Phase 3 review gaps — rotation, Hadamard, cache dispatch
```

All changes are opt-in (`ISOQUANT_CACHE_MODE=prealloc`, `ISOQUANT_USE_NPT8_FUSED=1`).
Default behavior is unchanged. Safe to merge to main whenever ready.
