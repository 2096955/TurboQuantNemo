# Phase 2 — Incremental Packed-Cache Append + Preallocated Buffer: Closeout

**Date:** 2026-04-24
**Status:** COMPLETE — O(1) preallocated-buffer append implemented and verified.
**Updated:** 2026-04-26 (prealloc optimization committed, end-to-end A/B captured)

## Implementation summary

Phase 2 was delivered in two steps:

1. **Incremental packed-cache append** (commit `d699e6f`): Pack only the new
   token per decode step and concatenate to the existing packed cache. Eliminated
   lazy rebuild (`packed_cache_misses` 2000 → 0).

2. **O(1) preallocated buffer** (commit `8a9b830`): Replace per-step
   `mx.concatenate` (O(T) copy) with preallocated buffers extended in 256-slot
   steps. Decode updates use `arr[:, pos:end, :] = val` slice assignment — O(1)
   amortized. Opt-in via `ISOQUANT_CACHE_MODE=prealloc`; default remains
   `concat_append`.

## Implementation correctness: PASSED

11 tests across both cache modes:

```
$ cd mlx-lm && python3 -m pytest tests/test_iso_incremental_pack.py -v
test_incremental_append_matches_rebuild_short[concat_append]    PASSED
test_incremental_append_matches_rebuild_short[prealloc]         PASSED
test_incremental_append_matches_rebuild_long[concat_append]     PASSED
test_incremental_append_matches_rebuild_long[prealloc]          PASSED
test_prealloc_matches_concat_append                             PASSED
test_buffer_extension_at_step_boundary                          PASSED
test_fused_attention_equivalence                                PASSED
test_reconstruct_values_ignore_padding                          PASSED
test_state_serialization_roundtrip                              PASSED
test_trim_invalidates_packed_cache                              PASSED
test_restore_then_fused_attention_and_decode                    PASSED
11 passed
```

Key invariants verified:
- Bit-exact match between incremental packed cache and fresh rebuild (short + long)
- Bit-exact match between `concat_append` and `prealloc` modes
- Buffer extension at capacity boundary (128 + 256 → 128 + 512)
- Fused Metal attention produces identical output in both modes (0 kernel failures)
- `reconstruct_keys` / `get_values` return only valid tokens (ignore padding)
- State serialization round-trip excludes padding, restores prealloc buffers
- `trim()` invalidates and re-extends packed caches
- `from_state` restore → fused attention → continued decode (full pipeline)

## Cache-update microbenchmark: 28.3x speedup

```
$ cd mlx-lm && python3 tests/bench_prealloc_vs_concat.py
Benchmark: 1024 decode steps from T=8192
  H=32, D=256

  concat_append  : 11.046s  (92.7 updates/s)
  prealloc       :  0.390s  (2626.7 updates/s)

Speedup: 28.3x (11.046s -> 0.390s)
Per-step delta: 10.40 ms
PASS: 28.3x improvement, 10.40 ms/step saved
```

This is a **cache-update gate** number, not a decode throughput claim. It measures
only the compressed KV update path (6 slice assignments vs 6 concatenations).
The speedup scales with H_kv × head_dim; models with more KV heads benefit more.

Exit-code gate: benchmark fails (`SystemExit(1)`) if speedup < 1.05x.

## End-to-end model A/B: no regression

Same-session A/B on Qwen3.6-35B-A3B-nvfp4 (H_kv=2), 8K context:

| Mode | decode tok/s | peak memory MB | Metal failures | packed cache hit |
|---|---|---|---|---|
| concat_append | 44.11 | 19580 | 0 | 1.0 |
| prealloc | 44.27 | 19580 | 0 | 1.0 |
| **Δ** | **+0.36%** | **0** | **—** | **—** |

**Interpretation:** No measurable decode throughput difference on this model.
Expected — Qwen3.6 has only H_kv=2 KV heads, so the cache-update path is a tiny
fraction of total decode time. The optimization's per-step savings (~10 ms at
H=32) scale linearly with H_kv; at H_kv=2 the saving is ~0.6 ms/step, within
measurement noise.

The A/B confirms:
- No regression — safe to ship as opt-in
- Identical peak memory — prealloc padding does not leak
- Zero Metal failures — stride-aware kernels work correctly on padded buffers
- 100% packed cache hit rate — no lazy rebuilds in either mode

## End-to-end model A/B: Gemma 4 (H_kv=8) — +28% decode throughput

Same-session A/B on Gemma 4-26B-A4B-IT-4bit (H_kv=8, head_dim=256), 4K context:

| Mode | decode tok/s | peak memory MB | Metal failures | packed cache hit |
|---|---|---|---|---|
| concat_append | 20.82 | 15182 | 0 | 1.0 |
| prealloc | 26.65 | 15216 | 0 | 1.0 |
| **Δ** | **+28.0%** | **+34 MB** | **—** | **—** |

**Interpretation:** With 4x more KV heads than Qwen3.6, the cache-update path
consumes a meaningful fraction of decode time. Prealloc's O(1) slice assignment
eliminates the O(T) concatenation cost across 28 layers × 8 KV heads = 224
cache updates per decode step.

- Swap unchanged (4666 MB both runs) — not a thermal/swap artifact
- Peak memory delta +34 MB — negligible padding overhead
- Same model, same session, same swap state — clean paired comparison

**H_kv scaling confirmed:**

| Model | H_kv | decode Δ | Signal |
|---|---|---|---|
| Qwen3.6-35B-A3B | 2 | +0.36% | noise |
| Gemma 4-26B-A4B | 8 | +28.0% | real win |

The prealloc optimization is a **product-level decode throughput improvement**
for models with H_kv >= 8. For H_kv=2 models it remains a safe no-regression
opt-in with cache-update-path benefits only.

## Prealloc technical design

### Six managed arrays

| Array | dtype | dim 2 |
|---|---|---|
| `compressed_keys["indices"]` | uint8 | head_dim |
| `compressed_keys["x_norm"]` | float16 | 1 |
| `compressed_values["indices"]` | uint8 | head_dim |
| `compressed_values["x_norm"]` | float16 | 1 |
| `_packed_keys_cache` | uint8 | ceil(head_dim × bits / 8) |
| `_packed_values_cache` | uint8 | ceil(head_dim × bits / 8) |

### Buffer lifecycle

1. `finalize_deferred_prefill` → `_extend_buffers_by_step()` pads all 6 arrays
   by `_PREALLOC_STEP=256` slots
2. Each decode step: `_ensure_buffer_capacity(seq_len)` checks and extends if
   needed; then slice-assigns new data at `[offset:offset+seq_len]`
3. `state` getter trims to `[:offset]` for serialization
4. `state` setter + `meta_state` setter rebuild packed caches and re-extend
5. `trim(n)` invalidates and re-extends

### Metal kernel stride handling

Padded buffers pass `storage_stride = indices.shape[1]` to Metal kernels.
Kernels index as `base[head * storage_stride * D + t * D + d]` instead of
assuming compact layout. This allows the fused attention path to operate
directly on padded buffers without copying.

## Phase 2 historical evidence (pre-prealloc)

### Cache wiring (Phase 2 step 1: incremental append)

```
Phase 0 (baseline):  packed_cache_misses=2000  decompress_calls=0
Phase 2 step 1:      packed_cache_misses=0     decompress_calls=0
```

### 4K context (3 repeats, Phase 0 vs Phase 2 step 1)

| Cell | Phase 0 mean | Phase 2 mean | Δ % | Phase 2 CV |
|---|---|---|---|---|
| baseline_default | 49.77 | 58.50 | +17.6% | 21.5% |
| baseline_isoquant | 13.65 | 16.91 | +23.9% | 23.7% |
| nvfp4_default | 49.64 | 54.94 | +10.7% | 27.8% |
| **nvfp4_isoquant** | 23.49 | **32.53** | **+38.5%** | **17.4%** |

Default-KV cells (code NOT touched) improved 11-18% — environmental drift.

### 8K context (3 repeats, post-reboot, 0 MB swap)

| Cell | P0 mean | P2 mean | Δ % |
|---|---|---|---|
| baseline_default | 69.1 | 80.9 | +17.0% |
| baseline_isoquant | 16.2 | 24.9 | +54.0% |
| nvfp4_default | 59.3 | 84.8 | +43.1% |
| **nvfp4_isoquant** | 26.4 | **39.1** | **+48.1%** |

Within-session ratio: nvfp4_iso / nvfp4_default = 44.5% (P0) → 46.1% (P2).
The ratio barely moved — Phase 2 step 1 eliminated lazy rebuilds but still
used O(T) `mx.concatenate`. The prealloc step (commit `8a9b830`) fixes this.

## Caveats

1. **Cross-session noise.** Phase 0 ran under 60+ GB swap; Phase 2 step 1
   post-reboot. Default-KV cells moved +17-43%. Within-session ratios and
   same-session A/B are the trustworthy signals.
2. **Within-session CV.** 4K: 17-28%; 8K: 13-34%. Report mean±CV, not single
   repeat numbers.
3. **H_kv sensitivity.** Prealloc's cache-update speedup (28.3x at H=32) scales
   linearly with H_kv. On Qwen3.6 (H_kv=2), the end-to-end effect is within
   noise. Gemma 4 (H_kv=8) showed +28% decode throughput — confirmed by
   same-session A/B. Models with even higher H_kv would show larger gains.

## Phase 3 decision: DEFERRED

The prealloc optimization completes Phase 2's intended scope. The residual gap
(nvfp4_isoquant at ~46% of nvfp4_default throughput) is dominated by:
- Per-token compress (SO(4) rotation + quantize)
- Fused Metal kernel decode (qk_dot + value_accum + inverse_rotation)

Phase 3 (NPT=8 kernel fusion) targets the Metal kernel dispatch overhead.
Phase 1 showed fused V-accum at 5.5% peak bandwidth — execution/dispatch-bound,
not bandwidth-bound. Phase 3 might help via dispatch reduction but the lever
is limited.

**Do not start Phase 3 yet.** The Gemma 4 A/B shows prealloc already delivers
a real decode throughput win (+28%) on higher-H_kv models without kernel fusion.
Phase 3's remaining value is limited to the per-token compress + fused-kernel
overhead, which is the dominant cost on H_kv=2 models where prealloc has no
effect. Next steps:
1. Keep prealloc as opt-in — it is a proven win for H_kv >= 8
2. Proceed to Phase 5 decision (documentation, honest gap characterization)
3. Reassess Phase 3 only if a specific H_kv=2 throughput target is set

## Saved artifacts

- `kernel_attribution_4k.json`, `kernel_attribution_8k.json` — per-kernel
  attribution (Phase 2 step 1; interpret per wrapper-artifact caveat)
- `matrix_T4096_d512_r{1,2,3}.json` — 4K matrix, 3 repeats
- `matrix_T8192_d1024_r{1,2,3}.json` — 8K matrix, 3 repeats (post-reboot)
- `../prealloc-ab/concat_append.json` — Qwen3.6 same-session A/B, concat_append mode
- `../prealloc-ab/prealloc.json` — Qwen3.6 same-session A/B, prealloc mode
- `../prealloc-ab-gemma4/concat_append.json` — Gemma 4 same-session A/B, concat_append
- `../prealloc-ab-gemma4/prealloc.json` — Gemma 4 same-session A/B, prealloc (+28%)
- `mlx-lm/tests/test_iso_incremental_pack.py` — 11 correctness tests
- `mlx-lm/tests/bench_prealloc_vs_concat.py` — A/B benchmark with exit-code gate
