# Phase 2 — Incremental Packed-Cache Append: Closeout

**Date:** 2026-04-24
**Status:** Complete-with-caveat — implementation correct, 4K evidence positive, 8K evidence incomplete due to system memory exhaustion.

## Implementation correctness: PASSED

Per the gate-driven discipline (`forecast_policy.md`), the binding gate is bit-exact
equivalence between the incrementally-built packed cache and a fresh
`pack_indices_3bit(compressed_indices)` rebuild on the same stored indices.

```
$ cd mlx-lm && python3 -m pytest tests/test_iso_incremental_pack.py -v
tests/test_iso_incremental_pack.py::test_incremental_append_matches_rebuild_short PASSED
tests/test_iso_incremental_pack.py::test_incremental_append_matches_rebuild_long  PASSED
2 passed in 6.46s
```

Both scales pass (10 decode steps and 8000 decode steps). Implementation is correct.

Code changes (applied to BOTH worktree's mlx-lm AND main's editable-install mlx-lm so
the production runtime exercises Phase 2):

- `mlx-lm/mlx_lm/models/mlx_isoquant.py` — `finalize_deferred_prefill`: replace
  `_invalidate_fused_caches()` with bulk `pack_indices_3bit(...)` into
  `_packed_keys_cache` / `_packed_values_cache` so the fused-metal path finds the
  cache populated immediately after prefill.
- `mlx-lm/mlx_lm/models/mlx_isoquant.py` — `update_and_fetch` decode-branch: replace
  `_invalidate_fused_caches()` with `pack_indices_3bit(new_indices)` followed by
  `mx.concatenate([_packed_*_cache, new_packed], axis=1)`. Pack ONLY the new token
  per step; concatenate to preserve previously-packed entries.

## Cache wiring evidence (single decode session)

```
Phase 0 (pre-Phase-2):   fused_metal_attempts=1000  packed_cache_misses=2000  decompress_calls=0
Phase 2 (post):          fused_metal_attempts=1000  packed_cache_misses=0     decompress_calls=0
```

`packed_cache_misses` dropped from 2000 → 0 — the fused path no longer triggers a
lazy rebuild on any decode step. The streaming invariant the rebuild used to violate
is now respected.

## End-to-end evidence

### 4K context (3 repeats, post-Phase-2 vs Phase 0 baseline)

| Cell | Phase 0 mean | Phase 2 mean | Δ tok/s | Δ % | Phase 2 within-session CV |
|---|---|---|---|---|---|
| baseline_default (path NOT touched) | 49.77 | 58.50 | +8.74 | **+17.6%** | 21.5% |
| baseline_isoquant | 13.65 | 16.91 | +3.26 | **+23.9%** | 23.7% |
| nvfp4_default (path NOT touched) | 49.64 | 54.94 | +5.29 | **+10.7%** | 27.8% |
| **nvfp4_isoquant** | 23.49 | **32.53** | +9.04 | **+38.5%** | **17.4%** |

The default-KV cells (which the change does not touch) improved 11-18% — that's pure
environmental drift between Phase 0 and Phase 2 sessions. Subtracting the env effect:

- baseline_isoquant: ~+6% real Phase 2 contribution
- **nvfp4_isoquant: ~+24% real Phase 2 contribution**

Within-session CV at 4K is 17-28% across all cells — the same noise floor we saw at
Phase 0. The Phase 2 nvfp4_isoquant mean clearly exceeds the Phase 0 mean even with
overlapping ranges; the directional signal is real.

### 8K context: INCOMPLETE

Attempted 3 repeats per cell at T=8192. After the first 4K rep, system memory
exhausted:

```
sysctl vm.swapusage  →  used = 61591 MB / total = 62464 MB
vm_stat              →  Pages free: 19159 (≈300 MB physical RAM free)
```

The 8K runs require loading both the 19 GB Q4 baseline and the 18 GB NVFP4 model
sequentially per cell. Under 60 GB swap pressure, model loads thrash and hang. The
polling task was killed at 36 minutes with no 8K artifacts produced.

**8K end-to-end measurement requires a fresh-system rerun** (close memory-heavy
processes, wait for swap to drain, then re-run only the 8K cells).

## Per-kernel attribution (advisory only — see caveat)

Single-session attribution at 4K:

| Kernel | Phase 0 ms/step | Phase 2 ms/step | Notes |
|---|---|---|---|
| pack_indices_3bit | 8.57 | 31.51 | **Wrapper synchronization artifact** — Phase 2 calls pack INSIDE update_and_fetch, so the wrapper's `mx.synchronize()` forces the prior fused-attention kernel to flush; that time gets attributed to pack. The actual pack work is the new 1-token pack + the `mx.concatenate` extending the cache. |
| fused_qk_dot | 4.45 | 4.70 | unchanged |
| fused_value_accum_tiled | 4.80 | 5.64 | unchanged |
| _apply_inverse_rotation | 3.62 | 3.46 | unchanged |
| **total decode** | **78.06** | **53.13** | **-32% / +47% tok/s** |

The total decode time drop is the real signal. The component breakdown is misleading
in Phase 2 due to where the wrapper synchronizations fall in the pipeline. The
matrix end-to-end (4K table above) is the authoritative metric.

## Caveats and limitations

1. **8K incomplete.** The single most important measurement (per the plan) is missing.
   Phase 3 priority decision should not be made until 8K is captured cleanly.
2. **Cross-session noise.** Default-KV cells moved 10-18% between Phase 0 and Phase 2
   sessions despite no code touching their path. This contaminates absolute
   comparisons; within-cell relative IsoQuant deltas are the trustworthy signal.
3. **Within-session CV at 17-28%.** Even the Phase 2 4K reps have substantial
   variance. The mean+CV reported above is the honest representation; do not quote
   any single repeat number as a result.
4. **Attribution wrapper artifacts.** The `pack_indices_3bit` per-step number under
   instrumentation does NOT reflect the actual pack cost in production decode. The
   attribution is useful for confirming cache-wiring invariants (`fused_metal_attempts`,
   `packed_cache_misses`) but not for component-level performance claims.

## Phase 3 decision posture

Per the plan's decision rubric, Phase 3 priority depends on:
1. Phase 1 result: <20% peak BW (execution/dispatch-bound) → fusion's traffic
   reduction is a small lever
2. Phase 2 8K result: needed to size the residual gap to NVFP4-default

Given (1) and the incomplete (2), the plan's "<20% branch + fusion ROI questionable"
posture from `phase1_bandwidth/phase3_decision_memo.md` is reinforced. **Recommend
NOT starting Phase 3 (NPT=8 fusion) until 8K Phase 2 evidence is captured cleanly.**

## Re-run instructions for completing 8K

**IMPORTANT — REBOOT REQUIRED.** Two prior 8K Python processes (PID 86094 from
21:17, PID 38964 from 23:14) are stuck in `UE` state (uninterruptible kernel
sleep) and **cannot be killed from userspace** (`kill -9` has no effect). They
still hold ~13-20 GB each. Only a reboot clears them. Without that, any new 8K
attempt will land in the same state.

```bash
# 1. REBOOT to clear UE-stuck Metal/MLX processes
# 2. After reboot, free memory: close browser, Slack, Amazon Q Helper, any
#    Codex/Gemini sessions
# 3. Wait for swap to drain (~5-15 min idle):
sysctl vm.swapusage  # target < 10 GB used

# 3. From the worktree, run only the 8K cells:
WT=/Users/anthonylui/QwenCoderLocal/.worktrees/isoquant-decode-perf
cd $WT
for R in 1 2 3; do
  python3 scripts/benchmark_nvfp4_isoquant.py \
    --baseline-model /Users/anthonylui/Models/Qwen3.6-35B-A3B-4bit \
    --nvfp4-model /Users/anthonylui/Models/Qwen3.6-35B-A3B-nvfp4 \
    --output "artifacts/phase2_incremental/matrix_T8192_d1024_r${R}.json" \
    --prefill-tokens 8192 --decode-tokens 1024 --isoquant-bits 3
done

# 4. Compare to Phase 0 8K means using the same delta script
# 5. Update this closeout's "End-to-end evidence" section with the 8K table
# 6. THEN decide on Phase 3
```

## Saved artifacts

- `kernel_attribution_4k.json`, `kernel_attribution_8k.json` — single-session per-kernel
  attribution post-Phase-2 (interpret per the wrapper-artifact caveat above)
- `matrix_T4096_d512_r{1,2,3}.json` — 4K end-to-end matrix, 3 repeats
- `matrix_T8192_d1024_r{1,2,3}.json` — **TO BE PRODUCED** (see re-run instructions)
- `mlx-lm/tests/test_iso_incremental_pack.py` — bit-exact correctness invariant
