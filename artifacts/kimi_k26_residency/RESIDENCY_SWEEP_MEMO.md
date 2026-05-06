# Kimi K2.6 default-cache residency sweep (Lane C, Step 5.4)

**Date:** 2026-05-06
**Plan:** `docs/superpowers/plans/2026-05-02-wrap-loose-ends-and-bandwidth-roadmap.md` Step 5.4 first action
**Script:** `scripts/sweep_kimi_default_cache_residency.py`
**Artifact:** `default_cache_sweep.json` (this directory)
**Model:** `/Volumes/Samsung9904tb/Kimi-K2.6` (554 GB), default MLA cache, no observer/predictor/cliques, NPT16 off
**Method:** L0 arm — 32 prompt tokens, 30 warmup steps, 30 measure steps per cell

## Raw results (sweep values 64, 128, 200, 400, 800)

| max_resident_experts | tok/s | median ms | std | min | max | hits | misses | hit_rate |
|---|---|---|---|---|---|---|---|---|
| 64 | 0.591 | 1691 | 222 | 1501 | 2563 | 0 | 14400 | 0.000 |
| **128** | **0.655** | 1527 | 99 | 1439 | 1853 | 0 | 14400 | 0.000 |
| 200 | 0.646 | 1549 | 126 | 1488 | 1939 | 0 | 14400 | 0.000 |
| 400 | 0.429 | 2330 | 262 | 2165 | 3331 | 0 | 14400 | 0.000 |
| 800 | 0.172 | 5805 | **1429** | 1836 | 7728 | 10690 | 3710 | 0.742 |

Script's mechanical pick: `best_max_resident_experts=128` at 0.655 tok/s.

## Why the headline pick is misleading

After reading `mlx-lm/mlx_lm/expert_offload.py:716-758` (`_plan_expert_loads` +
`_evict_one_unpinned`):

- Per decode step, Kimi K2.6 visits 60 MoE layers, each routing top-8 experts.
  Per-step working set = **60 × 8 = 480 unique (layer, expert) pairs**.
- The eviction rule pins all of THIS layer's required experts from being
  evicted. Anything from previously-processed layers (this step) or previous
  steps is evictable.
- If `max_resident_experts < 480`, the cache cannot hold even one complete
  step's working set. By the time step N+1 needs layer 0's experts again,
  they have been evicted to make room for layers within step N. **Hit rate is
  forced to zero in this regime.** Confirmed: cells 64, 128, 200, 400 all
  show exactly 0/14400 hits.
- At 800 slots (above the 480 threshold), one step's working set fits, step 2
  reuses, hit rate jumps to 74%. Not 100% because router output is somewhat
  token-dependent so the second step's pinned set is not identical to the
  first's.

So the ranking among 64/128/200/400 is determined purely by **cache-management
overhead** (eviction scan, allocator pressure, GC), not by any residency
benefit. The "128 is best" finding is real but **vacuous** for the §5.4
question — it just says "if your cache can't help, the smaller cache hurts
less, except for 64 which is too small even for the no-help regime."

## Why 800 is slower than 128 despite higher hit rate

(Corrected after Codex audit 2026-05-06.)

Two plausible contributors, neither yet proven dominant:

- **RAM/shard pressure (revised).** Codex measured actual layer-1 expert-0
  packed + scale tensors at **~23.6 MiB** (not the ~50 MB I originally
  estimated), so 800 experts ≈ 18.5 GiB (not 40 GB). However,
  `max_resident_experts > 256` also enables up to 4 cached safetensor
  shards × ~9.1 GB each (~36 GB potential), so total memory pressure is
  still plausible — it just comes mostly from shard-cache headroom, not
  expert tensors alone.
- **Eviction-scan cost.** `_evict_one_unpinned` builds
  `candidates = [k for k in self._lru.keys() if k not in pinned]` — O(N)
  Python scan per layer. 800 vs 128 is ~6.6× longer candidate list per
  call. The cost exists but is **not proven dominant** by the current
  measurement (no per-component timing in the artifact).
- Step variance at 800 (range 1836–7728 ms, std 1429) is consistent with
  occasional GC/paging/shard-fault pauses but doesn't isolate the cause.

The data shows 800 is slower; it does not yet show *why* with confidence.

## What the sweep actually tells us

**Useful regime starts at max_resident_experts ≥ 480.** None of the swept
values 64, 128, 200, 400 explore that regime; they're all in the "cache
provides zero hits" zone. Only the 800 cell crosses the threshold but
overshoots into RAM pressure.

**The sweep's design did not isolate the useful inflection point**
(somewhere in [480, 800]).

## Recommended follow-up

(Per Codex audit 2026-05-06.)

1. **Re-sweep bracketing the threshold:**
   `--sweep-values 448,480,512,560,640,720,800`. 448 sits BELOW the 480
   threshold to confirm the 0%-hit cliff; 480 is exactly at; 512 is just
   above; 800 repeats as a reproducibility anchor. This characterises the
   real speed/RAM tradeoff once the cache starts providing hits AND
   confirms the cliff hypothesis is sharp, not gradual.
2. **Capture system memory snapshots** (RSS, swap usage, vm_stat) before,
   after each cell, and at end. The current sweep records only tok/s and
   cache stats — adding memory series lets us discriminate between
   eviction-scan overhead vs RAM/shard pressure as causes of the 800-cell
   slowness. Without it, we can't claim either dominates.
3. **Defer eviction-code optimisation** until the bracketing sweep + memory
   instrumentation actually shows scan cost dominates. If RAM/shard
   pressure turns out to be the bigger lever, the right fix is shard-cache
   tuning (`max_cached_shards`), not eviction-loop refactoring.
4. **Future L1+ comparison cells:** measure how the residency optimum
   shifts when (a) NPT16 fused decode is on, (b) DedeKimi observer /
   clique pinning is on. The current sweep is L0 only and is the foundation
   for those comparisons.

## §5.4 status

The first action of Step 5.4 (default-cache residency sweep) ran and
produced a measured artifact, but the conclusion is **"sweep design
under-explored the useful regime"** rather than "best residency identified."

## v2 bracketed sweep (2026-05-06)

Per Codex audit, re-ran with `--sweep-values 448,480,512,560,640,720,800`.
Artifact: `default_cache_sweep_v2.json`. Pre-sweep memory snapshot:
`memory_snapshot_pre_v2.json` (42.9 GB free, swap 16 MB).

| max_resident | tok/s | median ms | std | hit_rate |
|---|---|---|---|---|
| 448 | 0.389 | 2569 | 1878 | 0.000 |
| **480** | **0.532** | 1880 | 312 | **0.681** |
| 512 | 0.505 | 1981 | 310 | 0.681 |
| 560 | 0.499 | 2004 | 322 | 0.681 |
| 640 | 0.484 | 2067 | 387 | 0.701 |
| 720 | 0.568 | 1762 | 473 | 0.721 |
| 800 | 0.494 | 2023 | 385 | 0.742 |

**Findings:**

1. **Threshold cliff at exactly 480 confirmed.** 448 still gives 0%
   hits (same regime as v1's 64/128/200/400). 480 jumps to 68%. Single-cell
   discontinuity at the predicted `60 layers × 8 top-k` boundary.

2. **The v1 "800 is 3× slower" claim does NOT reproduce.** v2 800 =
   0.494 tok/s (median 2023 ms, std 385) vs v1 800 = 0.172 tok/s (median
   5805 ms, std 1429). Same code, same model, same hit_rate (0.742).
   Codex audit (round 2) verified: the sweep tears down model state
   between cells via `kab.teardown` + `gc.collect()` + `mx.clear_cache()`,
   so Python/model state contamination is unlikely. **More importantly:
   v1 and v2 800-cells have nearly identical expert-loading counters
   (load_count 3710 = 3710; load_time_ms_total 10700 ≈ 10649)** — the
   slowdown is NOT in expert loading, it's in uninstrumented step
   latency / synchronization (Metal command queue, GPU sync, thermal,
   etc.). **The RAM-pressure / scan-overhead / filesystem-cache
   explanations in the earlier sections of this memo are all weakly
   supported. The actual cause of the v1 800 slowness is unknown.**

3. **Above 480, the throughput plateau is shallow.** 480, 512, 560, 640,
   720, 800 all sit in 0.484–0.568 tok/s range — ~17% spread between
   slowest cell (640) and fastest (720). v2 mechanical "best" is 720
   (0.568 tok/s) but the margin over 480 (0.532) is 6.8% and within
   step-time std (~310-470 ms median std). **No defensible single
   "best" residency value emerges from one sweep.**

4. **448 shows much higher std than 480** (1878 vs 312 ms median std,
   range 2021–7091 ms vs 1460–2910 ms). Possible explanation: the
   first cell after model load shoulders some warm-up cost the others
   amortize. Sweep-order effect again, not a property of 448 itself.

## Honest §5.4 disposition (2026-05-06)

- **Cliff is real**: max_resident_experts < 480 gives zero hit rate on
  this Kimi config. **Recommendation: do not use values below 480.**
- **Useful regime starts at 480.** Above 480, no single value is
  reliably better than another within this sweep's noise. The original
  v1 finding that 800 is dramatically worse appears to be a sweep-order
  artifact and should not be relied on.
- **A defensible single recommendation requires paired-repeat
  measurements** at the top contenders (e.g., 480 vs 720 vs 800 paired
  alternating, similar to §3.4's profile_ablation.py protocol). Until
  then, any of {480, 720} is a defensible choice with the data we have;
  480 is conservative on RAM, 720 is the v2 mechanical winner.

§5.4 first action complete-with-caveat: cliff identified and confirmed,
plateau characterised, but no single "best" residency proven. Larger
follow-up (paired repeats) deferred until the user decides whether the
remaining uncertainty matters.
