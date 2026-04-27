# IsoQuant Kernel C — Next-Phase Execution Plan

> **Status:** This is a summary/companion to the canonical plan at
> [2026-04-24-isoquant-decode-performance.md](2026-04-24-isoquant-decode-performance.md).
> For detailed task steps, refer to the canonical plan. Phases 0-3b complete and merged to main.

**Date:** 2026-04-24
**Status:** Phases 0-3b complete; Phase 4-5 pending
**Owner:** Implementation track (this repo)

## Context

Kernel C (V-accumulation) was tiled this session: 32.66 → 4.52 ms at 4K context (7.2×), per
properly-attributed instrumentation after Codex's profiler fix. End-to-end at 8K context,
nvfp4+isoquant moved from 12.21 → 30.84 tok/s (+153%). Reference cells at 8K:

| Cell | tok/s | ms/step |
|---|---|---|
| nvfp4+default | 92.45 | 10.8 |
| **nvfp4+isoquant (post-tiling)** | **30.84** | **32.4** |
| baseline+isoquant (post-tiling) | 22.01 | 45.4 |

Decode budget at 4K, post-tiling, per-component:

| Component | ms/step | Notes |
|---|---|---|
| `pack_indices_3bit` | ~14 | 28-op MLX chain, dispatch-bound (intercept 0.43 ms/call × 20 calls + slope) |
| Kernel fragmentation (3-kernel pipeline) | ~5–15 | Inter-kernel sync + memory round-trips |
| Tiled `fused_value_accum` | 4.52 | Now amortised — was 32.66 |
| Model compute (DeltaNet, MoE, MLPs, dispatch) | ~39 | Out of IsoQuant scope |
| `fused_qk_dot` + inverse rotation + residual | ~7 | Minor terms |

## Goal

Bring NVFP4+IsoQuant to **parity-or-better** with NVFP4+default at 8K (revised from earlier
"within <2×" — the math at 8K supports a stronger claim, see Forecast Correction below).
Validate the depth-reduction / Brent-bound scaling model at 32K. Produce the measurement
record that backs the paper's central claim.

## Forecast (not result) — corrected

Earlier summary said "within <2× at 8K". At 8K the actual arithmetic points stronger:

- Current measured: nvfp4+iso = 32.4 ms/step (saved artifact)
- Phase 2 (incremental append): ~9 ms saved → ~23 ms/step (~43 tok/s)
- Phase 3 (fused NPT=8 + T-tile): ~5–15 ms saved → ~13–18 ms/step (~55–75 tok/s)
- nvfp4+default reference: 10.8 ms/step (~92 tok/s)

**Forecast (not result) for post-Phase-3 at 8K: parity is plausible in the optimistic
branch, ~1.5–1.7× is a reasonable conservative branch.** Don't undersell the target — parity
closes the paper story cleanly. Equally important: **the saved ms wins are not guaranteed to
add linearly** (Phase 2 may amortise differently once Phase 3's fused kernel removes some of
the inter-kernel sync overhead). Treat as a forecast that Phase 4 measurements either
validate or refute.

## Hard architectural constraint (must be acknowledged in any write-up)

The existing single-kernel fully-fused path
([`fused_kv_decode_kernels.py:527`](../../../mlx-lm/mlx_lm/models/fused_kv_decode_kernels.py#L527))
asserts `NPT=4`, i.e. `head_dim=128`. Our test model (Qwen3.6-35B-A3B / qwen3_5_moe) has
`head_dim=256` → `NPT=8`. This means:

1. The existing fully-fused kernel **does not apply** to this model checkpoint at all
2. The 3-kernel pipeline was the only viable path even before any tiling work
3. **Phase 3 (NPT=8 generalisation + T-tiling)** is therefore not "re-enable an existing
   path" — it's "generalise the existing path AND add tiling on top", two distinct changes
4. This strengthens the case that the **tiled V-accumulation was the correct first move**
   (no NPT constraint, smaller engineering surface, immediate scaling win)

---

## Phase 0 — Clean baseline + persisted attribution evidence (~0.5 day)

The 4K cross-session numbers from this session were noisy (default-KV cells swung +77% at 8K
and -47% at 4K despite no code touching their path). Need a clean reference before any delta
measurement. **Also need to persist the post-Codex-fix per-kernel attribution as an artifact
— the saved 4K/8K matrix JSONs only contain end-to-end tok/s, not the per-component
breakdown that proves the V-accum tiled win.**

- [x] Close background processes (Amazon Q Helper, Comet renderers, Chrome renderers, etc.)
- [x] Wait for thermal settle; confirm no swap pressure via `vm_stat`
- [x] Re-run full 4-cell matrix (`{baseline, nvfp4} × {default, isoquant}`) at T ∈ {4096, 8192,
  16384, 32768}, 100 decode steps, 3 repeats per cell
- [x] Record P50 / P95 / P99 per-token latency + per-kernel attribution
- [x] Commit raw data to `artifacts/phase0_baseline_<date>/*.json`
- [x] **Persist one post-fix attribution run as a permanent artifact:** run the corrected
  `instrument_isoquant_decode.py` (with the `fused_value_accum_tiled` wrapper from Codex's
  fix) at `T=4096` and `T=8192` with `ISOQUANT_VACCUM_TILE=128`. Capture the per-kernel
  table to `artifacts/phase0_baseline_<date>/kernel_attribution_4k.json` and
  `kernel_attribution_8k.json`. This is the evidence backing the
  "V-accum 32.66 → 4.52 ms" claim — without it, the writeup has end-to-end numbers but no
  per-component breakdown for review.

**Gate:** within-cell variance < 5% across 3 repeats at 8K. If not, find source first.

**Output:** single trustworthy reference table + persisted per-kernel attribution JSON. Every
subsequent claim measured against the table; the attribution JSONs are the citable
provenance for the V-accum win.

---

## Phase 1 — Bandwidth sanity check (~0.5 day)

Pre-fusion check from DeepSeek's review. The roofline analysis put the tiled kernel 450× above
the memory-bound floor — that's a big gap, and the size of Phase 3's expected gain depends on
which floor we're actually near.

- [x] Instrument tiled Kernel C with bytes-moved counters (packed V read + attention weights
  read + output written)
- [x] Divide by measured kernel time at T ∈ {4096, 8192, 16384}
- [x] Plot achieved GB/s vs T; compare to M4 Max realistic peak (~300 GB/s for this access
  pattern)

**Decision rule for Phase 3 priority:**

| Achieved BW | Phase 3 value | Action |
|---|---|---|
| ≥ 60% of peak | Traffic reduction (upper-end gain) | Proceed with fusion as planned |
| 20–60% | Mixed (dispatch + traffic) | Proceed; expect mid-range gain |
| < 20% | Instruction-level waste | Investigate centroid-load patterns first; consider software-pipelined Kernel C variant *before* fusing |

**Output:** one-page memo with achieved-bandwidth curve and revised Phase 3 forecast.

---

## Phase 2 — Incremental packed-cache append (~1–2 days)

The deterministic ~9 ms win, attacks `pack_indices_3bit`'s O(T) repeat work. Note: our
characterization showed the kernel is dispatch-bound (intercept 0.43 ms/call dominates), so
this win is largely the slope component (~5 ms at 4K, ~12 ms at 8K). The intercept stays
unless we also fuse the pack into a single Metal kernel — flag for a later sub-phase if
Phase 1 BW data motivates it.

- [x] In `IsoQuantKVCache`, replace per-step full-rebuild of `pack_indices_3bit` with
  incremental append: each new token writes its 3-bit packed indices into the next slot of a
  pre-allocated `[H_kv, max_T, packed_bytes]` buffer
- [x] Invalidate only the appended region for downstream consumers; previously-packed entries
  immutable
- [x] Handle edge cases:
  - [x] Cache eviction (currently none for these models, but assert)
  - [x] Sliding-window layers on Gemma 4 (skip — they don't use IsoQuant per design)
  - [x] Deferred-prefill bulk-compress transition (still writes packed form once at end of
    prefill; only decode steps change)
- [x] Correctness: bit-exact equality between incrementally-built packed cache and
  freshly-packed reference on same KV vectors, ≥ 10,000 tokens
- [x] Re-run Phase 0's full matrix; measure delta

**Expected at 8K:** ~32 ms → ~23 ms/step (31 → ~43 tok/s)

**Gate:** observed delta within 50% of predicted ~9 ms on Kernel C attribution. If smaller,
find leakage (likely MLX buffer-allocation amortisation) before Phase 3.

**Output:** PR, updated baseline table, one-paragraph writeup of observed vs predicted delta.

---

## Phase 3 — Fused NPT=8 + T-tile pipeline (~3–5 days)

Structural cleanup. Single fused kernel replacing the 3-kernel pipeline. Scope depends on
Phase 1's BW measurement — bandwidth-bound → fusion geometry matters more, latency-bound →
the merge itself is the win.

- [x] **Design doc first** (don't start coding without it):
  - Single fused kernel: `{packed K, packed V, Q, attention weights buffer} → {output vector}`
  - Target: one Metal command buffer, NPT=8, pre-computed softmax scores in threadgroup
    memory, tiled reduction over T into register accumulators, final simdgroup reduction
- [x] Validate on small reference first ($d_k=128$, $T=256$, single head) against 3-kernel
  path. Bit-exact up to FP16 re-association
- [x] Scale up: multi-head, GQA via `kv_head_map`, variable T
- [x] Run existing 9-test correctness suite for the 3-kernel path
- [x] Benchmark at T ∈ {4096, 8192, 16384, 32768}
- [x] Xcode GPU capture → check occupancy. If register pressure dropped occupancy below tiled
  kernel, reduce per-thread state and re-test

**Expected at 8K:** ~23 ms → ~13–18 ms/step (43 → ~55–75 tok/s)

**Gate:** at 8K, within 1.7× of nvfp4+default's 10.8 ms/step. If not, residual is either
model-compute-adjacent (out of scope) or a genuine constant-factor tax on scalar
quantisation (relevant for Phase 5 Branch C).

**Risks:**
- NPT=8 increases register pressure; threadgroup memory for per-head attention weights can
  collide with Kernel A's Q-cache. Budget ~half a day for tuning.
- If Phase 3 takes longer than 5 days, it's probably the occupancy/register-pressure rabbit
  hole. Set hard 7-day budget; if blown, ship Phase 2 alone and write paper against
  partially-fused result.

**Output:** PR, benchmark table across contexts, Xcode capture showing occupancy, short
writeup of where measured time lands relative to bandwidth-bound floor from Phase 1.

---

## Phase 4 — Scaling validation at 32K (~1 day)

The datapoint that separates this from standard KV-quant work. Most published benchmarks
stop at 8K. This is the **falsifiable prediction of the depth-reduction model**.

Brent's-theorem prediction at M4 Max with $B=32$:
- $T=4K$: $T/B = 128$, modest oversubscription
- $T=8K$: $T/B = 256$, good oversubscription
- $T=32K$: $T/B = 1024$, well past saturation ($P_\text{concurrent} \approx 160$)

**Actions:**
- [x] Re-run full 4-cell matrix at T ∈ {4096, 8192, 16384, 32768} post-Phase 3
- [x] Plot tiled-Kernel-C speedup vs T. Overlay theoretical: $S(T) = T / (B + \log_2(T/B))$
- [x] Predict: speedup grows 4K → 8K → 16K, then flattens around 32K

**Decision tree:**

| Observed at 32K | Interpretation |
|---|---|
| Roughly flat vs 8K speedup | Theory confirmed → paper headline |
| Still growing | Tile size $B$ wrong; revisit |
| Collapsed | Second-order effect (probably thermal or memory pressure) — identify |

- [x] **Also at 32K:** long-context PPL regression. Compare isoquant vs default on a 32K
  held-out document. Flag divergence > 5%.

**Output:** the headline chart for the paper + one-paragraph statement on whether the Brent
bound was validated.

---

## Phase 5 — Decision gate (~0.5 day, no coding)

After Phase 4 you have the data to answer: *do we need a representation change, or are we
done?*

**Branch A — parity or better at 8K+.**
- Write the paper. Don't touch representation.
- Core contribution: depth-reduction framing, 7.2× kernel gain, phase-transition narrative,
  scaling validation at 32K.
- Submit to MLSys or EuroMLSys.

**Branch B — small residual gap (1.3–1.7×) at 8K+.**
- Paper still strong; honest about gap.
- Discussion: flag representation change as future work.
- Stop at: *"compression granularity governs decode scaling; scalar QQ is asymptotically
  pinned; alternative granularities are future work."*
- **Do not invent PPL studies you haven't run.**

**Branch C — large residual gap (>1.7×) at 8K+.**
Investigate representation properly:
1. PPL study: 4-dim SO(4)-block VQ vs current scalar QQ, matched bit-budget, on
   Qwen3-30B-A3B (small, fast iteration) first.
2. Verify histogram-kernel arithmetic against actual bit budget *before* committing —
   $K=8$ at 4-dim is 0.75 bits/dim; matching current 3-bits/dim needs $K=4096$, which breaks
   the histogram-size assumption. (This was the math error in the earlier 4D VQ reviewer
   reply.)
3. Only if PPL holds, scope kernel redesign as separate project with own gate conditions.

**Output:** short decision memo with branch + reasoning.

---

## Paper-writing track (parallel from Phase 2)

Don't leave writing until Phase 5. Draft as you go.

- After Phase 2: methodology section (IsoQuant architecture, scalar QQ, SO(4)+WHT, DKV
  constraint)
- After Phase 3: results section through 8K, depth-reduction framing with Brent's theorem,
  roofline analysis
- After Phase 4: scaling figure and the saturation prediction
- After Phase 5: discussion, future work, decision on inclusion of VQ/PQ material

**Keep out of the paper until you have data:**
- Anything about $O(T+d)$ being achievable at this $d_k$
- Any claim that 4-dim block VQ is a free win
- Any throughput forecast that hasn't been measured

---

## Timeline

| Phase | Calendar | Risk |
|---|---|---|
| 0. Clean baseline | 0.5 day | low |
| 1. Bandwidth check | 0.5 day | low |
| 2. Incremental append | 1–2 days | low |
| 3. Fused pipeline | 3–5 days | medium |
| 4. 32K validation | 1 day | low |
| 5. Decision gate | 0.5 day | — |
| **Total** | **6–10 working days** | |

**Critical path: Phase 3.** Hard 7-day budget on it; if blown, ship Phase 2 + write paper
against partially-fused result.

---

## Success criteria

- Kernel C: 32.66 → ~1–3 ms/step at 4K, a **10–30× total** gain from original serial version
- Decode step at 8K: ~32 ms → ~13–18 ms, i.e. nvfp4+isoquant at 55–75 tok/s
- Gap to nvfp4+default: **1× to 1.5×**, acknowledged as constant-factor residual of scalar
  quantisation
- Validated Brent-bound scaling model with empirically-identified saturation point
- Paper draft submittable to EuroMLSys with **measurement, not forecast**, as the backbone

---

## Appendix: review pattern (file for future use)

This session's adversarial-review loop produced four reply types with different signal/noise
ratios:

| Reply type | Outcome |
|---|---|
| Histogram trick | Confidently wrong; corrected cleanly when arithmetic checked |
| Concession | Accurate and useful |
| Flattery + 4D VQ | Inflated claims AND arithmetic error wrapped in superlatives |
| Roadmap agreement | Calibrated and useful |

**Rule for the multi-LLM loop:** *any reply whose novel contribution rests on a derivation
gets that derivation independently re-checked before it enters a rider document.* The
flattery reply is the most dangerous — it wraps errors in the most superlatives. The more a
reply tells you the idea is publishable, the more carefully check the equations.

---

## Out-of-scope (this plan)

- MoE / DeltaNet kernel optimisations (the ~39 ms model-compute floor)
- Kimi MLA + IsoQuant DKV validation (blocked on 128 GB hardware + Kimi checkpoint)
- NVFP4 throughput hardware investigation (M5 Max FP4 path question)
- 1T-class model benchmarks
