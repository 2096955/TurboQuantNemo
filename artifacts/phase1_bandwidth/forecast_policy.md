# Forecast policy (plan vs evidence)

## Rules

1. **Forecasts in the written plan** (e.g. post-Phase-2 / Phase-3 ms ranges, parity talk) are **hypotheses and prioritization heuristics**, not acceptance criteria.

2. **Implementation correctness** is established by:
   - Bit-exact or bounded tests (e.g. Phase 2: incremental pack vs full `pack_indices_3bit`).
   - Equivalence to a stable reference path (e.g. Phase 3: fused NPT=8 vs 3-kernel pipeline, within agreed tolerance).

3. **Performance correctness** is established by:
   - Saved **artifacts** (JSON, logs) and, where available, **per-kernel attribution** in a single session.
   - **Phase-to-phase measured deltas** (after each phase), not by comparing to hand-estimated savings from the plan.

4. **Roadmap / go-no-go** should use **measurable gates**:
   - Good: "After Phase 2, `pack_indices_3bit` per-step ms in `instrument_isoquant_decode.py` output drops by X% vs pre-Phase-2 JSON."
   - Weak as sole gate: "Save ~9 ms" without tying to a recorded attribution or repeat run.

5. If the end-to-end matrix is **noisy** (variance gate fail, high swap), label results **advisory** and rely on attribution + pre/post diffs for engineering decisions.

## Phase 0 / 1 in this worktree

- **Phase 0 matrix:** complete-with-caveat (8K repeatability failed under load; see [PHASE0_CLOSEOUT.md](../phase0_baseline_2026-04-24/PHASE0_CLOSEOUT.md)).
- **Phase 1:** measured `max_pct_peak_best` &lt; 20% in [tiled_v_accum_bw.json](tiled_v_accum_bw.json) — see [phase3_decision_memo.md](phase3_decision_memo.md). Original optimistic **bandwidth / traffic-reduction** story for **Phase 3 fusion is demoted**; Phase 3 may still be worth it for dispatch / sync, but that is a **separate** hypothesis to test with equivalence + benchmarks after Phase 2.
