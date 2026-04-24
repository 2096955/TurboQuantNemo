# Phase 0 / Phase 1 — completion status (worktree)

**Location:** `.worktrees/isoquant-decode-perf`  
**Date recorded:** 2026-04-24 (closeout pass)

## Phase 0

| Criterion | Status |
|-----------|--------|
| 12 matrix JSONs present | **OK** — `phase0_baseline_2026-04-24/matrix_*.json` |
| All cells `status == ok` in matrix runs | **OK** (as produced by benchmark harness) |
| `kernel_attribution_4k.json` / `kernel_attribution_8k.json` | **OK** |
| Variance gate (T=8192, CV &lt; 5%) | **NOT MET** — documented in [phase0_baseline_2026-04-24/PHASE0_CLOSEOUT.md](phase0_baseline_2026-04-24/PHASE0_CLOSEOUT.md); run `python3 scripts/phase0_variance_check.py` → exit 1 |
| “Clean” system state (plan 0.1) | **NOT MET** — high swap / speed run; matrix is **advisory** |
| **Closeout label** | **Complete-with-caveat** — attribution + [evidence_summary.md](phase0_baseline_2026-04-24/evidence_summary.md) are the high-SNR baselines for Phase 2 |

## Phase 1

| Criterion | Status |
|-----------|--------|
| `scripts/instrument_bandwidth_iso.py` + run | **OK** |
| `artifacts/phase1_bandwidth/tiled_v_accum_bw.json` | **OK** — `max_pct_peak_best` &lt; 20 |
| Decision memo + forecast policy | **OK** — [phase1_bandwidth/phase3_decision_memo.md](phase1_bandwidth/phase3_decision_memo.md), [phase1_bandwidth/forecast_policy.md](phase1_bandwidth/forecast_policy.md) |
| **Closeout label** | **Complete** — rubric says &lt;20% → execution/dispatch-bound; fusion forecast in plan **demoted** to hypothesis per memo |

## Verification commands (re-run)

```bash
cd .worktrees/isoquant-decode-perf
python3 -m py_compile scripts/phase0_variance_check.py scripts/phase0_matrix_cv_summary.py
python3 scripts/phase0_matrix_cv_summary.py
python3 scripts/phase0_variance_check.py; echo "gate_exit=$?"
python3 scripts/phase0_variance_check.py --per-t; echo "gate_exit=$?"
python3 -c "import json; p='artifacts/phase1_bandwidth/tiled_v_accum_bw.json'; d=json.load(open(p)); assert d['max_pct_peak_best']<20, d"
```
