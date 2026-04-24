# Phase 0 — Closeout (2026-04-24 worktree)

## Scope

- Worktree: `.worktrees/isoquant-decode-perf`, branch `isoquant-decode-perf`
- Phase 0 plan: [docs/superpowers/plans/2026-04-24-isoquant-decode-performance.md](../../../../docs/superpowers/plans/2026-04-24-isoquant-decode-performance.md) (Task 0.1–0.5)

## Artifact inventory (complete)

| Item | Count / path | Notes |
|------|----------------|-------|
| Baseline matrix JSON | 12 | `matrix_T{4096,8192,16384,32768}_d{512|1024}_r{1,2,3}.json` |
| Task 0.5 attribution 4K | 1 | `kernel_attribution_4k.json` |
| Task 0.5 attribution 8K | 1 | `kernel_attribution_8k.json` |
| Per-kernel high-SNR baseline | 2 (above) | Single-session, instrumented decode; use for Phase 2/3 delta measurement |

## Matrix quality: variance gate (8K) — **FAILED (documented)**

The plan’s Task 0.4 gate is **&lt; 5% CV** across 3 repeats at **T = 8192** for all four cells.

- Run: `python3 scripts/phase0_variance_check.py` (default; same artifact directory as this note).
- **Result:** `GATE FAIL` (exit code 1). Measured CV% at T=8192 (3 repeats, `python3 scripts/phase0_matrix_cv_summary.py`): `baseline_default` 29.5%, `baseline_isoquant` 18.9%, `nvfp4_default` 18.9%, `nvfp4_isoquant` 13.2%. Default-KV rows are the noisiest; the IsoQuant row of interest is still &gt;5%.

**Interpretation:** the end-to-end 2x2 matrix under **high swap / non-clean** conditions is **exploratory / advisory**, not a publication-grade stable baseline. For roadmap decisions, prefer:

1. **Per-kernel attribution** (this directory’s `kernel_attribution_*.json`) — high SNR inside one run.
2. **Phase-to-phase measured deltas** after each implementation (Phase 2+), not plan forecasts.

A clean-room rerun of the 8K×3 matrix (low swap, quiescent system) is **deferred** until publication-style numbers are required. Until then, Phase 0 is **complete-with-caveat**.

## System-state caveat (user-approved speed run)

- Task 0.1 clean-state targets (low swap, thermal, no heavy background) were not met for the full matrix sweep; the run proceeded with “speed over rigor” as agreed. This is **explicit** so later readers do not over-trust the matrix alone.

## Handoff to Phase 1 / Phase 2

- **Phase 1** closeout: see [../phase1_bandwidth/phase3_decision_memo.md](../phase1_bandwidth/phase3_decision_memo.md) and [../phase1_bandwidth/forecast_policy.md](../phase1_bandwidth/forecast_policy.md).
- **Next implementation priority:** Phase 2 — attribution at 8K shows `pack_indices_3bit` as the largest IsoQuant-attributable per-step cost; see [evidence_summary.md](evidence_summary.md).

## Related scripts

- `scripts/run_phase0_baseline.sh` — full matrix (uses `PYTHONPATH` for worktree `mlx-lm` when set by that script)
- `scripts/phase0_variance_check.py` — default: 8K gate; use `--per-t` for all T
- `scripts/instrument_isoquant_decode.py` — Task 0.5 attribution
