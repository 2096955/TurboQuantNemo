# Phase 0 — Completion Summary

**Completed:** 2026-04-21
**Spec:** `docs/superpowers/specs/2026-04-20-research-reality-program-design.md` §Phase 0
**Plan:** `docs/superpowers/plans/2026-04-20-phase-0-cleanup-and-invariance.md`
**Worktree:** `.worktrees/phase-0-codegen` on branch `feature/phase-0-codegen` @ `8c9bd56`

## Deliverables

- [x] **mojo-bench/ on main** — 23 JSON results in `mojo-bench/results/` (≥21 required), reproducible from clean checkout via the harness committed in earlier Phase 0 tasks (Tasks 1–5, merged to main pre-this-thread).
- [x] **Worktrees triaged** — 3 deleted (`nemotron-gemma-eval`, `qwen3-wiring`, `qwen3-deferred-dedekimi-impl`; salvageable commits cherry-picked to main where applicable), 4 held for later (`attnres-expert-management`, `gemma-nemotron-val`, `qwen3-deferred-dedekimi`, `rotorquant`). Decision log at `docs/superpowers/recommendations/worktree-cleanup-log.md`. The `mojo-vs-mlx-benchmark` worktree was merged to main but its directory remains on disk pending a future `git worktree remove` pass.
- [x] **δ memo verdict** — `reopen_for_narrow_case_X` (X = "speculative decoding using a shared-expert-only draft model on a target model that pays expert offload cost in steady state"). Memo: `docs/superpowers/notes/2026-04-21-spec-decode-rejection-audit.md`. Rejection paragraph in the published paper stays — it correctly describes default behavior — but the project's mental model treats δ as deferred-with-conditions, not closed.
- [x] **Prior specs marked DONE** — `2026-04-13-mojo-vs-mlx-kernel-benchmark-design.md` and `2026-04-16-qwen36-mixed-precision-pathway.md` both annotated as superseded by the 2026-04-20 research-reality program (commit `856314a`).
- [x] **Invariance harness passes on three standard configs** — `dense_llama_3_2_3b`, `moe_qwen36_35b_a3b`, `moe_qwen36_35b_a3b_offload`. Baseline at `artifacts/phase-0/invariance-baseline.json`. All 10 unit tests in `tests/test_invariance_check.py` pass.

## Findings worth carrying into Phase 1

- **Offload is bit-exact in steady state** (stronger than the spec required). MoE no-offload (`moe_qwen36_35b_a3b`) and MoE+offload at `--max-resident-experts 2048` produce *identical* sha256 hashes (`725097d8d5b26a838cc2597d18050ac778c21c0d6d7ea5dbd218af7eab4a345c`, len 203). The spec only required pair-internal consistency within a single config. This means Phase 1's async prefetch implementation has a tighter invariant to preserve: any drift in the offload-config hash (not just intra-pair drift) signals a real regression. Worth reflecting in 1-α acceptance criteria.

- **Subprocess stdout is not the model's output**. The first cut of `invariance_check.py` hashed `mlx_lm.generate`'s entire stdout, which always failed because the `Prompt: X tokens, Y tokens-per-sec` footer varies every run from perf jitter. Fixed by `_extract_generated_text` parsing between `==========` delimiters with a fail-loud `ValueError` if the format changes. Phase 1 instrumentation (e.g., hit-rate logs in `expert_offload.py`) MUST land outside the delimiters or they will defeat the harness silently.

- **argparse `nargs="*"` cannot capture `--`-prefixed flags**. The plan as written used `nargs="*"` for `--extra-args`; this rejected `--expert-offload` as "unrecognized." Replaced with a single quoted string + `shlex.split()`. Generic plan-style command fragments need a sanity check before being shipped in tasks.

- **No top-level Makefile means no CI gate yet**. Verified: `ls Makefile mlx-lm/Makefile deer-flow/Makefile` returns no matches at the root. The invariance gate is enforced by convention via `docs/superpowers/notes/invariance-manual-run-convention.md` until the project gains release engineering in Phase 2. Phase 1 PRs that touch `expert_offload.py`, `generate.py`, or new async prefetch code MUST manually run `python scripts/invariance_check.py --config-suite` before merge.

- **δ (speculative decoding) is conditionally deferred, not closed**. Phase 1's 1-α design need not address speculative pathways. The reopening conditions are scoped in the δ memo (shared-expert-only draft + measurement-backed hit-rate target ≥95% of baseline). This lowers activation energy for a future sprint without authorising the work.

## Phase 1 unblocked.
