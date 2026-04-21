# Invariance Check — Manual Run Convention

**Established:** 2026-04-21 (Phase 0 Task 24)
**Why this exists instead of CI wiring:** the repo has no top-level `Makefile` (verified: `ls Makefile mlx-lm/Makefile deer-flow/Makefile` returns no matches at the root), so adding a `make invariance-check` target would either require introducing a new top-level Makefile (out of scope for Phase 0) or burying the check in a sub-project Makefile that PR reviewers do not run by reflex. Until CI integration is added, this note is the binding gate.

## The convention

Until CI integration is added (deferred), Phase 1 PRs that touch:

- `mlx-lm/mlx_lm/expert_offload.py`
- `mlx-lm/mlx_lm/generate.py` (anything touching the generation loop or KV-cache flow)
- Any new async prefetch / scheduling code

…MUST manually run:

```bash
python scripts/invariance_check.py --config-suite
```

…and confirm exit 0 before merge. Failure blocks the PR.

## What the gate verifies

The harness runs each of the three standard configs (`dense_llama_3_2_3b`, `moe_qwen36_35b_a3b`, `moe_qwen36_35b_a3b_offload`) twice with greedy decoding and a fixed seed (`temp=0`, `seed=42`), then asserts both runs produce identical generated text (sha256 of the text between mlx_lm.generate's `==========` delimiters).

Baseline at `artifacts/phase-0/invariance-baseline.json` is the canonical reference; PR runs do not need to match that hash exactly (different `--max-tokens`, different mlx-lm versions can shift the output) but must match across the two runs *within their own invocation*.

## Failure protocol

If `--config-suite` exits non-zero on a PR:

1. Read the JSON output at `artifacts/phase-0/invariance-latest.json` (or wherever `--json` was directed). The `diff_chars` block shows the first character index where the two runs diverged plus 20-char context windows.
2. The likely causes, in order of probability:
   - Async expert prefetch raced against the inference thread (Phase 1's central risk; this is exactly what the harness is designed to catch).
   - A new RNG was seeded from wallclock somewhere instead of being deterministic.
   - mlx_lm.generate's stdout format changed and `_extract_generated_text` is failing loud (would surface as `ValueError`, not as `passed: false`).
3. Do not weaken the gate. Either fix the nondeterminism or escalate the regression to whoever owns the offload pathway.

## Future CI work

When the project gains a top-level Makefile (likely as part of Phase 2 release engineering), add:

```makefile
.PHONY: invariance-check
invariance-check:
	python scripts/invariance_check.py --config-suite --json artifacts/phase-0/invariance-latest.json
```

…and wire `make invariance-check` into the PR-required check list. At that point this convention note can be deleted in favor of the automated gate.
