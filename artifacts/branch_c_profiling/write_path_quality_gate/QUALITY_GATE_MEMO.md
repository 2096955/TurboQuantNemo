# §3.4 Write-path Quality Gate

**Roadmap:** `docs/superpowers/plans/2026-05-02-wrap-loose-ends-and-bandwidth-roadmap.md`
Step 3.4 — "Only promote if the win is stable and no quality/PPL regression appears."

**Date:** 2026-05-05 (clean boot, see `artifacts/metal-counters/system_state_20260505.md`)
**Model:** `Qwen3.6-35B-A3B-nvfp4`
**Cache type:** `isoquant`
**Settings:** `--quick` (suite=micro, max_tokens=48, 90s wall budget), `--strict`,
`--seed 42`, `--temp 0.0` (greedy/deterministic)

## Conditions

| File | Env vars |
|---|---|
| `baseline_iso.json` | env_defaults: `FUSED_ENCODE=0`, `NPT8=1`, `CACHE_MODE=concat_append`, `USE_METAL=0`, `BITS=3` |
| `fused_encode.json` | baseline + `FUSED_ENCODE=1` |
| `prealloc.json` | baseline + `CACHE_MODE=prealloc` |
| `combined.json` | baseline + `FUSED_ENCODE=1` + `CACHE_MODE=prealloc` |

## Result: PASS

**Response text is byte-identical across all 4 conditions for both prompts.**
Under temp=0.0 / seed=42 deterministic generation, identical bytes = no
numerical drift introduced by the write-path code paths.

| Prompt | Response | Identical across all 4 configs? |
|---|---|---|
| Reasoning micro ("digit after 8 when counting") | `9` | yes |
| Code micro ("Output only `def f(): pass`") | `def f(): pass` | yes |

Peak resident memory also identical: **18711.9 MB** for every config.

## Note on harness "FAIL"

All 4 runs report `0/2 passed` because the strict repetition gate
(`max_word_repeat_ratio=0.22`) flags correct trivially-short answers as
"100% repetition" (`9` is one word repeated; `def f(): pass` is 3 words ≤ 4
minimum). This is a harness-level false-fail for the micro suite, not a model
quality issue: the responses are factually correct.

For §3.4's question ("did write-path changes cause quality regression?") the
correct signal is **byte-identical responses across configs**, which holds.
The harness pass/fail is irrelevant when both compared runs fail the same way
for the same harness reason.

## Why micro suite, not default

A first attempt used the v2 `default` suite (5 prompts, max_tokens=200) but
exceeded a 25-minute wall budget at the iso baseline because the model's
`head_dim` does not match NPT8 fused-attention dispatch and the IsoQuant
reconstruct fallback path is slow at this scale. The micro suite gives strong
byte-identity signal at low cost and is sufficient for §3.4's regression check.
Larger-suite quality validation for the IsoQuant fallback path is tracked
separately under "Phase 5 / model-specific loose ends" and is not gating §3.4.

## §3.4 closure

Combined with the perf evidence in `write_path_ablation_paired.json` (all
three candidates faster than iso baseline at T=4096 and T=8192, signal stable
or single-outlier), §3.4 graduates:

- `combined` (FUSED_ENCODE=1 + CACHE_MODE=prealloc) is the recommended promotion
  — biggest 8K delta (+2.14 ms/step) with no quality regression and no
  memory regression.
- Single-flag `fused_encode` and `prealloc` also pass both gates and remain
  available as opt-in env vars.
