# Execution Board

This board splits the remaining work into:

- **Agent can do now**
- **Agent builds, human runs**
- **Human sign-off only**

The intent is to separate code/documentation work from real-hardware validation and final release judgment.

See also: [`PRODUCTION_ROADMAP.md`](./PRODUCTION_ROADMAP.md) (full checklist), [`RELEASE_CANDIDATE_CHECKLIST.md`](./RELEASE_CANDIDATE_CHECKLIST.md) (concrete RC commands + pass/fail), and **Release checklist** at the bottom of the roadmap.

## Agent Can Do Now

- [x] Remove stale Nemotron RoPE material — **`mlx-lm/tests/test_nemotron_h_rope.py` is not present** in this fork; no removal needed. Prefer regression coverage in `test_quantized_offload.py` / `test_nemotron_quantized_offload_config.py`.
- [x] Tighten `scripts/eval_quality_gate.py`: harder coding tasks, **`--strict`** repetition gate, per-task latency in JSON artifacts (v2).
- [x] Improve machine-readable artifacts from `eval_quality_gate.py` (`latency_s`, `strict`, `max_word_repeat_ratio_default`).
- [x] Extend `scripts/benchmark_moe_offload.py`: **`--warm-second-pass`**, **`--repeat-runs`**, richer JSON (warm vs cold cache stats).
- [x] Strengthen `scripts/checkpoint_integrity.py`: **`--expect-repack`**, **`--expect-expert-keys`**, mixed-repack warning.
- [x] Harden `mlx_lm.server`: bounded queue + worker-aware readiness, structured POST validation, **`--max-request-body-bytes`**, optional **`--api-key`**, explicit CORS allowlist, dead-worker detection, hardening tests.
- [ ] Per-request generation **timeout** and **cancellation** (client disconnect already partially handled for SSE keepalive; full abort of MLX work is not done).
- [ ] Expand structured **metrics** (histograms, Prometheus) beyond `/metrics` JSON.
- [x] Expand `.github/workflows/mlx-lm-offload-ci.yml` — added `test_repack_experts.py` and `test_server_hardening.py`.
- [x] Merge `README_TurboQuantNemo.md` into the publishable repo README structure.
- [x] `docs/PRODUCTION_ROADMAP.md` — **Release checklist** section added (tracked alongside this board).

## Agent Builds, Human Runs

- [ ] Prepare a fixed-seed quality matrix runner for `2-bit`, `3-bit`, and `4-bit`, plus optional `layer-aware` — **script:** `scripts/run_quality_matrix.sh` (set `MODEL_2BIT` / `MODEL_3BIT` / `MODEL_4BIT` and optionally `MODEL_LAYER_AWARE`).
- [ ] Prepare cold-cache and warm-cache benchmark wrappers for the real 32 GB machine — use `benchmark_moe_offload.py --warm-second-pass --split-decode-timing --json-output`.
- [ ] Prepare 30-60+ minute soak automation wrappers from `docs/SOAK_TESTING.md`.
- [ ] Prepare real-repo coding eval harnesses for:
  - bug fixing
  - traceback debugging
  - multi-file edits
  - refactors
- [ ] Prepare artifact capture for:
  - peak RSS
  - swap activity
  - prefill latency
  - decode tok/s
  - cache hit rate
  - long-session failures
- [x] Prepare single-user server validation commands/checklist for target hardware — see `docs/RELEASE_CANDIDATE_CHECKLIST.md`.
- [x] Prepare release checklist humans can execute on target hardware — see `docs/RELEASE_CANDIDATE_CHECKLIST.md`.

## Human Sign-Off Only

- [ ] Approve acceptable quality thresholds for coding tasks.
- [ ] Approve acceptable memory and latency envelope on 32 GB hardware.
- [ ] Decide whether the supported public claim is:
  - real code testing
  - single-user production-ready
  - experimental
- [ ] Review long-session soak results on the target 32 GB machine.
- [ ] Review real-repo coding behavior and failure cases.
- [ ] Decide whether warm-cache behavior is good enough for daily use.
- [ ] Decide whether serving remains single-user only or expands to queued multi-user.
- [ ] Freeze one known-good release profile:
  - checkpoint recipe
  - conversion command
  - repack step
  - eval command
  - benchmark command
- [ ] Approve final public wording for reliability and support claims.

## Recommended Order

1. Complete the **Agent Can Do Now** section.
2. Run the **Agent Builds, Human Runs** tasks on the real 32 GB machine.
3. Finish the **Human Sign-Off Only** decisions before making production claims.
