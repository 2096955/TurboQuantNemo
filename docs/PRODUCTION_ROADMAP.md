# Production-grade roadmap (post “works on 32 GB”)

This document is the **actionable checklist** from milestone closure to **production-grade** service and release discipline. The MoE expert offload V2 **technical milestone** is complete; items below are **product / ops / QA** work.

For day-to-day tracking, use:

- [`docs/EXECUTION_BOARD.md`](./EXECUTION_BOARD.md) for owner-oriented execution
- [`docs/RELEASE_CANDIDATE_CHECKLIST.md`](./RELEASE_CANDIDATE_CHECKLIST.md) for a **concrete RC gate** (commands, pass/fail, single-user local)
- this roadmap for the full production checklist

## What is already done

- Mixed-quant conversion path works.
- Repack path works (`repack_experts`).
- Quantized expert offload works (`mx.load` load path + HF `time_step_limit`).
- 120B quality gate passed at **2-bit, 3-bit, and 4-bit** expert widths.
- 32 GB single-machine operation validated for the intended profile.
- Root-cause fixes for loader dtype handling and `time_step_limit` are in place.

---

## 1. Quality validation

- [ ] Turn `scripts/eval_quality_gate.py` into a **real regression suite**, not only a smoke check.
- [ ] Add **fixed-seed** runs for 2-bit, 3-bit, and 4-bit checkpoints (same hardware, same script version).
- [ ] Add **real coding tasks**: bug fixing, refactors, multi-file edits, traceback debugging, test-writing.
- [ ] Add **long-context** tasks using real repositories, not only short prompts.
- [ ] Add explicit **pass/fail thresholds** for repetition, truncation, malformed code, and wrong answers.
- [ ] Save **known-good baseline outputs** (JSON) per release for automated comparison.

**Repo support:** `eval_quality_gate.py` supports `--suite`, `--output-json`, and `--compare-baseline` (see `--help`).

---

## 2. Long-session testing

- [ ] Run **30–60+ minute** soak tests on real coding workflows.
- [ ] Test **many-turn** sessions with **context growth**.
- [ ] Verify quality does not degrade after **repeated expert shard** swaps.
- [ ] Verify recovery after **interrupted** generations or **repeated model reloads**.

**Repo support:** See `docs/SOAK_TESTING.md` (template).

---

## 3. Performance and memory

- [ ] Re-benchmark **2-bit, 3-bit, and 4-bit** on the **same** hardware.
- [ ] Measure **cold-cache** and **warm-cache** behavior separately (see benchmark flags).
- [ ] Record: peak RSS, MLX peak memory, swap, prefill latency, decode tok/s, expert cache hit rate.
- [ ] Validate behavior near the **32 GB** limit under **sustained** decode, not only short runs.
- [ ] Confirm `mx.load()` shard caching stays stable over long sessions.

**Repo support:** `scripts/benchmark_moe_offload.py` supports `--json-output` for artifacts.

---

## 4. Reliability hardening

- [ ] **Fast-fail** startup checks: missing shards, stale `model.safetensors.index.json`, incomplete repacks, incompatible configs.
- [ ] Checkpoint **integrity** after conversion and repack.
- [ ] Regression tests for **`mx.load()`** offload path (**uint32** + **bfloat16**) — see `mlx-lm/tests/test_quantized_offload.py`.
- [ ] Regression tests for Nemotron **`time_step_limit`** — see `test_quantized_offload.py` / `test_nemotron_quantized_offload_config.py`.
- [ ] Remove **stale** tests (e.g. obsolete Nemotron RoPE tests if present).
- [ ] Failure-path tests for **partial** checkpoints and shard mismatches.

**Repo support:** `scripts/checkpoint_integrity.py`.

---

## 5. Serving and operational readiness

- [ ] Decide and document **single-user** vs **queued multi-user** as the first supported mode.
- [x] **Queueing** and basic cooperative cancellation handling for server mode.
- [ ] Hard generation **timeout** / preemption for server mode.
- [x] **Health** and **readiness** checks.
- [ ] **Structured logs** for generation failures, expert loads, memory faults.
- [x] Basic JSON **metrics**: queue depth and expert cache summary.
- [x] **Concurrency limits** explicit in docs and config.
- [x] Optional auth and explicit CORS policy for local / controlled deployments.

---

## 6. Documentation and release discipline

- [ ] Main **README** points third parties to `README_TurboQuantNemo.md` and this roadmap.
- [ ] Docs clearly state the **validated path** is **Nemotron-H + quantized expert offload** (not hybrid TurboQuant KV until validated).
- [ ] Exact **reproduction commands** for convert, repack, quality gate, benchmark.
- [ ] Freeze one **known-good deployment profile** for third-party testing.
- [ ] **Release checklist** and **rollback** procedure.

---

## 7. CI / automation

- [ ] Automated test runs for **offload** and **repack** paths (where MLX is available).
- [ ] One **quality regression** job (fixed prompts + seed) with artifacts.
- [ ] One **benchmark sanity** run with thresholds (optional strict gates).
- [ ] **Artifact capture** for benchmark JSON and quality-gate JSON.

**Repo support:** `.github/workflows/mlx-lm-offload-ci.yml` (subset tests on macOS).

---

## 8. Nice-to-have after productionization

- Warm-cache benchmark mode (extended metrics) — **partial:** `benchmark_moe_offload.py --warm-second-pass --repeat-runs`.
- Richer eval suite for **real coding** quality.
- Multi-user serving.
- Fully validated **Nemotron hybrid TurboQuant KV** path.

---

## Release checklist (tracked)

Use this as a **go/no-go** list before tagging a release or inviting third-party testers.

**Executable RC gate (recommended):** follow [`RELEASE_CANDIDATE_CHECKLIST.md`](./RELEASE_CANDIDATE_CHECKLIST.md) end-to-end on target hardware. Owner split and backlog: [`EXECUTION_BOARD.md`](./EXECUTION_BOARD.md). High-level commands also appear in [`README.md`](../README.md).

- [ ] All **Agent Can Do Now** items on the execution board are done or explicitly waived.
- [ ] **Hardware matrix**: fixed-seed eval for 2-bit / 3-bit / 4-bit on the **32 GB** machine (`scripts/run_quality_matrix.sh`).
- [ ] **Benchmark**: cold + warm metrics captured (`benchmark_moe_offload.py --warm-second-pass --json-output`).
- [ ] **Checkpoint**: `checkpoint_integrity.py` passes with flags appropriate to your layout (`--expect-repack` if using repacked shards).
- [ ] **Serving** (if applicable): `/health`, `/ready`, `/metrics` verified; queue limits documented.
- [ ] **Baselines**: store `eval_quality_gate` JSON artifacts for the release tag.
- [ ] **Human sign-off**: quality and latency thresholds approved (see execution board).

---

## Scope boundaries

| Claim | Status |
|--------|--------|
| Nemotron-H + **quantized expert offload** on ~32 GB Apple Silicon, single-user | **Validated** (see `README_TurboQuantNemo.md`) |
| Nemotron + **TurboQuant KV** on hybrid attention | **Not** a production claim until separately gated |
| Multi-tenant **serving** | **Out of scope** until §5 is implemented |
