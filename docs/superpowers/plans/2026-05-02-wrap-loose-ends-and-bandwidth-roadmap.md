# Loose-Ends Closure + Bandwidth Math Roadmap

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` for execution, and use `superpowers:subagent-driven-development` only for independent code/test slices. This document is the planning source of truth; do not mark tasks complete from docs alone. Close tasks only with measured artifacts, passing tests, committed code, or explicit human sign-off.

**Date:** 2026-05-02
**Branch:** `main`
**Mode:** planning / consolidation

## Goal

Converge the current MLX constrained-memory work into one release-grade path while preserving a small, measurement-first research lane for bandwidth-limit ideas. The immediate priority is not to add more speculative mechanisms. It is to close validation gaps, stabilize the NPT8/tiled decode state, finish write-path attribution, and decide which math ideas deserve implementation.

## Current Ground Truth

- **Core stack exists:** mixed/layer-aware quantization, expert repack/offload, IsoQuant KV, deferred prefill, Qwen/Gemma/Nemotron cache reconstruction, benchmark/eval/soak harnesses, and server hardening.
- **Best validated path:** Nemotron-H 120B mixed checkpoint has pinned quality, benchmark, memory, and soak artifacts under the intended 32 GB-class envelope.
- **Not release-complete:** current quality artifacts are mostly `strict=false`; the RC checklist still requires strict fixed-seed matrix runs, checkpoint integrity, cold/warm benchmarks, serving sanity, and human threshold sign-off.
- **Qwen3-30B:** runtime/memory path is viable, but quality remains blocked at 8/12.
- **Gemma4:** layer-aware path is strong, but the real 16 GB hardware proof remains open.
- **Decode tiling:** NPT8 + T-tiled decode is merged and opt-in. It improves read-side scaling but does not close the IsoQuant/default KV gap.
- **Profiling conclusion:** the next material performance lever is the write path: `_compress_batch` and 3-bit packing, not more read-kernel tiling.
- **Current machine caveat:** the latest focused NPT8 pytest run aborted and left a Python process in `UEs`; reboot before trusting any new MLX/Metal benchmark numbers.

## File Map

| File / Artifact | Role |
|---|---|
| `docs/PATHWAY_PROVEN_CHECKLIST.md` | Pinned pathway status and blockers |
| `docs/RELEASE_CANDIDATE_CHECKLIST.md` | Single-user RC gate commands |
| `docs/PRODUCTION_ROADMAP.md` | Product/ops/QA checklist |
| `docs/EXECUTION_BOARD.md` | Human sign-off and owner split |
| `artifacts/branch_c_profiling/profiling_memo.md` | NPT8/tiled decomposition and write-path priority |
| `docs/superpowers/plans/2026-04-29-metal-profiling-handover.md` | Write-path profiler handover |
| `mlx-lm/mlx_lm/models/mlx_isoquant.py` | IsoQuant cache, fused/tiled decode dispatch |
| `mlx-lm/mlx_lm/models/fused_kv_decode_npt8_tiled.py` | T-tiled NPT8 decode kernel |
| `scripts/profile_metal_counters.py` | Synthetic read/write component profiler |
| `scripts/eval_quality_gate.py` | Quality gate and strict-mode artifacts |
| `scripts/benchmark_moe_offload.py` | Throughput/memory benchmark |

---

## Phase 0: Sanity Gate Before More Measurement

**Goal:** avoid collecting invalid MLX/Metal data after a stuck GPU/session state.

- [ ] **Step 0.1: Reboot or verify clean system state**
  - Required before any new Metal benchmark or NPT8 test.
  - Check for stuck `Python -m pytest`, `benchmark_nvfp4_isoquant.py`, or `mlx_lm.generate` processes.
  - Record result in `artifacts/metal-counters/system_state_YYYYMMDD.json` or a short markdown note.

- [ ] **Step 0.2: Re-run the focused NPT8 tests after clean boot**
  - Command:
    ```bash
    PYTHONPATH=mlx-lm python -m pytest \
      mlx-lm/tests/test_fused_npt8.py \
      mlx-lm/tests/test_fused_npt8_tiled.py -q
    ```
  - Pass condition: tests complete without Metal abort or stuck `UEs` process.
  - If this fails again, stop decode-kernel benchmarking and open a focused stability task.

- [ ] **Step 0.3: Capture current git/artifact hygiene**
  - Save `git status --short`, current head, and the NPT8 test outcome.
  - Do not clean/delete artifacts unless the operator explicitly approves.

---

## Phase 1: Close Existing Release Gates

**Goal:** get one honest release candidate path before pursuing new math.

- [ ] **Step 1.1: Freeze the intended RC profile**
  - Decide the supported profile: model path, quant recipe, repack step, offload settings, KV mode, max resident experts, and server mode.
  - Candidate default: Nemotron-H 120B mixed, quantized expert offload, IsoQuant KV, single-user local.
  - Artifact: `docs/RELEASE_PROFILE_NEMOTRON120B.md`.

- [ ] **Step 1.2: Run checkpoint integrity on the frozen profile**
  - Use `scripts/checkpoint_integrity.py` with `--require-config`, `--expect-repack`, and `--expect-expert-keys` where applicable.
  - Artifact: `artifacts/rc/checkpoint_integrity_*.json`.

- [ ] **Step 1.3: Run strict fixed-seed quality matrix**
  - Use `scripts/run_quality_matrix.sh` or direct `scripts/eval_quality_gate.py --strict --seed 42`.
  - Required for 2-bit / 3-bit / 4-bit if those checkpoints are part of the claim.
  - Artifact: `artifacts/quality-matrix/*.json`.
  - Pass condition: every claimed checkpoint reports full pass under strict mode.

- [ ] **Step 1.4: Run cold/warm benchmark matrix**
  - Use `scripts/benchmark_moe_offload.py --warm-second-pass --repeat-runs 3 --split-decode-timing`.
  - Artifact: `artifacts/rc/bench_*.json`.
  - Pass condition: memory stays under the approved 32 GB envelope, decode throughput is stable, and hit rate remains high.

- [ ] **Step 1.5: Run 30-60+ minute single-user soak**
  - Artifact: `artifacts/rc/soak_*.json`.
  - Pass condition: no OOM/crash, RSS drift bounded, no progressive quality decay.

- [ ] **Step 1.6: Run server sanity gate if serving is claimed**
  - Verify `/health`, `/ready`, authenticated `/metrics`, unauthenticated rejection, queue-full behavior, and one coherent chat completion.
  - Artifact: `artifacts/rc/server_sanity_*.json` or markdown transcript.

- [ ] **Step 1.7: Update pathway docs only after artifacts exist**
  - Update `docs/PATHWAY_PROVEN_CHECKLIST.md`, `docs/EXECUTION_BOARD.md`, and `docs/PRODUCTION_ROADMAP.md`.
  - Do not convert "prepared commands" into "done" without JSON artifacts or explicit human sign-off.

---

## Phase 2: Decode/Tiling Loose Ends

**Goal:** make the current NPT8/tiled state boring and accurately documented.

- [ ] **Step 2.1: Verify NPT8 dispatch with instrumentation**
  - Add or run an instrumented check that proves `_fused_attention_npt8_tiled` is called for `D=256`, `T >= 512`, `ISOQUANT_USE_NPT8_FUSED=1`.
  - Existing token-count proof is useful but indirect.
  - Artifact: `artifacts/phase3b-tiled-smoke/dispatch_verified_*.json`.

- [ ] **Step 2.2: Re-run tiled vs 3-kernel after clean boot**
  - Run short and long contexts with paired/repeated measurements, not single runs.
  - Artifact: `artifacts/phase3b-tiled-smoke/e2e_paired_*.json`.
  - Pass condition: no regression beyond noise; any claimed speedup must survive paired repeats.

- [ ] **Step 2.3: Keep NPT8/tiled opt-in unless paired E2E is positive**
  - Default remains conservative.
  - `ISOQUANT_USE_NPT8_FUSED=1` stays the explicit switch until release data supports changing it.

- [ ] **Step 2.4: Document the honest performance statement**
  - Expected wording: tiled NPT8 fixes read-side T-scaling and roughly doubles stable 8K IsoQuant throughput versus the pre-fusion baseline, but default KV remains about 1.8-1.9x faster per decode step at 8K.
  - Link `artifacts/phase4_scaling/validation_memo.md` and `artifacts/branch_c_profiling/profiling_memo.md`.

---

## Phase 3: Finish Write-Path Attribution

**Goal:** close the attribution gap before building any new compression math.

- [ ] **Step 3.1: Finish `scripts/profile_metal_counters.py` write-path wiring**
  - Complete the handover from `docs/superpowers/plans/2026-04-29-metal-profiling-handover.md`.
  - Add write components into `run_synthetic_phase()`:
    - `compress_python`
    - `compress_fused_metal`
    - `pack_3bit`
    - `cache_concat`
    - `cache_prealloc`
  - Account for K and V writes as `2x` per decode step.

- [ ] **Step 3.2: Run synthetic write-path profile**
  - Command:
    ```bash
    python scripts/profile_metal_counters.py \
      --model /Users/anthonylui/Models/Qwen3.6-35B-A3B-nvfp4 \
      --output artifacts/metal-counters/profile_with_write.json \
      --prefill 4096 8192 --skip-e2e --skip-traces
    ```
  - Artifact: `artifacts/metal-counters/profile_with_write.json`.

- [ ] **Step 3.3: Compare read + 2x write prediction against E2E gap**
  - If prediction explains the gap, optimize the dominant component.
  - If residual remains large, treat Python/MLX dispatch and allocation as the bottleneck.
  - Artifact: `artifacts/metal-counters/write_path_attribution_memo.md`.

- [ ] **Step 3.4: Decide whether fused encode and prealloc graduate**
  - Gate with paired repeats at 4K and 8K:
    - `ISOQUANT_FUSED_ENCODE=1`
    - `ISOQUANT_CACHE_MODE=prealloc`
    - both together
  - Artifact: `artifacts/branch_c_profiling/write_path_ablation_paired.json`.
  - Only promote if the win is stable and no quality/PPL regression appears.

---

## Phase 4: Math-First Bandwidth Experiments

**Goal:** run tiny, falsifiable experiments for left-field ideas before committing engineering time.

### Experiment A: Transform Search for Lower Bits

- [ ] **Step 4.1: Build an offline transform-error harness**
  - Input: captured K/V activation samples from representative layers/heads.
  - Compare:
    - current SO(4) + scalar Lloyd-Max
    - pure WHT
    - block-WHT, e.g. 32/64/128
    - SO(4) then WHT
    - WHT then SO(4)
  - Metrics:
    ```text
    MSE, max error, cosine similarity, q^T k score error, output reconstruction error
    ```
  - Artifact: `artifacts/bandwidth_math/transform_error_matrix.json`.

- [ ] **Step 4.2: Gate the only question that matters**
  - Decision question:
    ```text
    Can 2-bit WHT/block-WHT match or beat 3-bit SO(4) PPL?
    ```
  - Run PPL at 512, 2048, 8192 if feasible.
  - Artifact: `artifacts/bandwidth_math/wht_2bit_ppl.json`.
  - Promote only if PPL delta is inside the same gate used for current IsoQuant claims.

### Experiment B: Asymmetric K/V Bit Allocation

- [ ] **Step 4.3: Add offline K/V sensitivity sweep**
  - Test schedules:
    - `K3/V2`
    - `K2/V3`
    - `K4/V2` on high-sensitivity layers
    - layer-aware K/V schedule from calibration error
  - Artifact: `artifacts/bandwidth_math/kv_asym_sweep.json`.

- [ ] **Step 4.4: Run PPL gate for the best K/V schedule**
  - Promote only if bytes moved drop materially and PPL remains within gate.
  - Artifact: `artifacts/bandwidth_math/kv_asym_ppl.json`.

### Experiment C: Block/Product Quantization

- [ ] **Step 4.5: Prototype 4D/8D block quantization offline**
  - Compare scalar Lloyd-Max at equal bits/value against small-block vector quantization.
  - Start with offline NumPy/MLX calibration only; do not write Metal kernels yet.
  - Artifact: `artifacts/bandwidth_math/block_pq_error.json`.

- [ ] **Step 4.6: Kill or promote block/PQ**
  - Promote only if it beats scalar quantization at 2 bits/value by enough to justify codebook lookup complexity.
  - Otherwise document as a negative result.

### Parked Ideas

- [ ] **Step 4.7: Document but do not implement entropy/top-k sparse attention**
  - Treat as architecture/approximation research, not current kernel work.
  - Requires attention entropy traces and long-context retrieval validation.

- [ ] **Step 4.8: Document but do not implement linear attention / NSA-style attention**
  - Model rewrite / retraining risk; out of scope for this release cycle.

---

## Phase 5: Model-Specific Loose Ends

**Goal:** keep non-release paths honest without letting them block the RC.

- [ ] **Step 5.1: Qwen3-30B quality recovery decision**
  - Current blocker: 8/12 quality despite viable runtime.
  - Options:
    - rerun after latest KV reconstruction fix and strict gate updates,
    - tune decode/prompt settings,
    - apply SSD/LoRA only after RC path is stable,
    - mark Qwen3-30B as runtime proof but quality-blocked.
  - Artifact: `artifacts/qwen3_quality_recovery/decision.md`.

- [ ] **Step 5.2: Gemma4 physical 16 GB proof**
  - Re-run quality, benchmark, and soak on actual 16 GB-class hardware.
  - Artifact paths must replace capped-host caveat in `docs/PATHWAY_PROVEN_CHECKLIST.md`.

- [ ] **Step 5.3: Qwen3.6 KV fidelity close-out**
  - Run the deferred KV fidelity/PPL test for the mixed 16 GB profile.
  - Artifact: `results/qwen36_kv_fidelity.json` or `artifacts/qwen36/kv_fidelity.json`.

- [ ] **Step 5.4: Kimi format gap remains parked**
  - Do not resume Kimi until disk gate, hardware gate, and format decision are explicitly reopened.
  - If reopened, start from `docs/superpowers/handovers/2026-05-01-phase-2-quantize-format-gap.md`.

- [ ] **Step 5.5: SSD/self-distillation remains optional**
  - Create `docs/PHASE7_SSD_OUTLINE.md` only after the RC path is stable.
  - Frame SSD as quality/post-training work, not memory/runtime work.

---

## Phase 6: Release Claim and Cleanup

**Goal:** make the repo state match the public claim.

- [ ] **Step 6.1: Decide the public claim**
  - Choices:
    - experimental research artifact,
    - serious single-user local testing,
    - production-grade serving.
  - Current evidence supports at most serious single-user local testing after strict RC gates pass.

- [ ] **Step 6.2: Commit or deliberately park tracked changes**
  - The current worktree has many tracked modifications and many untracked artifacts.
  - Before inviting third-party testing, split into commits:
    - runtime/code changes,
    - tests,
    - docs,
    - measured artifacts.

- [ ] **Step 6.3: Write final release note**
  - Include exact model path/checkpoint recipe, commands, hardware, memory envelope, throughput, quality gate version, and known limitations.
  - Artifact: `docs/RELEASE_NOTES_NEMOTRON120B_RC.md`.

- [ ] **Step 6.4: Preserve negative results**
  - Qwen3-30B quality block, NPT8 residual gap, Kimi format gap, and speculative/predictor regressions should be documented as measured constraints, not hidden.

---

## Decision Rules

1. **No phase closes from documentation alone.** Require JSON artifacts, passing tests, committed code, or explicit human sign-off.
2. **Do not add new math to the release path until Phase 1 passes.** Math experiments may run offline, but they do not change the RC profile.
3. **Prefer fewer mechanisms.** A release profile with offload + IsoQuant + one quant recipe is stronger than a profile requiring stacked approximations.
4. **Measure paired when comparing performance.** Single-run speedups are directional only.
5. **Treat MLX/Metal hangs as invalidating the session.** Reboot before new benchmark claims.

## Recommended Next Action

Start with Phase 0 and Phase 3:

1. clean system state / reboot,
2. rerun focused NPT8 tests,
3. finish write-path profiler wiring,
4. run `profile_with_write.json`,
5. decide whether fused encode + prealloc should graduate.

Only after that should Phase 4 bandwidth-math prototypes get engineering time.
