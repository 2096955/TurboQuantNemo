# Loose-Ends Closure + Bandwidth Math Roadmap

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` for execution, and use `superpowers:subagent-driven-development` only for independent code/test slices. This document is the planning source of truth; do not mark tasks complete from docs alone. Close tasks only with measured artifacts, passing tests, committed code, or explicit human sign-off.

**Date:** 2026-05-02
**Branch:** `main`
**Mode:** planning / consolidation

## Goal

Converge the current MLX constrained-memory work into one release-grade path while preserving two explicitly scoped research lanes:

1. bandwidth / KV-cache math for models where IsoQuant is the active path, and
2. Kimi K2.6 decode acceleration on 128 GB Apple Silicon.

The immediate priority is not to add more speculative mechanisms. It is to close validation gaps, stabilize the MLX/Metal measurement environment, run the already-wired write-path attribution, and decide which math or Kimi speed ideas deserve implementation.

## Current Ground Truth

- **Core stack exists:** mixed/layer-aware quantization, expert repack/offload, IsoQuant KV, deferred prefill, Qwen/Gemma/Nemotron cache reconstruction, benchmark/eval/soak harnesses, and server hardening.
- **Best validated path:** Nemotron-H 120B mixed checkpoint has pinned quality, benchmark, memory, and soak artifacts under the intended 32 GB-class envelope.
- **Not release-complete:** current quality artifacts are mostly `strict=false`; the RC checklist still requires strict fixed-seed matrix runs, checkpoint integrity, cold/warm benchmarks, serving sanity, and human threshold sign-off.
- **Qwen3-30B:** runtime/memory path is viable, but quality remains blocked at 8/12.
- **Gemma4:** layer-aware path is strong, but the real 16 GB hardware proof remains open.
- **Decode tiling:** NPT8 + T-tiled decode is merged and opt-in. It improves read-side scaling but does not close the IsoQuant/default KV gap.
- **Profiling conclusion:** the next material performance lever is the write path: `_compress_batch` and 3-bit packing, not more read-kernel tiling.
- **Write-path profiler status:** `scripts/profile_metal_counters.py` already includes read components, write components (`compress_python`, `compress_fused_metal`, `pack_3bit`, `cache_concat`, `cache_prealloc`), active-mode selection, and K/V `2x` write accounting. The remaining work is to run it in a clean MLX state and compare artifacts.
- **Kimi K2.6 status:** Kimi is an active exploratory lane, not a release path. Load/decode smoke, expert offload, Kimi MLA IsoQuant cache, `trim()`, NPT16 synthetic kernel tests, Kimi profiling, predictor/clique A/B, and speculative harness scaffolding exist. Current measurements favor the default MLA cache over Kimi IsoQuant for speed; Kimi speed work should focus on default-cache expert residency/offload, 2-bit expert regeneration, and speculative decoding rather than more KV compression by default.
- **Apple `ml-ssd` status:** useful only as optional post-training / code-quality work. It is not a runtime, memory, expert-residency, or bandwidth lever.
- **Current machine caveat:** the latest focused pytest run aborted while importing `mlx.core` during test collection and left a Python process in `UEs`. Treat this first as an MLX/Metal import/runtime-health blocker, not as proven NPT8-kernel failure. Reboot before trusting any new MLX/Metal benchmark numbers.

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
| `mlx-lm/mlx_lm/models/fused_kv_decode_npt16.py` | Kimi MLA D=512 NPT16 fused latent-attention kernel |
| `mlx-lm/mlx_lm/models/kimi_mla_isoquant_dkv.py` | Kimi MLA cache: compress 512-D `kv_latent`, keep 64-D `k_pe` raw, support `trim()` |
| `scripts/profile_metal_counters.py` | Synthetic read/write component profiler |
| `scripts/run_write_path_attribution_gate.sh` | One-shot MLX smoke → NPT8 tests → synthetic profile (+ optional `RUN_E2E=1`) |
| `artifacts/metal-counters/profile_with_write_e2e.json` | Synthetic + default vs IsoQuant E2E + `comparison{}` |
| `artifacts/metal-counters/write_path_attribution_memo.md` | Narrative attribution + E2E vs prediction |
| `artifacts/kimi_k26_profiling/npt16_synthetic_gate.json` | Kimi NPT16 pytest smoke (`run_kimi_npt16_parity_gate.py --synthetic-only`) |
| `scripts/ab_kimi_layered_stack.py` | Kimi layered A/B harness for default/IsoQuant/expert levers |
| `scripts/sweep_kimi_default_cache_residency.py` | L0-only sweep of `max_resident_experts` → `default_cache_sweep.json` |
| `scripts/run_kimi_npt16_parity_gate.py` | Records synthetic `test_fused_npt16.py` result + real-weight MLA parity placeholder |
| `scripts/profile_kimi_speculative.py` | Kimi target/draft speculative decode smoke harness |
| `docs/KIMI_K26_FULL_STACK.md` | Kimi runbook, evidence, and remaining blockers |
| `scripts/eval_quality_gate.py` | Quality gate and strict-mode artifacts |
| `scripts/benchmark_moe_offload.py` | Throughput/memory benchmark |

---

## Phase 0: Sanity Gate Before More Measurement

**Goal:** avoid collecting invalid MLX/Metal data after a stuck GPU/session state.

- [ ] **Step 0.1: Reboot or verify clean system state**
  - Required before any new Metal benchmark or NPT8 test.
  - Check for stuck `Python -m pytest`, `benchmark_nvfp4_isoquant.py`, or `mlx_lm.generate` processes.
  - Record result in `artifacts/metal-counters/system_state_YYYYMMDD.json` or a short markdown note.

- [ ] **Step 0.2: Run minimal MLX import smoke before focused tests**
  - Command:
    ```bash
    PYTHONPATH=mlx-lm python -c "import mlx.core as mx; print(mx.default_device())"
    ```
  - Pass condition: import exits cleanly and leaves no stuck `UEs` process.
  - If this fails, stop all Metal benchmark/test work and treat the session as MLX/runtime unhealthy.

- [ ] **Step 0.3: Re-run the focused NPT8 tests only after import smoke passes**
  - Command:
    ```bash
    PYTHONPATH=mlx-lm python -m pytest \
      mlx-lm/tests/test_fused_npt8.py \
      mlx-lm/tests/test_fused_npt8_tiled.py -q
    ```
  - Pass condition: tests complete without Metal abort or stuck `UEs` process.
  - If this fails after the import smoke passes, then open a focused NPT8/tiled stability task.

- [ ] **Step 0.4: Capture current git/artifact hygiene**
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

## Phase 3: Run Write-Path Attribution

**Goal:** close the attribution gap before building any new compression math.

- [x] **Step 3.1: Confirm `scripts/profile_metal_counters.py` write-path wiring exists**
  - The profiler includes write components in `run_synthetic_phase()`:
    - `compress_python`
    - `compress_fused_metal`
    - `pack_3bit`
    - `cache_concat`
    - `cache_prealloc`
  - It accounts for K and V writes as `2x` per decode step.
  - This marks code wiring complete only; it does not close the measurement gate.

- [x] **Step 3.2: Run synthetic write-path profile**
  - Command:
    ```bash
    python scripts/profile_metal_counters.py \
      --model /Users/anthonylui/Models/Qwen3.6-35B-A3B-nvfp4 \
      --output artifacts/metal-counters/profile_with_write.json \
      --prefill 4096 8192 --skip-e2e --skip-traces
    ```
    Or: `./scripts/run_write_path_attribution_gate.sh`.
  - Artifact: `artifacts/metal-counters/profile_with_write.json` (pinned for Qwen3.6 nvfp4 run).

- [x] **Step 3.3: Compare read + 2x write prediction against E2E gap**
  - If prediction explains the gap, optimize the dominant component.
  - If residual remains large, treat Python/MLX dispatch and allocation as the bottleneck.
  - Artifacts: **`artifacts/metal-counters/profile_with_write_e2e.json`** (`comparison`≈95–102% of E2E gap @ T=4096/8192, Qwen3.6 nvfp4) and **`artifacts/metal-counters/write_path_attribution_memo.md`**.

- [x] **Step 3.4: Decide whether fused encode and prealloc graduate** — **GRADUATE** (closed 2026-05-05 with v2 reproduction)
  - Perf evidence v2 (current code, clean boot 2026-05-05T15:51Z):
    `artifacts/branch_c_profiling/write_path_ablation_paired_v2.json`. All
    three candidates reproduce as wins. **Sign reproduces 6/6 cells (all
    positive); magnitude is 5/6 cells equal or larger.** The exception is
    `prealloc @ T=8192` which is smaller in v2 (+0.75 vs v1 +1.27, ratio
    0.58×). v2 deliberately ran only the three §3.4 candidates
    (`--configs fused_enc prealloc combined`); sanity configs `no_npt8` and
    `metal_fwd` were not re-measured and the v1 rows below remain pinned
    for context. **`combined` is stable signal at both T=4096 and T=8192**
    (was outlier×1 at both in v1):
    - `fused_enc`: T=4096 +2.89 (outlier×2), T=8192 +2.70 (outlier×1)
    - `prealloc`: T=4096 +0.68 (noisy), T=8192 +0.75 (outlier×2)
    - `combined`: T=4096 **+2.93 (stable)**, T=8192 **+2.73 (stable)**
    - T=8192 paired_gap is harness-flagged INVALID (2 outliers in gap itself
      → % gap suppressed; absolute Δ remains valid)
  - Perf evidence v1 (historical, PROVISIONAL — pre-commit code):
    `artifacts/branch_c_profiling/write_path_ablation_paired.json`
    (merged from `ablation_paired.json` 2026-04-28T21:03Z + `ablation_combined.json`
    2026-04-28T21:35Z — see memo `write_path_ablation_memo.md`). Per decode
    step paired delta vs iso baseline (positive = variant faster):
    - `fused_enc`: T=4096 +1.23 (stable), T=8192 +1.92 (1 outlier)
    - `prealloc`: T=4096 +0.61 (stable), T=8192 +1.27 (2 outliers)
    - `combined`: T=4096 +1.33 (1 outlier), T=8192 **+2.14** (1 outlier)
    - `no_npt8` sanity: T=4096 -1.08, T=8192 -2.55 (NPT8 helps; keep on)
  - Quality evidence (FINAL — current code): `artifacts/branch_c_profiling/write_path_quality_gate/`
    (`baseline_iso.json`, `fused_encode.json`, `prealloc.json`, `combined.json`,
    `QUALITY_GATE_MEMO.md`). Greedy/seed=42 micro-suite: response text
    **byte-identical** across all 4 configs; peak memory identical at
    18711.9 MB. Harness `FAIL` is false-fail of strict repetition gate on
    correct trivially-short answers and applies equally to baseline.
  - **Code-identity gap (Codex audit 2026-05-05):** the FUSED_ENCODE Metal
    path was committed AFTER the 2026-04-28 perf runs:
    `mlx-lm/mlx_lm/models/fused_kv_compress.py` first appears in `57cb5f5`
    (2026-05-02); the import was added in `40501b2` (2026-04-29). The
    2026-04-28 ablation ran against uncommitted local code which cannot now
    be reconstructed bit-exactly. The activation receipts confirm
    FUSED_ENCODE was active at run time, but the perf delta numbers cannot
    be tied to the currently-committed code without a rerun.
  - **Promotion (final):** `combined` (FUSED_ENCODE=1 + CACHE_MODE=prealloc)
    promoted as the recommended default for IsoQuant write path on this
    model class. Single-flag variants remain opt-in. v2 reproduction
    confirms the wins on currently-committed code.
  - **Caveat (quality):** signal is from the micro suite (2 prompts, 48 token
    cap) because the IsoQuant reconstruct fallback path on Qwen3.6
    head_dim=128 is too slow for the v2 default suite (5×200 tokens) —
    initial default-suite run exceeded 25 min wall budget and was cancelled.
    Larger-suite validation of the IsoQuant fallback path is a separate item,
    not gating §3.4.

---

## Phase 4: Math-First Bandwidth Experiments

**Goal:** run tiny, falsifiable experiments for left-field ideas before committing engineering time.

**Scope note:** these experiments are not automatically Kimi work. Kimi MLA already has a compressed latent representation, and current Kimi measurements favor default MLA cache over Kimi IsoQuant for speed. For Kimi, bandwidth math only becomes active if long-context or memory-envelope evidence shows default MLA cache is no longer acceptable.

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
  - Kimi variant: if included, apply transforms only to `kv_latent`; never rotate or quantize `k_pe`.

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
  - Apple `ml-ssd` can motivate attention-distribution inspection as a quality/post-training reference, but it does not implement this runtime mechanism.

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

- [ ] **Step 5.4: Kimi K2.6 decode acceleration lane**
  - Current evidence:
    - source checkpoint: `/Volumes/Samsung9904tb/Kimi-K2.6` (~554 GB),
    - offload load/decode smoke passes,
    - Kimi MLA IsoQuant cache and `trim()` exist,
    - NPT16 D=512 kernel is synthetic-correct and speeds Kimi IsoQuant vs unfused Kimi IsoQuant,
    - default MLA cache is still the faster measured Kimi decode path,
    - predictor/cliques did not produce a stable speed win,
    - 2-bit expert conversion scripts exist, but the generated checkpoint was deleted and must be regenerated before speculative testing.
  - Next actions:
    1. Run a clean default-cache `max_resident_experts` sweep before further KV work:
       ```bash
       PYTHONPATH=mlx-lm python scripts/sweep_kimi_default_cache_residency.py \
         --model /Volumes/Samsung9904tb/Kimi-K2.6 \
         --sweep-values 64,128,200,400,800,1200 \
         --output artifacts/kimi_k26_residency/default_cache_sweep.json
       ```
    2. If disk/time is approved, regenerate the 2-bit per-expert checkpoint and load-smoke it after the `.scales` gate fix.
    3. Run `scripts/profile_kimi_speculative.py` only after a working draft checkpoint exists.
    4. Run the NPT16 parity gate (synthetic now; real-weight MLA block documented in JSON) before treating NPT16 as more than synthetic-correct:
       ```bash
       PYTHONPATH=mlx-lm python scripts/run_kimi_npt16_parity_gate.py \
         --output artifacts/kimi_k26_profiling/npt16_real_weight_logit_parity.json
       ```
  - Artifacts:
    - `artifacts/kimi_k26_residency/default_cache_sweep.json`,
    - regenerated 2-bit checkpoint integrity/load-smoke JSON,
    - `artifacts/kimi_k26_speculative/speculative_smoke.json`,
    - `artifacts/kimi_k26_profiling/npt16_real_weight_logit_parity.json`.
  - Do not update pathway-proven docs for Kimi until quality, memory, throughput, and soak artifacts exist.

- [ ] **Step 5.5: SSD/self-distillation remains optional**
  - Create `docs/PHASE7_SSD_OUTLINE.md` only after the RC path is stable.
  - Frame SSD as quality/post-training work, not memory/runtime work.
  - Do not use Apple `ml-ssd` as evidence for faster Kimi decode; it can only support a later code-quality/fine-tuning branch.

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
  - Qwen3-30B quality block, NPT8 residual gap, Kimi default-vs-IsoQuant result, Kimi 2-bit regeneration state, and speculative/predictor regressions should be documented as measured constraints, not hidden.

---

## Decision Rules

1. **No phase closes from documentation alone.** Require JSON artifacts, passing tests, committed code, or explicit human sign-off.
2. **Do not add new math to the release path until Phase 1 passes.** Math experiments may run offline, but they do not change the RC profile.
3. **Prefer fewer mechanisms.** A release profile with offload + IsoQuant + one quant recipe is stronger than a profile requiring stacked approximations.
4. **Measure paired when comparing performance.** Single-run speedups are directional only.
5. **Treat MLX/Metal hangs as invalidating the session.** Reboot before new benchmark claims.

## Recommended Next Action

Start with Phase 0, then choose one of three non-overlapping lanes:

1. clean system state / reboot,
2. run the minimal `import mlx.core` smoke,
3. if clean, rerun focused NPT8 tests,
4. lane A: close the Nemotron RC gate,
5. lane B: synthetic + E2E write-path attribution is **pinned** for Qwen3.6 nvfp4 (`profile_with_write_e2e.json`, memo); next run **§3.4** (`--fused-encode`, `--prealloc`, paired repeats) or re-run `./scripts/run_write_path_attribution_gate.sh` with `RUN_E2E=1` after any MLX wedge,
6. lane C: run Kimi default-cache residency sweep before any more Kimi KV-compression or `ml-ssd` work.

Only after that should Phase 4 bandwidth-math prototypes get engineering time.
