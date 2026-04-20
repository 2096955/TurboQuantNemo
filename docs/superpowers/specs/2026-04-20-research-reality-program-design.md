# Mojo + RotaryQuant — Research Reality Program

**Date:** 2026-04-20
**Status:** Design draft, awaiting user review
**Sequencing:** Serial — Phase 0 → Phase 1 → Phase 2 (no hard deadline; correctness over speed)

---

## Two parallel research objectives

This program serves **two distinct objectives**, run as serial phases after cleanup:

| # | Objective | Phase | What "done" looks like |
|---|-----------|-------|------------------------|
| 1 | **Speed** — push offload-pathway tok/s as high as physically possible on Apple Silicon | Phase 1 | Async expert prefetch shipped + measured; offload pathway tok/s improved or floor proven; ≥5 additional models with quality data |
| 2 | **Portable RotaryQuant in Mojo** — prove the full RotaryQuant pipeline (IsoQuant + WHT + SO(4) + MoE separation) works *for real* in Mojo, end-to-end, on Apple Silicon | Phase 2 | Real model runs through Mojo RotaryQuant pipeline, output cross-validated vs MLX reference; portability claim grounded in Mojo's language-level multi-backend guarantee |

**Publication is not a phase.** Once results from Phase 1 and Phase 2 exist, blog posts go up on the existing GitHub blog as wrap-up. No publication scaffolding sprint.

**Non-Apple GPU testing is out of scope** — no NVIDIA/AMD infrastructure available. The portability claim stands on (a) Mojo end-to-end working on Apple Silicon, plus (b) Mojo's compilation model targeting NVIDIA/AMD/Apple from the same source. Cross-vendor empirical validation is honest future work.

---

## Consolidation of prior specs

This spec supersedes future planning around speed and Mojo work. Prior specs stay in the repo as historical foundation:

| Prior spec | Status now | Role going forward |
|------------|------------|-------------------|
| `2026-04-13-mojo-vs-mlx-kernel-benchmark-design.md` | DONE — kernel-level benchmark complete (21 result JSONs, 5 adversarial reviews, MLX 8.4× faster geomean on decode-shape) | Cited as Phase 2 foundation; provides per-kernel performance characterization and stats harness. **No longer planning ground.** |
| `2026-04-16-qwen36-mixed-precision-pathway.md` | DONE — Qwen3.6-35B-A3B mixed-precision (4-bit dense / 2-bit experts / 8-bit shared / IsoQuant 3-bit KV) converted and validated | Cited as B-γ baseline; Qwen3.6 used as primary speed test target in Phase 1. **No longer planning ground.** |

Phase 0 includes a small task to update each prior spec's status block to `DONE — superseded by 2026-04-20-research-reality-program-design.md`.

---

## Audit findings driving Phase 1 scope

Codebase audit on 2026-04-20 invalidated two earlier B candidates:

**MTP / speculative decoding** is already implemented at `mlx-lm/mlx_lm/generate.py:530-714` (`speculative_generate_step`), with CLI flags (`--draft-model`, `--num-draft-tokens`), server integration in `mlx-lm/mlx_lm/server.py`, cache rewinding, and `expert_mgr.set_phase()` hooks. **Explicitly rejected for the offload pathway** in `docs/FROM_ATTENTION_TO_CONSUMER_HARDWARE.md:515-520`: each speculative token can route to different experts, dropping prefetch hit rate below 70%.

**Block AttnRes** is already implemented at `mlx-lm/mlx_lm/models/kimi_linear.py:59-71` (`BlockAttnRes` class with softmax over layer outputs). The AttnRes-based expert predictor at `mlx-lm/mlx_lm/expert_offload.py:76-122` was **measured at 10.6–11.2% throughput regression** and marked No-go.

The audit-identified decode bottleneck on offload pathways is **expert disk I/O at ~3.5 ms/token**, not arithmetic. RotaryQuant has exhausted low-hanging fruit in decode-phase compute optimization; remaining headroom is in I/O scheduling.

---

## Realistic speed bar (Phase 1)

| Configuration | Documented tok/s |
|---------------|------------------|
| Initial Qwen baseline (likely raw fp16, all resident) | 131 |
| Qwen3.6-35B-A3B with full RotaryQuant + offload | 15.6 |
| Nemotron-H 120B with full RotaryQuant + offload | 14.85 |
| Gemma4-26B with full RotaryQuant + offload | 12.85 |

The bulk of the gap from 131 → ~15 is **the value proposition** (fitting models that wouldn't otherwise fit), not a regression. Realistic recoverable ceiling on the offload pathway is **~25–40 tok/s** (1.5–2.5× current). Phase 1 measures success against that bar; the 131 figure is not a target.

---

# Phase 0 — Cleanup (1 week)

Pay down accumulated debt before any new work to prevent further DRY violations.

## 0.1 Land Mojo benchmark worktree

**Worktree:** `.claude/worktrees/mojo-vs-mlx-benchmark/mojo-bench`

**Status today:**
- 21 result JSONs in `mojo-bench/results/`
- 7 kernel files: `bench_matmul`, `bench_softmax`, `bench_rope`, `bench_isoquant_rotate`, `bench_kv_compress`, `bench_fused_attention`, `bench_vec_add`
- 5 adversarial reviews complete (Manual, Gemini, Codex×2, Superpowers code-reviewer, Human A−)
- DW (Durbin-Watson) code fix verified in source; result JSONs predate fix → re-run sweep needed before merge

**Action:**
1. Re-run benchmark sweep with current DW-fixed code (≤4 hours wall-clock)
2. Diff old vs new JSONs; if results materially differ, surface a note for the eventual blog post
3. Rebase worktree onto current `main`
4. Resolve any conflicts (none expected — `mojo-bench/` is a new top-level directory)
5. Run smoke test: `cd mojo-bench && pixi run bench-vec-add` to confirm environment is intact post-rebase
6. Merge to `main`, delete worktree

**Acceptance:**
- `git log main` shows merge commit
- `mojo-bench/` directory present in main tree with 21 fresh JSONs
- `pixi run bench-all` reproduces results from a clean checkout
- Worktree directory removed; branch deleted

## 0.2 Delete abandoned/placeholder worktrees

Audit identified 4 abandoned/placeholder worktrees. Implementation plan will enumerate them by name with last-touched date and rationale for deletion.

**Process per worktree:**
1. `git diff <worktree-branch> main --stat` — confirm zero substantive changes (or only changes already merged)
2. Capture branch HEAD SHA in implementation plan log (in case later recovery needed)
3. `git worktree remove` then `git branch -D`

**Acceptance:** `git worktree list` shows only worktrees with active intent (mojo-bench post-merge → 0; remaining = current dev worktrees only).

## 0.3 Triage 2 dirty worktrees

| Worktree | Issue |
|----------|-------|
| `qwen3-wiring` | 32 commits behind main; dirty edits overlap committed work |
| `qwen3-deferred-dedekimi-impl` | Test file duplicates one already on main |

**Process per worktree:**
1. Generate diff vs main: `git diff main...<branch>` and `git status`
2. Classify each hunk:
   - `duplicates_main` — already on main; discard
   - `genuinely_new` — not on main; keep
   - `ambiguous` — needs case-by-case review
3. Produce one-page recommendation document: per-hunk classification + recommended action
4. **User ratifies the recommendation document before any destructive action**
5. Execute approved actions: rebase salvageable hunks onto a clean branch, abandon the rest

**Acceptance:**
- Zero worktrees with dirty edits overlapping main
- Salvaged work either merged, in a PR, or in a clearly-named follow-up branch
- Discarded work logged with SHA + rationale (recoverable if needed)

## 0.4 δ note audit (1 day, tail of Phase 0)

The earlier brainstorm raised a hypothetical "B-δ: routing-aware draft model for spec-decode on offload." User correctly noted that auditing the existing rejection notes is cheaper than building speculatively.

**Scope:**
- Read `docs/FROM_ATTENTION_TO_CONSUMER_HARDWARE.md:515-520` rejection text in full
- Cross-reference commit history (`git log --all -p -- docs/FROM_ATTENTION_TO_CONSUMER_HARDWARE.md`) for when the rejection was added and why
- Search related branches/specs for any prior measurement (`grep -r "speculative" docs/superpowers/`)
- Determine: was rejection global, or only specific to 120B + offload? Are there narrow cases (smaller models, partial-offload, draft-on-shared-experts) worth a future attempt?

**Output:** `docs/superpowers/notes/2026-04-2X-spec-decode-rejection-audit.md` containing:
- Original rejection text + commit context
- Measurement evidence (or lack thereof)
- Verdict, one of: `kill_permanently` / `reopen_for_narrow_case_X` / `inconclusive_need_measurement`
- If `reopen_for_narrow_case_X`: scope estimate for a future sprint

## 0.5 Mark prior specs DONE

Two-line edit each. Update front-matter Status to `DONE — superseded by 2026-04-20-research-reality-program-design.md`.

## 0.6 Numerical invariance harness (correctness gate before any optimization)

**Why before Phase 1:** Async expert prefetch (1-α) can change expert dispatch ordering under race conditions you don't anticipate. If that silently corrupts output, you'll see it as benchmark noise — or not at all. Anchor correctness with a bit-exactness gate that exists *before* you start optimising.

**Deliverable:** `scripts/invariance_check.py`
- Runs the same prompt twice through the same model + same seed
- Asserts identical output token-for-token (and identical logits within fp16 epsilon)
- Runs as a CI smoke test before any 1-α PR merges
- Three model configs covered: dense (Llama 3.2 3B), MoE (Qwen3.6-35B-A3B), MoE-with-offload (same Qwen, expert offload enabled)

**Acceptance:** Three configs pass invariance check; failures block any 1-α merge until resolved.

## Phase 0 exit criteria

- Mojo bench merged to main with fresh JSONs; reproducible from clean checkout
- All worktrees clean (no DRY violations)
- δ memo committed with verdict
- Both prior specs marked DONE
- Invariance harness in place and passing on three model configs
- Phase 1 starts on a clean, correctness-gated foundation

---

# Phase 1 — Speed (Objective 1) — 2-3 weeks

**Sequencing within Phase 1:** Option 1 — **B-α primary, B-γ background, B-β deferred to next sprint**.

## 1-α  Async expert prefetch + scheduling (PRIMARY)

**Target:** the 3.5 ms/token expert I/O bottleneck on the offload pathway.

### Architecture

```
Decode loop (current):
  for each token:
    layer.compute()  ──► router.select_experts()  ──► load_experts_from_disk()  ──► experts.compute()
                                                              │
                                                              └─ STALL: ~3.5 ms here

Decode loop (after 1-α):
  prefetch_thread:  predicts next-layer experts → loads them in background
  main thread:
    for each token:
      layer.compute()  ──► router.select_experts()  ──► (already loaded)  ──► experts.compute()
                                                              │
                                                              └─ STALL eliminated when prefetch hit
```

### Components

**Profiler (Step 1, mandatory before any optimization):**
- File: `mlx-lm/mlx_lm/expert_offload.py` (extend existing)
- Adds: per-layer stall histograms, expert reuse-distance distribution, prefetch-hit-rate counter
- Runs as `--expert-profile` flag on `mlx_lm.generate`
- Output: JSON per run, summary table CLI

**Prefetch scheduler (Step 2):**
- File: `mlx-lm/mlx_lm/expert_prefetcher.py` (new) OR extension of `expert_offload.py` (decision: implementation plan)
- Uses next-layer router prediction (cheap forward pass on routing weights only) to enqueue prefetch
- Configurable look-ahead depth (1-4 layers)
- Integration point: `expert_mgr.set_phase()` already exists at `mlx-lm/mlx_lm/expert_offload.py`

**Eviction policy update (Step 3):**
- Modify LRU in `expert_offload.py`: `LRU + write-back-aware` — never evict an expert in the prefetch queue
- Add hysteresis to prevent thrashing under high look-ahead depth

**Optional: mmap-backed weights (Step 4, decided by profiler output):**
- If profiler shows OS page cache is helping (high hit rate without explicit prefetch), invest in `mmap`-based loader
- If profiler shows OS page cache is cold (every load is a real disk read), explicit threadpool prefetch wins
- Decision lives in implementation plan, informed by Step 1 data

### Data flow

```
                ┌──────────────────────────────────────────┐
                │  Main inference thread                   │
                │                                          │
   Token N  ──► │  Layer L compute → Router.predict_L+1   │
                │                          │               │
                │                          ▼               │
                │                    [ Predicted exp set ] │
                │                          │               │
                └──────────────────────────┼───────────────┘
                                           │ enqueue
                                           ▼
                ┌──────────────────────────────────────────┐
                │  Prefetch thread (separate executor)     │
                │                                          │
                │  Dequeue → Check LRU → If miss: SSD read │
                │                                  │       │
                │                                  ▼       │
                │                          [ Insert LRU ]  │
                │                                          │
                └──────────────────────────────────────────┘
                                           │
                                           ▼
   Token N+1 ─► Layer L+1 compute → experts already resident → no stall
```

### Test strategy

| Test | What it proves |
|------|---------------|
| Profiler unit tests on synthetic stall pattern | Profiler correctly attributes stalls to expert load events |
| Prefetcher correctness: outputs match no-prefetch path bit-for-bit | Prefetch is a perf optimization, not a numerical change |
| End-to-end benchmark on Qwen3.6-35B-A3B (15.6 tok/s baseline) | Headline number |
| End-to-end benchmark on Nemotron-H 120B (14.85 baseline) | Generalization to larger model |
| End-to-end benchmark on Gemma4-26B (12.85 baseline) | Generalization to different MoE topology |
| Stress test: long context (8K+) with high expert diversity | Confirms no thrash, no memory leak |
| Failure mode: prefetch queue saturation | Confirms backpressure prevents OOM |

### Acceptance bands

| Band | Criterion | Outcome |
|------|-----------|---------|
| Stretch | +50% tok/s on Qwen3.6-35B-A3B (15.6 → 23+) | Headline result; goes in blog |
| Minimum | +20% tok/s on at least 2 of 3 test models | Phase 1-α success |
| Floor | <10% improvement on all 3 models | Phase 1-α failure path: ship profiler as standalone tool, write up findings as "expert I/O is fundamental SSD latency" paper contribution |

### Risk

**High.** May discover that bottleneck is fundamental SSD random-read latency. If profiler reveals look-ahead can't hide stalls (e.g., expert prediction accuracy too low), software prefetch can't help. The profiler itself is shippable value if optimization plateaus.

## 1-γ  Wider model coverage + quality validation (BACKGROUND)

Runs concurrently with 1-α as automated benchmark sweeps; minimal active developer time.

### Approach

1. Extend `scripts/eval_quality_gate.py` to apply RotaryQuant to 5+ additional models
2. Candidates (final list in implementation plan; criteria: HF availability, MLX-compatible, quality baseline measurable):
   - Llama 3.x small variants (1B, 3B, 8B)
   - Phi-4
   - Mistral 7B / 22B variants
   - Qwen 2.5 7B / 14B (older line, sanity check)
3. Run the 5-prompt quality gate per model (same harness as `eval_quality_gate.py`)
4. Measure PPL drift vs uncompressed baseline
5. Document failure modes per model in `docs/RotaryQuant_paper.md` Appendix

### Components

- `scripts/eval_quality_gate.py` — extend to take a model list
- `scripts/run_coverage_sweep.sh` — orchestrator (sequential, log-aggregating)
- `docs/RotaryQuant_paper.md` Appendix — coverage table

### Acceptance

- ≥5 models pass the quality gate (PPL drift <5% per existing gate threshold)
- Coverage table appended with: model, PPL drift, tok/s, memory peak, pass/fail, notes
- Any failures reproducible from a single command

### Risk

**Low.** Mostly automation + waiting. Worst case: a model has a tokenizer/architecture quirk that requires a small adapter — captured as known limitation in coverage table.

## 1-β  Mojo prefill kernels (DEFERRED — next sprint)

Captured for future reference, not in scope this cycle. Phase 0 Mojo benchmark showed MLX 8.4× faster geomean on **decode-shaped** workloads. Prefill is compute-bound and may be a different story; worth re-benchmarking on prefill-shaped tensors before committing engineering effort. Belongs to a future spec.

## Phase 1 exit criteria

- 1-α: profiler shipped + outcome (stretch / minimum / floor) committed to repo with reproducible benchmarks
- 1-γ: ≥5 additional models with published quality data
- All tok/s claims have reproducible benchmark scripts in repo
- Invariance harness still passing — async prefetch did not introduce nondeterminism
- **1-α profiler findings feed forward into M2.2 scoping** — if I/O is fundamental, MoE-in-Mojo target may shrink to "MoE works correctly" rather than "MoE matches MLX speed"
- `docs/RotaryQuant_paper.md` updated with Phase 1 results (no separate publication step)
- Phase 2 unblocks (no shared infrastructure dependency, but serial execution to avoid context-switching cost)

---

# Phase 2 — End-to-End Mojo RotaryQuant — 4-6 weeks

**The contribution is end-to-end Mojo inference of the full RotaryQuant pipeline.** Portability across vendors is the *implication* this contribution enables (covered in M2.3), not the frame of the work.

Phase 0's Mojo work was kernel-level — proves Mojo *can* execute IsoQuant rotation, KV compression, attention. Phase 2 proves the *full pipeline* works *for real*: real model weights → real inference loop → real tokens, with output cross-validated against MLX reference, on Apple Silicon.

Portability framing (developed in M2.3, not the goal of M2.1/M2.2):
1. **Empirical:** End-to-end RotaryQuant in Mojo runs on Apple Silicon (this phase delivers)
2. **Architectural:** Modular Platform 25.6+ compiles Mojo to KGEN → LLVM IR → backend bitcode for Apple Metal, NVIDIA CUDA, AMD ROCm from the same source. Cross-vendor deployment is a build-target switch, not a rewrite.

We do not test on non-Apple GPUs in this phase (no infrastructure). The portability claim is honestly framed: *the work runs on Apple Silicon today; the language guarantee makes other backends a deployment exercise rather than a research exercise.* Future work, infrastructure permitting, validates empirically.

## 2.0 Interop decision (Phase 2 first task — must precede any kernel work)

The Python ↔ Mojo handoff for `model_loader.py` is a load-bearing decision. Three candidates:

| Approach | Pros | Cons | Risk |
|----------|------|------|------|
| PyMojo binding | Fewest IPC hops; ergonomic | Bleeding-edge; tied to Modular SDK churn | High — can swallow a week if it regresses |
| Subprocess (Python loads weights, pipes tensors) | Easy to swap | Tensor serialisation overhead; complex stdio handling | Medium |
| **File-based handoff** (Python writes safetensors → Mojo mmaps) | **Most debuggable; no live IPC; works today** | One disk round-trip on model load (acceptable — once per session) | **Low** |

**Default decision: file-based handoff.** Mojo reads weights via `mmap` from a safetensors file written by an existing Python script. No live process coupling. If Phase 2 measurements later show this dominates first-token latency (and it shouldn't — it's a one-time load, not per-token), revisit.

**Acceptance for 2.0:** decision documented in `mojo-bench/rotaryquant/INTEROP.md` before any other Phase 2 work begins. Implementation plan locks the contract: file format, where Mojo expects to find it, how Python signals completion.

## 2.1 Architecture

```
mojo-bench/                                    (existing — kernels DONE in Phase 0)
└── kernels/                                   ← bench_isoquant_rotate, bench_kv_compress, etc.

mojo-bench/rotaryquant/                        ← NEW: end-to-end pipeline
├── pixi.toml                                   ← shared with mojo-bench
├── isoquant.mojo                               ← IsoQuant 4D rotation (port from kernel; productionize)
├── wht.mojo                                    ← Walsh-Hadamard Transform (new in Mojo)
├── so4.mojo                                    ← SO(4) block rotation (extracted from isoquant_rotate kernel)
├── moe_separation.mojo                         ← Per-component quantization (gate / up / down / shared)
├── kv_cache.mojo                               ← KV cache management (allocation, compress, retrieve, evict)
├── attention.mojo                              ← Multi-head attention orchestration (calls so4 + wht + kv_cache)
├── model_loader.py                             ← Python-side weight loader (interop, OK to use Python initially)
├── inference.mojo                              ← End-to-end decode loop
├── validate/
│   ├── compare_logits.py                       ← Cross-validate Mojo logits vs MLX logits per token
│   └── tolerance.py                            ← Numerical tolerance defs (top-k, KL divergence)
└── README.md                                   ← Reproduction
```

## 2.2 Reference implementation (what to port)

Source of truth for RotaryQuant in MLX:
- `mlx-lm/mlx_lm/models/mlx_isoquant.py` (35.9 KB) — main IsoQuant impl with WHT + SO(4) combined path
- `mlx-lm/mlx_lm/models/isoquant_metal_kernels.py` (18.6 KB) — Metal kernels (this is what the Mojo port mirrors)
- `mlx-lm/mlx_lm/models/kimi_mla_isoquant_dkv.py` (1 KB) — MLA-specific variant (out of scope for Phase 2)

Mojo port targets feature-parity with `mlx_isoquant.py` (excluding MLA). The Metal kernels file is the closest analogue to what Mojo writes.

## 2.3 Milestones

### M2.1 — IsoQuant + WHT + SO(4) end-to-end on small dense model (~2 weeks)

**Why dense first:** Dense model is a clean test for the rotation/quantization stack. MoE adds expert dispatch complexity that should be layered on top of a working dense pipeline, not co-developed.

**Target model:** Llama 3.2 3B (small, MLX-compatible, no MoE, IsoQuant works in MLX as control)

**Deliverables:**
- `isoquant.mojo`, `wht.mojo`, `so4.mojo` — feature-equivalent to MLX kernels
- `kv_cache.mojo` — handles compress on store, decompress on load
- `attention.mojo` — multi-head attention with IsoQuant KV
- `inference.mojo` — decode loop, single-prompt
- Cross-validation: per-token logits within tolerance vs MLX run on same model + same prompt

**Acceptance for M2.1:**
- Single prompt completes (e.g., "The capital of France is" → coherent continuation)
- **Top-1 token match rate target ≥98% vs MLX reference over 50 generated tokens; ≥95% acceptable with documented drift analysis** (tracks where divergence accumulates: layer index, head index, op type — fma ordering vs softmax overflow vs reciprocal approximation)
- Top-5 logit KL divergence <0.05 vs MLX reference per token (mean over sequence)
- No memory growth across 256-token generation (no leak)
- Documented tok/s on M4 Max (no expectation of beating MLX — this is feature-parity, not speed)
- **Staged-buffer pattern documented in deliverable** — Mojo kernels do not assume zero-copy semantics on unified memory; explicit copy boundaries between Python-loaded weights and Mojo-resident tensors. Mitigates the silent-wrong-results failure mode where Mojo and Python view the same buffer with different lifetimes

### M2.2 — MoE separation, scale to small MoE (~2 weeks)

**Why second:** MoE separation is the RotaryQuant feature that distinguishes it from generic 4-bit quant. Scaling to MoE proves the *full* RotaryQuant pipeline.

**Target model:** Qwen3.6-35B-A3B (the program's existing focus model; MoE; mixed-precision recipe already validated in MLX per the DONE qwen36 spec)

**Deliverables:**
- `moe_separation.mojo` — per-component quantization (gate 4-bit, up 4-bit, down 4-bit, shared 8-bit, routed 2-bit)
- Extension of `inference.mojo` for MoE routing
- Expert dispatch in Mojo (no offload at this milestone — fits-in-RAM is fine for proof of concept)

**Acceptance for M2.2:**
- Same prompt completes on Qwen3.6-35B-A3B via Mojo
- Top-1 token match rate ≥95% vs MLX reference (lower threshold than M2.1 because MoE introduces nondeterministic-but-acceptable routing variance)
- Documented tok/s on M4 Max
- All 5 quality-gate prompts pass (no garbage output, no infinite loops)

### M2.3 — Documentation + portability claim writeup (~1 week)

**Deliverables:**
- `mojo-bench/rotaryquant/README.md` — reproduction instructions
- `docs/MOJO_END_TO_END.md` — architecture writeup, lessons learned, performance comparison vs MLX
- Section in `docs/RotaryQuant_paper.md` covering portable Mojo implementation
- Honest portability claim: empirical (Apple Silicon today) + architectural (Mojo language guarantee for other backends), with future-work acknowledgment

## 2.4 Cross-validation harness

### Tolerance definitions (`validate/tolerance.py`)

```
TOLERANCE = {
    "top_1_match_rate": 0.98 for dense, 0.95 for MoE,
    "top_5_kl_divergence": 0.05 per token (mean over generated sequence),
    "logit_l_inf": 0.1 (max absolute difference on top-1 logit),
    "tok_s_ratio": "report-only (no pass/fail)",
}
```

### Test protocol

For each milestone, run side-by-side:
1. MLX reference: `python -m mlx_lm.generate --model <X> --kv-cache-type isoquant --prompt <P> --max-tokens 50 --seed 42`
2. Mojo implementation: `pixi run inference --model <X> --prompt <P> --max-tokens 50 --seed 42`
3. Diff logits per generated token; check against tolerance
4. Report pass/fail + per-token diff plot

### Test corpus

5 prompts spanning: factual ("capital of France"), reasoning ("if 2x+5=15, x="), code (a small Python function), instruction-following (summarize a paragraph), open-ended creative (one sentence). Same as `eval_quality_gate.py` corpus where possible.

## 2.5 Risk register

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| Modular Platform 25.6+ silently produces wrong results on unified-memory zero-copy paths | **High** | Two-layer mitigation: (1) pin to Mojo 0.26.3.0.dev2026040805 (already known good per Phase 0); (2) explicit staged-buffer pattern in M2.1 deliverable — never assume zero-copy semantics across Python ↔ Mojo boundary; documented copy points; invariance harness from Phase 0.6 catches symptom even if pattern slips |
| Modular Platform 25.6+ tooling regressions (GPU detection, compiler bugs) | Medium | Pinned compiler version; no upgrades mid-phase |
| Python+Mojo interop has high overhead, dominates runtime | Medium | Acceptable for M2.1 (proof of concept); document as known issue; not the bottleneck for portability claim |
| Numerical equivalence to MLX too tight; Mojo float ops differ subtly | Medium | Tolerance bands defined upfront; report failures with KL plots, not pass/fail booleans |
| MoE expert dispatch in Mojo is more work than allocated | High | M2.2 is the riskier milestone; honest scope cut: deliver dense-only M2.1 + scope M2.2 to a future sprint if blocked |
| 4-6 weeks underestimates Mojo learning curve / tooling friction | Medium | "No deadline, get it right" stance applies; phase length is estimate, not commitment |
| MLX reference is moving target (active dev) | Low | Pin MLX commit at start of Phase 2; document version |

## 2.6 Failure modes and stop conditions

If at end of week 3 of Phase 2:
- M2.1 incomplete and no clear path to top-1 ≥98% → cut to **honest goal**: ship working IsoQuant kernel + cross-validation harness on tiny model, document specific blocking issues, end Phase 2 with this scope
- M2.1 done but M2.2 path unclear → ship M2.1, defer M2.2 to next sprint; portability claim still holds (RotaryQuant runs end-to-end in Mojo, just not the MoE flavor yet)

Phase 2 has a soft kill switch at the M2.1/M2.2 boundary; user gets a checkpoint review there.

## Phase 2 exit criteria

- M2.1: dense model end-to-end in Mojo, cross-validated against MLX, on Apple Silicon
- M2.2: MoE model end-to-end in Mojo, cross-validated, on Apple Silicon (or scoped to next sprint with explicit reason)
- M2.3: Portability claim documented with empirical + architectural framing
- Reproduction instructions tested from a fresh checkout
- Phase 3 (Codex review) unblocked

---

# Phase 3 — Codex full-work review (1-2 days, gate before any external updates)

The whole program — Phase 0 cleanup + Phase 1 speed work + Phase 2 Mojo work — gets an independent adversarial review by Codex via `delegate-to-codex` (per the global delegation pattern in `~/.claude/CLAUDE.md`) before any update lands in `docs/RotaryQuant_paper.md` or the GitHub blog.

## 3.1 Scope

Codex reviews the program as shipped, not the spec. The brief:

1. **Cross-program coherence** — does what shipped actually match this spec? Where it diverged, is the divergence justified and documented?
2. **Phase 0 correctness**
   - Worktree triage executed without losing genuine work (compare against pre-triage SHA log)
   - δ memo verdict is grounded in cited evidence
   - Invariance harness covers all three model configs and is genuinely bit-exact (not just "passes today by accident")
3. **Phase 1 correctness**
   - 1-α profiler measures what it claims to measure (no off-by-one in stall attribution; no double-counting)
   - Prefetcher correctness: tokens identical to no-prefetch path under varying look-ahead depth + concurrent load
   - Eviction policy doesn't introduce thrash under the documented stress test
   - Tok/s claims have variance bands, not single runs
   - Floor-case profiler is genuinely shippable as a standalone tool if 1-α didn't hit minimum
4. **Phase 2 correctness**
   - File-based interop contract is what was actually built (no silent drift to a different mechanism)
   - Staged-buffer pattern is in place at every Python ↔ Mojo boundary, not just where convenient
   - Cross-validation harness is honest — drift analysis is real per-layer/per-head data, not a hand-wave
   - Portability writeup doesn't overclaim (no "works on NVIDIA" without the test data to back it)
5. **Documentation review** — proposed paper + blog updates accurately reflect what shipped, with no unsupported claims

## 3.2 Process

1. Generate a review packet: spec + diff vs main + test outputs + benchmark JSONs + proposed paper/blog updates
2. Invoke `delegate-to-codex "<full-work review prompt>" --context <packet files>`
3. Codex returns review with severity-tagged findings (`blocker` / `concern` / `suggestion`)
4. **Blockers** must be resolved before paper/blog updates ship
5. **Concerns** are addressed or explicitly accepted with rationale committed to repo
6. **Suggestions** logged to a follow-up backlog

## 3.3 Acceptance

- Codex review committed to `docs/superpowers/reviews/2026-XX-XX-codex-program-review.md`
- All `blocker` findings resolved (resolution diff + commit referenced in review file)
- All `concern` findings either resolved or accepted with reason
- Sign-off line added: "Codex review passed at SHA <X>; paper/blog updates cleared to ship"

## 3.4 Failure handling

If Codex returns a blocker that can't be resolved within the 1-2 day window, this phase pauses and escalates:
- Block the paper/blog update
- Open an issue capturing the blocker
- Triage with the user: fix-and-retry, scope-cut-and-document, or accept-with-explicit-caveat in publication

## Phase 3 exit criteria

- Codex review file committed
- Blockers resolved; concerns resolved or accepted
- `docs/RotaryQuant_paper.md` updated with Phase 1 + Phase 2 results
- GitHub blog post drafted and committed
- Program closed

---

## Cross-cutting concerns

### Testing standard across phases

- Every benchmark or claim must have a reproducible script in `scripts/` or `mojo-bench/`
- Quality claims use `scripts/eval_quality_gate.py` (PPL drift + 5-prompt corpus)
- Speed claims report mean ± stdev over ≥3 runs (use existing `scripts/run_variance_study.sh` pattern)
- Cross-validation against MLX uses `validate/compare_logits.py` (Phase 2)

### Logging and artifacts

- Per-phase artifact directory: `artifacts/phase-{0,1,2}/` for benchmark JSONs, profiler output, plots
- All commit messages reference phase + milestone (e.g., `phase-1/1-α: profiler initial implementation`)
- δ memo, prior-spec status updates, triage recommendations all committed before destructive actions

### Delegation

- Phase 0 cleanup: solo (no delegation needed; mostly git operations + small Python edits)
- Phase 1-α: profiler implementation candidate for Codex via `delegate-to-codex`; main scheduler stays solo (architecture-sensitive)
- Phase 1-γ: solo (configuration + sweep orchestration)
- Phase 2: Mojo work is novel per `delegate-to-council` candidacy criteria — Gemini drafts kernel ports, Ollama critiques, Codex review, ARB final. Cross-validation harness solo.
- Phase 3: Codex full-work review is the gate — independent of any code Codex helped author during Phase 1/2 (a fresh review pass on the shipped state, not a confirmation of its own contributions)

---

## Risks (program-level)

| Risk | Mitigation |
|------|-----------|
| Phase 1 floor outcome (no meaningful tok/s improvement) | Floor case still ships profiler as standalone contribution; honest paper section |
| Phase 2 M2.2 slip (MoE in Mojo too hard) | M2.1/M2.2 boundary checkpoint; honest scope cut |
| Mojo + Modular Platform churn (compiler bugs, GPU detection regressions) | Pinned compiler version; no upgrades during phase |
| User context drift over 7-10 weeks | Each phase has clear exit criteria; checkpoints between phases |
| Discovery that prior assumptions were wrong (as happened with MTP/AttnRes pre-audit) | Phase 0 audit tasks built in; willingness to invalidate plan and rewrite (this spec is itself v2) |

## Out of scope (explicit)

- MTP / speculative decoding implementation (already exists; rejected for offload pathway)
- Block AttnRes Mojo kernel (already exists in Python; predictor measured No-go)
- Routing-aware draft model B-δ (deferred pending δ note audit verdict)
- Mojo prefill kernels (1-β, next sprint)
- Non-Apple GPU validation (no infrastructure; portability claim grounds on language guarantee + future work)
- Public RotaryQuant repo curation (no publication scaffolding sprint per user direction)
- Workshop paper (in flight separately, not part of this program)
- 131 tok/s as a target (unreachable on offload pathway; realistic ceiling ~25-40)
- arXiv preprint (out of scope; tracked elsewhere)
- MLA / Kimi-specific IsoQuant Mojo port (M2 targets dense + standard MoE; MLA is its own variant)

## Open questions for the implementation plan

These are deliberately unresolved at the spec level; implementation plan decides based on Phase 0 outputs:

- **1-α decision: mmap vs explicit threadpool prefetch?** Profiler output drives this.
- **1-γ model list:** Final 5+ models for coverage sweep — selected based on baseline availability and HF stability.
- **2-α layout: kernels in `mojo-bench/rotaryquant/` flat vs nested by component?** Decided once first kernels port.

(The Python ↔ Mojo interop decision was promoted out of open questions — see §2.0; default is file-based handoff, locked before any kernel work.)

---

## Architecture diagram (program level)

```
                ┌────────────────────────────────────────┐
                │  Phase 0  Cleanup (1 wk)               │
                │  ├─ 0.1 Mojo bench merge (re-run DW)   │
                │  ├─ 0.2 Abandoned worktrees deleted    │
                │  ├─ 0.3 Dirty triage (user-ratified)   │
                │  ├─ 0.4 δ note audit                   │
                │  └─ 0.5 Mark prior specs DONE          │
                └─────────────────┬──────────────────────┘
                                  │ clean foundation
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │  Phase 1   SPEED   (Objective 1)   2-3 wk                   │
   │  ├─ 1-α  Async expert prefetch (PRIMARY)                    │
   │  │       Target: 3.5 ms/tok I/O bottleneck                  │
   │  │       Bands: stretch +50% / min +20% / floor <10%        │
   │  ├─ 1-γ  Wider model coverage  (BACKGROUND)                 │
   │  │       ≥5 models with PPL drift data                      │
   │  └─ 1-β  Mojo prefill kernels  (DEFERRED next sprint)       │
   └─────────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
   ┌─────────────────────────────────────────────────────────────┐
   │  Phase 2   END-TO-END MOJO ROTARYQUANT             4-6 wk   │
   │  ├─ 2.0   Interop decision (file-based handoff default)     │
   │  ├─ M2.1  Dense (Llama 3.2 3B) end-to-end in Mojo           │
   │  │        IsoQuant + WHT + SO(4) + KV cache + attention     │
   │  │        Cross-validate ≥98% target / ≥95% acceptable      │
   │  ├─ M2.2  MoE (Qwen3.6-35B-A3B) end-to-end in Mojo          │
   │  │        + MoE separation; ≥95% top-1; checkpoint here     │
   │  └─ M2.3  Portability writeup                                │
   │           Empirical (Apple) + architectural (Mojo language) │
   └─────────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
   ┌─────────────────────────────────────────────────────────────┐
   │  Phase 3   CODEX FULL-WORK REVIEW                  1-2 days │
   │  Independent adversarial review (delegate-to-codex)         │
   │  Gates paper + blog updates                                 │
   │  Blockers must resolve; concerns resolved or accepted       │
   └─────────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
                    Paper + blog updates ship.
                    Program closed.
```

Total: **~7-10.5 weeks serial, no hard deadline.**
