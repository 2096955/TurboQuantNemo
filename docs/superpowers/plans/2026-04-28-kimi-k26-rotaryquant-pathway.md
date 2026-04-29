# Kimi K2.6 RotaryQuant Pathway Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`
> for implementation slices or `superpowers:executing-plans` for serial execution. Do not
> mark a task complete from code inspection alone; completion requires the named artifact,
> test output, or benchmark JSON.

**Date:** 2026-04-28  
**Status:** Phases 0-2 complete (expert offload wired + load/decode smoke passed). Phase 3 next.  
**Target checkpoint:** `/Volumes/Samsung9904tb/Kimi-K2.6`  
**Target hardware:** M4 Max, 128 GB unified memory  
**Goal:** Run Kimi K2.6 on M4 Max 128 GB. Text non-expert weights (attention BF16, shared experts
BF16, embeddings, router gates, norms) consume ~23.4 GB resident. Shared experts use `moe_intermediate_size` (2048),
not `intermediate_size` (18432). Routed experts at 4-bit total ~570.8 GB — ~3,830 of 23,040
instances fit in remaining RAM (~16.6%). Expert offload is therefore a runtime necessity.
RotaryQuant/IsoQuant on MLA latent targets the KV cache axis. Primary objective: get Kimi
K2.6 loading, decoding, and offloading correctly at interactive speed, then profile and
optimize.

---

## Ground Truth From Checkpoint Census

Local checkpoint facts already verified:

| Field | Value |
|---|---|
| Path | `/Volumes/Samsung9904tb/Kimi-K2.6` |
| Size | ~554 GB |
| Shards | `model-00001-of-000064.safetensors` ... `model-00064-of-000064.safetensors` |
| Top-level `model_type` | `kimi_k25` |
| Text `model_type` | `kimi_k2` |
| Architecture | `KimiK25ForConditionalGeneration` |
| Text architecture | `DeepseekV3ForCausalLM` |
| Layers | 61 |
| Routed experts | 384 |
| Experts/token | 8 |
| `kv_lora_rank` | 512 |
| `qk_rope_head_dim` | 64 |
| `qk_nope_head_dim` | 128 |
| `v_head_dim` | 128 |
| Weight map count | 208,550 tensors |

Important correction to the older placeholder: K2.6's config and current `DeepseekV3Attention`
code expose `kv_a_proj_with_mqa` as `kv_lora_rank + qk_rope_head_dim` = `512 + 64`.
That suggests the positional RoPE part is already the separate `k_pe` tensor, while
`kv_latent` is the 512-D non-RoPE MLA latent. Do not assume the old `448 + 64` split is
correct until it is verified against Kimi's released modeling code and a numerical test.

---

## Non-Negotiable Constraints

1. **Do not generic-enable IsoQuant on Kimi MLA.** Kimi caches `(kv_latent, k_pe)`, not
   ordinary `(K, V)`. RoPE state must remain raw.
2. **Expert offload comes first.** A 554 GB checkpoint cannot be validated on the target
   memory path if routed experts are resident-loaded or stacked.
3. **No full-run claims without artifacts.** Kimi support is not "done" until quality,
   memory, and throughput JSONs exist.
4. **No ALU/BW claims from CPU fences.** Use Xcode Instruments before classifying Kimi
   fused kernels as ALU-bound or bandwidth-bound.
5. **Treat Branch C as guidance, not a transplant.** Branch C proves `compress_batch` +
   `pack_indices_3bit` dominate Qwen3.6 NPT=8. Kimi's MLA latent is 512-D, so the fused
   path likely needs NPT=16 or a separate MLA-specific kernel.

---

## Phase 0 — Checkpoint And Loader Census

**Purpose:** Persist the K2.6 facts above and prove we can inspect the checkpoint without
accidentally loading routed experts.

**Files:**
- Create: `results/kimi_k26_checkpoint_census.json`
- Create: `docs/KIMI_K26_FULL_STACK.md`
- Read-only: `/Volumes/Samsung9904tb/Kimi-K2.6/config.json`
- Read-only: `/Volumes/Samsung9904tb/Kimi-K2.6/model.safetensors.index.json`

- [x] **Step 1: Write census script or one-shot command**

Capture:
- model types and architectures
- MLA dimensions
- expert counts
- shard count and total indexed tensor size
- expert-key count by projection
- non-expert tensor count

- [x] **Step 2: Save census artifact**

Output: `results/kimi_k26_checkpoint_census.json`

- [x] **Step 3: Add Kimi runbook stub**

Create `docs/KIMI_K26_FULL_STACK.md` with:
- checkpoint path
- current blockers
- exact command templates
- artifact list

**Gate:** census JSON exists and confirms `model_type=kimi_k25`, text `model_type=kimi_k2`,
61 layers, 384 experts, top-k 8, and 60 routed-expert layers. **PASSED** (2026-04-28).

---

## Phase 1 — Kimi Expert Offload Wiring

**Purpose:** Make Kimi K2.6 load through the existing offload pathway without resident-loading
all routed experts.

**Files:**
- Modify: `mlx-lm/mlx_lm/expert_offload.py`
- Modify: `mlx-lm/mlx_lm/expert_weight_loader.py`
- Modify: `mlx-lm/mlx_lm/utils.py`
- Test: `mlx-lm/tests/test_kimi_k26_offload.py`

### Task 1.1 — Add Kimi expert-key parsing

- [x] **Step 1: Write failing parser tests**

Test these keys:

```python
language_model.model.layers.1.mlp.experts.0.gate_proj.weight_packed
language_model.model.layers.1.mlp.experts.0.gate_proj.weight_scale
language_model.model.layers.1.mlp.experts.0.gate_proj.weight_shape
language_model.model.layers.60.mlp.experts.383.down_proj.weight_packed
```

Expected parser result: `(layer_idx, expert_id, proj, suffix)` for `kimi_k25` and `kimi_k2`.

- [x] **Step 2: Add Kimi model types to offload sets**

Add `kimi_k25` and `kimi_k2` to:
- `EXPERT_OFFLOAD_MODEL_TYPES`
- `MOE_MODEL_TYPES`
- `_offload_supported_types` in `utils.py`

- [x] **Step 3: Reuse or alias Qwen-style expert table builder**

Kimi K2.6 routed experts use the same `mlp.experts.{E}.{gate,up,down}_proj.*` shape as
Qwen-style keys, with the `language_model.` prefix. Either:
- make `_QWEN3_EXPERT_KEY_RE` a generic switch-GLU expert regex, or
- add `_KIMI_EXPERT_KEY_RE` and `build_kimi_expert_key_table()`.

**Gate:** test proves table has 60 × 384 entries for the real K2.6 weight map. **PASSED** (2026-04-28).

### Task 1.2 — Exclude Kimi experts from non-expert loading

- [x] **Step 1: Write failing test for `load_non_expert_weights` selection**

Use a synthetic weight map with one Kimi expert key and one non-expert key. Assert only the
non-expert key is selected.

- [x] **Step 2: Remove hard-coded architecture list or add Kimi**

`expert_weight_loader.py` currently hard-codes known model types. Replace this with a helper
that checks all supported expert regexes, or explicitly include Kimi.

**Gate:** no Kimi `mlp.experts.*` key is included in non-expert load selection. **PASSED** (2026-04-28).

### Task 1.3 — Swap Kimi `SwitchGLU` to offload modules

- [x] **Step 1: Write failing attach/swap test**

Instantiate the Kimi/DeepSeek-V3 model from config only. With `expert_offload=True`, assert
each routed layer's `mlp.switch_mlp` becomes `OffloadQuantizedSwitchGLU`.

- [x] **Step 2: Generalize `_swap_qwen3_offload_modules`**

Kimi K2.6 uses DeepSeek-V3-style `DeepseekV3MoE.switch_mlp`, so the Qwen3 swap logic is
structurally reusable. Rename to a generic helper or call it for `kimi_k25` / `kimi_k2`.

- [x] **Step 3: Handle nested quantization config**

Kimi stores `quantization_config` under `text_config`, not top-level config. Added hoisting
in `utils.py` to lift `text_config.quantization_config` to top-level before quantization
dispatch.

**Gate:** Kimi offload modules are swapped before weights load, and no routed expert tensors
are required by `model.load_weights(..., strict=False)`. **PASSED** (2026-04-28).

### Task 1.4 — Attach manager to Kimi layers

- [x] **Step 1: Add attach test**

After attach, every routed Kimi layer has manager references on gate/up/down offload linears.

- [x] **Step 2: Route Kimi through Qwen-style attach helper**

`attach_expert_offload_manager()` routes `kimi_k25`/`kimi_k2` through `_attach_qwen3_moe()`.
Loader dispatch resolves Kimi's `model.model` property chain via `getattr(model, "model", model)`.

**Gate:** attach count is 60 routed layers. **PASSED** (2026-04-28).

### Task 1.5 — Compressed-tensors key aliasing for on-demand expert loads

Kimi's raw safetensor keys are `weight_packed`, `weight_scale`, `weight_shape`. The full-load
path in `deepseek_v3.py:398-410` remaps these to `weight` (uint32 view of packed), `scales`,
and `biases` (synthetic: `-8 * scale`). The offload path reads raw safetensors directly via
`ExpertOffloadManager._load_expert_pair_tensors` (expert_offload.py:519+), which expects the aliased
`{proj}_weight`, `{proj}_scales`, `{proj}_biases` keys in the expert spec.

- [x] **Step 1: Write failing test for compressed-tensors expert load**

Load one expert from a real K2.6 shard. Assert the loaded tensors have `weight`, `scales`, and
`biases` keys matching the shapes the `OffloadQuantizedSwitchGLU` forward pass expects.

- [x] **Step 2: Add on-demand aliasing in the expert load path**

Compressed-tensors aliasing: `weight_packed`→`.view(uint32)`, `biases=-8*scale`.

- [x] **Step 3: Add Kimi expert key regex**

`_KIMI_EXPERT_KEY_RE` and `build_kimi_expert_key_table()` added with raw suffix capture;
load path remaps at load time.

**Gate:** a single Kimi expert round-trips through offload load → `OffloadQuantizedSwitchGLU`
forward → produces a finite output tensor. **PASSED** (2026-04-28).

---

## Phase 2 — Kimi Load Smoke Without RotaryQuant

**Purpose:** Prove the 554 GB checkpoint can be loaded through expert offload and can decode
a tiny prompt before touching MLA compression.

**Files:**
- Create: `results/kimi_k26_offload_load_smoke.json`
- Create: `results/kimi_k26_default_decode_smoke.json`

- [x] **Step 1: Load-only smoke**

Ran with `--max-resident-experts 32 --max-tokens 1`. Load: 8.4s, peak RSS: 22.66 GB,
expert table: 23,040 entries, attached layers: 60/61.
Artifact: `results/kimi_k26_offload_load_smoke.json`.

- [x] **Step 2: Decode smoke**

Ran two prompts (16 + 32 tokens) with default MLA cache. Coherent text at 0.60-0.64 tok/s,
peak RSS 22.88 GB.
Artifact: `results/kimi_k26_default_decode_smoke.json`.

**Gate:** coherent decode completes without OOM, with expert offload attached and routed
expert resident count bounded by `max_resident_experts`. **PASSED** (2026-04-28).

---

## Phase 3 — Kimi MLA Cache Design

**Purpose:** Implement the correct cache abstraction for Kimi MLA before any fused speed work.

**Files:**
- Create/modify: `mlx-lm/mlx_lm/models/kimi_mla_isoquant_dkv.py`
- Modify: `mlx-lm/mlx_lm/models/deepseek_v3.py` or a Kimi-specific wrapper
- Test: `mlx-lm/tests/test_kimi_mla_isoquant_dkv.py`

### Task 3.1 — Verify actual MLA split

- [ ] **Step 1: Compare current MLX code to Kimi local modeling files**

Read:
- `/Volumes/Samsung9904tb/Kimi-K2.6/configuration_deepseek.py`
- `/Volumes/Samsung9904tb/Kimi-K2.6/configuration_kimi_k25.py`
- any local modeling file if present

Confirm whether the compressible latent is:
- all `kv_lora_rank=512`, with RoPE in separate `k_pe=64`, or
- old placeholder `448 content + 64 RoPE` inside a 512-D latent.

- [ ] **Step 2: Update placeholder constants**

The module must not keep stale `448 + 64` assumptions if K2.6 uses `512 + 64`.

**Gate:** test encodes the verified split from real K2.6 config.

### Task 3.2 — Build `KimiMLAIsoQuantCache`

The cache should store:
- compressed `kv_latent`
- raw `k_pe`

It should not quantize or rotate `k_pe`.

- [ ] **Step 1: Synthetic update/fetch test**

Given synthetic `(kv_latent, k_pe)` inputs, assert:
- latent is compressed/decompressed or stored by the selected mode
- `k_pe` round-trips exactly
- offsets match default cache behavior

- [ ] **Step 2: Deferred prefill boundary**

Implement FP16 latent accumulation during prefill and bulk compression at decode boundary,
matching existing IsoQuant deferred prefill semantics.

**Gate:** default MLA attention and Kimi compressed-MLA attention match within tolerance on
synthetic tensors for prefill + decode.

---

## Phase 4 — Correctness First: Unfused Kimi RotaryQuant

**Purpose:** Get Kimi K2.6 correctness with RotaryQuant/IsoQuant before adding NPT=16 fused
kernels.

**Files:**
- Modify: Kimi/DeepSeek-V3 attention path
- Create: `results/kimi_k26_mla_isoquant_correctness.json`

- [ ] **Step 1: Wire cache selection**

`make_prompt_cache(..., kv_cache_type="isoquant")` should return Kimi MLA-specific cache
objects for Kimi/DeepSeek-V3 MLA layers.

- [ ] **Step 2: Use reconstruction path first**

For first correctness pass, reconstruct compressed `kv_latent` and run the existing MLA
attention math. This is expected to be slower; it proves the representation is correct.

- [ ] **Step 3: Run synthetic and small real-model quality smoke**

Run default MLA vs Kimi-MLA-IsoQuant on a short prompt.

**Gate:** no RoPE corruption, finite logits, deterministic smoke output, and no quality
collapse on a 5-10 prompt mini-gate.

---

## Phase 5 — Kimi Decode Profiling

**Purpose:** Measure where Kimi decode time goes before assuming Qwen Branch C applies.

**Files:**
- Create: `scripts/profile_kimi_k26_decode.py`
- Create: `artifacts/kimi_k26_profiling/kimi_k26_decode_profile.json`
- Create: `artifacts/kimi_k26_profiling/profiling_memo.md`

- [ ] **Step 1: Profile default offload MLA**

Measure:
- expert load / ensure_loaded
- routed MLP compute
- MLA attention
- cache update
- `kv_latent` reconstruction, if using compressed path

- [ ] **Step 2: Profile Kimi MLA IsoQuant reconstruct path**

Use fresh cache per phase and clear component timings after warmup, following the Branch C
fixes.

- [ ] **Step 3: Decide if fused cache work is on critical path**

If Kimi's decode remains dominated by expert I/O, prioritize offload scheduling. If MLA
cache work is material, proceed to Phase 6.

**Gate:** profiling JSON has no negative residuals, includes call counts, and records
default vs compressed latency.

---

## Phase 6 — Fused Kimi MLA Attention

**Purpose:** Speed compressed MLA attention without reconstructing `kv_latent`.

**Architectural note:** Kimi latent attention uses `head_dim=kv_lora_rank=512` and effectively
one latent KV head broadcast across query heads. Existing Branch C fused kernels cover NPT=8
for `head_dim=256`; Kimi likely requires NPT=16 support or a dedicated MLA fused kernel.

- [ ] **Step 1: Add NPT=16 synthetic tests**

Extend fused kernel tests to `head_dim=512`, repeats `64`, and Kimi-style additive `pe_scores`
mask.

- [ ] **Step 2: Implement Kimi fused latent attention**

Compute attention over compressed `kv_latent`, add raw RoPE positional scores, and emit latent
output for `unembed_out`.

- [ ] **Step 3: Validate against unfused reference**

Compare fused vs reconstructed reference over T ∈ `{128, 512, 4096}`.

**Gate:** max error within tolerance, no fallback, and `tiled_kernel_observed=true`.

---

## Phase 7 — Apply Branch C Write-Path Lessons

**Purpose:** Optimize the likely Kimi compressed-cache bottleneck after Phase 5/6 identify it.

Branch C result on Qwen3.6:
- `compress_batch`: ~13 ms/step
- `pack_indices_3bit`: ~5 ms/step
- Metal attention kernel: not dominant

Kimi may show a similar write-path bottleneck, but Kimi's latent is 512-D, so measure first.

- [ ] **Step 1: Profile Kimi compress + pack separately**
- [ ] **Step 2: If dominant, fuse latent `_compress_batch` into Metal**
- [ ] **Step 3: Fold `pack_indices_3bit` into the compression kernel**
- [ ] **Step 4: Re-run Kimi end-to-end decode matrix**

**Gate:** committed before/after artifacts show real tok/s gain and no quality regression.

---

## Phase 8 — Full Pathway Evidence

**Files:**
- Create: `results/kimi_k26_default_quality.json`
- Create: `results/kimi_k26_rotaryquant_quality.json`
- Create: `results/kimi_k26_default_memory.json`
- Create: `results/kimi_k26_rotaryquant_memory.json`
- Create: `results/kimi_k26_pathway_benchmark.json`
- Update: `docs/KIMI_K26_FULL_STACK.md`
- Update: `docs/PATHWAY_PROVEN_CHECKLIST.md` only if gates pass

- [ ] **Step 1: Quality gate**

Run the same 12-prompt gate used for Qwen/Gemma/Nemotron. Kimi RotaryQuant must not regress
relative to default beyond the accepted tolerance.

- [ ] **Step 2: Memory gate**

Record peak RSS, swap, resident expert count, shard cache size, and KV cache mode.

- [ ] **Step 3: Throughput gate**

Run repeatable decode benchmark at representative context lengths.

- [ ] **Step 4: Soak gate**

Run long enough to catch shard-cache leaks, expert-residency drift, and delayed Metal failures.

**Gate:** only update pathway-proven docs when JSON artifacts exist and pass the criteria.

---

## Commit Strategy

1. `test(kimi): capture K2.6 checkpoint census and expert key assumptions`
2. `feat(kimi): add expert offload support for kimi_k25/kimi_k2`
3. `test(kimi): load K2.6 through offload path without resident experts`
4. `feat(kimi): add MLA-specific IsoQuant cache split`
5. `evidence(kimi): validate K2.6 MLA IsoQuant correctness`
6. `feat(kimi): fused MLA latent attention for compressed cache`
7. `evidence(kimi): K2.6 pathway benchmark and quality gate`

Keep docs/evidence commits separate from runtime commits where possible.

---

## Current Known Blockers

- ~~Existing expert offload does not list `kimi_k25` or `kimi_k2`.~~ **Resolved** in Phase 1.
- `kimi_mla_isoquant_dkv.py`, its test, and `check_kimi_disk_prereq.sh` are currently
  untracked; decide whether to adopt or replace them in Phase 3.
- `mlx-lm/mlx_lm/models/kimi_linear.py` has uncommitted formatting/predictor edits; do not
  overwrite them blindly.
- Existing generic IsoQuant cache is not Kimi-MLA-safe (Phase 3).
- Existing fused NPT=8 path does not directly cover Kimi's 512-D MLA latent (Phase 6).
