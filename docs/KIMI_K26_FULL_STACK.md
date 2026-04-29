# Kimi K2.6 Full Stack Runbook

**Checkpoint:** `/Volumes/Samsung9904tb/Kimi-K2.6`
**Target hardware:** M4 Max, 128 GB unified memory
**Status:** Phases 0-6 throughput-validated. Phase 6: NPT=16 kernel correct (13/13 synthetic) + throughput A/B shows -1318 ms (56% faster). MLA logit parity test on real weights not yet run.

---

## Architecture Summary

| Field | Value |
| --- | --- |
| Top-level model_type | `kimi_k25` |
| Text model_type | `kimi_k2` |
| Text architecture | `DeepseekV3ForCausalLM` |
| Layers | 61 (layer 0 dense, layers 1-60 MoE) |
| Routed experts | 384 per layer, top-8 |
| Shared experts | 1 per MoE layer |
| Hidden size | 7168 |
| MoE intermediate | 2048 |
| Dense intermediate | 18432 |
| Shared expert intermediate | 2048 (`moe_intermediate_size * n_shared_experts`) |
| Attention | MLA (Multi-head Latent Attention) |
| kv_lora_rank | 512 |
| qk_rope_head_dim | 64 |
| qk_nope_head_dim | 128 |
| v_head_dim | 128 |
| q_lora_rank | 1536 |
| Heads | 64 query, 64 KV |
| Vocab | 163,840 |
| Max context | 262,144 (YaRN RoPE, factor 64) |
| Quantization | 4-bit INT4 group-32 on routed experts only |
| Unquantized | Attention, shared experts, dense MLP, embeddings (BF16) |

## Memory Budget (128 GB M4 Max)

| Component | Size |
| --- | --- |
| Attention weights (BF16) | 12.3 GB |
| Shared experts (BF16, moe_intermediate=2048) | 5.3 GB |
| Dense MLP layer 0 (BF16) | 0.8 GB |
| Embeddings (BF16) | 4.7 GB |
| **Text non-expert resident total** | **~23.4 GB** |
| Non-routed safetensors payload | ~24.4 GB before sanitizer drops vision/projector tensors |
| Routed experts (4-bit, all 23,040) | 570.8 GB |
| Available for expert cache | ~94.6 GB |
| Max resident experts | ~3,816 (~16.6%) |
| Expert offload required | **Yes** |

Each expert instance is ~24.8 MB at 4-bit with group-32 scales.
Note: shared experts use `moe_intermediate_size` (2048), not `intermediate_size` (18432).

## Phase 2 Smoke Results

Artifacts:
- `results/kimi_k26_offload_load_smoke.json`
- `results/kimi_k26_default_decode_smoke.json`

Measured on 2026-04-28:
- Load smoke: 21.84 GB RSS after load, 22.66 GB peak after 1-token decode.
- Decode smoke: 16-token and 32-token prompts completed at 0.60-0.64 tok/s, 22.88 GB peak RSS.
- Expert offload: 60/61 layers attached; layer 0 is dense.
- Manager: `max_resident_experts=32`, `max_cached_shards=1`, expert table size 23,040.

This is a load/decode smoke only. It does not validate Kimi MLA IsoQuant on the real
checkpoint, long-context behavior, full quality gates, repeatability, or pathway-proven status.

Programmatic callers must pass `model_config={"expert_offload": True, ...}`. The CLI
`--expert-offload` flag sets this automatically; omitting it follows the normal resident-load
path, which is not viable for this 554 GB checkpoint.

## Phase 5 Profiling Results (v2, 2026-04-29)

Artifact: `artifacts/kimi_k26_profiling/kimi_k26_decode_profile_v2.json`

Config: `max_resident_experts=2000`, 32-token prefill, 20 decode steps, 3 warmup.

| Metric | Default cache | IsoQuant cache |
| --- | --- | --- |
| Median step | 820.6 ms (1.22 tok/s) | 2648.9 ms (0.38 tok/s) |
| Step range | 426-5317 ms | 1750-7543 ms |
| Std | 1451.6 ms | 1905.9 ms |
| Attention share | 315.9 ms (23.9%) | 932.2 ms (29.7%) |
| MLP/MoE block share | 1004.7 ms (76.1%) | 2201.7 ms (70.3%) |
| Expert loads | 1042 | 964 |
| Avg load time | 4.41 ms | 4.76 ms |
| Expert I/O per step | 230.0 ms (22.9% of MLP/MoE) | 229.7 ms (10.4% of MLP/MoE) |
| Decode hit rate | 89.1% | 90.0% |

Key findings:

- Expert I/O is 22.9% of the MLP/MoE block (default cache), **not** 90.5% as previously overclaimed.
  The prior Phase 5 profile timed the whole MLP/MoE block; this rerun separates expert I/O via
  `ExpertOffloadManager.stats_summary()` counters.
- IsoQuant adds ~616 ms to attention per step (decompression cost for D=512).
- Both configs have variance exceeding the median — results are directional, not precise.
- Expert I/O cost is cache-type-independent (~230 ms/step in both).

## Phase 6 A/B: Fused vs Unfused (2026-04-29)

Artifact: `artifacts/kimi_k26_profiling/kimi_k26_fused_ab_v2.json`

| Metric | Unfused (NPT16_FUSED=0) | Fused (NPT16_FUSED=1) |
| --- | --- | --- |
| Median step | 3686.3 ms (0.27 tok/s) | 2368.6 ms (0.42 tok/s) |
| Range | 1563-7886 ms | 1709-6498 ms |
| Std | 2102.5 ms | 1591.5 ms |
| Fused latent layers | 59 | 59 |

Delta: **-1317.7 ms (56% faster with fused path).**

The v1 A/B (prior run) showed fused +304 ms *slower* due to a bug in `fused_latent_attention`:
`_packed_keys_cache` and `_packed_values_cache` were nulled every call, forcing O(T) full
`pack_indices_3bit` rebuild × 59 layers × every decode step, and packing K and V separately
despite MLA's K=V identity. Fix: maintain packed cache on the outer `KimiMLAIsoQuantCache`,
extend incrementally when a new token appends, share one buffer for K and V.

## Remaining Blockers

1. ~~Kimi MLA IsoQuant has synthetic cache tests only~~ — Phase 4 correctness passed (2026-04-29).
2. NPT=16 kernel for D=512 is synthetic-correct (13/13 tests) and throughput A/B shows -1318 ms (56% faster). MLA logit parity on real weights not yet validated. Gate: `ISOQUANT_USE_NPT16_FUSED=1`.
3. Tokenizer emitted TikToken fallback / regex warnings in the smoke runs; decode completed, but
   tokenizer handling should be pinned before quality gates.
4. DeepseekV3 attention auto-routes to fused path when cache supports it, but fused path has no real-weight logit parity validation yet. Synthetic shared-K=V tests verify the kernel handles MLA's K=V pattern, not that the absorbed-weight trick preserves logits end-to-end.
5. Phase 4 used the default `TURBOQUANT_SKIP_LAYERS=2`, so layers 0-1 used `KVCache`
   and layers 2-60 used `KimiMLAIsoQuantCache`.

## MLA Cache Constraint (DKV)

Kimi MLA caches two tensors per token:

- `kv_latent`: 512-D compressible content (kv_lora_rank)
- `k_pe`: 64-D positional RoPE state (qk_rope_head_dim)

IsoQuant/RotaryQuant must compress **only** `kv_latent`. The `k_pe` dimensions carry RoPE phase
and must never be rotated or quantized. This is the DKV constraint from the paper (Section 8).

## Command Templates

### Phase 2: Load smoke rerun

```bash
PYTHONPATH=mlx-lm python3 -m mlx_lm.generate \
  --model /Volumes/Samsung9904tb/Kimi-K2.6 \
  --expert-offload \
  --max-resident-experts 32 \
  --max-tokens 1 \
  --prompt "Hello"
```

### Phase 4: Correctness with IsoQuant (after Phase 3 MLA cache)

```bash
PYTHONPATH=mlx-lm python3 -m mlx_lm.generate \
  --model /Volumes/Samsung9904tb/Kimi-K2.6 \
  --expert-offload \
  --max-resident-experts 2000 \
  --kv-cache-type isoquant \
  --max-tokens 64 \
  --prompt "Explain the concept of attention in transformers."
```

### Phase 8: Full benchmark

```bash
PYTHONPATH=mlx-lm python3 scripts/eval_quality_gate.py \
  --model /Volumes/Samsung9904tb/Kimi-K2.6 \
  --expert-offload \
  --max-resident-experts 2000 \
  --kv-cache-type isoquant
```

## Artifact Checklist

| Artifact | Phase | Status |
| --- | --- | --- |
| `results/kimi_k26_checkpoint_census.json` | 0 | Done |
| `results/kimi_k26_offload_load_smoke.json` | 2 | Done (load 8.4s, 22.66 GB RSS, 23040 experts, 60/61 layers) |
| `results/kimi_k26_default_decode_smoke.json` | 2 | Done (0.60-0.64 tok/s, decode completed, 22.88 GB peak) |
| `mlx-lm/tests/test_kimi_mla_isoquant_dkv.py` | 3 | Done (synthetic MLA cache, dispatch, save/load tests) |
| `results/kimi_k26_mla_isoquant_correctness.json` | 4 | Done (3/3 prompts pass, no quality collapse, reconstruction-path IsoQuant semantically equivalent to default MLA; default skip kept layers 0-1 uncompressed) |
| `artifacts/kimi_k26_profiling/kimi_k26_decode_profile.json` | 5 | v1 directional only (high variance, MLP/MoE block not decomposed) |
| `artifacts/kimi_k26_profiling/kimi_k26_decode_profile_v2.json` | 5 | Done (expert I/O isolated: 230 ms/step = 22.9% of MLP/MoE block; attention 23.9% default, 29.7% iso; still high variance: std > median) |
| `mlx-lm/mlx_lm/models/fused_kv_decode_npt16.py` | 6 | Kernel correct (13/13 synthetic tests) |
| `mlx-lm/tests/test_fused_npt16.py` | 6 | Synthetic only (shared-KV verifies kernel handles K=V, not that MLA absorbed-weight trick preserves logits on real weights) |
| `artifacts/kimi_k26_profiling/kimi_k26_fused_ab_v2.json` | 6 | Done (fused -1318ms faster, 0.42 vs 0.27 tok/s; v1 showed regression from O(N²) repack bug now fixed) |
| Phase 6 MLA logit parity test | 6 | Needed: default vs fused path on real Kimi weights, tight tolerance |
| `results/kimi_k26_pathway_benchmark.json` | 8 | Not started |

## Plan Reference

Full phased plan: `docs/superpowers/plans/2026-04-28-kimi-k26-rotaryquant-pathway.md`
