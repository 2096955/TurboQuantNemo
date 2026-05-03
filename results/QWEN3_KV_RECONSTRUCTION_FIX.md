# Qwen3 MoE — TurboQuant / IsoQuant attention fix (2026-04-08)

## Issue

`qwen3_moe.Attention` passed only the **latest chunk** from `cache.update_and_fetch()` into `scaled_dot_product_attention`, same failure mode as Gemma before the `reconstruct_keys()` / `get_values()` fix.

Symptom: **0/N quality gate** with `--kv-cache-type turboquant` or `isoquant` (garbage / repetition).

## Fix

- `mlx_lm/models/qwen3_moe.py`: after `update_and_fetch`, if `isinstance(cache, TurboQuantKVCache)`, use `cache.reconstruct_keys()` and `cache.get_values()` and call SDPA with `cache=None` (matches `qwen3_next.py`).
- `mlx_lm/models/qwen2.py`: `isinstance(cache, TurboQuantKVCache)` so **`IsoQuantKVCache`** (subclass) is included (was `type(cache).__name__ == "TurboQuantKVCache"`).
- `mlx_lm/models/mlx_isoquant.py`: **`IsoQuantKVCache`** overrides `reconstruct_keys()` / `get_values()` so that during **deferred prefill** (FP16 chunks only, before `finalize_deferred_prefill`), attention still receives the **full** concatenated K/V tensor instead of hitting empty `compressed_keys["indices"]`.

## Tests

- `tests/test_qwen3_offload_module_selection.py::TestQwen3TurboQuantAttentionPath::test_attention_prefill_uses_reconstructed_keys_not_latest_chunk`

## Re-measurement

Re-run pinned JSON after this commit:

```bash
# Example — adjust model path
python scripts/eval_quality_gate.py --model <Qwen3-30B-A3B-4bit> --suite quick \
  --kv-cache-type isoquant --output-json results/qwen3_offload_isoquant_quality.json
```

Existing `results/qwen3_offload_*_quality.json` from before this fix are **invalid** for TurboQuant/IsoQuant claims.
