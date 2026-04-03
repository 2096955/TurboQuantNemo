# TurboQuant KV cache (optional)

TurboQuant compresses attention KV using a rotation + Lloyd-Max quantizer + QJL residual
signs. This fork wires it for **hybrid** models that expose `layer.is_linear` (full
attention vs linear/SSM blocks).

## Usage

```bash
python -m mlx_lm.generate --model <path> --prompt "Hi" --kv-cache-type turboquant
mlx_lm.server --model <path> --kv-cache-type turboquant
```

Programmatic: pass `kv_cache_type="turboquant"` to `generate_step` / `stream_generate`.

## Layout

- Implementation: `mlx_lm/models/mlx_turboquant.py`
- Lloyd–Max tables: `mlx_lm/models/turboquant_codebooks/dim_128_{1,2,3,4}bit.npz`
- Attention bypass: `mlx_lm/models/qwen3_next.py` (`Qwen3NextAttention`; also used by Qwen3.5 text)

## Limits

- Batch server paths that merge `BatchKVCache` do not support `TurboQuantKVCache` yet.
- Speculative decoding: TurboQuant applies to the main model cache; draft model stays default.

## Benchmark

`scripts/turboquant_benchmark.py` compares prefill time and `sum(c.nbytes)` for
`default` vs `turboquant`.
