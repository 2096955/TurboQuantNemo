# Layer-Aware Quantization for MoE Expert Offloading

## Overview

Layer-aware quantization assigns different bit-widths to MoE expert projections based on their position in the network. Edge layers (near input/output) use higher precision; middle layers use lower precision. This is informed by kurtosis measurements showing edge layers have more outlier-sensitive weight distributions.

## Implementation

### Recipe format

JSON recipes in `mlx-lm/mlx_lm/recipes/` define per-layer bit-width bands:

```json
{
  "schema_version": 1,
  "model_family": "qwen3_moe",
  "layer_count": 48,
  "bands": [
    {"start": 0, "end": 4, "routed_bits": 4},
    {"start": 5, "end": 8, "routed_bits": 3},
    {"start": 9, "end": 38, "routed_bits": 2},
    {"start": 39, "end": 42, "routed_bits": 3},
    {"start": 43, "end": 47, "routed_bits": 4}
  ],
  "dense": {"bits": 4, "group_size": 64}
}
```

### CLI usage

```bash
# Layer-aware with recipe
python -m mlx_lm convert --hf-path <model> --mlx-path <output> \
    -q --q-bits 4 --q-group-size 64 --expert-recipe qwen3_moe_layer_bands_v1

# Uniform mixed-expert bits
python -m mlx_lm convert --hf-path <model> --mlx-path <output> \
    -q --q-bits 4 --q-group-size 64 --mixed-expert-bits 2

# APEX-style (shared Q8_0 + edge/middle gradient)
python -m mlx_lm convert --hf-path <model> --mlx-path <output> \
    -q --q-bits 4 --q-group-size 64 --expert-recipe apex
```

### Key files

- `mlx-lm/mlx_lm/convert.py` — `_build_apex_expert_quant_predicate()`, `_compile_routed_band_schedule()`
- `mlx-lm/mlx_lm/recipes/qwen3_moe_layer_bands_v1.json` — Qwen3 48-layer recipe
- `mlx-lm/tests/test_apex_quant_predicate.py` — predicate validation tests

### APEX findings (partially superseded)

From the APEX paper (mudler/apex-quant):

- **KEEP:** Shared expert Q8_0 (structural kurtosis gap — always active, high sensitivity)
- **KEEP:** Stock quantization algorithms are optimal (don't modify kernels)
- **DROP:** Static 3-tier layer gradient — superseded by AttnRes dynamic block importance signal

Static layer bands remain as cold-start fallback only. The AttnRes predictor provides per-input, per-token importance that static bands cannot.

## Relationship to expert offload

Layer-aware quantization is orthogonal to expert offloading. The quant predicate runs at convert time to set per-layer bit-widths. The offload manager loads these pre-quantized experts on demand at inference time. Both systems use the same `SwitchLinear` / `QuantizedSwitchLinear` module hierarchy.
