# Phase 0 — Evidence summary (attribution)

Source JSON (same run, ISOQUANT 3-bit, `ISOQUANT_VACCUM_TILE=128`):

- [kernel_attribution_4k.json](kernel_attribution_4k.json) — T_prefill 4096, 100 decode steps
- [kernel_attribution_8k.json](kernel_attribution_8k.json) — T_prefill 8192, 100 decode steps

Model: `Qwen3.6-35B-A3B-4bit` (baseline checkpoint path as recorded in JSON).

## Per-kernel per-step (ms) and % of total decode

### T = 4096 prefill (after decode-only segment)

| Component | per_step_ms | % of decode |
|-----------|------------:|------------:|
| `pack_indices_3bit` | 8.57 | 10.99 |
| `fused_qk_dot` | 4.45 | 5.70 |
| `fused_value_accum_tiled` | 4.80 | 6.14 |
| `_apply_inverse_rotation` | 3.63 | 4.65 |

- End-to-end decode (this run): **~78.06 ms/step** (~12.8 tok/s) — remainder is model compute and unattributed work.

### T = 8192 prefill

| Component | per_step_ms | % of decode |
|-----------|------------:|------------:|
| `pack_indices_3bit` | **11.81** | **16.95** |
| `fused_qk_dot` | 5.97 | 8.58 |
| `fused_value_accum_tiled` | 5.48 | 7.86 |
| `_apply_inverse_rotation` | 3.49 | 5.01 |

- End-to-end decode (this run): **~69.65 ms/step** (~14.4 tok/s).

## Phase 2 target (engineering)

At **8K context**, **pack** (`pack_indices_3bit`) is the **largest IsoQuant-attributable** line item among the four wrapped kernels — larger than tiled V-accum, QK, and inverse rotation individually. **Phase 2 incremental packed-cache append** is therefore the **primary** next step per measured attribution, independent of pre-plan ms forecasts.

## Relation to 2x2 matrix JSON

- The 12 `matrix_*.json` files report end-to-end tok/s across four cells. Under load they failed the 8K repeatability gate; use them as **coarse** comparators, not as proof of a single clean tok/s. Attribution JSON above is the **high-SNR** baseline for kernel-level work.
