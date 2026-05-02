# Gemma 4-26B-A4B — full pathway stack (single session)

Combines: **expert offload** + **IsoQuant KV** (primary) + **deferred prefill**.

DedeKimi observer, predictor, and task cliques are optional enhancements — not required for the pathway-proven gate.

## Prerequisites

- Recommended MLX model dir: `gemma4-layer-aware/` (mixed 2/3/4-bit expert schedule).
- Fallback baseline: `gemma-4-26b-a4b-it-4bit/`.
- `mlx-lm` installed in editable mode: `cd mlx-lm && pip install -e .`

## One-shot commands

### Quality gate

```bash
export GEMMA4_MODEL="gemma4-layer-aware"

python scripts/eval_quality_gate.py \
  --model "$GEMMA4_MODEL" --suite all \
  --expert-offload --max-resident-experts 2048 \
  --kv-cache-type isoquant \
  --output-json results/gemma4_layer_aware_quality.json
```

### Benchmark (16GB envelope)

```bash
python scripts/benchmark_moe_offload.py \
  --model "$GEMMA4_MODEL" --profile A --expert-offload \
  --max-resident-experts 2048 \
  --kv-cache-type isoquant --target-envelope-mb 12800 \
  --json-output results/gemma4_layer_aware_benchmark.json
```

Current measured result: `12.85 tok/s`, `5419.8 MB` peak, `98.8%` decode hit rate.

### 2h stability soak

```bash
python scripts/run_stability_soak.py \
  --model "$GEMMA4_MODEL" --duration-mins 120 \
  --expert-offload --max-resident-experts 2048 \
  --kv-cache-type isoquant \
  --memory-limit-mb 12800 --output-dir results/soak
```

Final artifact path for the layer-aware checkpoint: `results/soak/gemma4-layer-aware_soak_final.json`.

### KV fidelity / PPL

```bash
python scripts/measure_kv_fidelity.py \
  --model "$GEMMA4_MODEL" \
  --depths 512,2048 --seed 42 \
  --expert-offload --max-resident-experts 2048 \
  --output-json results/gemma4_layer_aware_kv_fidelity.json
```

Current measured result: IsoQuant `delta_ppl_vs_default = 0.0` at both `512` and `2048` tokens.

## Artifacts

| Artifact | Path | Acceptance |
|----------|------|------------|
| Quality gate | `results/gemma4_layer_aware_quality.json` | All checks pass (exit 0). Current measured result: `12/12`. |
| Benchmark | `results/gemma4_layer_aware_benchmark.json` | `peak_memory_mb` ≤ 12800. Decode tok/s ≥ 5. Expert decode hit rate ≥ 90%. Current measured result: `12.85 tok/s`, `5419.8 MB`, `98.8%` decode hit. |
| KV fidelity / PPL | `results/gemma4_layer_aware_kv_fidelity.json` | IsoQuant delta vs default should stay near zero at target depths. Current measured result: `0.0` at `512` and `2048`. |
| 2h soak | `results/soak/gemma4-layer-aware_soak_final.json` | P99/P50 < 3.0. RSS drift ratio < 1.5x. No OOM. |

## Decode profiler (Phase 2)

```bash
python scripts/decode_profiler.py \
  --model "$GEMMA4_MODEL" --expert-offload \
  --kv-cache-type isoquant --warm-repeat \
  --output-json results/profile/gemma4_decode_breakdown.json
```

Reports `kv_attention_ms`, `routed_expert_ms`, `dense_ffn_ms`, `other_ms` per decode token. Gemma4 dual-pathway MoE layers classify `self.mlp` as dense FFN and `router`+`experts` as routed expert separately.

## Model notes

- Architecture: Gemma4 with interleaved attention + MoE layers. Dense `MLP` pathway runs alongside routed `Router`+`Experts` in MoE layers.
- The `enable_moe` flag per layer controls whether a layer has the dual pathway.
- Expert offload uses `SwitchGLU` via `expert_offload.py` LRU cache.
- IsoQuant KV applies quaternion rotation compression to attention-layer caches only (not Mamba/SSM).
- The layer-aware checkpoint uses a mixed expert schedule: selected layers are quantized to 2-bit or 3-bit while the rest stay at 4-bit. Expert offload must read per-layer bit metadata correctly for this checkpoint to run.

## Memory budget

`--target-envelope-mb 12800` = 80% of 16GB. This is the default starting point, not a sacred cap. If throughput improves materially with higher expert residency, it is acceptable to push toward **14400 MB** (90% of 16GB) provided the run remains stable and does not enter memory-pressure collapse. The soak uses `mx.set_memory_limit()` to enforce the chosen cap. For a true 16GB pathway proof, run on 16GB-class hardware.

**Expert residency:** `--max-resident-experts 2048` is the tuned layer-aware setting. Current measured run used `1646` resident slots, reached `98.8%` decode hit rate, and delivered `12.85 tok/s` at `5419.8 MB` peak. The default of 16 is far too low: it produces 0% hit rate and sub-1 tok/s, which is a failed configuration even if it technically fits.
