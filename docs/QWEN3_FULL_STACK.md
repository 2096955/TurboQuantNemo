# Qwen3-30B-A3B — full pathway stack (single session)

Combines: **expert offload** + **IsoQuant KV** (primary) + **deferred prefill**.

DedeKimi observer, predictor, and task cliques are optional enhancements — not required for the pathway-proven gate.

## Prerequisites

- Local MLX model dir: `mlx-community/Qwen3-30B-A3B-4bit` (or equivalent local snapshot).
- `mlx-lm` installed in editable mode: `cd mlx-lm && pip install -e .`

## One-shot commands

### Quality gate

```bash
export QWEN3_MODEL="mlx-community/Qwen3-30B-A3B-4bit"

python scripts/eval_quality_gate.py \
  --model "$QWEN3_MODEL" --suite all \
  --expert-offload --max-resident-experts 4096 \
  --kv-cache-type isoquant \
  --output-json results/qwen3_pathway_quality.json
```

### Benchmark (16GB envelope)

```bash
python scripts/benchmark_moe_offload.py \
  --model "$QWEN3_MODEL" --profile A --expert-offload \
  --max-resident-experts 4096 \
  --kv-cache-type isoquant --target-envelope-mb 12800 \
  --json-output results/qwen3_pathway_benchmark.json
```

### 2h stability soak

```bash
python scripts/run_stability_soak.py \
  --model "$QWEN3_MODEL" --duration-mins 120 \
  --expert-offload --max-resident-experts 4096 \
  --kv-cache-type isoquant \
  --memory-limit-mb 12800 --output-dir results/soak
```

## Artifacts

| Artifact | Path | Acceptance |
|----------|------|------------|
| Quality gate | `results/qwen3_pathway_quality.json` | All checks pass (exit 0). No degenerate repetition. |
| Benchmark | `results/qwen3_pathway_benchmark.json` | `peak_memory_mb` ≤ 12800. Decode tok/s ≥ 5. Expert decode hit rate ≥ 90%. |
| 2h soak | `results/soak/qwen3_2h_soak_final.json` | P99/P50 < 3.0. RSS drift ratio < 1.5x. No OOM. |

## Decode profiler (Phase 2)

```bash
python scripts/decode_profiler.py \
  --model "$QWEN3_MODEL" --expert-offload --max-resident-experts 4096 \
  --kv-cache-type isoquant --warm-repeat \
  --output-json results/profile/qwen3_decode_breakdown.json
```

Reports `kv_attention_ms`, `routed_expert_ms`, `dense_ffn_ms`, and `other_ms` per decode token. Qwen3 MoE layers classify `self.mlp` via `switch_mlp`/`gate`; dense fallback layers land in `dense_ffn_ms`.

## Model notes

- Architecture: Qwen3 MoE with 48 MoE layers, 128 experts, top-8 routing.
- Expert offload uses `SwitchGLU` via `expert_offload.py` LRU cache.
- IsoQuant KV applies quantized attention-cache compression on the attention path; deferred prefill finalizes before decode.

## Memory budget

`--target-envelope-mb 12800` = 80% of 16GB. This is the default starting point, not a sacred cap. If throughput improves materially with higher expert residency, it is acceptable to push toward **14400 MB** (90% of 16GB) provided the run remains stable and does not enter memory-pressure collapse. The soak uses `mx.set_memory_limit()` to enforce the chosen cap. For a true 16GB pathway proof, run on 16GB-class hardware.

**Expert residency:** `--max-resident-experts 4096` reached ~9.5GB peak with 96.4% decode hit rate and 9.87 tok/s. The lower-slot default produced 0% hit rate and ~0.66 tok/s, which is not an acceptable configuration even though it technically fit the envelope.
