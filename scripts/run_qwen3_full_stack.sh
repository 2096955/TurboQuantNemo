#!/usr/bin/env bash
# Full pathway stack for Qwen3 MoE: offload + predictor + IsoQuant + DedeKimi + cliques.
# Usage:
#   export QWEN3_MODEL=/path/to/Qwen3-30B-A3B-4bit
#   bash scripts/run_qwen3_full_stack.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${QWEN3_MODEL:-}"
if [[ -z "$MODEL" ]]; then
  echo "Set QWEN3_MODEL to your local MLX model directory." >&2
  exit 2
fi

CLIQUES="${QWEN3_TASK_CLIQUE_JSON:-results/gemma4_task_expert_cliques_min.json}"
OUT_BENCH="${QWEN3_FULLSTACK_BENCH_JSON:-results/qwen3_fullstack_benchmark.json}"
OUT_QUAL="${QWEN3_FULLSTACK_QUALITY_JSON:-results/qwen3_fullstack_isoquant_quality.json}"

export TURBOQUANT_BITS="${TURBOQUANT_BITS:-3}"

echo "=== benchmark_moe_offload (isoquant, full stack flags) ==="
python scripts/benchmark_moe_offload.py \
  --model "$MODEL" \
  --profile A \
  --expert-offload \
  --kv-cache-type isoquant \
  --turboquant-bits "${TURBOQUANT_BITS}" \
  --use-predictor \
  --use-dedekimi-observer \
  ${CLIQUES:+--task-expert-cliques-file "$CLIQUES"} \
  --json-output "$OUT_BENCH"

echo "=== eval_quality_gate (isoquant, full stack flags) ==="
python scripts/eval_quality_gate.py \
  --model "$MODEL" \
  --suite all \
  --expert-offload \
  --kv-cache-type isoquant \
  --use-predictor \
  --use-dedekimi-observer \
  ${CLIQUES:+--task-expert-cliques-file "$CLIQUES"} \
  --output-json "$OUT_QUAL"

echo "Wrote $OUT_BENCH and $OUT_QUAL"
