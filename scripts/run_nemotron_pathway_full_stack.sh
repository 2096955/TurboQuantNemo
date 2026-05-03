#!/usr/bin/env bash
# Nemotron-H pathway: offload + isoquant KV + DedeKimi observer.
# Predictor / task cliques remain optional experiments and are not part of the
# default pathway-proven gate.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${NEMOTRON_MODEL:-}"
if [[ -z "$MODEL" ]]; then
  echo "Set NEMOTRON_MODEL to your local Nemotron MLX checkpoint directory." >&2
  exit 2
fi

CFG="$MODEL/config.json"
if [[ ! -f "$CFG" ]]; then
  echo "Expected config at $CFG" >&2
  exit 2
fi

python - "$CFG" <<'PY'
import json
import sys
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text())
model_type = cfg.get("model_type")
layers = int(cfg.get("num_hidden_layers", -1))
routed = int(cfg.get("n_routed_experts", -1))
topk = int(cfg.get("num_experts_per_tok", -1))

if model_type != "nemotron_h":
    raise SystemExit(f"Expected model_type nemotron_h, got: {model_type!r}")

# Validate Nemotron-H architecture. Accept both 30B (52 layers) and 120B (96 layers).
if routed not in {64, 128} or topk < 6:
    raise SystemExit(
        "Checkpoint does not look like a Nemotron-H MoE model. "
        f"Found layers={layers}, routed_experts={routed}, topk={topk}."
    )

print(
    f"Preflight OK: model_type={model_type}, layers={layers}, "
    f"routed_experts={routed}, topk={topk}"
)
PY

CLIQUES="${NEMOTRON_TASK_CLIQUE_JSON:-}"
OUT_BENCH="${NEMOTRON_PATHWAY_BENCH_JSON:-results/nemotron_pathway_benchmark.json}"
OUT_QUAL="${NEMOTRON_PATHWAY_QUALITY_JSON:-results/nemotron_pathway_quality.json}"
MRE="${MAX_RESIDENT_EXPERTS:-48}"
USE_PREDICTOR="${NEMOTRON_USE_PREDICTOR:-0}"

export TURBOQUANT_BITS="${TURBOQUANT_BITS:-3}"

EXTRA_ARGS=()
if [[ "$USE_PREDICTOR" == "1" ]]; then
  EXTRA_ARGS+=(--use-predictor)
fi

echo "=== benchmark (Profile B, memory-mode 120b-32gb forces isoquant in harness) ==="
python scripts/benchmark_moe_offload.py \
  --model "$MODEL" \
  --profile B \
  --memory-mode 120b-32gb \
  --expert-offload \
  --max-resident-experts "$MRE" \
  --turboquant-bits "${TURBOQUANT_BITS}" \
  --use-dedekimi-observer \
  "${EXTRA_ARGS[@]}" \
  ${CLIQUES:+--task-expert-cliques-file "$CLIQUES"} \
  --json-output "$OUT_BENCH"

echo "=== eval_quality_gate (isoquant) ==="
python scripts/eval_quality_gate.py \
  --model "$MODEL" \
  --suite all \
  --expert-offload \
  --max-resident-experts "$MRE" \
  --kv-cache-type isoquant \
  --use-dedekimi-observer \
  "${EXTRA_ARGS[@]}" \
  ${CLIQUES:+--task-expert-cliques-file "$CLIQUES"} \
  --output-json "$OUT_QUAL"

echo "Wrote $OUT_BENCH and $OUT_QUAL"
