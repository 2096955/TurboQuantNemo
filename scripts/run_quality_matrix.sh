#!/usr/bin/env bash
# Fixed-seed quality matrix for uniform and optional layer-aware checkpoints
# (human-run on target hardware).
# Usage:
#   MODEL_2BIT=/path/to/nemotron-120b-mixed \
#   MODEL_3BIT=/path/to/nemotron-120b-mixed-3bit \
#   MODEL_4BIT=/path/to/nemotron-120b-4bit \
#   MODEL_LAYER_AWARE=/path/to/nemotron-120b-layer-aware \
#   ./scripts/run_quality_matrix.sh
#
# Writes JSON artifacts to OUT_DIR (default: ./artifacts/quality-matrix).
# Backward compatibility: MODEL_APEX is accepted as an alias for MODEL_LAYER_AWARE.

set -euo pipefail

OUT_DIR="${OUT_DIR:-./artifacts/quality-matrix}"
SEED="${SEED:-42}"
SUITE="${SUITE:-all}"

mkdir -p "${OUT_DIR}"

run_one() {
  local label="$1"
  local model="$2"
  if [[ -z "${model}" ]]; then
    echo "Skip ${label}: path not set" >&2
    return 0
  fi
  echo "=== ${label} ==="
  python3 scripts/eval_quality_gate.py \
    --model "${model}" \
    --expert-offload \
    --suite "${SUITE}" \
    --seed "${SEED}" \
    --strict \
    --output-json "${OUT_DIR}/qg-${label}-seed${SEED}.json"
}

run_one "2bit" "${MODEL_2BIT:-}"
run_one "3bit" "${MODEL_3BIT:-}"
run_one "4bit" "${MODEL_4BIT:-}"
run_one "layer-aware" "${MODEL_LAYER_AWARE:-${MODEL_APEX:-}}"

echo "Artifacts under ${OUT_DIR}"
