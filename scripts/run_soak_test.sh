#!/usr/bin/env bash
# Long-session soak automation wrapper
# Usage:
#   MODEL=/path/to/nemotron-120b-layer-aware \
#   DURATION_MINS=30 \
#   ./scripts/run_soak_test.sh
#
# Runs benchmark at the start and end, and loops quality gates.

set -euo pipefail

MODEL="${MODEL:-}"
DURATION_MINS="${DURATION_MINS:-30}"
OUT_DIR="${OUT_DIR:-./artifacts/soak}"

if [[ -z "${MODEL}" ]]; then
  echo "Error: MODEL path must be set" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
model_name=$(basename "${MODEL}")
timestamp=$(date +"%Y%m%d_%H%M%S")

echo "=== Soak Test for ${model_name} (${DURATION_MINS} mins) ==="

# 1. Start Benchmark
echo "[Start] Benchmarking cold+warm cache..."
python3 scripts/benchmark_moe_offload.py \
  --model "${MODEL}" \
  --expert-offload \
  --warm-second-pass \
  --json-output "${OUT_DIR}/soak_start_bench_${timestamp}.json"

echo "=== Starting Soak Loop ==="
end_time=$((SECONDS + DURATION_MINS * 60))
iter=1

while [ $SECONDS -lt $end_time ]; do
  echo "--- Soak Iteration ${iter} ($(date +%H:%M:%S)) ---"
  
  # Run quality gate to exercise expert loading and context
  python3 scripts/eval_quality_gate.py \
    --model "${MODEL}" \
    --expert-offload \
    --suite all \
    --output-json "${OUT_DIR}/soak_qg_iter${iter}_${timestamp}.json" || true
    
  echo "Iteration ${iter} complete."
  iter=$((iter + 1))
done

echo "=== Soak Loop Complete ==="

# 2. End Benchmark
echo "[End] Benchmarking warm cache..."
python3 scripts/benchmark_moe_offload.py \
  --model "${MODEL}" \
  --expert-offload \
  --warm-second-pass \
  --json-output "${OUT_DIR}/soak_end_bench_${timestamp}.json"

echo "=== Soak Test Complete. Artifacts in ${OUT_DIR} ==="
