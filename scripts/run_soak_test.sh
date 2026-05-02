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
MAX_RESIDENT="${MAX_RESIDENT:-128}"
KV_CACHE_TYPE="${KV_CACHE_TYPE:-default}"

if [[ -z "${MODEL}" ]]; then
  echo "Error: MODEL path must be set" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
model_name=$(basename "${MODEL}")
timestamp=$(date +"%Y%m%d_%H%M%S")

snapshot_system() {
  local label="$1"
  {
    echo "timestamp=$(date --iso-8601=seconds 2>/dev/null || date)"
    echo "pid=$$"
    echo "--- ps ---"
    ps -o pid=,rss=,etime= -p $$ || true
    echo "--- vm_stat ---"
    vm_stat || true
    echo "--- swapusage ---"
    sysctl vm.swapusage || true
  } > "${OUT_DIR}/system_${label}_${timestamp}.txt"
}

echo "=== Soak Test for ${model_name} (${DURATION_MINS} mins) ==="
echo "max_resident=${MAX_RESIDENT} kv_cache_type=${KV_CACHE_TYPE}"
snapshot_system "start"

# 1. Start Benchmark
echo "[Start] Benchmarking cold+warm cache..."
python3 scripts/benchmark_moe_offload.py \
  --model "${MODEL}" \
  --expert-offload \
  --max-resident-experts "${MAX_RESIDENT}" \
  --kv-cache-type "${KV_CACHE_TYPE}" \
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
    --max-resident-experts "${MAX_RESIDENT}" \
    --kv-cache-type "${KV_CACHE_TYPE}" \
    --suite all \
    --output-json "${OUT_DIR}/soak_qg_iter${iter}_${timestamp}.json" || true
  snapshot_system "iter${iter}"
    
  echo "Iteration ${iter} complete."
  iter=$((iter + 1))
done

echo "=== Soak Loop Complete ==="

# 2. End Benchmark
echo "[End] Benchmarking warm cache..."
python3 scripts/benchmark_moe_offload.py \
  --model "${MODEL}" \
  --expert-offload \
  --max-resident-experts "${MAX_RESIDENT}" \
  --kv-cache-type "${KV_CACHE_TYPE}" \
  --warm-second-pass \
  --json-output "${OUT_DIR}/soak_end_bench_${timestamp}.json"

snapshot_system "end"

echo "=== Soak Test Complete. Artifacts in ${OUT_DIR} ==="
