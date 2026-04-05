#!/usr/bin/env bash
# Benchmark runner for cold-cache and warm-cache MoE offload metrics
# Usage:
#   MODEL=/path/to/nemotron-120b-layer-aware \
#   ./scripts/run_benchmarks.sh
#
# Writes JSON artifacts to OUT_DIR (default: ./artifacts/benchmarks).

set -euo pipefail

OUT_DIR="${OUT_DIR:-./artifacts/benchmarks}"
MODEL="${MODEL:-}"

if [[ -z "${MODEL}" ]]; then
  echo "Error: MODEL path must be set" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

model_name=$(basename "${MODEL}")
timestamp=$(date +"%Y%m%d_%H%M%S")
outfile="${OUT_DIR}/bench-${model_name}-${timestamp}.json"

echo "=== Benchmarking ${model_name} ==="
python3 scripts/benchmark_moe_offload.py \
  --model "${MODEL}" \
  --expert-offload \
  --warm-second-pass \
  --json-output "${outfile}"

echo "Benchmark artifact written to ${outfile}"
