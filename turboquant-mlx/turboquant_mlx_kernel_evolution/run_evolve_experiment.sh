#!/usr/bin/env bash
# TurboQuant Phase 6 — OpenEvolve launcher (mirrors mlx_metal_kernel_opt pattern).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

ITERATIONS="${ITERATIONS:-25}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/openevolve_output_turboquant}"

echo "Repo root: $ROOT"
echo "Iterations: $ITERATIONS  Output: $OUTPUT_DIR"

python3 -m openevolve.cli \
  turboquant_mlx_kernel_evolution/initial_program.py \
  turboquant_mlx_kernel_evolution/evaluator.py \
  --config turboquant_mlx_kernel_evolution/config.yaml \
  --iterations "$ITERATIONS" \
  --output "$OUTPUT_DIR" \
  "$@"
