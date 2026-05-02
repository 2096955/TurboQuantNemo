#!/usr/bin/env bash
# Reproduce roadmap Phase 0 (MLX + NPT8) and Phase 3.2 synthetic / optional 3.2+3.3 E2E.
#
# Usage:
#   ./scripts/run_write_path_attribution_gate.sh
#   MODEL=/path/to/mlx MODEL ./scripts/run_write_path_attribution_gate.sh
#   RUN_E2E=1 DECODE_STEPS=35 ./scripts/run_write_path_attribution_gate.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:-${HOME}/Models/Qwen3.6-35B-A3B-nvfp4}"
RUN_E2E="${RUN_E2E:-0}"
DECODE_STEPS="${DECODE_STEPS:-35}"

export PYTHONPATH="${REPO_ROOT}/mlx-lm${PYTHONPATH:+:${PYTHONPATH}}"

echo "== MLX import smoke =="
python3 -c "import mlx.core as mx; print(mx.default_device())"

echo "== NPT8 focused pytest =="
python3 -m pytest \
  mlx-lm/tests/test_fused_npt8.py \
  mlx-lm/tests/test_fused_npt8_tiled.py -q

echo "== Synthetic write-path profile (Phase 3.2 fast path) =="
python3 scripts/profile_metal_counters.py \
  --model "$MODEL" \
  --output artifacts/metal-counters/profile_with_write.json \
  --prefill 4096 8192 --skip-e2e --skip-traces

if [[ "$RUN_E2E" == "1" ]]; then
  echo "== E2E default vs IsoQuant gap + synthetic comparison (Phase 3.3) =="
  python3 scripts/profile_metal_counters.py \
    --model "$MODEL" \
    --output artifacts/metal-counters/profile_with_write_e2e.json \
    --prefill 4096 8192 \
    --skip-traces \
    --decode-steps "$DECODE_STEPS"
else
  echo "== Skipping E2E (set RUN_E2E=1 to populate profile_with_write_e2e.json) =="
fi

echo "Done."
