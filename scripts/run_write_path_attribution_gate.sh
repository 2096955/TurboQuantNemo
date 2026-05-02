#!/usr/bin/env bash
# Reproduce roadmap Phase 0 (MLX + NPT8) and Phase 3.2 synthetic / optional 3.2+3.3 E2E.
#
# IMPORTANT: -h / --help is handled before set -euo pipefail and before any Python/MLX work,
# so help never triggers an MLX import or wedges the GPU driver.

case "${1:-}" in
-h | --help)
  cat <<'EOF'
Usage: scripts/run_write_path_attribution_gate.sh [-h|--help]

  Runs (in order): MLX import smoke → NPT8 pytest → synthetic profile_metal_counters
  (Phase 3.2). Optionally runs E2E default vs IsoQuant + comparison (Phase 3.3).

Environment (optional):
  MODEL         Path to MLX model dir (default: $HOME/Models/Qwen3.6-35B-A3B-nvfp4)
  PYTHON        Python interpreter to use (default: python3)
  RUN_E2E       Set to 1 to also write profile_with_write_e2e.json (default: 0)
  DECODE_STEPS  Decode steps for E2E phase when RUN_E2E=1 (default: 35)
  PYTHONPATH    Prepended with repo mlx-lm/ automatically

Examples:
  ./scripts/run_write_path_attribution_gate.sh
  MODEL=/path/to/model RUN_E2E=1 ./scripts/run_write_path_attribution_gate.sh
  PYTHON=/opt/homebrew/bin/python3.12 ./scripts/run_write_path_attribution_gate.sh
EOF
  exit 0
  ;;
esac

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:-${HOME}/Models/Qwen3.6-35B-A3B-nvfp4}"
RUN_E2E="${RUN_E2E:-0}"
DECODE_STEPS="${DECODE_STEPS:-35}"
PYTHON="${PYTHON:-python3}"

export PYTHONPATH="${REPO_ROOT}/mlx-lm${PYTHONPATH:+:${PYTHONPATH}}"

echo "== MLX import smoke =="
"$PYTHON" -c "import mlx.core as mx; print(mx.default_device())"

echo "== NPT8 focused pytest =="
"$PYTHON" -m pytest \
  mlx-lm/tests/test_fused_npt8.py \
  mlx-lm/tests/test_fused_npt8_tiled.py -q

echo "== Synthetic write-path profile (Phase 3.2 fast path) =="
"$PYTHON" scripts/profile_metal_counters.py \
  --model "$MODEL" \
  --output artifacts/metal-counters/profile_with_write.json \
  --prefill 4096 8192 --skip-e2e --skip-traces

if [[ "$RUN_E2E" == "1" ]]; then
  echo "== E2E default vs IsoQuant gap + synthetic comparison (Phase 3.3) =="
  "$PYTHON" scripts/profile_metal_counters.py \
    --model "$MODEL" \
    --output artifacts/metal-counters/profile_with_write_e2e.json \
    --prefill 4096 8192 \
    --skip-traces \
    --decode-steps "$DECODE_STEPS"
else
  echo "== Skipping E2E (set RUN_E2E=1 to populate profile_with_write_e2e.json) =="
fi

echo "Done."
