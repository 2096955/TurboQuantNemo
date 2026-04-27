#!/usr/bin/env bash
# Phase 0 baseline: 3 repeats per (T, cell) at 4K/8K/16K/32K with tiled V-accum.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATE="${DATE:-$(date +%Y-%m-%d)}"
OUT_DIR="$REPO_ROOT/artifacts/phase0_baseline_${DATE}"
mkdir -p "$OUT_DIR"
BENCH_SCRIPT="$REPO_ROOT/scripts/benchmark_nvfp4_isoquant.py"
export PYTHONPATH="$REPO_ROOT/mlx-lm${PYTHONPATH:+:$PYTHONPATH}"

BASELINE="${BASELINE:-/Users/anthonylui/Models/Qwen3.6-35B-A3B-4bit}"
NVFP4="${NVFP4:-/Users/anthonylui/Models/Qwen3.6-35B-A3B-nvfp4}"

for T in 4096 8192 16384 32768; do
  D=$(( T < 8192 ? 512 : 1024 ))
  for REPEAT in 1 2 3; do
    OUT="$OUT_DIR/matrix_T${T}_d${D}_r${REPEAT}.json"
    echo "=== T=$T D=$D repeat=$REPEAT -> $OUT ==="
    python3 "$BENCH_SCRIPT" \
      --baseline-model "$BASELINE" \
      --nvfp4-model "$NVFP4" \
      --output "$OUT" \
      --prefill-tokens "$T" \
      --decode-tokens "$D" \
      --isoquant-bits 3
  done
done

echo "Baseline complete: $OUT_DIR"
