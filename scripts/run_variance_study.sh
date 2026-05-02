#!/usr/bin/env bash
# Inter-run variance study: runs benchmark N times per model/config and
# reports mean ± stdev for tok/s and peak memory.
#
# Uses the existing --repeat-runs flag in benchmark_moe_offload.py.
#
# Usage:
#   bash scripts/run_variance_study.sh
#   MODEL=./nemotron-30b-mixed RUNS=5 bash scripts/run_variance_study.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/artifacts/variance}"
RUNS="${RUNS:-5}"
SEED="${SEED:-42}"
PROFILE="${PROFILE:-A}"
PYTHON_BIN="${PYTHON:-python}"

mkdir -p "$OUT_DIR"

declare -A MODELS
if [[ -n "${MODEL:-}" ]]; then
    MODELS[custom]="$MODEL"
else
    [[ -d "$REPO_ROOT/gemma4-layer-aware" ]] && MODELS[gemma4-layer-aware]="$REPO_ROOT/gemma4-layer-aware"
    [[ -d "$REPO_ROOT/nemotron-30b-mixed" ]] && MODELS[nemotron-30b-mixed]="$REPO_ROOT/nemotron-30b-mixed"
fi

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "ERROR: No models found. Set MODEL=<path> or ensure gemma4-layer-aware/nemotron-30b-mixed exist." >&2
    exit 1
fi

KV_MODES=("default" "isoquant")

for label in "${!MODELS[@]}"; do
    model_path="${MODELS[$label]}"
    for kv in "${KV_MODES[@]}"; do
        tag="${label}_${kv}_${RUNS}runs"
        out_file="$OUT_DIR/variance_${tag}.json"
        echo "=== $tag: $RUNS runs, profile $PROFILE, seed $SEED ==="
        "$PYTHON_BIN" "$SCRIPT_DIR/benchmark_moe_offload.py" \
            --model "$model_path" \
            --profile "$PROFILE" \
            --seed "$SEED" \
            --kv-cache-type "$kv" \
            --expert-offload \
            --repeat-runs "$RUNS" \
            --json-output "$out_file"
        echo "  -> $out_file"

        # Print summary if jq is available
        if command -v jq &>/dev/null && [[ -f "$out_file" ]]; then
            echo "  Summary:"
            jq -r '.multi_run_summary | "    tok/s: \(.decode_tok_per_s_mean | tostring) ± \(.decode_tok_per_s_stdev | tostring)  |  peak MB: \(.peak_memory_mb_mean | tostring) (max \(.peak_memory_mb_max | tostring))"' "$out_file" 2>/dev/null || true
        fi
        echo ""
    done
done

echo "All variance runs complete. Artifacts in $OUT_DIR/"
