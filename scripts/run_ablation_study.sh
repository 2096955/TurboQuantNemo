#!/usr/bin/env bash
# Ablation study: isolate contribution of each compression axis.
#
# Runs a 2x2 matrix (expert offload × KV compression) plus quality gate
# for each configuration. Reports throughput, memory, and quality per cell.
#
# The third axis (weight quantization) cannot be toggled at runtime —
# it's baked into the checkpoint. If a higher-precision checkpoint exists,
# set MODEL_FP16 to include it; otherwise that cell reports "N/A (no checkpoint)".
#
# Usage:
#   bash scripts/run_ablation_study.sh
#   MODEL=./gemma4-layer-aware bash scripts/run_ablation_study.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/artifacts/ablation}"
SEED="${SEED:-42}"
PROFILE="${PROFILE:-A}"
SUITE="${SUITE:-default}"
MODEL="${MODEL:-$REPO_ROOT/gemma4-layer-aware}"
MAX_TOKENS="${MAX_TOKENS:-500}"
MEMORY_LIMIT="${MEMORY_LIMIT:-}"
PYTHON_BIN="${PYTHON:-python}"

mkdir -p "$OUT_DIR"

if [[ ! -d "$MODEL" ]]; then
    echo "ERROR: Model not found at $MODEL" >&2
    exit 1
fi

model_label="$(basename "$MODEL")"
timestamp="$(date +%Y%m%d_%H%M%S)"

echo "=== Ablation Study: $model_label ==="
echo "    Profile: $PROFILE  Seed: $SEED  Suite: $SUITE"
echo "    Output:  $OUT_DIR/"
echo ""

# Define the 2x2 matrix: (expert_offload, kv_cache_type)
# Each cell name follows: {offload}_{kv}
declare -a CELLS=(
    "offload_isoquant"
    "offload_default"
    "noffload_isoquant"
    "noffload_default"
)

run_cell() {
    local cell_name="$1"
    local bench_out="$OUT_DIR/bench_${model_label}_${cell_name}_${timestamp}.json"
    local qg_out="$OUT_DIR/qg_${model_label}_${cell_name}_${timestamp}.json"

    # Parse cell name into flags
    local offload_flag=""
    local kv_type="default"

    case "$cell_name" in
        offload_isoquant)  offload_flag="--expert-offload"; kv_type="isoquant" ;;
        offload_default)   offload_flag="--expert-offload"; kv_type="default" ;;
        noffload_isoquant) offload_flag="";                 kv_type="isoquant" ;;
        noffload_default)  offload_flag="";                 kv_type="default" ;;
    esac

    local mem_flag=""
    if [[ -n "$MEMORY_LIMIT" ]]; then
        mem_flag="--memory-limit-mb $MEMORY_LIMIT"
    fi

    echo "--- Cell: $cell_name (offload=$([ -n "$offload_flag" ] && echo yes || echo no), kv=$kv_type) ---"

    # Benchmark run
    echo "  [1/2] Benchmark..."
    # shellcheck disable=SC2086
    if "$PYTHON_BIN" "$SCRIPT_DIR/benchmark_moe_offload.py" \
        --model "$MODEL" \
        --profile "$PROFILE" \
        --seed "$SEED" \
        --kv-cache-type "$kv_type" \
        $offload_flag \
        --json-output "$bench_out" 2>&1; then
        echo "    -> $bench_out"
    else
        echo "    FAILED (likely OOM or model too large without offload)"
        echo '{"error": "benchmark failed — likely OOM without expert offload"}' > "$bench_out"
    fi

    # Quality gate run
    echo "  [2/2] Quality gate (suite=$SUITE)..."
    # shellcheck disable=SC2086
    if "$PYTHON_BIN" "$SCRIPT_DIR/eval_quality_gate.py" \
        --model "$MODEL" \
        --suite "$SUITE" \
        --seed "$SEED" \
        --max-tokens "$MAX_TOKENS" \
        --kv-cache-type "$kv_type" \
        $offload_flag \
        $mem_flag \
        --output-json "$qg_out" 2>&1; then
        echo "    -> $qg_out"
    else
        echo "    Quality gate returned non-zero (some prompts may have failed)"
        [[ -f "$qg_out" ]] && echo "    -> $qg_out (partial)"
    fi
    echo ""
}

# Run all 4 cells
for cell in "${CELLS[@]}"; do
    run_cell "$cell"
done

# Summary table
echo ""
echo "=== Ablation Summary: $model_label ==="
printf "%-25s  %-10s  %-12s  %-10s\n" "Configuration" "tok/s" "Peak MB" "Quality"
printf "%-25s  %-10s  %-12s  %-10s\n" "-------------------------" "----------" "------------" "----------"

for cell in "${CELLS[@]}"; do
    bench_file="$OUT_DIR/bench_${model_label}_${cell}_${timestamp}.json"
    qg_file="$OUT_DIR/qg_${model_label}_${cell}_${timestamp}.json"

    if command -v jq &>/dev/null; then
        toks=$(jq -r '.decode_tok_per_s // .error // "N/A"' "$bench_file" 2>/dev/null || echo "N/A")
        peak=$(jq -r '.peak_memory_mb // "N/A"' "$bench_file" 2>/dev/null || echo "N/A")
        if [[ -f "$qg_file" ]] && jq -e '.n_total' "$qg_file" &>/dev/null; then
            quality=$(jq -r '"\(.n_pass)/\(.n_total)"' "$qg_file" 2>/dev/null || echo "?/?")
        else
            quality="N/A"
        fi
    else
        toks="(install jq)"; peak="(install jq)"; quality="(install jq)"
    fi

    printf "%-25s  %-10s  %-12s  %-10s\n" "$cell" "$toks" "$peak" "$quality"
done

echo ""
echo "Ablation artifacts in $OUT_DIR/"
