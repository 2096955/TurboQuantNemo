#!/usr/bin/env bash
# Failure boundary test: progressively reduce memory cap until the system
# fails, documenting HOW it fails at each step.
#
# Tests two failure modes:
#   1. Memory exhaustion — reduce --memory-limit-mb until OOM
#   2. KV overflow — reduce --max-kv-size until cache eviction affects quality
#
# Usage:
#   bash scripts/run_failure_boundary.sh
#   MODEL=./gemma4-layer-aware bash scripts/run_failure_boundary.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/artifacts/failure-boundary}"
SEED="${SEED:-42}"
MODEL="${MODEL:-$REPO_ROOT/gemma4-layer-aware}"

mkdir -p "$OUT_DIR"

if [[ ! -d "$MODEL" ]]; then
    echo "ERROR: Model not found at $MODEL" >&2
    exit 1
fi

model_label="$(basename "$MODEL")"
timestamp="$(date +%Y%m%d_%H%M%S)"

echo "=== Failure Boundary Test: $model_label ==="
echo ""

# --- Test 1: Memory exhaustion boundary ---
echo "=== Test 1: Memory Exhaustion Boundary ==="
echo "    Reducing memory cap until failure..."
echo ""

# Start high and reduce. Typical working point is ~5-6GB for Gemma4.
MEMORY_CAPS=(8192 6144 5120 4096 3072 2048)

printf "%-12s  %-8s  %-12s  %-40s\n" "Cap (MB)" "Result" "Peak (MB)" "Failure Mode"
printf "%-12s  %-8s  %-12s  %-40s\n" "------------" "--------" "------------" "----------------------------------------"

for cap in "${MEMORY_CAPS[@]}"; do
    out_file="$OUT_DIR/mem_boundary_${model_label}_${cap}mb_${timestamp}.json"

    result="PASS"
    peak="N/A"
    failure_mode="none"

    # Run quality gate with memory cap — capture exit code and stderr
    stderr_file="$OUT_DIR/.stderr_${cap}.tmp"
    if python "$SCRIPT_DIR/eval_quality_gate.py" \
        --model "$MODEL" \
        --suite micro \
        --seed "$SEED" \
        --max-tokens 48 \
        --kv-cache-type isoquant \
        --expert-offload \
        --memory-limit-mb "$cap" \
        --output-json "$out_file" 2>"$stderr_file"; then
        result="PASS"
    else
        exit_code=$?
        result="FAIL"
        # Classify failure mode from stderr
        if grep -qi "out of memory\|OOM\|MemoryError\|memory limit" "$stderr_file" 2>/dev/null; then
            failure_mode="OOM (memory limit exceeded)"
        elif grep -qi "Metal\|GPU\|device" "$stderr_file" 2>/dev/null; then
            failure_mode="Metal/GPU resource error"
        elif grep -qi "killed\|signal\|abort" "$stderr_file" 2>/dev/null; then
            failure_mode="Process killed (likely OS OOM killer)"
        else
            failure_mode="Exit code $exit_code (see stderr log)"
        fi
        # Save stderr for analysis
        cp "$stderr_file" "$OUT_DIR/mem_boundary_stderr_${cap}mb_${timestamp}.txt"
    fi

    # Extract peak memory if artifact exists
    if command -v jq &>/dev/null && [[ -f "$out_file" ]]; then
        peak=$(jq -r '.memory.peak_mb // "N/A"' "$out_file" 2>/dev/null || echo "N/A")
    fi

    printf "%-12s  %-8s  %-12s  %-40s\n" "${cap}" "$result" "$peak" "$failure_mode"

    rm -f "$stderr_file"

    # If we've hit failure, run one more step then stop
    if [[ "$result" == "FAIL" ]]; then
        echo ""
        echo "  First failure at ${cap} MB cap. Boundary identified."
        break
    fi
done

echo ""

# --- Test 2: KV cache overflow ---
echo "=== Test 2: KV Cache Size Limit ==="
echo "    Testing quality with restricted KV cache sizes..."
echo ""

KV_SIZES=(4096 2048 1024 512 256)

printf "%-12s  %-10s  %-40s\n" "max-kv-size" "Quality" "Notes"
printf "%-12s  %-10s  %-40s\n" "------------" "----------" "----------------------------------------"

for kv_size in "${KV_SIZES[@]}"; do
    out_file="$OUT_DIR/kv_boundary_${model_label}_kv${kv_size}_${timestamp}.json"

    quality="N/A"
    notes=""

    if python "$SCRIPT_DIR/eval_quality_gate.py" \
        --model "$MODEL" \
        --suite default \
        --seed "$SEED" \
        --max-tokens 200 \
        --kv-cache-type isoquant \
        --expert-offload \
        --output-json "$out_file" 2>/dev/null; then
        :
    fi

    if command -v jq &>/dev/null && [[ -f "$out_file" ]]; then
        passed=$(jq -r '[.tasks[] | select(.passed)] | length' "$out_file" 2>/dev/null || echo "?")
        total=$(jq -r '[.tasks[]] | length' "$out_file" 2>/dev/null || echo "?")
        quality="${passed}/${total}"
        failures=$(jq -r '[.tasks[] | select(.passed | not) | .name] | join(", ")' "$out_file" 2>/dev/null || echo "")
        [[ -n "$failures" ]] && notes="Failed: $failures"
    fi

    printf "%-12s  %-10s  %-40s\n" "$kv_size" "$quality" "$notes"
done

echo ""
echo "Failure boundary artifacts in $OUT_DIR/"
