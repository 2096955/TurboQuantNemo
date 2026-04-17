#!/usr/bin/env bash
# Failure boundary test: progressively reduce --memory-limit-mb until the
# system fails, documenting HOW it fails (graceful error / Metal crash /
# OS OOM kill / silent corruption).
#
# Note: a "KV cache size" sweep would require eval_quality_gate.py to expose
# a --max-kv-size flag. It currently does not, so that test is omitted. To
# vary KV behaviour, use scripts/run_ablation_study.sh which toggles
# --kv-cache-type {default,isoquant} at a fixed memory cap.
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

# Per-run scratch dir auto-cleaned via trap so stderr files don't leak on abort
TMPDIR_RUN="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_RUN"' EXIT

model_label="$(basename "$MODEL")"
timestamp="$(date +%Y%m%d_%H%M%S)"

echo "=== Failure Boundary Test: $model_label ==="
echo "    Reducing memory cap until failure..."
echo ""

# Start high and reduce. Typical working point is ~5-6GB for Gemma4.
MEMORY_CAPS=(8192 6144 5120 4096 3072 2048)

printf "%-12s  %-8s  %-12s  %-40s\n" "Cap (MB)" "Result" "Peak (MB)" "Failure Mode"
printf "%-12s  %-8s  %-12s  %-40s\n" "------------" "--------" "------------" "----------------------------------------"

for cap in "${MEMORY_CAPS[@]}"; do
    out_file="$OUT_DIR/mem_boundary_${model_label}_${cap}mb_${timestamp}.json"
    stderr_file="$TMPDIR_RUN/stderr_${cap}"

    # 100 tokens is enough for the micro suite prompts to pass quality when
    # memory permits — keeps "memory failed" distinct from "tokens too few".
    set +e
    python "$SCRIPT_DIR/eval_quality_gate.py" \
        --model "$MODEL" \
        --suite micro \
        --seed "$SEED" \
        --max-tokens 100 \
        --kv-cache-type isoquant \
        --expert-offload \
        --memory-limit-mb "$cap" \
        --output-json "$out_file" 2>"$stderr_file"
    exit_code=$?
    set -e

    peak="N/A"
    fits_cap="unknown"
    if command -v jq &>/dev/null && [[ -f "$out_file" ]]; then
        peak=$(jq -r '.memory.peak_mb // "N/A"' "$out_file" 2>/dev/null || echo "N/A")
        # `// "unknown"` would coerce boolean false -> "unknown"; use tostring.
        fits_cap=$(jq -r '.memory.fits_cap | tostring' "$out_file" 2>/dev/null || echo "unknown")
    fi

    # Distinguish three outcomes from exit_code + JSON:
    #   exit 0                  -> PASS
    #   exit !=0 and fits_cap=false -> OOM (the boundary we're looking for)
    #   exit !=0 and fits_cap=true  -> non-memory failure (quality, etc.)
    if [[ $exit_code -eq 0 ]]; then
        result="PASS"
        failure_mode="none"
    elif [[ "$fits_cap" == "false" ]]; then
        result="OOM"
        failure_mode="Memory cap exceeded (peak ${peak} > cap ${cap})"
        mv "$stderr_file" "$OUT_DIR/mem_boundary_stderr_${cap}mb_${timestamp}.txt"
    else
        result="QUAL"
        if grep -qi "out of memory\|MemoryError\|killed\|signal SIGKILL\|abort()" "$stderr_file" 2>/dev/null; then
            # OOM via OS killer wouldn't write the JSON, so this catches the
            # rare case where the process partially wrote then crashed.
            result="CRASH"
            failure_mode="Process crash (OS killer or fatal error)"
        else
            failure_mode="Quality gate prompts failed (memory was fine)"
        fi
        mv "$stderr_file" "$OUT_DIR/mem_boundary_stderr_${cap}mb_${timestamp}.txt"
    fi

    printf "%-12s  %-8s  %-12s  %-40s\n" "${cap}" "$result" "$peak" "$failure_mode"

    # Stop only on memory-related failure — keep going through cap reductions
    # if quality is the issue (that's a different test).
    if [[ "$result" == "OOM" || "$result" == "CRASH" ]]; then
        echo ""
        echo "  First memory failure at ${cap} MB cap. Boundary identified."
        break
    fi
done

echo ""
echo "Failure boundary artifacts in $OUT_DIR/"
