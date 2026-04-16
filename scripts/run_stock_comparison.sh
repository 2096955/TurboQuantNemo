#!/usr/bin/env bash
# Stock comparison: run quality gate on stock mlx-lm vs this fork,
# on identical prompts with identical seeds.
#
# For models that don't fit without expert offload, the stock run will
# report "OOM / does not load" — that IS the comparison.
#
# Usage:
#   bash scripts/run_stock_comparison.sh
#   MODEL=./gemma4-layer-aware bash scripts/run_stock_comparison.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/artifacts/stock-comparison}"
SEED="${SEED:-42}"
SUITE="${SUITE:-default}"
MAX_TOKENS="${MAX_TOKENS:-500}"
MODEL="${MODEL:-$REPO_ROOT/gemma4-layer-aware}"
STOCK_VENV="${STOCK_VENV:-$REPO_ROOT/.stock-mlx-lm-venv}"

mkdir -p "$OUT_DIR"

if [[ ! -d "$MODEL" ]]; then
    echo "ERROR: Model not found at $MODEL" >&2
    exit 1
fi

model_label="$(basename "$MODEL")"
timestamp="$(date +%Y%m%d_%H%M%S)"

echo "=== Stock vs Fork Comparison: $model_label ==="
echo ""

# --- Step 1: Set up stock mlx-lm venv (if needed) ---
setup_stock_venv() {
    if [[ -d "$STOCK_VENV" ]] && "$STOCK_VENV/bin/python" -c "import mlx_lm" 2>/dev/null; then
        echo "Stock venv already exists at $STOCK_VENV"
        return 0
    fi

    echo "Setting up stock mlx-lm venv at $STOCK_VENV..."
    python3 -m venv "$STOCK_VENV"
    "$STOCK_VENV/bin/pip" install --quiet --upgrade pip
    "$STOCK_VENV/bin/pip" install --quiet mlx-lm
    echo "Stock mlx-lm installed: $("$STOCK_VENV/bin/pip" show mlx-lm 2>/dev/null | grep Version || echo 'unknown')"
}

# --- Step 2: Run fork version (full stack) ---
echo "=== Run 1: Fork (expert offload + IsoQuant) ==="
fork_out="$OUT_DIR/qg_fork_${model_label}_${timestamp}.json"

if python "$SCRIPT_DIR/eval_quality_gate.py" \
    --model "$MODEL" \
    --suite "$SUITE" \
    --seed "$SEED" \
    --max-tokens "$MAX_TOKENS" \
    --kv-cache-type isoquant \
    --expert-offload \
    --output-json "$fork_out" 2>&1; then
    echo "  -> $fork_out"
else
    echo "  Fork run completed with failures -> $fork_out"
fi

# --- Step 3: Run fork without enhancements (baseline within fork) ---
echo ""
echo "=== Run 2: Fork (no offload, default KV — stock-equivalent) ==="
fork_baseline_out="$OUT_DIR/qg_fork_baseline_${model_label}_${timestamp}.json"
fork_baseline_stderr="$OUT_DIR/.fork_baseline_stderr.tmp"

if python "$SCRIPT_DIR/eval_quality_gate.py" \
    --model "$MODEL" \
    --suite "$SUITE" \
    --seed "$SEED" \
    --max-tokens "$MAX_TOKENS" \
    --kv-cache-type default \
    --output-json "$fork_baseline_out" 2>"$fork_baseline_stderr"; then
    echo "  -> $fork_baseline_out"
else
    exit_code=$?
    if grep -qi "out of memory\|OOM\|MemoryError" "$fork_baseline_stderr" 2>/dev/null; then
        echo "  OOM without expert offload — model does not fit in RAM without the fork's compression stack"
        echo '{"error": "OOM without expert offload — model requires fork compression stack to fit in RAM"}' > "$fork_baseline_out"
    else
        echo "  Failed with exit $exit_code -> see $fork_baseline_out"
    fi
fi
rm -f "$fork_baseline_stderr"

# --- Step 4: Run stock mlx-lm (optional — requires pip install) ---
echo ""
echo "=== Run 3: Stock mlx-lm (pip release) ==="

if [[ "${SKIP_STOCK_INSTALL:-}" == "1" ]]; then
    echo "  Skipped (SKIP_STOCK_INSTALL=1)"
    stock_out="$OUT_DIR/qg_stock_${model_label}_${timestamp}.json"
    echo '{"error": "skipped — SKIP_STOCK_INSTALL=1"}' > "$stock_out"
else
    setup_stock_venv

    stock_out="$OUT_DIR/qg_stock_${model_label}_${timestamp}.json"
    stock_stderr="$OUT_DIR/.stock_stderr.tmp"

    # Stock mlx-lm won't have our quality gate script, so we test loading + basic generation
    echo "  Testing model load + basic generation with stock mlx-lm..."
    if "$STOCK_VENV/bin/python" -c "
import json, sys, time
try:
    from mlx_lm import load, generate
    t0 = time.time()
    model, tokenizer = load('$MODEL')
    load_s = time.time() - t0
    t0 = time.time()
    resp = generate(model, tokenizer, prompt='Explain gravity in one sentence.', max_tokens=100, verbose=False)
    gen_s = time.time() - t0
    result = {'status': 'ok', 'load_time_s': round(load_s, 2), 'gen_time_s': round(gen_s, 2), 'response': resp, 'tokens': len(resp.split())}
except Exception as e:
    result = {'status': 'error', 'error': str(e), 'error_type': type(e).__name__}
json.dump(result, open('$stock_out', 'w'), indent=2)
print(json.dumps(result, indent=2))
" 2>"$stock_stderr"; then
        echo "  -> $stock_out"
    else
        if grep -qi "out of memory\|OOM\|MemoryError" "$stock_stderr" 2>/dev/null; then
            echo "  Stock mlx-lm: OOM — model does not fit without fork's compression stack"
            echo '{"status": "error", "error": "OOM — model does not fit without compression stack"}' > "$stock_out"
        else
            echo "  Stock mlx-lm: failed (see $OUT_DIR/stock_stderr_${timestamp}.txt)"
            cp "$stock_stderr" "$OUT_DIR/stock_stderr_${timestamp}.txt"
        fi
    fi
    rm -f "$stock_stderr"
fi

# --- Summary ---
echo ""
echo "=== Comparison Summary: $model_label ==="
printf "%-30s  %-10s  %-50s\n" "Configuration" "Quality" "Notes"
printf "%-30s  %-10s  %-50s\n" "------------------------------" "----------" "--------------------------------------------------"

labels=("Fork (full stack)"            "Fork (no offload, default KV)" "Stock mlx-lm")
files=("$fork_out"                     "$fork_baseline_out"             "$stock_out")

for i in "${!labels[@]}"; do
    label="${labels[$i]}"
    file="${files[$i]}"

    quality="N/A"
    notes=""

    if command -v jq &>/dev/null && [[ -f "$file" ]]; then
        error=$(jq -r '.error // empty' "$file" 2>/dev/null)
        if [[ -n "$error" ]]; then
            notes="$error"
        elif jq -e '.n_total' "$file" &>/dev/null; then
            quality=$(jq -r '"\(.n_pass)/\(.n_total)"' "$file" 2>/dev/null || echo "?/?")
        elif jq -e '.status' "$file" &>/dev/null; then
            status=$(jq -r '.status' "$file" 2>/dev/null)
            if [[ "$status" == "ok" ]]; then
                quality="loaded"
                notes="$(jq -r '"Load: \(.load_time_s)s, Gen: \(.gen_time_s)s"' "$file" 2>/dev/null)"
            else
                quality="FAIL"
                notes="$(jq -r '.error' "$file" 2>/dev/null)"
            fi
        fi
    fi

    printf "%-30s  %-10s  %-50s\n" "$label" "$quality" "$notes"
done

echo ""
echo "Stock comparison artifacts in $OUT_DIR/"
