#!/usr/bin/env bash
# Run the IsoQuant submission bundle under a synthetic memory cap.
#
# Uses mx.set_memory_limit() via a wrapper that injects the cap before
# model loading. This simulates 16GB / 32GB hardware on a larger host.
#
# The cap is applied to the MLX Metal allocator. It does not limit
# Python heap or OS-level allocations, so it's a lower bound — real
# 16GB/32GB hardware may have tighter constraints from OS pressure.
# Still much better than "measured on 128GB host" with no cap.
#
# Usage:
#   scripts/run_memory_capped_validation.sh \
#     --target 16gb --model-kind gemma4 --model gemma-4-26b-a4b-it-4bit
#
#   scripts/run_memory_capped_validation.sh \
#     --target 32gb --model-kind nemotron120b
#
# This writes artifacts to results/capped_{target}/{prefix}_*.json
# and records system snapshots (RSS, swap, pageouts) at each stage.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_memory_capped_validation.sh --target TARGET --model-kind KIND [options]

Required:
  --target TARGET        One of: 16gb, 32gb
  --model-kind KIND      One of: qwen3, gemma4, nemotron120b

Optional:
  --model PATH           Model directory (falls back to env: QWEN3_MODEL, GEMMA4_MODEL, NEMOTRON_MODEL)
  --output-dir DIR       Default: results/capped_{target}
  --skip-quality         Skip quality gate
  --skip-stress          Skip stress tests
  --skip-longctx         Skip long-context eval
  --skip-benchmark       Skip benchmark
  --help
EOF
}

TARGET=""
MODEL_KIND=""
MODEL_PATH=""
OUTPUT_DIR=""
RUN_QUALITY="1"
RUN_STRESS="1"
RUN_LONGCTX="1"
RUN_BENCHMARK="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="${2:-}"; shift 2 ;;
    --model-kind) MODEL_KIND="${2:-}"; shift 2 ;;
    --model) MODEL_PATH="${2:-}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:-}"; shift 2 ;;
    --skip-quality) RUN_QUALITY="0"; shift ;;
    --skip-stress) RUN_STRESS="0"; shift ;;
    --skip-longctx) RUN_LONGCTX="0"; shift ;;
    --skip-benchmark) RUN_BENCHMARK="0"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${TARGET}" || -z "${MODEL_KIND}" ]]; then
  echo "Error: --target and --model-kind are required" >&2
  usage >&2
  exit 2
fi

# ── Resolve memory cap ──
CAP_BYTES=""
CAP_MB=""
case "${TARGET}" in
  16gb|16GB)
    # 80% of 16GB = 12.8GB for MLX Metal allocator
    # Reserve ~3.2GB for OS, Python, and system overhead
    CAP_BYTES=$((12800 * 1024 * 1024))
    CAP_MB="12800"
    ;;
  32gb|32GB)
    # 80% of 32GB = 25.6GB
    CAP_BYTES=$((25600 * 1024 * 1024))
    CAP_MB="25600"
    ;;
  *)
    echo "Error: unsupported --target '${TARGET}'. Use 16gb or 32gb." >&2
    exit 2
    ;;
esac

# ── Resolve model ──
MAX_RESIDENT=""
PREFIX=""
case "${MODEL_KIND}" in
  qwen3)
    PREFIX="qwen3"
    MODEL_PATH="${MODEL_PATH:-${QWEN3_MODEL:-mlx-community/Qwen3-30B-A3B-4bit}}"
    MAX_RESIDENT="128"
    ;;
  gemma4)
    PREFIX="gemma4"
    MODEL_PATH="${MODEL_PATH:-${GEMMA4_MODEL:-gemma-4-26b-a4b-it-4bit}}"
    MAX_RESIDENT="128"
    ;;
  nemotron120b|nemotron)
    PREFIX="nemotron120b"
    MODEL_PATH="${MODEL_PATH:-${NEMOTRON_MODEL:-sjug/Nemotron-3-Super-120B-A12B-MLX-4bit}}"
    MAX_RESIDENT="4096"
    ;;
  *)
    echo "Error: unsupported --model-kind '${MODEL_KIND}'" >&2
    exit 2
    ;;
esac

# ── Resolve shard cache ──
MAX_CACHED_SHARDS=""
case "${TARGET}" in
  16gb|16GB) MAX_CACHED_SHARDS="1" ;;
  32gb|32GB) MAX_CACHED_SHARDS="2" ;;
esac

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="results/capped_${TARGET}"
fi
mkdir -p "${OUTPUT_DIR}"

echo "=== Memory-Capped Validation ==="
echo "target=${TARGET} (${CAP_MB} MB MLX cap)"
echo "model_kind=${MODEL_KIND}"
echo "model_path=${MODEL_PATH}"
echo "max_resident=${MAX_RESIDENT}"
echo "output_dir=${OUTPUT_DIR}"
echo ""

# ── Helper: run a Python script with memory cap injected ──
run_capped() {
  local script="$1"
  shift
  python3 -c "
import mlx.core as mx
mx.set_memory_limit(${CAP_BYTES})
print(f'MLX memory limit set to ${CAP_MB} MB')
import runpy, sys
sys.argv = ['${script}'] + sys.argv[1:]
runpy.run_path('${script}', run_name='__main__')
" "$@"
}

# ── System snapshot ──
snapshot() {
  local label="$1"
  local ts
  ts=$(date +%Y%m%d_%H%M%S)
  {
    echo "=== ${label} at ${ts} ==="
    echo "--- ps rss ---"
    ps -o pid,rss,vsz -p $$ 2>/dev/null || true
    echo "--- vm_stat ---"
    vm_stat 2>/dev/null || true
    echo "--- swapusage ---"
    sysctl vm.swapusage 2>/dev/null || true
  } >> "${OUTPUT_DIR}/system_snapshots.txt"
}

snapshot "start"

# ── Quality gate (default + isoquant) ──
if [[ "${RUN_QUALITY}" == "1" ]]; then
  for kv in default isoquant; do
    echo ""
    echo "==> Quality gate: ${kv}"
    run_capped scripts/eval_quality_gate.py \
      --model "${MODEL_PATH}" \
      --expert-offload \
      --max-resident-experts "${MAX_RESIDENT}" \
      --max-cached-shards "${MAX_CACHED_SHARDS}" \
      --seed 42 --temp 0.0 \
      --max-tokens 500 \
      --kv-cache-type "${kv}" \
      --output-json "${OUTPUT_DIR}/${PREFIX}_${kv}_500tok_quality.json" \
      || echo "WARNING: quality gate ${kv} failed"
    snapshot "quality_${kv}"
  done
fi

# ── Stress tests (default + isoquant) ──
if [[ "${RUN_STRESS}" == "1" ]]; then
  for kv in default isoquant; do
    echo ""
    echo "==> Stress tests: ${kv}"
    run_capped scripts/moe_stress_tests.py \
      --model "${MODEL_PATH}" \
      --expert-offload \
      --max-resident-experts "${MAX_RESIDENT}" \
      --seed 42 --temp 0.0 \
      --test all \
      --max-tokens 200 \
      --kv-cache-type "${kv}" \
      --output-json "${OUTPUT_DIR}/${PREFIX}_stress_${kv}.json" \
      || echo "WARNING: stress tests ${kv} failed"
    snapshot "stress_${kv}"
  done
fi

# ── Long-context eval (default + isoquant) ──
if [[ "${RUN_LONGCTX}" == "1" ]]; then
  for kv in default isoquant; do
    echo ""
    echo "==> Long-context eval: ${kv}"
    run_capped scripts/long_context_kv_eval.py \
      --model "${MODEL_PATH}" \
      --expert-offload \
      --max-resident-experts "${MAX_RESIDENT}" \
      --seed 42 --temp 0.0 \
      --test all \
      --max-tokens 200 \
      --kv-cache-type "${kv}" \
      --output-json "${OUTPUT_DIR}/${PREFIX}_longctx_${kv}.json" \
      || echo "WARNING: long-context eval ${kv} failed"
    snapshot "longctx_${kv}"
  done
fi

# ── Benchmark (default + isoquant) ──
if [[ "${RUN_BENCHMARK}" == "1" ]]; then
  PROFILE="A"
  if [[ "${MODEL_KIND}" == "nemotron120b" || "${MODEL_KIND}" == "nemotron" ]]; then
    PROFILE="B"
  fi
  for kv in default isoquant; do
    echo ""
    echo "==> Benchmark: ${kv}"
    run_capped scripts/benchmark_moe_offload.py \
      --model "${MODEL_PATH}" \
      --expert-offload \
      --max-resident-experts "${MAX_RESIDENT}" \
      --profile "${PROFILE}" \
      --seed 42 \
      --kv-cache-type "${kv}" \
      --target-envelope-mb "${CAP_MB}" \
      --split-decode-timing \
      --json-output "${OUTPUT_DIR}/${PREFIX}_${kv}_bench.json" \
      || echo "WARNING: benchmark ${kv} failed"
    snapshot "bench_${kv}"
  done
fi

snapshot "end"

echo ""
echo "=== Memory-Capped Validation Complete ==="
echo "Artifacts in: ${OUTPUT_DIR}"
echo "System snapshots: ${OUTPUT_DIR}/system_snapshots.txt"
echo ""
echo "To summarize:"
echo "  python scripts/summarize_submission_results.py --input-dir ${OUTPUT_DIR}"
