#!/usr/bin/env bash
# Run the agent-memory coding evaluator across Gemma4/Qwen3 and KV modes.
#
# Default target is 24GB-class using the repo's existing 80% MLX allocator
# budgeting convention: 24GB -> 19,200 MB cap.
#
# Example:
#   scripts/run_agent_memory_kv_matrix.sh --model-kind all
#   scripts/run_agent_memory_kv_matrix.sh --model-kind qwen3 --include-no-memory

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_agent_memory_kv_matrix.sh --model-kind KIND [options]

Required:
  --model-kind KIND          One of: qwen3, gemma4, all

Optional:
  --target TARGET            One of: 24gb, none. Default: 24gb
  --output-dir DIR           Default: results/agent_memory_eval
  --memory-mode MODE         One of: scratchpad, none. Default: scratchpad
  --include-no-memory        Also run a no-memory baseline per KV mode
  --task NAME                Task name or all. Default: all
  --max-tokens N             Default: 260
  --repair-attempts N        Default: 1
  --max-run-latency-s N      Default: 300
  --max-task-latency-s N     Default: 300
  --max-step-latency-s N     Optional per-step latency cap
  --max-resident-experts N   Override model default (128)
  --max-cached-shards N      Optional shard-cache cap
  --no-predictor             Disable predictor
  --help
EOF
}

MODEL_KIND=""
TARGET="24gb"
OUTPUT_DIR="results/agent_memory_eval"
MEMORY_MODE="scratchpad"
INCLUDE_NO_MEMORY="0"
TASK_NAME="all"
MAX_TOKENS="260"
REPAIR_ATTEMPTS="1"
MAX_RUN_LATENCY_S="300"
MAX_TASK_LATENCY_S="300"
MAX_STEP_LATENCY_S=""
MAX_RESIDENT_EXPERTS=""
MAX_CACHED_SHARDS=""
USE_PREDICTOR="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-kind)
      MODEL_KIND="${2:-}"
      shift 2
      ;;
    --target)
      TARGET="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --memory-mode)
      MEMORY_MODE="${2:-}"
      shift 2
      ;;
    --include-no-memory)
      INCLUDE_NO_MEMORY="1"
      shift
      ;;
    --task)
      TASK_NAME="${2:-}"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="${2:-}"
      shift 2
      ;;
    --repair-attempts)
      REPAIR_ATTEMPTS="${2:-}"
      shift 2
      ;;
    --max-task-latency-s)
      MAX_TASK_LATENCY_S="${2:-}"
      shift 2
      ;;
    --max-run-latency-s)
      MAX_RUN_LATENCY_S="${2:-}"
      shift 2
      ;;
    --max-step-latency-s)
      MAX_STEP_LATENCY_S="${2:-}"
      shift 2
      ;;
    --max-resident-experts)
      MAX_RESIDENT_EXPERTS="${2:-}"
      shift 2
      ;;
    --max-cached-shards)
      MAX_CACHED_SHARDS="${2:-}"
      shift 2
      ;;
    --no-predictor)
      USE_PREDICTOR="0"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${MODEL_KIND}" ]]; then
  echo "Error: --model-kind is required" >&2
  usage >&2
  exit 2
fi

CAP_MB=""
case "${TARGET}" in
  24gb|24GB)
    CAP_MB="19200"
    ;;
  none)
    CAP_MB=""
    ;;
  *)
    echo "Error: unsupported --target '${TARGET}'" >&2
    exit 2
    ;;
esac

mkdir -p "${OUTPUT_DIR}"

run_cmd() {
  echo
  echo "==> $*"
  "$@"
}

run_one() {
  local prefix="$1"
  local model_path="$2"
  local kv="$3"
  local memory_mode="$4"
  local max_resident="$5"
  local out="${OUTPUT_DIR}/${prefix}_${kv}_${memory_mode}.json"

  local args=(
    python3 scripts/agent_memory_coding_eval.py
    --model "${model_path}"
    --task "${TASK_NAME}"
    --expert-offload
    --max-resident-experts "${max_resident}"
    --max-tokens "${MAX_TOKENS}"
    --repair-attempts "${REPAIR_ATTEMPTS}"
    --max-run-latency-s "${MAX_RUN_LATENCY_S}"
    --max-task-latency-s "${MAX_TASK_LATENCY_S}"
    --kv-cache-type "${kv}"
    --memory-mode "${memory_mode}"
    --output-json "${out}"
  )

  if [[ -n "${MAX_STEP_LATENCY_S}" ]]; then
    args+=(--max-step-latency-s "${MAX_STEP_LATENCY_S}")
  fi
  if [[ -n "${MAX_CACHED_SHARDS}" ]]; then
    args+=(--max-cached-shards "${MAX_CACHED_SHARDS}")
  fi
  if [[ -n "${CAP_MB}" ]]; then
    args+=(--memory-limit-mb "${CAP_MB}")
  fi
  if [[ "${USE_PREDICTOR}" == "1" ]]; then
    args+=(--use-predictor)
  fi

  run_cmd "${args[@]}"
}

run_model() {
  local prefix="$1"
  local model_path="$2"
  local max_resident="$3"

  for kv in turboquant isoquant; do
    run_one "${prefix}" "${model_path}" "${kv}" "${MEMORY_MODE}" "${max_resident}"
    if [[ "${INCLUDE_NO_MEMORY}" == "1" && "${MEMORY_MODE}" != "none" ]]; then
      run_one "${prefix}" "${model_path}" "${kv}" "none" "${max_resident}"
    fi
  done
}

echo "=== Agent Memory KV Matrix ==="
echo "model_kind=${MODEL_KIND}"
echo "target=${TARGET}"
echo "output_dir=${OUTPUT_DIR}"
echo "memory_mode=${MEMORY_MODE}"
echo "include_no_memory=${INCLUDE_NO_MEMORY}"
echo "task=${TASK_NAME}"
echo "max_tokens=${MAX_TOKENS}"
echo "repair_attempts=${REPAIR_ATTEMPTS}"
echo "max_run_latency_s=${MAX_RUN_LATENCY_S}"
echo "max_task_latency_s=${MAX_TASK_LATENCY_S}"
if [[ -n "${MAX_STEP_LATENCY_S}" ]]; then
  echo "max_step_latency_s=${MAX_STEP_LATENCY_S}"
fi
if [[ -n "${CAP_MB}" ]]; then
  echo "memory_limit_mb=${CAP_MB}"
fi

case "${MODEL_KIND}" in
  qwen3)
    run_model "qwen3" "${QWEN3_MODEL:-mlx-community/Qwen3-30B-A3B-4bit}" "${MAX_RESIDENT_EXPERTS:-128}"
    ;;
  gemma4)
    run_model "gemma4" "${GEMMA4_MODEL:-gemma-4-26b-a4b-it-4bit}" "${MAX_RESIDENT_EXPERTS:-128}"
    ;;
  all)
    run_model "qwen3" "${QWEN3_MODEL:-mlx-community/Qwen3-30B-A3B-4bit}" "${MAX_RESIDENT_EXPERTS:-128}"
    run_model "gemma4" "${GEMMA4_MODEL:-gemma-4-26b-a4b-it-4bit}" "${MAX_RESIDENT_EXPERTS:-128}"
    ;;
  *)
    echo "Error: unsupported --model-kind '${MODEL_KIND}'" >&2
    exit 2
    ;;
esac

echo
echo "Artifacts written under ${OUTPUT_DIR}"
