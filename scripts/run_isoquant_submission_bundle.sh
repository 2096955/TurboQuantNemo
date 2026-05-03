#!/usr/bin/env bash
# Generate the pinned IsoQuant-vs-baseline artifact bundle for one pathway model.
#
# This emits the naming scheme referenced in:
#   results/isoquant_submission/
#
# Coverage in this runner:
# - per-KV benchmark triad: default / turboquant / isoquant
# - per-KV quality triad
# - paired turboquant-vs-isoquant benchmark JSON
# - KV fidelity JSON
# - MoE stress suite JSON per KV mode
# - optional separate full-stack IsoQuant benchmark + quality JSON
#
# Long-context/state-retention is included via scripts/long_context_kv_eval.py.
# Soak should remain an explicitly human-triggered follow-up once the shorter
# artifact bundle passes.
#
# Usage:
#   scripts/run_isoquant_submission_bundle.sh \
#     --model-kind nemotron120b \
#     --model /path/to/model
#
# Environment fallbacks:
#   qwen3       -> $QWEN3_MODEL
#   gemma4      -> $GEMMA4_MODEL
#   nemotron120b -> $NEMOTRON_MODEL

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_isoquant_submission_bundle.sh --model-kind KIND [--model PATH] [options]

Required:
  --model-kind KIND          One of: qwen3, gemma4, nemotron120b

Model path:
  --model PATH               Local MLX model directory

Options:
  --output-dir DIR           Default: results/isoquant_submission
  --max-resident-experts N   Default: 128 for qwen3/gemma4, 48 for nemotron120b
  --task-expert-cliques-file PATH
                             Optional task clique JSON for full-stack and other runs
  --seed N                   Default: 42
  --temp FLOAT               Default: 0.0
  --turboquant-bits N        Default: 3
  --quality-suite NAME       Default: default
  --quality-harness-version V
                            Default: v2
  --quality-max-tokens N     Default: 500
  --quality-memory-limit-mb N
                            Optional mx.set_memory_limit cap for quality runs
  --fidelity-max-tokens N    Default: 256
  --stress-max-tokens N      Default: 200
  --fullstack-suite NAME     Default: all
  --fullstack-harness-version V
                            Default: same as --quality-harness-version
  --fullstack-max-tokens N   Default: 500
  --fullstack-memory-limit-mb N
                            Optional mx.set_memory_limit cap for full-stack quality run
  --prefill-step-size N      Default: 64
  --no-predictor             Disable predictor on runs that support it
  --no-dedekimi-observer     Disable DedeKimi observer on runs that support it
  --skip-benchmarks
  --skip-quality
  --skip-pair
  --skip-fidelity
  --skip-stress
  --skip-fullstack
  --skip-longctx
  --longctx-max-tokens N     Default: 200
  --help                     Show this message

Examples:
  scripts/run_isoquant_submission_bundle.sh \
    --model-kind qwen3 \
    --model "$QWEN3_MODEL"

  scripts/run_isoquant_submission_bundle.sh \
    --model-kind nemotron120b \
    --model "$NEMOTRON_MODEL" \
    --task-expert-cliques-file results/gemma4_task_expert_cliques_min.json
EOF
}

MODEL_KIND=""
MODEL_PATH=""
OUTPUT_DIR="results/isoquant_submission"
MAX_RESIDENT_EXPERTS=""
TASK_CLIQUES_FILE=""
SEED="42"
TEMP="0.0"
TURBOQUANT_BITS="3"
QUALITY_SUITE="default"
QUALITY_HARNESS_VERSION="v2"
QUALITY_MAX_TOKENS="500"
QUALITY_MEMORY_LIMIT_MB=""
FIDELITY_MAX_TOKENS="256"
STRESS_MAX_TOKENS="200"
FULLSTACK_SUITE="all"
FULLSTACK_HARNESS_VERSION=""
FULLSTACK_MAX_TOKENS="500"
FULLSTACK_MEMORY_LIMIT_MB=""
PREFILL_STEP_SIZE="64"
USE_PREDICTOR="1"
USE_DEDEKIMI="1"
RUN_BENCHMARKS="1"
RUN_QUALITY="1"
RUN_PAIR="1"
RUN_FIDELITY="1"
RUN_STRESS="1"
RUN_FULLSTACK="1"
RUN_LONGCTX="1"
LONGCTX_MAX_TOKENS="200"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-kind)
      MODEL_KIND="${2:-}"
      shift 2
      ;;
    --model)
      MODEL_PATH="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --max-resident-experts)
      MAX_RESIDENT_EXPERTS="${2:-}"
      shift 2
      ;;
    --task-expert-cliques-file)
      TASK_CLIQUES_FILE="${2:-}"
      shift 2
      ;;
    --seed)
      SEED="${2:-}"
      shift 2
      ;;
    --temp)
      TEMP="${2:-}"
      shift 2
      ;;
    --turboquant-bits)
      TURBOQUANT_BITS="${2:-}"
      shift 2
      ;;
    --quality-suite)
      QUALITY_SUITE="${2:-}"
      shift 2
      ;;
    --quality-harness-version)
      QUALITY_HARNESS_VERSION="${2:-}"
      shift 2
      ;;
    --quality-max-tokens)
      QUALITY_MAX_TOKENS="${2:-}"
      shift 2
      ;;
    --quality-memory-limit-mb)
      QUALITY_MEMORY_LIMIT_MB="${2:-}"
      shift 2
      ;;
    --fidelity-max-tokens)
      FIDELITY_MAX_TOKENS="${2:-}"
      shift 2
      ;;
    --stress-max-tokens)
      STRESS_MAX_TOKENS="${2:-}"
      shift 2
      ;;
    --fullstack-suite)
      FULLSTACK_SUITE="${2:-}"
      shift 2
      ;;
    --fullstack-harness-version)
      FULLSTACK_HARNESS_VERSION="${2:-}"
      shift 2
      ;;
    --fullstack-max-tokens)
      FULLSTACK_MAX_TOKENS="${2:-}"
      shift 2
      ;;
    --fullstack-memory-limit-mb)
      FULLSTACK_MEMORY_LIMIT_MB="${2:-}"
      shift 2
      ;;
    --prefill-step-size)
      PREFILL_STEP_SIZE="${2:-}"
      shift 2
      ;;
    --no-predictor)
      USE_PREDICTOR="0"
      shift
      ;;
    --no-dedekimi-observer)
      USE_DEDEKIMI="0"
      shift
      ;;
    --skip-benchmarks)
      RUN_BENCHMARKS="0"
      shift
      ;;
    --skip-quality)
      RUN_QUALITY="0"
      shift
      ;;
    --skip-pair)
      RUN_PAIR="0"
      shift
      ;;
    --skip-fidelity)
      RUN_FIDELITY="0"
      shift
      ;;
    --skip-stress)
      RUN_STRESS="0"
      shift
      ;;
    --skip-fullstack)
      RUN_FULLSTACK="0"
      shift
      ;;
    --skip-longctx)
      RUN_LONGCTX="0"
      shift
      ;;
    --longctx-max-tokens)
      LONGCTX_MAX_TOKENS="${2:-}"
      shift 2
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

PROFILE=""
PROFILE_TAG=""
PREFIX=""
ENV_MODEL=""
case "${MODEL_KIND}" in
  qwen3)
    PROFILE="A"
    PROFILE_TAG="profileA"
    PREFIX="qwen3"
    ENV_MODEL="${QWEN3_MODEL:-}"
    if [[ -z "${MAX_RESIDENT_EXPERTS}" ]]; then
      MAX_RESIDENT_EXPERTS="128"
    fi
    ;;
  gemma4)
    PROFILE="A"
    PROFILE_TAG="profileA"
    PREFIX="gemma4"
    ENV_MODEL="${GEMMA4_MODEL:-}"
    if [[ -z "${MAX_RESIDENT_EXPERTS}" ]]; then
      MAX_RESIDENT_EXPERTS="128"
    fi
    ;;
  nemotron120b|nemotron)
    PROFILE="B"
    PROFILE_TAG="profileB"
    PREFIX="nemotron120b"
    ENV_MODEL="${NEMOTRON_MODEL:-}"
    if [[ -z "${MAX_RESIDENT_EXPERTS}" ]]; then
      MAX_RESIDENT_EXPERTS="48"
    fi
    ;;
  *)
    echo "Error: unsupported --model-kind '${MODEL_KIND}'" >&2
    exit 2
    ;;
esac

if [[ -z "${MODEL_PATH}" ]]; then
  MODEL_PATH="${ENV_MODEL}"
fi

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Error: model path is required (--model or model-specific env var)" >&2
  exit 2
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Error: model path does not exist: ${MODEL_PATH}" >&2
  exit 2
fi

if [[ -n "${TASK_CLIQUES_FILE}" && ! -f "${TASK_CLIQUES_FILE}" ]]; then
  echo "Error: task clique file not found: ${TASK_CLIQUES_FILE}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
export TURBOQUANT_BITS

if [[ -z "${FULLSTACK_HARNESS_VERSION}" ]]; then
  FULLSTACK_HARNESS_VERSION="${QUALITY_HARNESS_VERSION}"
fi

COMMON_MODEL_ARGS=(
  --model "${MODEL_PATH}"
  --expert-offload
  --max-resident-experts "${MAX_RESIDENT_EXPERTS}"
)

PREDICTOR_ARGS=()
if [[ "${USE_PREDICTOR}" == "1" ]]; then
  PREDICTOR_ARGS+=(--use-predictor)
fi

DEDEKIMI_ARGS=()
if [[ "${USE_DEDEKIMI}" == "1" ]]; then
  DEDEKIMI_ARGS+=(--use-dedekimi-observer)
fi

CLIQUE_ARGS=()
if [[ -n "${TASK_CLIQUES_FILE}" ]]; then
  CLIQUE_ARGS+=(--task-expert-cliques-file "${TASK_CLIQUES_FILE}")
fi

QUALITY_HARNESS_ARGS=(--harness-version "${QUALITY_HARNESS_VERSION}")
if [[ -n "${QUALITY_MEMORY_LIMIT_MB}" ]]; then
  QUALITY_HARNESS_ARGS+=(--memory-limit-mb "${QUALITY_MEMORY_LIMIT_MB}")
fi

FULLSTACK_HARNESS_ARGS=(--harness-version "${FULLSTACK_HARNESS_VERSION}")
if [[ -n "${FULLSTACK_MEMORY_LIMIT_MB}" ]]; then
  FULLSTACK_HARNESS_ARGS+=(--memory-limit-mb "${FULLSTACK_MEMORY_LIMIT_MB}")
fi

run_cmd() {
  echo
  echo "==> $*"
  "$@"
}

run_benchmark_one() {
  local kv="$1"
  local out="${OUTPUT_DIR}/${PREFIX}_${kv}_${PROFILE_TAG}_bench.json"
  run_cmd python3 scripts/benchmark_moe_offload.py \
    "${COMMON_MODEL_ARGS[@]}" \
    --profile "${PROFILE}" \
    --seed "${SEED}" \
    --kv-cache-type "${kv}" \
    --turboquant-bits "${TURBOQUANT_BITS}" \
    --prefill-step-size "${PREFILL_STEP_SIZE}" \
    --split-decode-timing \
    --warm-second-pass \
    "${PREDICTOR_ARGS[@]}" \
    "${DEDEKIMI_ARGS[@]}" \
    "${CLIQUE_ARGS[@]}" \
    --json-output "${out}"
}

run_quality_one() {
  local kv="$1"
  local out="${OUTPUT_DIR}/${PREFIX}_${kv}_${QUALITY_MAX_TOKENS}tok_quality.json"
  run_cmd python3 scripts/eval_quality_gate.py \
    "${COMMON_MODEL_ARGS[@]}" \
    --seed "${SEED}" \
    --temp "${TEMP}" \
    --suite "${QUALITY_SUITE}" \
    --max-tokens "${QUALITY_MAX_TOKENS}" \
    --kv-cache-type "${kv}" \
    "${QUALITY_HARNESS_ARGS[@]}" \
    "${PREDICTOR_ARGS[@]}" \
    "${DEDEKIMI_ARGS[@]}" \
    "${CLIQUE_ARGS[@]}" \
    --output-json "${out}"
}

run_pair() {
  local out="${OUTPUT_DIR}/${PREFIX}_turbo_vs_iso_pair.json"
  run_cmd python3 scripts/compare_pathway_kv_modes.py \
    "${COMMON_MODEL_ARGS[@]}" \
    --profile "${PROFILE}" \
    --seed "${SEED}" \
    --turboquant-bits "${TURBOQUANT_BITS}" \
    "${PREDICTOR_ARGS[@]}" \
    "${DEDEKIMI_ARGS[@]}" \
    "${CLIQUE_ARGS[@]}" \
    --output "${out}"
}

run_fidelity() {
  local out="${OUTPUT_DIR}/${PREFIX}_kv_fidelity.json"
  run_cmd python3 scripts/measure_kv_fidelity.py \
    --model "${MODEL_PATH}" \
    --expert-offload \
    --max-resident-experts "${MAX_RESIDENT_EXPERTS}" \
    --max-tokens "${FIDELITY_MAX_TOKENS}" \
    --output-json "${out}"
}

run_stress_one() {
  local kv="$1"
  local out="${OUTPUT_DIR}/${PREFIX}_stress_${kv}.json"
  run_cmd python3 scripts/moe_stress_tests.py \
    "${COMMON_MODEL_ARGS[@]}" \
    --seed "${SEED}" \
    --temp "${TEMP}" \
    --test all \
    --max-tokens "${STRESS_MAX_TOKENS}" \
    --kv-cache-type "${kv}" \
    "${PREDICTOR_ARGS[@]}" \
    "${DEDEKIMI_ARGS[@]}" \
    --output-json "${out}"
}

run_fullstack_isoquant() {
  local bench_out="${OUTPUT_DIR}/${PREFIX}_fullstack_isoquant_benchmark.json"
  local qual_out="${OUTPUT_DIR}/${PREFIX}_fullstack_isoquant_quality.json"
  run_cmd python3 scripts/benchmark_moe_offload.py \
    "${COMMON_MODEL_ARGS[@]}" \
    --profile "${PROFILE}" \
    --seed "${SEED}" \
    --kv-cache-type isoquant \
    --turboquant-bits "${TURBOQUANT_BITS}" \
    --prefill-step-size "${PREFILL_STEP_SIZE}" \
    --split-decode-timing \
    --warm-second-pass \
    "${PREDICTOR_ARGS[@]}" \
    "${DEDEKIMI_ARGS[@]}" \
    "${CLIQUE_ARGS[@]}" \
    --json-output "${bench_out}"

  run_cmd python3 scripts/eval_quality_gate.py \
    "${COMMON_MODEL_ARGS[@]}" \
    --seed "${SEED}" \
    --temp "${TEMP}" \
    --suite "${FULLSTACK_SUITE}" \
    --max-tokens "${FULLSTACK_MAX_TOKENS}" \
    --kv-cache-type isoquant \
    "${FULLSTACK_HARNESS_ARGS[@]}" \
    "${PREDICTOR_ARGS[@]}" \
    "${DEDEKIMI_ARGS[@]}" \
    "${CLIQUE_ARGS[@]}" \
    --output-json "${qual_out}"
}

run_longctx_one() {
  local kv="$1"
  local out="${OUTPUT_DIR}/${PREFIX}_longctx_${kv}.json"
  run_cmd python3 scripts/long_context_kv_eval.py \
    "${COMMON_MODEL_ARGS[@]}" \
    --seed "${SEED}" \
    --temp "${TEMP}" \
    --test all \
    --max-tokens "${LONGCTX_MAX_TOKENS}" \
    --kv-cache-type "${kv}" \
    "${PREDICTOR_ARGS[@]}" \
    --output-json "${out}"
}

echo "=== IsoQuant submission bundle ==="
echo "model_kind=${MODEL_KIND}"
echo "model_path=${MODEL_PATH}"
echo "profile=${PROFILE}"
echo "prefix=${PREFIX}"
echo "output_dir=${OUTPUT_DIR}"
echo "max_resident_experts=${MAX_RESIDENT_EXPERTS}"
echo "quality_harness_version=${QUALITY_HARNESS_VERSION}"
if [[ -n "${QUALITY_MEMORY_LIMIT_MB}" ]]; then
  echo "quality_memory_limit_mb=${QUALITY_MEMORY_LIMIT_MB}"
fi
echo "fullstack_harness_version=${FULLSTACK_HARNESS_VERSION}"
if [[ -n "${FULLSTACK_MEMORY_LIMIT_MB}" ]]; then
  echo "fullstack_memory_limit_mb=${FULLSTACK_MEMORY_LIMIT_MB}"
fi
echo "predictor=${USE_PREDICTOR}"
echo "dedekimi=${USE_DEDEKIMI}"
if [[ -n "${TASK_CLIQUES_FILE}" ]]; then
  echo "task_expert_cliques_file=${TASK_CLIQUES_FILE}"
fi

if [[ "${RUN_BENCHMARKS}" == "1" ]]; then
  for kv in default turboquant isoquant; do
    run_benchmark_one "${kv}"
  done
fi

if [[ "${RUN_QUALITY}" == "1" ]]; then
  for kv in default turboquant isoquant; do
    run_quality_one "${kv}"
  done
fi

if [[ "${RUN_PAIR}" == "1" ]]; then
  run_pair
fi

if [[ "${RUN_FIDELITY}" == "1" ]]; then
  run_fidelity
fi

if [[ "${RUN_STRESS}" == "1" ]]; then
  for kv in default turboquant isoquant; do
    run_stress_one "${kv}"
  done
fi

if [[ "${RUN_FULLSTACK}" == "1" ]]; then
  run_fullstack_isoquant
fi

if [[ "${RUN_LONGCTX}" == "1" ]]; then
  for kv in default isoquant; do
    run_longctx_one "${kv}"
  done
fi

echo
echo "Artifacts written under ${OUTPUT_DIR}"
