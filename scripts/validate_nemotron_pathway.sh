#!/bin/bash
set -e

MODEL_PATH="../nemotron-30b-mixed" # Placeholder for actual 120B path
RESULTS_DIR="results"
mkdir -p $RESULTS_DIR

echo "1. Running Nemotron TurboQuant Baseline..."
python scripts/eval_quality_gate.py --model $MODEL_PATH --expert-offload --kv-cache-type turboquant --output-json $RESULTS_DIR/nemotron_turboquant_baseline.json --suite all

echo "2. Running Nemotron IsoQuant Evaluation..."
python scripts/eval_quality_gate.py --model $MODEL_PATH --expert-offload --kv-cache-type isoquant --output-json $RESULTS_DIR/nemotron_isoquant_eval.json --suite all

echo "3. Running Nemotron Full Stack Soak Test..."
python scripts/eval_quality_gate.py --model $MODEL_PATH --expert-offload --kv-cache-type isoquant --use-predictor --max-tokens 1500 --suite soak --output-json $RESULTS_DIR/nemotron_full_stack_soak.json
