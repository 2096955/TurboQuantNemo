#!/bin/bash
set -e

MODEL_PATH="../gemma-4-26b-a4b-it-4bit"
RESULTS_DIR="results"
mkdir -p $RESULTS_DIR

echo "1. Running TurboQuant Baseline..."
python scripts/eval_quality_gate.py --model $MODEL_PATH --expert-offload --kv-cache-type turboquant --output-json $RESULTS_DIR/gemma4_turboquant_baseline.json --suite all

echo "2. Running IsoQuant Evaluation..."
python scripts/eval_quality_gate.py --model $MODEL_PATH --expert-offload --kv-cache-type isoquant --output-json $RESULTS_DIR/gemma4_isoquant_eval.json --suite all

echo "3. Running Predictor Evaluation..."
python scripts/eval_quality_gate.py --model $MODEL_PATH --expert-offload --use-predictor --output-json $RESULTS_DIR/gemma4_predictor_eval.json --suite all

echo "4. Running Full Stack Soak Test..."
python scripts/eval_quality_gate.py --model $MODEL_PATH --expert-offload --kv-cache-type isoquant --use-predictor --max-tokens 1500 --suite soak --output-json $RESULTS_DIR/gemma4_full_stack_soak.json
