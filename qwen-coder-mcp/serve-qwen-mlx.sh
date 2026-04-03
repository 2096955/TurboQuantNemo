#!/usr/bin/env bash
# Serve Qwen Coder 32B MLX model on port 8080.
# Run after conversion has finished (see README-mlx.md).

set -e
MODEL_DIR="${1:-qwen-coder-32b-mlx-q4}"
PORT="${2:-8080}"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Model dir not found: $MODEL_DIR"
  echo "Usage: $0 [model_dir] [port]"
  echo "Example: $0 qwen-coder-32b-mlx-q4 8080"
  exit 1
fi

echo "Serving model: $MODEL_DIR on port $PORT"
exec python -m mlx_lm server --model "$MODEL_DIR" --port "$PORT"
