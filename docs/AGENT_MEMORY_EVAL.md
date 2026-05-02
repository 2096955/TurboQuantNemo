# Agent-Memory Coding Eval

This benchmark answers a different question from the raw quality gate.

- `scripts/eval_quality_gate.py` measures raw prompt quality.
- `scripts/long_context_kv_eval.py` measures raw long-context / KV fidelity.
- `scripts/agent_memory_coding_eval.py` measures **agentic coding after context resets**, where only a persisted scratch-memory note survives between steps.

## What it tests

Three multi-step Python coding tasks:

1. `csv_pipeline_module`
2. `merge_sort_module`
3. `lru_cache_module`

For each task:

- Step 1 starts from a clean prompt.
- After the response, the harness extracts a compact memory note from the model's own output:
  - imports
  - classes
  - functions
  - tests
  - short code skeleton
- Step 2 and Step 3 run after a hard context reset.
- The next prompt receives only the saved memory note plus the new step instruction.
- Optional repair retries feed validator failures back to the model.

This is intentionally an **agent-memory** benchmark, not a raw long-context benchmark.

## Metrics

JSON artifacts include:

- per-step pass/fail
- attempts used
- preserve-hit rate for symbols that should survive across steps
- decode-hit rate (if expert offload stats are available)
- task pass rate
- total repair turns used
- MLX peak memory
- RSS / swap / pageout deltas

## Run One Model

```bash
PYTHONPATH=mlx-lm python3 scripts/agent_memory_coding_eval.py \
  --model gemma-4-26b-a4b-it-4bit \
  --task all \
  --expert-offload \
  --max-resident-experts 128 \
  --kv-cache-type isoquant \
  --memory-mode scratchpad \
  --repair-attempts 1 \
  --memory-limit-mb 19200 \
  --output-json results/agent_memory_eval/gemma4_isoquant_scratchpad.json
```

## Run 24GB Matrix

```bash
scripts/run_agent_memory_kv_matrix.sh --model-kind all
```

Default behavior:

- target `24gb` -> `19200 MB` MLX cap
- compares `turboquant` vs `isoquant`
- uses `scratchpad` external memory

Optional:

```bash
scripts/run_agent_memory_kv_matrix.sh \
  --model-kind all \
  --include-no-memory
```

That adds a no-memory baseline for the same tasks and KV modes.
