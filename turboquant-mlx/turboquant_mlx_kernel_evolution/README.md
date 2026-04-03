# Phase 6 — OpenEvolve: TurboQuant asymmetric attention scores

This directory mirrors the layout of [OpenEvolve `examples/mlx_metal_kernel_opt`](https://github.com/algorithmicsuperintelligence/openevolve/tree/main/examples/mlx_metal_kernel_opt), adapted for **Path A**: evolving the Python/MLX `asymmetric_attention_scores` hot path (see `../mlx_turboquant.py` for the production reference).

## Prerequisites

1. Repo root is **`turboquant-mlx/`** (parent of this folder).
2. Codebooks for your head dimension and bit width, e.g. `codebooks/dim_128_4bit.npz` — run `python codebook_precompute.py` from the repo root if missing.
3. `pip install openevolve` (see `requirements.txt`).
4. An OpenAI-compatible server for mutations (recommended: local `mlx_lm.server`) — adjust `config.yaml` `llm.api_base` and `primary_model`.

## Quick checks

```bash
cd /path/to/turboquant-mlx
export PYTHONPATH="$PWD"

# Fidelity + micro-throughput on initial_program.py
python turboquant_mlx_kernel_evolution/evaluator.py

# Same evaluation in a subprocess (implicit `eval` subcommand)
python turboquant_mlx_kernel_evolution/mlx_lm_generate_with_hook.py \
  --program turboquant_mlx_kernel_evolution/initial_program.py
```

## Real KV fixture (Phase 3 export)

Export tensors from a **live forward** (same capture path as `validate_real_kv.py`):

```bash
cd /path/to/turboquant-mlx
export PYTHONPATH="$PWD"
python turboquant_mlx_kernel_evolution/export_phase3_fixture.py \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --layer 0 --kv-head 0 --bits 4 \
  --output turboquant_mlx_kernel_evolution/fixtures/phase3_real.npz
```

Then:

```bash
export TURBOQUANT_FIXTURE_NPZ=turboquant_mlx_kernel_evolution/fixtures/phase3_real.npz
python turboquant_mlx_kernel_evolution/evaluator.py
```

The `.npz` includes extra keys (`layer_idx`, `query_head`, …) that the loader ignores.

## Synthetic fixture (no model)

```bash
python turboquant_mlx_kernel_evolution/save_fixture.py \
  --output turboquant_mlx_kernel_evolution/fixtures/tq_synth.npz
export TURBOQUANT_FIXTURE_NPZ=...
```

## End-to-end `mlx_lm.generate` with evolved scores

The **mlx-lm** fork loads an optional module when **`TURBOQUANT_ASYMMETRIC_SCORE_MODULE`** is set to an absolute or relative path of a `.py` file that defines `asymmetric_attention_scores(...)`. The file is reloaded when its mtime changes.

Run generation from **`turboquant-mlx/`** so `codebooks/` resolves (or ensure cwd matches `make_prompt_cache` turboquant layout):

```bash
cd /path/to/turboquant-mlx
export PYTHONPATH="/path/to/turboquant-mlx:$PYTHONPATH"
python turboquant_mlx_kernel_evolution/mlx_lm_generate_with_hook.py generate \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --prompt "Hello" \
  --max-tokens 32 \
  --program turboquant_mlx_kernel_evolution/initial_program.py
```

Omit `--program` to use the built-in reference implementation.

## Run evolution

```bash
chmod +x turboquant_mlx_kernel_evolution/run_evolve_experiment.sh
./turboquant_mlx_kernel_evolution/run_evolve_experiment.sh
# or
ITERATIONS=10 OUTPUT_DIR=/tmp/tq_oe ./turboquant_mlx_kernel_evolution/run_evolve_experiment.sh
```

Or invoke the CLI directly:

```bash
cd /path/to/turboquant-mlx
export PYTHONPATH="$PWD"
python -m openevolve.cli \
  turboquant_mlx_kernel_evolution/initial_program.py \
  turboquant_mlx_kernel_evolution/evaluator.py \
  --config turboquant_mlx_kernel_evolution/config.yaml \
  --iterations 25 \
  --output ./openevolve_output_turboquant
```

Override LLM endpoint without editing YAML:

```bash
python -m openevolve.cli ... --api-base http://localhost:8080/v1 --primary-model qwen
```

## Environment variables

| Variable | Purpose |
| -------- | ------- |
| `TURBOQUANT_CODEBOOK_DIR` | Codebook directory (default: `../codebooks`) |
| `TURBOQUANT_FIXTURE_NPZ` | Optional `.npz` fixture for fixed inputs |
| `TURBOQUANT_HEAD_DIM`, `TURBOQUANT_BITS`, `TURBOQUANT_SEQ_KV`, `TURBOQUANT_NUM_QUERIES` | Synthetic fixture sizing |
| `TURBOQUANT_COSINE_GATE` | Fidelity gate (default `0.995`) |
| `TURBOQUANT_BENCH_WARMUP`, `TURBOQUANT_BENCH_ITERS` | Throughput measurement |
| `TURBOQUANT_ASYMMETRIC_SCORE_MODULE` | Path to evolved `.py` (used by **mlx-lm** `TurboQuantKVCache` at inference) |

## Files

| File | Role |
| ---- | ---- |
| `initial_program.py` | Baseline; only `EVOLVE-BLOCK` may change |
| `evaluator.py` | Cosine gate vs `mlx_turboquant`, then eval/s |
| `turboquant_benchmark_suite.py` | Fixture builder + micro-bench |
| `config.yaml` | OpenEvolve + MAP-Elites features |
| `run_evolve_experiment.sh` | Launcher |
| `mlx_lm_generate_with_hook.py` | `eval` subprocess + `generate` wrapper (sets score-module env) |
| `export_phase3_fixture.py` | Capture KV → `.npz` for OpenEvolve |
| `save_fixture.py` | Synthetic `.npz` fixture |
| `best_program.py` | Promote winner here after a run |

## Integrating a winner

Copy the evolved `asymmetric_attention_scores` from the best OpenEvolve artifact into `mlx_turboquant.py` (or the mlx-lm fork copy), run `validate_real_kv.py` and your Phase 5 benchmarks before merging.
