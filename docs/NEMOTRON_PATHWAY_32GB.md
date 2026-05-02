# Nemotron-H — 32 GB-class pathway (IsoQuant + offload)

Use the same hooks as Qwen3; `model_type` is **`nemotron_h`** in MLX. The current practical target in this repo is the `nemotron-30b-mixed` checkpoint. The same script also accepts 120B-class Nemotron-H checkpoints if they match the architecture preflight.

## Environment

```bash
export NEMOTRON_MODEL="/path/to/nemotron-30b-mixed"    # your converted MLX dir
export TURBOQUANT_BITS=3
```

## Full stack (script)

```bash
bash scripts/run_nemotron_pathway_full_stack.sh
```

The script now includes a preflight check on `config.json` and accepts Nemotron-H MoE checkpoints that match the expected architecture shape, including the current 30B mixed checkpoint used for pathway validation.

## Manual: benchmark + quality

```bash
python scripts/benchmark_moe_offload.py \
  --model "$NEMOTRON_MODEL" --profile B --memory-mode 120b-32gb \
  --expert-offload --max-resident-experts "${MAX_RESIDENT_EXPERTS:-48}" \
  --turboquant-bits 3 \
  --json-output results/nemotron_pathway_benchmark.json
  # Add --use-dedekimi-observer if you want observer metrics (optional)

python scripts/eval_quality_gate.py \
  --model "$NEMOTRON_MODEL" --suite all \
  --expert-offload --kv-cache-type isoquant \
  --output-json results/nemotron_pathway_quality.json
  # Add --use-dedekimi-observer if you want observer metrics (optional)
```

Set `NEMOTRON_USE_PREDICTOR=1` only when explicitly testing the optional AttnRes-style predictor path. It is not part of the required pathway gate.

**32 GB validation target:** keep the workload within a 25.6 GB envelope and pin the JSON artifacts. The current pinned Nemotron artifacts were collected on a capped larger-memory host, so they prove the constrained runtime envelope and sustained stability, but they are still weaker than a native physical 32 GB rerun.

## Current pinned state

- `results/nemotron_pathway_benchmark.json`: 35.50 tok/s decode, 4348 MB benchmark peak, 99.98% decode hit rate, fits 25.6 GB target envelope.
- `results/soak/nemotron-30b-mixed_soak_final.json`: 375 iterations over 120 minutes, 32.6 tok/s P50, P99/P50 1.16, 12333 MB peak, RSS drift 1.03x, no OOM/crash.
- `results/nemotron_pathway_quality.json`: 10/12 (post-fence-fix rerun, responses persisted in artifact v4). Remaining failures: `Multi-file style refactor` (off-topic output) and `Long decode soak (1K+ tokens)` (model stopped at 973 words before test functions).

Benchmark and soak are therefore pinned. Pathway close-out is still blocked on the quality gate.

## IsoQuant vs TurboQuant pair

```bash
python scripts/compare_pathway_kv_modes.py \
  --model "$NEMOTRON_MODEL" \
  --output results/nemotron_kv_isoquant_vs_turboquant.json \
  --expert-offload --profile B --turboquant-bits 3
  # --use-dedekimi-observer  # optional
```
