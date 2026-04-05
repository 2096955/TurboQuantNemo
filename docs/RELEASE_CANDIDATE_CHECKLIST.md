# Release Candidate Checklist

This checklist is the practical release gate for **single-user local use** of
Nemotron-H with quantized expert offload on a ~32 GB Apple Silicon machine.

It is **not** a production multi-user serving checklist. That broader scope
remains in [PRODUCTION_ROADMAP.md](./PRODUCTION_ROADMAP.md)
and [EXECUTION_BOARD.md](./EXECUTION_BOARD.md).

## Scope

- Target: Nemotron-H mixed-quant / uniform-quant checkpoints with expert offload
- Machine class: ~32 GB Apple Silicon
- Goal: verify quality, memory envelope, throughput stability, and basic HTTP
  serving behavior before third-party testing

## Exit Criteria

Ship for serious single-user local testing only if all of the following are true:

- Checkpoint integrity passes for all intended checkpoints
- Fixed-seed quality matrix passes at 2-bit, 3-bit, and 4-bit
- If a layer-aware checkpoint is part of the release profile, that run passes too
- Benchmark stays comfortably within the 32 GB memory budget
- 30–60+ minute soak is clean
- Real coding prompts remain coherent
- HTTP server health, readiness, auth, and backpressure behave as expected

Do **not** claim production serving readiness from this checklist alone.

## 1. Checkpoint Integrity

Run for each checkpoint:

```bash
python3 scripts/checkpoint_integrity.py --model /path/to/nemotron-120b-mixed --require-config --expect-repack --expect-expert-keys --json
python3 scripts/checkpoint_integrity.py --model /path/to/nemotron-120b-mixed-3bit --require-config --expect-repack --expect-expert-keys --json
python3 scripts/checkpoint_integrity.py --model /path/to/nemotron-120b-4bit --require-config --expect-repack --expect-expert-keys --json
```

Pass if:

- exit code is `0` for each run
- `config.json` is present
- shard/index validation passes
- routed expert keys are detected

## 2. Fixed-Seed Quality Matrix

Run on the target machine:

```bash
MODEL_2BIT=/path/to/nemotron-120b-mixed \
MODEL_3BIT=/path/to/nemotron-120b-mixed-3bit \
MODEL_4BIT=/path/to/nemotron-120b-4bit \
MODEL_LAYER_AWARE=/path/to/nemotron-120b-layer-aware \
SEED=42 \
SUITE=all \
./scripts/run_quality_matrix.sh
```

Artifacts are written under `./artifacts/quality-matrix` unless `OUT_DIR` is set.

Pass if:

- the required uniform runs complete successfully
- if `MODEL_LAYER_AWARE` is set, that run completes successfully too
- all JSON artifacts report full pass under `--strict`
- there are no repetition failures
- outputs remain coherent across coding and reasoning prompts

## 3. Cold/Warm Benchmark

Run for each checkpoint:

```bash
python3 scripts/benchmark_moe_offload.py \
  --model /path/to/nemotron-120b-mixed \
  --profile B \
  --expert-offload \
  --memory-mode 120b-32gb \
  --prefill-step-size 64 \
  --split-decode-timing \
  --warm-second-pass \
  --repeat-runs 3 \
  --json-output /tmp/bench-2bit.json
```

Repeat for `3bit` and `4bit`.

Suggested healthy gate:

- `decode_tok_per_s >= 15` for the 2-bit profile
- `peak_memory_mb < 24576`
- no swap-driven collapse or unstable repeat-to-repeat behavior

Capture:

- first-pass decode tok/s
- warm-pass decode tok/s
- peak memory
- any warnings triggered by the benchmark harness

## 4. Soak Run

Start with a baseline artifact:

```bash
python3 scripts/eval_quality_gate.py \
  --model /path/to/nemotron-120b-mixed \
  --expert-offload \
  --suite all \
  --strict \
  --seed 42 \
  --output-json /tmp/soak-start.json
```

Then run repeated real prompts for **30–60+ minutes**, including:

- short coding prompts
- longer multi-paragraph prompts
- multi-turn chat
- interrupted / restarted requests
- prompts that force different expert usage patterns

Pass if:

- no hangs or dead workers occur
- no progressive quality decay appears
- memory does not trend upward without returning
- no shard/offload failures or repeated reload path errors appear

## 5. Real Code Evaluation

Run **10–20 repository-realistic tasks**, not only toy prompts.

Include:

- traceback diagnosis
- small function implementation
- pytest writing
- bug explanation
- refactor planning
- multi-file reasoning

Pass if:

- answers remain coherent at 2-bit
- there is no obvious practical regression versus 3-bit/4-bit for code work
- outputs are usable enough for real iteration, not just smoke prompts

## 6. HTTP Server Sanity Check

Start the server from `mlx-lm/`:

```bash
python3 -m mlx_lm.server \
  --model /path/to/nemotron-120b-mixed \
  --expert-offload \
  --host 127.0.0.1 \
  --api-key test-key \
  --max-pending-requests 8 \
  --queue-put-timeout-s 0 \
  --max-request-body-bytes 67108864
```

Verify:

```bash
# Operational endpoints
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/ready
curl -s -H 'Authorization: Bearer test-key' http://127.0.0.1:8080/metrics
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8080/metrics

# Inference through the server (verify the full generation path works)
curl -s -H 'Authorization: Bearer test-key' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":50}'
```

Pass if:

- `/health` returns `200`
- `/ready` returns `200`
- authenticated `/metrics` returns `200`
- unauthenticated `/metrics` returns `401`
- inference returns a coherent JSON response with `choices[0].message.content` containing "4"
- queue-full behavior returns `503` rather than hanging indefinitely

## 7. Release Decision

**Ready for third-party single-user testing** if all sections above pass.

**Not ready to claim production serving** if any of these remain true:

- no long soak has been run
- overload/disconnect behavior has not been validated on a live server
- hard generation timeout is still absent
- hard MLX cancellation / preemption is still absent
- multi-user policy / quotas / richer metrics are still unimplemented

## Notes

- Use fixed seeds where possible for reproducibility.
- Preserve the generated JSON artifacts from quality and benchmark runs.
- If a run fails, record the exact command, checkpoint path, and failure mode
  before retrying.
