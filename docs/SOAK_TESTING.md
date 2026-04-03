# Long-session (soak) testing — template

Use this checklist for **30–60+ minute** runs after code or checkpoint changes. Goals: stability, no quality drift, sane memory under sustained decode.

## Environment

- Fixed **seed** and **temperature** (match `eval_quality_gate.py` defaults unless testing sampling).
- Same **hardware** and **OS** as release validation.
- Log **MLX peak memory**, **RSS** (Activity Monitor / `ps`), and **swap** if applicable.

## Scenarios

1. **Repeated coding edits**  
   Multi-turn: small bugfix → refactor → add tests → explain traceback. Track whether outputs stay coherent and non-repetitive.

2. **Context growth**  
   Append prior assistant turns to the prompt (or use a real repo + file excerpts). Watch for slowdown or OOM near your RAM envelope.

3. **Expert churn**  
   Prompts that route across many experts (broad topics) to exercise **shard load/evict** repeatedly. Compare **expert cache hit rate** at start vs end of session (if logged).

4. **Interrupted generation**  
   Stop mid-generation (Ctrl+C or API cancel), then run another request. Confirm no crash and no persistent bad state.

5. **Reload**  
   Exit process and reload the same model path **multiple times** in one soak script. Confirm **checkpoint_integrity**-style checks still pass and no shard handle leaks.

## Pass criteria (tune per product)

- No unexplained **repetition** or empty/truncated answers on fixed smoke prompts (reuse quality gate tasks between turns).
- **Peak memory** stays within the validated envelope (e.g. 32 GB profile).
- No **monotonic slowdown** that suggests a leak (decode tok/s stable within noise).

## Automation (optional)

- Wrap a loop calling `scripts/eval_quality_gate.py` with `--output-json` every N minutes; diff against baselines.
- Pair with `scripts/benchmark_moe_offload.py --json-output` at start and end of soak for regression artifacts.
