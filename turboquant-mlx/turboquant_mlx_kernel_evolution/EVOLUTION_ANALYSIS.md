# TurboQuant score evolution — analysis log

Use this file after OpenEvolve runs to record:

- Hardware (chip, OS, MLX version)
- `TURBOQUANT_*` environment overrides (fixture path, cosine gate, bench iterations)
- Whether fidelity gate held across islands (min / mean cosine)
- Best `throughput_eval_per_sec` vs baseline `initial_program.py`
- LLM mutation failures (syntax, shape bugs) and mitigations
- Next steps (Path B Metal, end-to-end `mlx_lm.generate` hook in the mlx-lm fork)

## Template

| Run ID | Date | Iterations | Best combined_score | Notes |
| ------ | ---- | ---------- | -------------------- | ----- |
| —      | —    | —          | —                    | —     |
