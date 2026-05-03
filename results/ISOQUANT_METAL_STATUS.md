IsoQuant Metal backend status as of 2026-04-11

- Correctness: verified against dense rotation with the real MLX/Metal path.
- Test coverage: `tests/test_mlx_isoquant.py` includes Metal-specific checks behind `ISOQUANT_RUN_METAL_TESTS=1`.
- Runtime status: still experimental and opt-in via `ISOQUANT_USE_METAL=1`.
- Wall-clock status: after the threadgroup-resident FWHT update, Metal is faster than dense on all 9 tested microbenchmark shapes in `results/isoquant_rotation_benchmark.json` (1.08×–1.48× speedup).
- Recommendation: keep dense as the default runtime for now, but the Metal path is no longer a correctness-only branch; it is a plausible promotion candidate once end-to-end workload validation is completed.

Measured summary from `results/isoquant_rotation_benchmark.json`

| d_k | seq_len | dense (ms) | Metal (ms) | speedup | roundtrip err |
|-----|---------|-----------|-----------|---------|---------------|
| 128 | 1       | 0.183     | 0.140     | 1.31×   | 5.96e-07      |
| 128 | 32      | 0.161     | 0.129     | 1.25×   | 8.34e-07      |
| 128 | 256     | 0.117     | 0.108     | 1.08×   | 9.54e-07      |
| 256 | 1       | 0.124     | 0.115     | 1.08×   | 6.78e-07      |
| 256 | 32      | 0.117     | 0.109     | 1.08×   | 9.54e-07      |
| 256 | 256     | 0.131     | 0.119     | 1.10×   | 9.54e-07      |
| 512 | 1       | 0.146     | 0.099     | 1.48×   | 7.15e-07      |
| 512 | 32      | 0.137     | 0.119     | 1.15×   | 9.54e-07      |
| 512 | 256     | 0.206     | 0.160     | 1.28×   | 1.19e-06      |

Interpretation

- Metal is faster than dense on all 9 tested shapes (1.08×–1.48×).
- The backend is numerically sound (roundtrip error < 1.2e-06 across all shapes).
- The remaining work is end-to-end workload validation and promotion criteria, not kernel correctness.
