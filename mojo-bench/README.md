# Mojo GPU Kernel Benchmarks

Kernel-level benchmarks for Mojo GPU on Apple Silicon (Metal backend).
Part of the Mojo vs MLX framework comparison in `docs/FROM_ATTENTION_TO_CONSUMER_HARDWARE.md`.

## Setup

```bash
pixi install
```

## Run

```bash
pixi run bench-smoke    # Vector add smoke test
pixi run bench-all      # Full benchmark suite
```

## Results

Output JSON files are written to `results/`.
Compare with MLX results using `scripts/compare_mojo_vs_mlx.py`.

See `docs/superpowers/specs/2026-04-13-mojo-vs-mlx-kernel-benchmark-design.md` for methodology.
