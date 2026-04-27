# Phase 3b: T-tiled NPT=8 Kernel Microbenchmark

**Date:** 2026-04-27
**Hardware:** Apple Silicon (M-series), MLX Metal backend
**Config:** h_kv=8, h_q=16, D=256, 3-bit IsoQuant, tile_size=256
**Method:** Synthetic cache with identity block matrices, use_hadamard=False,
3 warmup + 10 timed iterations, median reported. Direct kernel calls — does
NOT exercise the IsoQuantKVCache dispatch path, cache-update overhead, or
non-identity rotations.

## Kernel-level decode latency (single attention step)

| T | 3-kernel ms | NPT=8 v1 ms | NPT=8 tiled ms | tiled vs 3-kernel |
|---|---|---|---|---|
| 64 | 0.357 | 0.320 | 0.451 | 0.79x (slower) |
| 128 | 0.289 | 0.227 | 0.286 | 1.01x |
| 256 | 0.296 | 0.298 | 0.379 | 0.78x |
| 512 | 0.447 | 0.458 | 0.379 | 1.18x |
| 1024 | 0.781 | 0.838 | 0.406 | 1.92x |
| 2048 | 1.311 | 1.479 | 0.399 | 3.28x |
| 4096 | 2.594 | 2.680 | 0.407 | 6.37x |

## Observations

1. **Tiled latency is near-constant in this range.** From T=512 to T=4096,
   tiled latency stays at ~0.4ms while v1/3-kernel grow linearly. Tiles
   execute in parallel across threadgroups.

2. **Crossover is between T=256 and T=512.** The current dispatch threshold
   of 512 is on the correct side of this crossover for this synthetic
   workload. Whether the same crossover holds in production (with non-identity
   rotations, Hadamard, cache overhead) has not been measured.

3. **v1 is faster at short T.** At T=64-128, v1 avoids tile merge overhead.
   The dispatch correctly routes short sequences to v1.

4. **3-kernel and v1 scale linearly with T,** as expected for serial-loop
   kernels with no T-parallelism.

## Limitations

- **Synthetic only.** Identity block matrices and no Hadamard — does not
  reflect production workloads where SO(4) rotation + WHT add overhead.
- **Kernel-only timing.** Excludes cache update, packed-index repacking,
  norm lookup, and dispatch overhead.
- **No 8K+ data.** The constant-time pattern *suggests* the tiled path
  scales favorably beyond T=4096, but this is not measured.
- **Single hardware config.** Results may differ across M-series generations.
