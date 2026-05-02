#!/usr/bin/env python3
"""Sweep T to find the real single-kernel vs 3-kernel crossover point.

Forces each path at every T value and reports where the 3-kernel path
starts winning. This gives us a data-driven threshold.

Usage:
    cd mlx-lm && python scripts/sweep_threshold.py
"""

from __future__ import annotations

import time
import numpy as np
import mlx.core as mx

from mlx_lm.models.mlx_isoquant import IsoQuantKVCache
from mlx_lm.models.mlx_turboquant import get_default_codebook_dir


def _make_cache(num_heads: int, head_dim: int, seq_len: int):
    cache = IsoQuantKVCache(
        num_heads=num_heads,
        head_dim=head_dim,
        bit_width=3,
        codebook_dir=get_default_codebook_dir(),
    )
    if cache._fallback_cache is not None:
        raise RuntimeError("No codebook found")

    rng = np.random.default_rng(42)
    keys = mx.array(
        rng.normal(size=(1, num_heads, seq_len, head_dim)).astype(np.float16)
    )
    values = mx.array(
        rng.normal(size=(1, num_heads, seq_len, head_dim)).astype(np.float16)
    )
    cache.update_and_fetch(keys, values)
    cache.finalize_deferred_prefill()
    return cache


def bench_path(
    cache,
    queries,
    scale,
    mask,
    H_q,
    H_kv,
    T,
    D,
    repeats,
    use_single: bool,
    warmup=10,
    iters=100,
):
    """Force single-kernel or 3-kernel path and measure latency."""
    from mlx_lm.models.fused_kv_decode_kernels import pack_indices_3bit

    if cache._packed_keys_cache is None:
        cache._packed_keys_cache = pack_indices_3bit(cache.compressed_keys["indices"])
    if cache._packed_values_cache is None:
        cache._packed_values_cache = pack_indices_3bit(
            cache.compressed_values["indices"]
        )

    k_packed = cache._packed_keys_cache
    v_packed = cache._packed_values_cache
    k_norms = cache.compressed_keys["x_norm"][:, :, 0].astype(mx.float32)
    v_norms = cache.compressed_values["x_norm"][:, :, 0].astype(mx.float32)
    centroids = cache.compressor.centroids.reshape(-1).astype(mx.float32)
    kv_head_map = mx.arange(H_q, dtype=mx.uint32) // repeats

    if use_single:

        def _run():
            return cache._fused_attention_single_kernel(
                k_packed,
                v_packed,
                centroids,
                k_norms,
                v_norms,
                queries[0, :, 0, :],  # q_rot shape: (H_q, D)
                kv_head_map,
                scale,
                mask,
                H_q,
                T,
                D,
            )
    else:

        def _run():
            return cache._fused_attention_3kernel(
                k_packed,
                v_packed,
                centroids,
                k_norms,
                v_norms,
                queries[0, :, 0, :],
                kv_head_map,
                scale,
                mask,
                H_q,
                H_kv,
                T,
                D,
                repeats,
            )

    for _ in range(warmup):
        out = _run()
        mx.eval(out)

    t0 = time.perf_counter()
    for _ in range(iters):
        out = _run()
        mx.eval(out)
    return (time.perf_counter() - t0) / iters


def main():
    H = 8  # Use 8 heads — representative of real models
    D = 128
    T_values = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048]

    print(f"Threshold sweep: H={H}, D={D}, 100 iters each")
    print(
        f"{'T':>6}  {'Single (ms)':>12}  {'3-kernel (ms)':>14}  {'Winner':>8}  {'Ratio':>8}"
    )
    print("-" * 62)

    crossover = None
    for T in T_values:
        cache = _make_cache(H, D, T)
        rng = np.random.default_rng(99)
        queries = mx.array(rng.normal(size=(1, H, 1, D)).astype(np.float32))
        scale = D**-0.5

        # Need q_rot — apply rotation like the real path does
        R_T = mx.swapaxes(cache.rotation_matrices, -2, -1)
        q_flat = queries[0, :, 0, :]
        q_rot = mx.squeeze(mx.matmul(q_flat[:, None, :], R_T), axis=1)

        # q_rot_full unused — bench_path extracts q_rot internally

        t_single = bench_path(
            cache, queries, scale, None, H, H, T, D, 1, use_single=True
        )
        t_3kernel = bench_path(
            cache, queries, scale, None, H, H, T, D, 1, use_single=False
        )

        winner = "SINGLE" if t_single <= t_3kernel else "3-KERN"
        ratio = t_3kernel / t_single if t_single > 0 else 0

        print(
            f"{T:>6}  {t_single * 1000:>12.3f}  {t_3kernel * 1000:>14.3f}  {winner:>8}  {ratio:>8.2f}x"
        )

        if crossover is None and winner == "3-KERN":
            crossover = T

    print()
    if crossover is not None:
        print(f"Crossover detected at T={crossover}")
        print(f"Recommended threshold: {crossover} (or slightly below)")
    else:
        print(
            "Single kernel won at all tested T values — threshold could be raised to max tested"
        )


if __name__ == "__main__":
    main()
