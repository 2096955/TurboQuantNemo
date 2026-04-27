"""Benchmark: NPT=8 tiled vs v1 vs 3-kernel at multiple T values.

Measures decode-only latency (single-step attention) at different cache
lengths to characterize the tiled vs v1 crossover point and validate
the T >= 512 dispatch threshold.

Uses synthetic IsoQuant cache (no model load) for isolated kernel timing.
"""

import os
import sys
import time

os.environ.setdefault("ISOQUANT_BITS", "3")

import json

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mlx-lm"))
from mlx_lm.models.fused_kv_decode_kernels import (
    fused_qk_dot,
    fused_value_accum,
    pack_indices_3bit,
)
from mlx_lm.models.fused_kv_decode_npt8 import fused_attention_npt8
from mlx_lm.models.fused_kv_decode_npt8_tiled import fused_attention_npt8_tiled
from mlx_lm.models.mlx_isoquant import structured_rotate_inverse


def build_synthetic_cache(T, h_kv, h_q, D, seed=42):
    rng = np.random.default_rng(seed)
    indices_k = mx.array(rng.integers(0, 8, (h_kv, T, D), dtype=np.uint8))
    indices_v = mx.array(rng.integers(0, 8, (h_kv, T, D), dtype=np.uint8))
    norms_k = mx.array(rng.standard_normal((h_kv, T)).astype(np.float32))
    norms_v = mx.array(rng.standard_normal((h_kv, T)).astype(np.float32))
    centroids = mx.array(np.linspace(-1.5, 1.5, 8, dtype=np.float32))
    q_rot = mx.array(rng.standard_normal((h_q, D)).astype(np.float32))
    kv_head_map = mx.arange(h_q, dtype=mx.uint32) // (h_q // h_kv)

    k_p = pack_indices_3bit(indices_k)
    v_p = pack_indices_3bit(indices_v)

    n_blocks = D // 4
    blocks = np.zeros((h_kv, n_blocks, 4, 4), dtype=np.float32)
    for h in range(h_kv):
        for b in range(n_blocks):
            blocks[h, b] = np.eye(4, dtype=np.float32)
    block_matrices = mx.array(blocks)

    mx.eval(k_p, v_p, norms_k, norms_v, centroids, q_rot, kv_head_map, block_matrices)
    return k_p, v_p, centroids, norms_k, norms_v, q_rot, kv_head_map, block_matrices


def bench_3kernel(
    k_p, v_p, c, nk, nv, q, kv_map, block_matrices, h_q, T, D, scale, warmup=3, iters=10
):
    expanded_blocks = mx.take(block_matrices, kv_map, axis=0)
    for _ in range(warmup):
        scores = fused_qk_dot(k_p, c, nk, q, kv_map, h_q, T, D) * scale
        attn = mx.softmax(scores, axis=-1)
        out = fused_value_accum(v_p, c, nv, attn, kv_map, h_q, T, D)
        out = structured_rotate_inverse(out, expanded_blocks, False)
        mx.eval(out)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        scores = fused_qk_dot(k_p, c, nk, q, kv_map, h_q, T, D) * scale
        attn = mx.softmax(scores, axis=-1)
        out = fused_value_accum(v_p, c, nv, attn, kv_map, h_q, T, D)
        out = structured_rotate_inverse(out, expanded_blocks, False)
        mx.eval(out)
        times.append(time.perf_counter() - t0)
    return times


def bench_npt8_v1(
    k_p, v_p, c, nk, nv, q, kv_map, block_matrices, h_q, T, D, scale, warmup=3, iters=10
):
    blocks_t = mx.swapaxes(block_matrices, -2, -1)
    for _ in range(warmup):
        out = fused_attention_npt8(
            k_p,
            v_p,
            c,
            nk,
            nv,
            q,
            kv_map,
            blocks_t=blocks_t,
            scale=scale,
            use_hadamard=False,
            mask=None,
            num_heads=h_q,
            seq_len=T,
            head_dim=D,
        )
        mx.eval(out)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fused_attention_npt8(
            k_p,
            v_p,
            c,
            nk,
            nv,
            q,
            kv_map,
            blocks_t=blocks_t,
            scale=scale,
            use_hadamard=False,
            mask=None,
            num_heads=h_q,
            seq_len=T,
            head_dim=D,
        )
        mx.eval(out)
        times.append(time.perf_counter() - t0)
    return times


def bench_npt8_tiled(
    k_p,
    v_p,
    c,
    nk,
    nv,
    q,
    kv_map,
    block_matrices,
    h_q,
    T,
    D,
    scale,
    tile_size=256,
    warmup=3,
    iters=10,
):
    for _ in range(warmup):
        out = fused_attention_npt8_tiled(
            k_p,
            v_p,
            c,
            nk,
            nv,
            q,
            kv_map,
            block_matrices=block_matrices,
            scale=scale,
            use_hadamard=False,
            mask=None,
            num_heads=h_q,
            seq_len=T,
            head_dim=D,
            tile_size=tile_size,
        )
        mx.eval(out)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fused_attention_npt8_tiled(
            k_p,
            v_p,
            c,
            nk,
            nv,
            q,
            kv_map,
            block_matrices=block_matrices,
            scale=scale,
            use_hadamard=False,
            mask=None,
            num_heads=h_q,
            seq_len=T,
            head_dim=D,
            tile_size=tile_size,
        )
        mx.eval(out)
        times.append(time.perf_counter() - t0)
    return times


def main():
    h_kv, h_q, D = 8, 16, 256
    scale = float(1.0 / np.sqrt(D))
    T_values = [64, 128, 256, 512, 1024, 2048, 4096]

    results = []
    print(
        f"{'T':>6} | {'3-kernel ms':>12} | {'NPT8 v1 ms':>12} | {'NPT8 tiled ms':>14} | {'v1/3k':>6} | {'tiled/3k':>8}"
    )
    print("-" * 75)

    for T in T_values:
        k_p, v_p, c, nk, nv, q, kv_map, blocks = build_synthetic_cache(T, h_kv, h_q, D)

        t_3k = bench_3kernel(k_p, v_p, c, nk, nv, q, kv_map, blocks, h_q, T, D, scale)
        t_v1 = bench_npt8_v1(k_p, v_p, c, nk, nv, q, kv_map, blocks, h_q, T, D, scale)
        t_tiled = bench_npt8_tiled(
            k_p, v_p, c, nk, nv, q, kv_map, blocks, h_q, T, D, scale
        )

        m_3k = np.median(t_3k) * 1000
        m_v1 = np.median(t_v1) * 1000
        m_tiled = np.median(t_tiled) * 1000

        ratio_v1 = m_v1 / m_3k if m_3k > 0 else 0
        ratio_tiled = m_tiled / m_3k if m_3k > 0 else 0

        print(
            f"{T:>6} | {m_3k:>10.3f}ms | {m_v1:>10.3f}ms | {m_tiled:>12.3f}ms | {ratio_v1:>5.2f}x | {ratio_tiled:>7.2f}x"
        )

        results.append(
            {
                "T": T,
                "3kernel_ms": round(m_3k, 3),
                "npt8_v1_ms": round(m_v1, 3),
                "npt8_tiled_ms": round(m_tiled, 3),
                "3kernel_times": [round(t * 1000, 3) for t in t_3k],
                "npt8_v1_times": [round(t * 1000, 3) for t in t_v1],
                "npt8_tiled_times": [round(t * 1000, 3) for t in t_tiled],
            }
        )

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "artifacts", "phase3b-tiled-smoke"
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "tiled_vs_v1_bench.json"), "w") as f:
        json.dump({"h_kv": h_kv, "h_q": h_q, "D": D, "results": results}, f, indent=2)
    print(f"\nResults saved to {out_dir}/tiled_vs_v1_bench.json")


if __name__ == "__main__":
    main()
