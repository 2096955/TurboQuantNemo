"""A/B benchmark for IsoQuant cache update modes.

Production-shape defaults:
- Prefill length: 8192
- Head dim: 256
- Decode KV generated outside the timed loop

Usage:
    uv run python tests/bench_prealloc_vs_concat.py
    uv run python tests/bench_prealloc_vs_concat.py --decode-steps 256
"""

import argparse
import os
import time

os.environ["ISOQUANT_BITS"] = "3"

import mlx.core as mx
import numpy as np

from mlx_lm.models.mlx_isoquant import IsoQuantKVCache, get_stats, reset_stats
from mlx_lm.models.mlx_turboquant import get_default_codebook_dir

CODEBOOK_DIR = get_default_codebook_dir()
DEFAULT_NUM_HEADS = 32
DEFAULT_HEAD_DIM = 256
DEFAULT_PREFILL_LEN = 8192
DEFAULT_DECODE_STEPS = 1024


def _make_cache(mode: str, num_heads: int, head_dim: int) -> IsoQuantKVCache:
    os.environ["ISOQUANT_CACHE_MODE"] = mode
    return IsoQuantKVCache(
        num_heads=num_heads,
        head_dim=head_dim,
        bit_width=3,
        layer_idx=0,
        codebook_dir=CODEBOOK_DIR,
    )


def _gen_kv(seq_len: int, *, num_heads: int, head_dim: int, seed: int):
    rng = np.random.default_rng(seed)
    keys = mx.array(
        rng.standard_normal((1, num_heads, seq_len, head_dim)).astype(np.float32)
    )
    values = mx.array(
        rng.standard_normal((1, num_heads, seq_len, head_dim)).astype(np.float32)
    )
    mx.eval(keys, values)
    return keys, values


def _pregenerate_decode_kv(
    decode_steps: int, *, num_heads: int, head_dim: int
) -> list[tuple[mx.array, mx.array]]:
    decode_kv = []
    for step in range(decode_steps):
        decode_kv.append(
            _gen_kv(1, num_heads=num_heads, head_dim=head_dim, seed=10_000 + step)
        )
    return decode_kv


def bench(
    mode: str,
    *,
    num_heads: int,
    head_dim: int,
    prefill_len: int,
    decode_kv: list[tuple[mx.array, mx.array]],
) -> float:
    reset_stats()
    cache = _make_cache(mode, num_heads, head_dim)
    keys, values = _gen_kv(prefill_len, num_heads=num_heads, head_dim=head_dim, seed=7)
    cache.update_and_fetch(keys, values)
    cache.finalize_deferred_prefill()
    if cache._packed_keys_cache is not None:
        mx.eval(cache._packed_keys_cache, cache._packed_values_cache)
    mx.synchronize()

    t0 = time.perf_counter()
    for k_new, v_new in decode_kv:
        cache.update_and_fetch(k_new, v_new)
    mx.synchronize()
    elapsed = time.perf_counter() - t0

    steps = len(decode_kv)
    stats = get_stats()
    print(f"  {mode:15s}: {elapsed:.3f}s  ({steps / elapsed:.1f} updates/s)")
    print(f"    buffer cap: {cache.compressed_keys['indices'].shape[1]}")
    print(f"    valid:      {cache.offset}")
    print(f"    ms/step:    {elapsed / steps * 1000:.2f}")
    print(f"    decode calls: {stats.decode_calls}")
    print(
        f"    metal: {stats.fused_metal_attempts} attempts, "
        f"{stats.fused_metal_failures} failures"
    )
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--prefill-len", type=int, default=DEFAULT_PREFILL_LEN)
    parser.add_argument("--decode-steps", type=int, default=DEFAULT_DECODE_STEPS)
    args = parser.parse_args()

    decode_kv = _pregenerate_decode_kv(
        args.decode_steps,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
    )
    mx.synchronize()

    print(f"Benchmark: {args.decode_steps} decode steps from T={args.prefill_len}")
    print(f"  H={args.num_heads}, D={args.head_dim}")
    print("  Decode KV is pre-generated outside the timed loop.")
    print()

    t_concat = bench(
        "concat_append",
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        prefill_len=args.prefill_len,
        decode_kv=decode_kv,
    )
    t_prealloc = bench(
        "prealloc",
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        prefill_len=args.prefill_len,
        decode_kv=decode_kv,
    )

    print()
    speedup = t_concat / t_prealloc if t_prealloc > 0 else float("inf")
    per_step_ms = (t_concat - t_prealloc) / args.decode_steps * 1000
    print(f"Speedup: {speedup:.3f}x ({t_concat:.3f}s -> {t_prealloc:.3f}s)")
    print(f"Per-step delta: {per_step_ms:.2f} ms")

    if speedup < 1.0:
        print(f"FAIL: prealloc is SLOWER by {-per_step_ms:.2f} ms/step")
        raise SystemExit(1)
    elif speedup < 1.05:
        print("FAIL: speedup < 5% -- does not justify complexity")
        raise SystemExit(1)
    else:
        print(f"PASS: {speedup:.1f}x improvement, {per_step_ms:.2f} ms/step saved")


if __name__ == "__main__":
    main()
