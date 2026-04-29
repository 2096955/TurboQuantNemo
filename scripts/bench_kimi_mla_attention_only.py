#!/usr/bin/env python3
"""Attention-only baseline for Kimi MLA IsoQuant at D=512.

Isolates the IsoQuant attention path from the MoE dispatch noise.
Measures fused vs unfused per-step attention time at varying context T.
No model load, no expert offload — synthetic queries against a real
KimiMLAIsoQuantCache built up by repeated update_and_fetch calls.

This gives the floor: how fast can the attention path possibly be on
M4 Max for the Kimi MLA latent shape, before any MoE infrastructure
enters the picture.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MLX_LM_ROOT = REPO_ROOT / "mlx-lm"
if MLX_LM_ROOT.exists() and str(MLX_LM_ROOT) not in sys.path:
    sys.path.insert(0, str(MLX_LM_ROOT))

import mlx.core as mx  # noqa: E402

from mlx_lm.models.kimi_mla_isoquant_dkv import KimiMLAIsoQuantCache  # noqa: E402
from mlx_lm.models.mlx_turboquant import get_default_codebook_dir  # noqa: E402


# Kimi K2.6 MLA shape (verified against /Volumes/Samsung9904tb/Kimi-K2.6/config.json)
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
NUM_QUERY_HEADS = 64  # MLA absorbed-weight decode: q is in latent space, H_kv=1


def build_cache_at(
    T: int, bit_width: int = 3, layer_idx: int = 0
) -> KimiMLAIsoQuantCache:
    """Build a KimiMLAIsoQuantCache pre-populated with T tokens of latent state."""
    cb = get_default_codebook_dir()
    cache = KimiMLAIsoQuantCache(
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        bit_width=bit_width,
        layer_idx=layer_idx,
        codebook_dir=cb,
    )

    rng = np.random.default_rng(42 + layer_idx)
    # Bulk-prefill via deferred path: append all tokens, then finalize.
    # This matches the production flow.
    chunk = 64
    for start in range(0, T, chunk):
        n = min(chunk, T - start)
        kv_latent = mx.array(
            rng.normal(size=(1, 1, n, KV_LORA_RANK)).astype(np.float32)
        )
        k_pe = mx.array(rng.normal(size=(1, 1, n, QK_ROPE_HEAD_DIM)).astype(np.float32))
        cache.update_and_fetch(kv_latent, k_pe)
    cache.finalize_deferred_prefill()
    mx.synchronize()
    return cache


def time_one_arm(
    T: int,
    fused_flag: str,
    decode_steps: int,
    warmup: int,
) -> dict:
    """Run a single arm: build cache at T, then time decode-step attention."""
    os.environ["ISOQUANT_USE_NPT16_FUSED"] = fused_flag

    cache = build_cache_at(T)
    rng = np.random.default_rng(99)
    scale = 1.0 / (KV_LORA_RANK**0.5)

    # Prepare a fixed query and PE-scores buffer big enough for all decode steps.
    q = mx.array(
        rng.normal(size=(1, NUM_QUERY_HEADS, 1, KV_LORA_RANK)).astype(np.float32)
    )

    # Warmup (first call also JIT-compiles the kernel)
    for _ in range(warmup):
        T_now = cache.offset
        pe = mx.array(
            rng.normal(size=(1, NUM_QUERY_HEADS, 1, T_now)).astype(np.float32)
        )
        out = cache.fused_latent_attention(q, pe, scale)
        mx.eval(out)
        # Append a token to grow the cache by one (mimics real decode)
        kv_latent = mx.array(
            rng.normal(size=(1, 1, 1, KV_LORA_RANK)).astype(np.float32)
        )
        k_pe = mx.array(rng.normal(size=(1, 1, 1, QK_ROPE_HEAD_DIM)).astype(np.float32))
        cache.update_and_fetch(kv_latent, k_pe)
    mx.synchronize()

    times_ms = []
    for _ in range(decode_steps):
        T_now = cache.offset
        pe = mx.array(
            rng.normal(size=(1, NUM_QUERY_HEADS, 1, T_now)).astype(np.float32)
        )
        mx.synchronize()
        t0 = time.perf_counter()
        out = cache.fused_latent_attention(q, pe, scale)
        mx.eval(out)
        mx.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

        kv_latent = mx.array(
            rng.normal(size=(1, 1, 1, KV_LORA_RANK)).astype(np.float32)
        )
        k_pe = mx.array(rng.normal(size=(1, 1, 1, QK_ROPE_HEAD_DIM)).astype(np.float32))
        cache.update_and_fetch(kv_latent, k_pe)

    del cache
    gc.collect()

    arr = np.array(times_ms)
    return {
        "T_start": T,
        "T_end": T + decode_steps,
        "fused_flag": fused_flag,
        "decode_steps": decode_steps,
        "times_ms": times_ms,
        "median_ms": float(np.median(arr)),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "p25_ms": float(np.percentile(arr, 25)),
        "p75_ms": float(np.percentile(arr, 75)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }


def theoretical_floor_ms(T: int, num_query_heads: int = NUM_QUERY_HEADS) -> dict:
    """Theoretical attention floor on M4 Max for one IsoQuant 3-bit MLA call.

    Memory-bound (cache reads dominate at IsoQuant 3-bit):
      Per token in cache: 512 dims * 3 bits = 192 bytes (packed)
      Plus k_norms (4 bytes) + v_norms (4 bytes) = 8 bytes
      Total per token: 200 bytes
      Cache read bytes: T * 200
      M4 Max bandwidth: 270 GB/s sustained, ~410 GB/s peak

    Compute (with absorbed-weight MLA decode):
      Q @ K: H_q * T * D = 64 * T * 512 = 32768T multiply-add = 65536T FLOPs
      Softmax: O(H_q * T) — negligible
      Score @ V: H_q * T * D = 32768T multiply-add = 65536T FLOPs
      Total: ~131KT FLOPs
      M4 Max: ~16 TFLOPs FP16 sustained

    Returns both bandwidth-bound and compute-bound floors.
    """
    cache_bytes = T * 200
    compute_flops = 131000 * T

    bw_sustained = 270e9  # 270 GB/s
    bw_peak = 410e9
    flops_sustained = 16e12

    return {
        "T": T,
        "cache_bytes": cache_bytes,
        "bandwidth_floor_ms_sustained": cache_bytes / bw_sustained * 1000,
        "bandwidth_floor_ms_peak": cache_bytes / bw_peak * 1000,
        "compute_floor_ms": compute_flops / flops_sustained * 1000,
        "first_principles_floor_ms": max(
            cache_bytes / bw_sustained * 1000,
            compute_flops / flops_sustained * 1000,
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--T-values",
        default="32,256,1024,4096,8192",
        help="Comma-separated context lengths to test",
    )
    p.add_argument("--decode-steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument(
        "--output",
        default="artifacts/kimi_k26_profiling/attention_only_baseline.json",
    )
    args = p.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    Ts = [int(t) for t in args.T_values.split(",")]

    results = {
        "shape": {
            "kv_lora_rank": KV_LORA_RANK,
            "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
            "num_query_heads": NUM_QUERY_HEADS,
            "bits": 3,
        },
        "decode_steps": args.decode_steps,
        "warmup": args.warmup,
        "runs": [],
        "theoretical": [theoretical_floor_ms(t) for t in Ts],
    }

    print("\n" + "=" * 70)
    print("Kimi MLA Attention-Only Microbench (no MoE, no model load)")
    print(f"Shape: H_q={NUM_QUERY_HEADS}, H_kv=1, D={KV_LORA_RANK}, IsoQuant 3-bit")
    print("=" * 70)

    for T in Ts:
        floor = theoretical_floor_ms(T)
        print(f"\n--- T={T} ---")
        print(
            f"  Theoretical floor: bandwidth={floor['bandwidth_floor_ms_sustained']:.3f} ms (sustained), "
            f"compute={floor['compute_floor_ms']:.3f} ms"
        )
        for label, flag in [
            ("3-kernel (NPT16=0)", "0"),
            ("single-pass (NPT16=1)", "1"),
        ]:
            r = time_one_arm(T, flag, args.decode_steps, args.warmup)
            r["label"] = label
            results["runs"].append(r)
            ratio = r["median_ms"] / floor["first_principles_floor_ms"]
            print(
                f"  {label}: median={r['median_ms']:.2f} ms, "
                f"std={r['std_ms']:.2f}, range=[{r['min_ms']:.2f}, {r['max_ms']:.2f}], "
                f"vs floor: {ratio:.0f}x"
            )

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
