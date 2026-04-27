"""Per-kernel ms attribution for IsoQuant decode steps."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("ISOQUANT_BITS", "3")

# Phase 2 lives in main's mlx-lm via the editable install. The worktree's
# mlx-lm copy has structural drift from main's _profile_mx_call infrastructure
# and would shadow the production runtime. Load from main (no sys.path
# override).
_ = Path(__file__)  # keep import live

import mlx.core as mx

import mlx_lm.models.fused_kv_decode_kernels as fkdk
import mlx_lm.models.fused_kv_decode_tiled as fkdt
from mlx_lm import load
from mlx_lm.models.cache import finalize_deferred_kv_caches, make_prompt_cache
from mlx_lm.models.mlx_isoquant import IsoQuantKVCache, reset_stats, stats_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("out_path")
    p.add_argument("--model", default="/Users/anthonylui/Models/Qwen3.6-35B-A3B-4bit")
    p.add_argument("--prefill", type=int, default=4096)
    p.add_argument("--decode", type=int, default=100)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    times: dict[str, list[float]] = defaultdict(list)

    def _wrap(name, fn):
        def wrapped(*a, **kw):
            mx.synchronize()
            t = time.perf_counter()
            out = fn(*a, **kw)
            if isinstance(out, mx.array):
                mx.eval(out)
            elif isinstance(out, (list, tuple)):
                for o in out:
                    if isinstance(o, mx.array):
                        mx.eval(o)
            mx.synchronize()
            times[name].append((time.perf_counter() - t) * 1000.0)
            return out

        return wrapped

    fkdk.pack_indices_3bit = _wrap("pack_indices_3bit", fkdk.pack_indices_3bit)
    fkdk.fused_qk_dot = _wrap("fused_qk_dot", fkdk.fused_qk_dot)
    fkdk.fused_value_accum = _wrap("fused_value_accum", fkdk.fused_value_accum)
    fkdt.fused_value_accum_tiled = _wrap(
        "fused_value_accum_tiled", fkdt.fused_value_accum_tiled
    )

    _orig_inv = IsoQuantKVCache._apply_inverse_rotation

    def _timed_inv(self, *a, **kw):
        mx.synchronize()
        t = time.perf_counter()
        out = _orig_inv(self, *a, **kw)
        mx.eval(out)
        mx.synchronize()
        times["_apply_inverse_rotation"].append((time.perf_counter() - t) * 1000.0)
        return out

    IsoQuantKVCache._apply_inverse_rotation = _timed_inv

    print(f"Loading model {args.model}...")
    model, _tok = load(args.model)
    cache = make_prompt_cache(model, kv_cache_type="isoquant")

    print(f"Running {args.prefill}-token prefill...")
    reset_stats()
    prompt = mx.array([1] * args.prefill)
    mx.synchronize()
    t0 = time.perf_counter()
    out = model(prompt[None, :], cache=cache)
    mx.eval(out)
    mx.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000.0

    times.clear()
    finalize_deferred_kv_caches(cache)
    mx.synchronize()

    print(f"Running {args.decode} decode steps...")
    y = mx.array([[42]])
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.decode):
        out = model(y, cache=cache)
        mx.eval(out)
    mx.synchronize()
    decode_total_ms = (time.perf_counter() - t0) * 1000.0
    per_step_ms = decode_total_ms / args.decode

    attribution = {
        "model": args.model,
        "T_prefill": args.prefill,
        "decode_steps": args.decode,
        "prefill_ms": prefill_ms,
        "total_decode_ms": decode_total_ms,
        "per_step_ms": per_step_ms,
        "tok_per_s": 1000.0 / per_step_ms,
        "kernels": {
            k: {
                "calls": len(v),
                "total_ms": sum(v),
                "avg_ms": sum(v) / len(v),
                "per_step_ms": sum(v) / args.decode,
                "pct_of_decode": sum(v) / decode_total_ms * 100.0,
            }
            for k, v in times.items()
            if v
        },
        "global_stats": stats_summary(),
    }
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(attribution, f, indent=2, default=str)
    print(
        f"Wrote {args.out_path}: per_step={per_step_ms:.2f} ms ({attribution['tok_per_s']:.1f} tok/s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
