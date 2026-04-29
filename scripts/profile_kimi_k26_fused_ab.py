#!/usr/bin/env python3
"""Phase 6 A/B: NPT=16 fused vs unfused throughput on real Kimi K2.6.

Runs the same decode workload twice:
  A) ISOQUANT_USE_NPT16_FUSED=0 (3-kernel pipeline)
  B) ISOQUANT_USE_NPT16_FUSED=1 (single-pass Metal kernel)

Both use isoquant cache type on the real Kimi K2.6 checkpoint.
Reports wall-clock step times, throughput, and expert cache stats.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MLX_LM_ROOT = REPO_ROOT / "mlx-lm"

if MLX_LM_ROOT.exists() and str(MLX_LM_ROOT) not in sys.path:
    sys.path.insert(0, str(MLX_LM_ROOT))


def load_model(model_path: str, max_resident_experts: int):
    from mlx_lm import load

    model, tokenizer = load(
        model_path,
        model_config={
            "expert_offload": True,
            "max_resident_experts": max_resident_experts,
        },
    )
    return model, tokenizer


def run_decode(
    model,
    cache_type: str,
    prompt_tokens: int,
    decode_steps: int,
    warmup: int,
    fused_flag: str,
) -> dict:
    import mlx.core as mx
    from mlx_lm.models.cache import finalize_deferred_kv_caches, make_prompt_cache

    os.environ["ISOQUANT_USE_NPT16_FUSED"] = fused_flag

    cache = make_prompt_cache(model, kv_cache_type=cache_type)
    prompt = mx.array([1] * prompt_tokens)
    mx.eval(model(prompt[None, :], cache=cache))
    finalize_deferred_kv_caches(cache)
    mx.synchronize()

    y = mx.array([[42]])

    for _ in range(warmup):
        out = model(y, cache=cache)
        mx.eval(out)
    mx.synchronize()

    expert_mgr = getattr(model, "expert_offload_manager", None)
    if expert_mgr is not None:
        expert_mgr.set_phase("decode")
        expert_mgr.reset_metrics()

    step_times = []
    for _ in range(decode_steps):
        mx.synchronize()
        t0 = time.perf_counter()
        out = model(y, cache=cache)
        mx.eval(out)
        mx.synchronize()
        step_times.append((time.perf_counter() - t0) * 1000.0)

    expert_stats = expert_mgr.stats_summary() if expert_mgr is not None else {}

    fused_latent_count = 0
    for c in cache:
        if hasattr(c, "supports_fused_latent_attention"):
            if c.supports_fused_latent_attention:
                fused_latent_count += 1

    del cache
    gc.collect()

    return {
        "fused_flag": fused_flag,
        "step_times_ms": step_times,
        "median_ms": float(np.median(step_times)),
        "mean_ms": float(np.mean(step_times)),
        "std_ms": float(np.std(step_times)),
        "p25_ms": float(np.percentile(step_times, 25)),
        "p75_ms": float(np.percentile(step_times, 75)),
        "min_ms": float(np.min(step_times)),
        "max_ms": float(np.max(step_times)),
        "tok_per_s": float(1000.0 / np.median(step_times)),
        "expert_cache_stats": expert_stats,
        "fused_latent_layers": fused_latent_count,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 6 A/B: NPT=16 fused vs unfused on real Kimi K2.6"
    )
    parser.add_argument(
        "--model",
        default="/Volumes/Samsung9904tb/Kimi-K2.6",
    )
    parser.add_argument("--max-resident-experts", type=int, default=2000)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--decode-steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--output",
        default="artifacts/kimi_k26_profiling/kimi_k26_fused_ab.json",
    )
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model}...")
    model, tokenizer = load_model(args.model, args.max_resident_experts)

    results = {"runs": []}

    for label, flag in [("unfused", "0"), ("fused", "1")]:
        print(f"\n{'=' * 60}")
        print(f"Run: {label} (ISOQUANT_USE_NPT16_FUSED={flag})")
        print(f"{'=' * 60}")

        run_result = run_decode(
            model,
            cache_type="isoquant",
            prompt_tokens=args.prompt_tokens,
            decode_steps=args.decode_steps,
            warmup=args.warmup,
            fused_flag=flag,
        )
        run_result["label"] = label
        results["runs"].append(run_result)

        print(f"  Median step: {run_result['median_ms']:.1f} ms")
        print(f"  Throughput:  {run_result['tok_per_s']:.2f} tok/s")
        print(f"  Std:         {run_result['std_ms']:.1f} ms")
        print(
            f"  Range:       {run_result['min_ms']:.1f} - {run_result['max_ms']:.1f} ms"
        )
        print(f"  Fused latent layers: {run_result['fused_latent_layers']}")
        es = run_result.get("expert_cache_stats", {})
        if es:
            print(
                f"  Expert cache: loads={es.get('load_count', 0)}, "
                f"avg_load={es.get('avg_load_ms', 0):.2f} ms, "
                f"hit_rate={es.get('decode_hit_rate', 0):.1%}"
            )

    unfused = results["runs"][0]
    fused = results["runs"][1]
    delta = fused["median_ms"] - unfused["median_ms"]

    print(f"\n{'=' * 60}")
    print("A/B COMPARISON")
    print(f"{'=' * 60}")
    print(
        f"  Unfused median: {unfused['median_ms']:.1f} ms ({unfused['tok_per_s']:.2f} tok/s)"
    )
    print(
        f"  Fused median:   {fused['median_ms']:.1f} ms ({fused['tok_per_s']:.2f} tok/s)"
    )
    print(f"  Delta:          {delta:+.1f} ms")
    print(f"  Fused latent layers activated: {fused['fused_latent_layers']}")

    if fused["fused_latent_layers"] == 0:
        print(
            "  WARNING: No layers activated fused latent attention. Check supports_fused_latent_attention."
        )

    results["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    results["model"] = args.model
    results["max_resident_experts"] = args.max_resident_experts
    results["prompt_tokens"] = args.prompt_tokens
    results["decode_steps"] = args.decode_steps
    results["summary"] = {
        "unfused_median_ms": unfused["median_ms"],
        "fused_median_ms": fused["median_ms"],
        "delta_ms": delta,
        "unfused_tok_s": unfused["tok_per_s"],
        "fused_tok_s": fused["tok_per_s"],
        "fused_latent_layers": fused["fused_latent_layers"],
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
