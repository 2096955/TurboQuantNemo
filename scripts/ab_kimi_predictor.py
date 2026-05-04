#!/usr/bin/env python3
"""A/B: Kimi K2.6 decode with vs without AttnResExpertPredictor.

The predictor is wired in OffloadQuantizedSwitchGLU.__call__
(switch_layers.py:712-794) but only attached when model_config has
use_predictor=True (utils.py:636-647). Prior Kimi profiles showed
prefetch_requests=0 because the flag was off.

This script compares decode hit-rate and per-step latency between:
  baseline:  use_predictor=False (LRU only)
  variant:   use_predictor=True  (SimulatedAttnResPredictor learns
             affinity from record_activation and prefetches via
             predict_experts before each layer's MoE compute)

Both arms run identical (warmup, decode_steps) sequences. Predictor
state in the variant is allowed to learn during warmup; measurement
window starts after warmup with metrics reset.

If the variant shows a material drop in load_count / load_time_ms_total
or a rise in decode_hit_rate, the predictor is paying off without
offline calibration. If not, calibration is the next step.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MLX_LM_ROOT = REPO_ROOT / "mlx-lm"
if MLX_LM_ROOT.exists() and str(MLX_LM_ROOT) not in sys.path:
    sys.path.insert(0, str(MLX_LM_ROOT))


def run_arm(
    model_path: str,
    use_predictor: bool,
    max_resident: int,
    prompt_tokens: int,
    warmup: int,
    decode_steps: int,
) -> dict:
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import finalize_deferred_kv_caches, make_prompt_cache

    cfg = {
        "expert_offload": True,
        "max_resident_experts": max_resident,
    }
    if use_predictor:
        cfg["use_predictor"] = True

    print(f"  loading {model_path} (use_predictor={use_predictor})", flush=True)
    t0 = time.time()
    model, tokenizer = load(model_path, model_config=cfg)
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    cache = make_prompt_cache(model)
    prompt = mx.array([1] * prompt_tokens)
    mx.eval(model(prompt[None, :], cache=cache))
    finalize_deferred_kv_caches(cache)
    mx.synchronize()

    y = mx.array([[42]])

    # warmup -- includes any predictor learning
    print(f"  warmup {warmup} steps", flush=True)
    for _ in range(warmup):
        out = model(y, cache=cache)
        mx.eval(out)
    mx.synchronize()

    expert_mgr = getattr(model, "expert_offload_manager", None)
    if expert_mgr is not None:
        expert_mgr.set_phase("decode")
        expert_mgr.reset_metrics()
        # Predictor (if any) keeps its learned affinity; we reset only the
        # cache hit/miss/prefetch counters. This is the right A/B: we
        # measure post-warmup steady state, not cold-start behavior.

    print(f"  measuring {decode_steps} steps", flush=True)
    times = []
    for _ in range(decode_steps):
        mx.synchronize()
        t = time.perf_counter()
        out = model(y, cache=cache)
        mx.eval(out)
        mx.synchronize()
        times.append((time.perf_counter() - t) * 1000.0)

    stats = expert_mgr.stats_summary() if expert_mgr is not None else {}

    arr = np.asarray(times)
    result = {
        "use_predictor": use_predictor,
        "step_times_ms": times,
        "median_ms": float(np.median(arr)),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "tok_per_s": float(1000.0 / np.median(arr)),
        "expert_stats": stats,
    }

    del model, tokenizer, cache
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass
    return result


def print_summary(label: str, r: dict):
    s = r["expert_stats"]
    print(f"\n--- {label} ---", flush=True)
    print(
        f"  step_median_ms: {r['median_ms']:.1f}  (tok/s {r['tok_per_s']:.3f})",
        flush=True,
    )
    print(f"  step_std_ms:    {r['std_ms']:.1f}", flush=True)
    print(f"  range_ms:       {r['min_ms']:.1f} - {r['max_ms']:.1f}", flush=True)
    print(f"  hit_rate:       {s.get('hit_rate', 0):.3f}", flush=True)
    print(f"  decode_hit_rate:{s.get('decode_hit_rate', 0):.3f}", flush=True)
    print(f"  load_count:     {s.get('load_count', 0)}", flush=True)
    print(f"  load_total_ms:  {s.get('load_time_ms_total', 0):.1f}", flush=True)
    print(f"  prefetch_req:   {s.get('prefetch_requests', 0)}", flush=True)
    print(f"  prefetch_cached:{s.get('prefetch_already_cached', 0)}", flush=True)
    pr = s.get("prefetch_requests", 0)
    pac = s.get("prefetch_already_cached", 0)
    if pr > 0:
        # Useful prefetches = predicted experts that weren't already cached at predict time.
        # Of those, some will arrive in time to convert a future miss into a hit.
        print(
            f"  prefetch_useful: {pr - pac} of {pr} ({(pr - pac) / pr:.1%})",
            flush=True,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/Volumes/Samsung9904tb/Kimi-K2.6")
    p.add_argument("--max-resident", type=int, default=200)
    p.add_argument("--prompt-tokens", type=int, default=32)
    p.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Steps before measurement; gives predictor time to learn affinity",
    )
    p.add_argument("--decode-steps", type=int, default=50)
    p.add_argument(
        "--output",
        default="artifacts/kimi_k26_predictor/predictor_ab.json",
    )
    args = p.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n========== BASELINE (use_predictor=False) ==========", flush=True)
    baseline = run_arm(
        args.model,
        use_predictor=False,
        max_resident=args.max_resident,
        prompt_tokens=args.prompt_tokens,
        warmup=args.warmup,
        decode_steps=args.decode_steps,
    )

    print("\n========== VARIANT (use_predictor=True) ==========", flush=True)
    variant = run_arm(
        args.model,
        use_predictor=True,
        max_resident=args.max_resident,
        prompt_tokens=args.prompt_tokens,
        warmup=args.warmup,
        decode_steps=args.decode_steps,
    )

    print_summary("baseline (no predictor)", baseline)
    print_summary("variant (predictor on)", variant)

    bs = baseline["expert_stats"]
    vs = variant["expert_stats"]
    diff = {
        "step_ms_delta": variant["median_ms"] - baseline["median_ms"],
        "step_ms_pct_delta": (
            (variant["median_ms"] - baseline["median_ms"])
            / max(baseline["median_ms"], 1e-9)
            * 100.0
        ),
        "tok_per_s_delta": variant["tok_per_s"] - baseline["tok_per_s"],
        "decode_hit_rate_delta": (
            vs.get("decode_hit_rate", 0) - bs.get("decode_hit_rate", 0)
        ),
        "load_count_delta": vs.get("load_count", 0) - bs.get("load_count", 0),
        "load_time_ms_total_delta": (
            vs.get("load_time_ms_total", 0) - bs.get("load_time_ms_total", 0)
        ),
        "prefetch_requests": vs.get("prefetch_requests", 0),
        "prefetch_already_cached": vs.get("prefetch_already_cached", 0),
    }

    print("\n========== DELTA (variant - baseline) ==========", flush=True)
    print(
        f"  step_median:     {diff['step_ms_delta']:+.1f} ms  "
        f"({diff['step_ms_pct_delta']:+.1f}%)",
        flush=True,
    )
    print(
        f"  tok/s:           {diff['tok_per_s_delta']:+.3f}",
        flush=True,
    )
    print(
        f"  decode_hit_rate: {diff['decode_hit_rate_delta']:+.3f}",
        flush=True,
    )
    print(
        f"  load_count:      {diff['load_count_delta']:+d} (fewer is better)",
        flush=True,
    )
    print(
        f"  load_time_ms:    {diff['load_time_ms_total_delta']:+.1f} (fewer is better)",
        flush=True,
    )

    payload = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": args.model,
        "max_resident_experts": args.max_resident,
        "warmup": args.warmup,
        "decode_steps": args.decode_steps,
        "baseline": baseline,
        "variant": variant,
        "diff": diff,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
