#!/usr/bin/env python3
"""Phase 5: Kimi K2.6 decode profiling.

Measures where decode time goes per step:
- MLA attention (including cache update/decompress)
- MoE routing + expert I/O + expert compute
- Norms + residuals
- Total step

The goal: determine if expert I/O dominates (making fused MLA attention
Phase 6 low-priority) or if MLA cache work is material.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def make_cache(model, cache_type: str, prompt_tokens: int):
    import mlx.core as mx
    from mlx_lm.models.cache import finalize_deferred_kv_caches, make_prompt_cache

    cache = make_prompt_cache(model, kv_cache_type=cache_type)
    prompt = mx.array([1] * prompt_tokens)
    mx.eval(model(prompt[None, :], cache=cache))
    finalize_deferred_kv_caches(cache)
    mx.synchronize()
    return cache


def profile_decode_steps(
    model, cache: list, steps: int, warmup: int = 3
) -> dict[str, Any]:
    """Time each decode step with per-component granularity.

    Uses mx.eval + mx.synchronize fences around the full forward pass.
    Component breakdown uses monkey-patched timing on the decoder layer.
    """
    import mlx.core as mx

    y = mx.array([[42]])

    # Warmup
    for _ in range(warmup):
        out = model(y, cache=cache)
        mx.eval(out)
    mx.synchronize()

    expert_mgr = getattr(model, "expert_offload_manager", None)
    if expert_mgr is not None:
        expert_mgr.set_phase("decode")
        expert_mgr.reset_metrics()

    # Full step timing (outer fence)
    step_times = []
    for _ in range(steps):
        mx.synchronize()
        t0 = time.perf_counter()
        out = model(y, cache=cache)
        mx.eval(out)
        mx.synchronize()
        step_times.append((time.perf_counter() - t0) * 1000.0)

    return {
        "step_times_ms": step_times,
        "median_ms": float(np.median(step_times)),
        "mean_ms": float(np.mean(step_times)),
        "std_ms": float(np.std(step_times)),
        "p25_ms": float(np.percentile(step_times, 25)),
        "p75_ms": float(np.percentile(step_times, 75)),
        "min_ms": float(np.min(step_times)),
        "max_ms": float(np.max(step_times)),
        "expert_cache_stats": (
            expert_mgr.stats_summary() if expert_mgr is not None else {}
        ),
    }


def profile_component_breakdown(
    model, cache: list, steps: int, warmup: int = 3
) -> dict[str, Any]:
    """Instrument the decoder layer to time attention vs MLP separately.

    This patches DeepseekV3DecoderLayer.__call__ to record per-component
    timings with eval+sync fences. The fences add overhead (~0.15ms each),
    so the sum of components will exceed the unfenced step time. The PURPOSE
    is relative attribution, not absolute timing.
    """
    import mlx.core as mx
    from mlx_lm.models.deepseek_v3 import DeepseekV3DecoderLayer

    original_call = DeepseekV3DecoderLayer.__call__

    attn_times: list[float] = []
    mlp_times: list[float] = []
    layer_call_count = 0
    recording = False

    def instrumented_call(self, x, mask=None, cache=None):
        nonlocal layer_call_count
        if not recording:
            return original_call(self, x, mask, cache)

        layer_call_count += 1

        # Attention
        mx.synchronize()
        t0 = time.perf_counter()
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        mx.eval(r)
        mx.synchronize()
        attn_times.append((time.perf_counter() - t0) * 1000.0)

        h = x + r

        # MLP (MoE with expert offload, or dense)
        mx.synchronize()
        t0 = time.perf_counter()
        r = self.mlp(self.post_attention_layernorm(h))
        mx.eval(r)
        mx.synchronize()
        mlp_times.append((time.perf_counter() - t0) * 1000.0)

        return h + r

    DeepseekV3DecoderLayer.__call__ = instrumented_call
    y = mx.array([[42]])

    try:
        # Warmup (uninstrumented)
        for _ in range(warmup):
            out = model(y, cache=cache)
            mx.eval(out)
        mx.synchronize()

        expert_mgr = getattr(model, "expert_offload_manager", None)
        if expert_mgr is not None:
            expert_mgr.set_phase("decode")
            expert_mgr.reset_metrics()

        # Instrumented steps
        recording = True
        layer_call_count = 0
        for _ in range(steps):
            out = model(y, cache=cache)
            mx.eval(out)
        mx.synchronize()
        recording = False

    finally:
        DeepseekV3DecoderLayer.__call__ = original_call

    num_layers = (
        len(model.model.layers) if hasattr(model, "model") else len(model.layers)
    )
    calls_per_step = layer_call_count // steps if steps > 0 else num_layers

    attn_arr = np.array(attn_times)
    mlp_arr = np.array(mlp_times)

    # Per-step aggregation: sum across layers for each step
    attn_per_step = []
    mlp_per_step = []
    for s in range(steps):
        start = s * calls_per_step
        end = start + calls_per_step
        attn_per_step.append(float(np.sum(attn_arr[start:end])))
        mlp_per_step.append(float(np.sum(mlp_arr[start:end])))

    # Per-layer stats (across all steps)
    attn_per_layer = []
    mlp_per_layer = []
    for layer_idx in range(calls_per_step):
        layer_attn = attn_arr[layer_idx::calls_per_step]
        layer_mlp = mlp_arr[layer_idx::calls_per_step]
        attn_per_layer.append(
            {
                "layer": layer_idx,
                "attn_median_ms": float(np.median(layer_attn)),
                "mlp_median_ms": float(np.median(layer_mlp)),
            }
        )

    return {
        "steps": steps,
        "layers_per_step": calls_per_step,
        "total_layer_calls": layer_call_count,
        "attn_per_step": {
            "median_ms": float(np.median(attn_per_step)),
            "mean_ms": float(np.mean(attn_per_step)),
            "std_ms": float(np.std(attn_per_step)),
            "values_ms": attn_per_step,
        },
        "mlp_per_step": {
            "median_ms": float(np.median(mlp_per_step)),
            "mean_ms": float(np.mean(mlp_per_step)),
            "std_ms": float(np.std(mlp_per_step)),
            "values_ms": mlp_per_step,
        },
        "attn_fraction": float(
            np.median(attn_per_step)
            / (np.median(attn_per_step) + np.median(mlp_per_step))
        ),
        "mlp_fraction": float(
            np.median(mlp_per_step)
            / (np.median(attn_per_step) + np.median(mlp_per_step))
        ),
        "expert_cache_stats": (
            expert_mgr.stats_summary() if expert_mgr is not None else {}
        ),
        "per_layer": attn_per_layer,
    }


def cache_info(cache: list) -> dict[str, Any]:
    from mlx_lm.models.mlx_isoquant import IsoQuantKVCache

    info = {
        "total_layers": len(cache),
        "iso_layers": 0,
        "kv_layers": 0,
        "mla_iso_layers": 0,
    }
    for c in cache:
        cls_name = type(c).__name__
        if cls_name == "KimiMLAIsoQuantCache":
            info["mla_iso_layers"] += 1
        elif isinstance(c, IsoQuantKVCache):
            info["iso_layers"] += 1
        else:
            info["kv_layers"] += 1
    return info


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Kimi K2.6 decode profiling (Phase 5)")
    parser.add_argument(
        "--model",
        default="/Volumes/Samsung9904tb/Kimi-K2.6",
        help="Path to Kimi K2.6 checkpoint",
    )
    parser.add_argument("--max-resident-experts", type=int, default=2000)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--decode-steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--output",
        default="artifacts/kimi_k26_profiling/kimi_k26_decode_profile.json",
    )
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model}...")
    model, tokenizer = load_model(args.model, args.max_resident_experts)

    results = {"configs": []}

    for cache_type in ["default", "isoquant"]:
        print(f"\n{'=' * 60}")
        print(f"Cache type: {cache_type}")
        print(f"{'=' * 60}")

        print(f"Building cache with {args.prompt_tokens}-token prefill...")
        cache = make_cache(model, cache_type, args.prompt_tokens)
        ci = cache_info(cache)
        print(f"  Cache: {ci}")

        print(f"\nStep timing ({args.decode_steps} steps, {args.warmup} warmup)...")
        step_profile = profile_decode_steps(
            model, cache, args.decode_steps, warmup=args.warmup
        )
        print(f"  Median step: {step_profile['median_ms']:.1f} ms")
        print(f"  Throughput: {1000.0 / step_profile['median_ms']:.2f} tok/s")
        expert_stats = step_profile.get("expert_cache_stats", {})
        if expert_stats:
            print(
                "  Expert cache: "
                f"decode_hit={expert_stats.get('decode_hit_rate', 0):.1%}, "
                f"loads={expert_stats.get('load_count', 0)}, "
                f"avg_load={expert_stats.get('avg_load_ms', 0):.2f} ms"
            )

        del cache
        gc.collect()

        print(f"\nComponent breakdown ({args.decode_steps} steps)...")
        cache2 = make_cache(model, cache_type, args.prompt_tokens)
        breakdown = profile_component_breakdown(
            model, cache2, args.decode_steps, warmup=args.warmup
        )

        attn_med = breakdown["attn_per_step"]["median_ms"]
        mlp_med = breakdown["mlp_per_step"]["median_ms"]
        print(
            f"  Attention total: {attn_med:.1f} ms/step ({breakdown['attn_fraction']:.1%})"
        )
        print(
            f"  MLP total:       {mlp_med:.1f} ms/step ({breakdown['mlp_fraction']:.1%})"
        )
        print(f"  Instrumented sum: {attn_med + mlp_med:.1f} ms/step")
        print(f"  Unfenced step:    {step_profile['median_ms']:.1f} ms/step")
        print(
            f"  Fence overhead:   {(attn_med + mlp_med) - step_profile['median_ms']:.1f} ms"
        )

        # Top-5 slowest layers
        layers_by_total = sorted(
            breakdown["per_layer"],
            key=lambda l: l["attn_median_ms"] + l["mlp_median_ms"],
            reverse=True,
        )
        print("\n  Top 5 slowest layers (attn + mlp):")
        for l in layers_by_total[:5]:
            total = l["attn_median_ms"] + l["mlp_median_ms"]
            print(
                f"    Layer {l['layer']:2d}: attn {l['attn_median_ms']:.2f} ms, "
                f"mlp {l['mlp_median_ms']:.2f} ms, total {total:.2f} ms"
            )

        config_result = {
            "cache_type": cache_type,
            "cache_info": ci,
            "prompt_tokens": args.prompt_tokens,
            "decode_steps": args.decode_steps,
            "step_profile": step_profile,
            "component_breakdown": breakdown,
        }
        results["configs"].append(config_result)

        del cache2
        gc.collect()

    # Summary comparison
    default_cfg = results["configs"][0]
    iso_cfg = results["configs"][1]
    d_step = default_cfg["step_profile"]["median_ms"]
    i_step = iso_cfg["step_profile"]["median_ms"]
    d_attn = default_cfg["component_breakdown"]["attn_per_step"]["median_ms"]
    i_attn = iso_cfg["component_breakdown"]["attn_per_step"]["median_ms"]
    d_mlp = default_cfg["component_breakdown"]["mlp_per_step"]["median_ms"]
    i_mlp = iso_cfg["component_breakdown"]["mlp_per_step"]["median_ms"]

    print(f"\n{'=' * 60}")
    print("COMPARISON: default vs isoquant")
    print(f"{'=' * 60}")
    print(
        f"  Step time:  default {d_step:.1f} ms vs isoquant {i_step:.1f} ms (delta {i_step - d_step:+.1f} ms)"
    )
    print(
        f"  Attention:  default {d_attn:.1f} ms vs isoquant {i_attn:.1f} ms (delta {i_attn - d_attn:+.1f} ms)"
    )
    print(
        f"  MLP/MoE:    default {d_mlp:.1f} ms vs isoquant {i_mlp:.1f} ms (delta {i_mlp - d_mlp:+.1f} ms)"
    )
    print(
        f"  Attn share: default {default_cfg['component_breakdown']['attn_fraction']:.1%} vs isoquant {iso_cfg['component_breakdown']['attn_fraction']:.1%}"
    )

    if d_mlp > 0:
        expert_dominance = d_mlp / (d_attn + d_mlp)
        print(f"\n  MLP/MoE block dominance (default): {expert_dominance:.1%}")
        if expert_dominance > 0.7:
            print(
                "  -> MLP/MoE dominates. Inspect expert cache stats before claiming pure expert I/O."
            )
        elif expert_dominance > 0.4:
            print(
                "  -> Mixed. Both expert I/O and attention contribute. Phase 6 has moderate value."
            )
        else:
            print(
                "  -> Attention DOMINATES. Fused MLA attention (Phase 6) is high-priority."
            )

    results["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    results["model"] = args.model
    results["max_resident_experts"] = args.max_resident_experts
    results["summary"] = {
        "default_step_ms": d_step,
        "isoquant_step_ms": i_step,
        "gap_ms": i_step - d_step,
        "default_attn_fraction": default_cfg["component_breakdown"]["attn_fraction"],
        "default_mlp_fraction": default_cfg["component_breakdown"]["mlp_fraction"],
        "isoquant_attn_fraction": iso_cfg["component_breakdown"]["attn_fraction"],
        "isoquant_mlp_fraction": iso_cfg["component_breakdown"]["mlp_fraction"],
        "interpretation_note": (
            "Component timing splits attention from the whole MLP/MoE block. "
            "Expert cache stats estimate the offload portion, but MLP/MoE includes "
            "routing, expert loading, and expert compute."
        ),
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
