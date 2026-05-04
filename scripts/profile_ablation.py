#!/usr/bin/env python3
"""Low-bias paired A/B ablation profiling for IsoQuant decode gap.

True paired measurement: for each repeat, runs iso baseline and one
variant back-to-back (alternating order), computes per-pair delta.
Reports paired deltas with median, p25/p75, and raw samples.

The default-vs-iso gap is also measured as a paired comparison so that
% gap denominators are consistent with the paired methodology.

Production ablation matrix (all env-var toggles, no monkey-patches):
  baseline    - default KV cache (no IsoQuant), paired with iso
  fused_enc   - FUSED_ENCODE=1
  no_npt8     - NPT8=0 (falls back to 3-kernel path)
  prealloc    - ISOQUANT_CACHE_MODE=prealloc
  metal_fwd   - ISOQUANT_USE_METAL=1 (forward rotation in compress)
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone

import mlx.core as mx
import numpy as np


def load_model(model_path):
    from mlx_lm import load

    model, tokenizer = load(model_path)
    return model, tokenizer


ENV_DEFAULTS = {
    "ISOQUANT_FUSED_ENCODE": "0",
    "ISOQUANT_USE_NPT8_FUSED": "1",
    "ISOQUANT_CACHE_MODE": "concat_append",
    "ISOQUANT_USE_METAL": "0",
    "ISOQUANT_BITS": "3",
}


def _set_env(overrides: dict):
    merged = {**ENV_DEFAULTS, **overrides}
    originals = {}
    for k, v in merged.items():
        originals[k] = os.environ.get(k)
        os.environ[k] = v
    return originals


def _restore_env(originals: dict):
    for k, v in originals.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _clear_kernel_caches():
    for mod_path, attr in [
        ("mlx_lm.models.fused_kv_compress", "_kernel_cache"),
        ("mlx_lm.models.fused_kv_decode_npt8_tiled", "_tiled_kernel_cache"),
        ("mlx_lm.models.fused_kv_decode_kernels", "_fused_kernel_cache"),
        ("mlx_lm.models.isoquant_metal_kernels", "_kernel_cache"),
    ]:
        try:
            import importlib

            mod = importlib.import_module(mod_path)
            getattr(mod, attr, {}).clear()
        except Exception:
            pass


def make_fresh_cache(model, cache_type, prefill_T):
    from mlx_lm.models.cache import finalize_deferred_kv_caches, make_prompt_cache

    cache = make_prompt_cache(model, kv_cache_type=cache_type)
    prompt = mx.array([1] * prefill_T)
    mx.eval(model(prompt[None, :], cache=cache))
    finalize_deferred_kv_caches(cache)
    mx.synchronize()
    return cache


def _cache_activation_state(cache):
    from mlx_lm.models.mlx_isoquant import IsoQuantKVCache

    info = {"iso_layers": 0, "total_layers": len(cache)}
    flags = {}
    for c in cache:
        if isinstance(c, IsoQuantKVCache):
            info["iso_layers"] += 1
            flags["head_dim"] = c.head_dim
            flags["bit_width"] = c.bit_width
            flags["fused_encode"] = getattr(c, "_use_fused_encode", False)
            flags["fused_metal_ok"] = getattr(c, "_fused_metal_ok", None)
            flags["supports_fused"] = getattr(c, "supports_fused_attention", False)
            flags["cache_mode"] = getattr(c, "_cache_mode", "unknown")
            flags["use_metal_runtime"] = getattr(c, "_use_metal_runtime", False)
    info["flags"] = flags
    return info


def _post_decode_state(cache):
    from mlx_lm.models.mlx_isoquant import IsoQuantKVCache

    flags = {}
    for c in cache:
        if isinstance(c, IsoQuantKVCache):
            flags["fused_encode_after"] = getattr(c, "_use_fused_encode", False)
            flags["fused_metal_ok_after"] = getattr(c, "_fused_metal_ok", None)
            flags["fallback_after"] = c._fallback_cache is not None
            break
    return flags


def run_one(model, cache_type, env_overrides, prefill_T, n_steps):
    orig_env = _set_env(env_overrides)
    _clear_kernel_caches()
    try:
        cache = make_fresh_cache(model, cache_type, prefill_T)
        pre = _cache_activation_state(cache)

        y = mx.array([[42]])
        out = model(y, cache=cache)
        mx.eval(out)
        mx.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_steps):
            out = model(y, cache=cache)
            mx.eval(out)
        mx.synchronize()
        ms_per_step = (time.perf_counter() - t0) / n_steps * 1000.0

        post = _post_decode_state(cache)
        del cache
    finally:
        _restore_env(orig_env)
    gc.collect()
    return ms_per_step, pre, post


def paired_ablation(model, ref_cfg, variant_cfg, prefill_T, n_steps, n_repeats):
    """Run n_repeats paired measurements: (ref, variant) alternating order.

    First pair is a throwaway priming run (not tallied).
    Returns list of dicts with ref_ms, variant_ms, delta_ms per pair.
    """
    ref_env = ref_cfg["env"]
    ref_cache = ref_cfg["cache_type"]
    var_env = variant_cfg["env"]
    var_cache = variant_cfg["cache_type"]

    pairs = []
    pre_ref = pre_var = post_ref = post_var = None

    for r in range(-1, n_repeats):  # r=-1 is throwaway priming
        ref_first = r % 2 == 0

        if ref_first:
            ref_ms, pr, por = run_one(model, ref_cache, ref_env, prefill_T, n_steps)
            var_ms, pv, pov = run_one(model, var_cache, var_env, prefill_T, n_steps)
        else:
            var_ms, pv, pov = run_one(model, var_cache, var_env, prefill_T, n_steps)
            ref_ms, pr, por = run_one(model, ref_cache, ref_env, prefill_T, n_steps)

        if pre_ref is None:
            pre_ref, post_ref = pr, por
        if pre_var is None:
            pre_var, post_var = pv, pov

        if r < 0:
            continue  # discard priming pair

        pairs.append(
            {
                "repeat": r,
                "ref_first": ref_first,
                "ref_ms": ref_ms,
                "variant_ms": var_ms,
                "delta_ms": ref_ms - var_ms,
            }
        )

    return pairs, pre_ref, post_ref, pre_var, post_var


def _outlier_check(deltas, threshold=3.0, min_deviation_ms=2.0):
    """Flag outliers using MAD (median absolute deviation).

    Returns (n_outliers, mad, outlier_mask).
    A sample is an outlier if BOTH conditions hold:
      - |delta - median| > threshold * MAD
      - |delta - median| > min_deviation_ms  (absolute floor)
    The absolute floor prevents flagging normal jitter (<2 ms) when MAD
    is small.
    """
    med = np.median(deltas)
    mad = float(np.median(np.abs(deltas - med)))
    if mad < 1e-6:
        return 0, mad, np.zeros(len(deltas), dtype=bool)
    deviations = np.abs(deltas - med)
    mask = (deviations > threshold * mad) & (deviations > min_deviation_ms)
    return int(np.sum(mask)), mad, mask


def stats_from_pairs(pairs):
    deltas = np.array([p["delta_ms"] for p in pairs])
    ref_samples = np.array([p["ref_ms"] for p in pairs])
    var_samples = np.array([p["variant_ms"] for p in pairs])

    n_outliers, mad, outlier_mask = _outlier_check(deltas)
    sign_consistent = bool(np.percentile(deltas, 25) * np.percentile(deltas, 75) > 0)
    has_sign_flip = bool(np.any(deltas > 0) and np.any(deltas < 0))

    return {
        "delta_median": float(np.median(deltas)),
        "delta_mean": float(np.mean(deltas)),
        "delta_std": float(np.std(deltas)),
        "delta_p25": float(np.percentile(deltas, 25)),
        "delta_p75": float(np.percentile(deltas, 75)),
        "delta_mad": float(mad),
        "outlier_count": n_outliers,
        "outlier_indices": [int(i) for i in np.where(outlier_mask)[0]],
        "sign_consistent": sign_consistent,
        "has_sign_flip": has_sign_flip,
        "ref_median": float(np.median(ref_samples)),
        "variant_median": float(np.median(var_samples)),
        "ref_samples": [float(x) for x in ref_samples],
        "variant_samples": [float(x) for x in var_samples],
        "delta_samples": [float(x) for x in deltas],
    }


# --- Configs ---

ISO_REF = {
    "label": "IsoQuant (FUSED_ENCODE=0, NPT8=1, concat, BITS=3)",
    "cache_type": "isoquant",
    "env": {},
}

DEFAULT_REF = {
    "label": "Default KV (no IsoQuant)",
    "cache_type": "default",
    "env": {},
}

PRODUCTION_CONFIGS = {
    "baseline": {
        "label": "Default KV (no IsoQuant)",
        "cache_type": "default",
        "env": {},
    },
    "fused_enc": {
        "label": "FUSED_ENCODE=1",
        "cache_type": "isoquant",
        "env": {"ISOQUANT_FUSED_ENCODE": "1"},
    },
    "no_npt8": {
        "label": "NPT8=0 (3-kernel fallback)",
        "cache_type": "isoquant",
        "env": {"ISOQUANT_USE_NPT8_FUSED": "0"},
    },
    "prealloc": {
        "label": "CACHE_MODE=prealloc",
        "cache_type": "isoquant",
        "env": {"ISOQUANT_CACHE_MODE": "prealloc"},
    },
    "metal_fwd": {
        "label": "USE_METAL=1 (forward rotation in compress)",
        "cache_type": "isoquant",
        "env": {"ISOQUANT_USE_METAL": "1"},
    },
    "combined": {
        "label": "FUSED_ENCODE=1 + prealloc",
        "cache_type": "isoquant",
        "env": {
            "ISOQUANT_FUSED_ENCODE": "1",
            "ISOQUANT_CACHE_MODE": "prealloc",
        },
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Low-bias paired A/B ablation profiling"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--decode-steps", type=int, default=80)
    parser.add_argument("--prefill", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Only run these configs (default: all except baseline)",
    )
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    print(f"Loading model {args.model}...")
    model, tokenizer = load_model(args.model)
    print(f"Decode steps: {args.decode_steps}, repeats: {args.repeats} (+1 priming)")
    print("Method: paired alternating with throwaway priming pair\n")

    results = {
        "model": args.model,
        "decode_steps": args.decode_steps,
        "repeats": args.repeats,
        "priming_pairs": 1,
        "env_defaults": ENV_DEFAULTS,
        "method": "paired_alternating_with_priming",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "configs_per_T": [],
    }

    for T in args.prefill:
        print(f"{'=' * 62}")
        print(f"Context T={T}")
        print(f"{'=' * 62}")

        t_result = {"T": T, "production": {}}

        # --- Paired gap: default vs iso ---
        print("\n  gap: Default vs IsoQuant (paired)...")
        gap_pairs, pre_iso, post_iso, pre_default, post_default = paired_ablation(
            model, ISO_REF, DEFAULT_REF, T, args.decode_steps, args.repeats
        )
        gap_stats = stats_from_pairs(gap_pairs)
        gap_valid = (
            gap_stats["outlier_count"] == 0
            and gap_stats["sign_consistent"]
            and not gap_stats["has_sign_flip"]
        )
        t_result["paired_gap"] = {
            "valid": gap_valid,
            "stats": gap_stats,
            "pairs": gap_pairs,
            "iso_activation_before": pre_iso,
            "iso_activation_after": post_iso,
            "default_activation_before": pre_default,
            "default_activation_after": post_default,
        }
        print(f"    iso median:     {gap_stats['ref_median']:.2f} ms/step")
        print(f"    default median: {gap_stats['variant_median']:.2f} ms/step")
        print(
            f"    paired gap:     {gap_stats['delta_median']:+.2f} ms "
            f"(p25={gap_stats['delta_p25']:+.2f}, p75={gap_stats['delta_p75']:+.2f})"
        )
        if gap_stats["outlier_count"] > 0:
            print(
                f"    WARNING: {gap_stats['outlier_count']} outlier(s) in gap "
                f"(MAD={gap_stats['delta_mad']:.2f}, indices={gap_stats['outlier_indices']})"
            )
        if gap_stats["has_sign_flip"]:
            print("    WARNING: gap samples cross sign — gap measurement contaminated")
        if not gap_valid:
            print("    >>> Gap INVALID — % gap will not be reported for this T")

        paired_gap = gap_stats["delta_median"]

        # --- Production paired ablations (each paired with iso) ---
        if args.configs:
            prod_keys = [k for k in args.configs if k in PRODUCTION_CONFIGS]
        else:
            prod_keys = [k for k in PRODUCTION_CONFIGS if k != "baseline"]
        for cfg_name in prod_keys:
            cfg = PRODUCTION_CONFIGS[cfg_name]
            print(f"\n  {cfg_name}: {cfg['label']} (paired with iso)...")

            pairs, _, _, pre_var, post_var = paired_ablation(
                model, ISO_REF, cfg, T, args.decode_steps, args.repeats
            )
            s = stats_from_pairs(pairs)

            if s["outlier_count"] > 0:
                signal = "outlier"
            elif s["sign_consistent"]:
                signal = "stable"
            else:
                signal = "noisy"

            t_result["production"][cfg_name] = {
                "label": cfg["label"],
                "signal": signal,
                "stats": s,
                "pairs": pairs,
                "activation_before": pre_var,
                "activation_after": post_var,
            }

            flags = (pre_var or {}).get("flags", {})
            flag_parts = []
            for fk in ["fused_encode", "cache_mode", "use_metal_runtime"]:
                if fk in flags:
                    flag_parts.append(f"{fk}={flags[fk]}")
            flag_str = f"  [{', '.join(flag_parts)}]" if flag_parts else ""

            print(f"    iso median:     {s['ref_median']:.2f} ms/step")
            print(f"    variant median: {s['variant_median']:.2f} ms/step")
            print(
                f"    paired delta:   {s['delta_median']:+.2f} ms "
                f"(p25={s['delta_p25']:+.2f}, p75={s['delta_p75']:+.2f}){flag_str}"
            )
            if s["outlier_count"] > 0:
                print(
                    f"    WARNING: {s['outlier_count']} outlier(s) "
                    f"(MAD={s['delta_mad']:.2f}, indices={s['outlier_indices']})"
                )

            post = post_var or {}
            if post.get("fallback_after"):
                print("    WARNING: fallback active after decode")
            if (
                pre_var
                and pre_var.get("flags", {}).get("fused_encode")
                and not post.get("fused_encode_after", True)
            ):
                print("    WARNING: fused_encode latched off during decode")

        # --- Summary table ---
        gap_tag = "VALID" if gap_valid else "INVALID"
        print(f"\n  {'─' * 72}")
        print(
            f"  Paired deltas vs iso  (paired gap = {paired_gap:+.2f} ms, "
            f"MAD={gap_stats['delta_mad']:.2f}, {gap_tag})"
        )
        print(f"  {'─' * 72}")
        print(
            f"  {'Config':<20} {'Δ median':>9} {'Δ p25':>8} {'Δ p75':>8} "
            f"{'~% gap':>8}  {'signal':>7}"
        )
        print(f"  {'─' * 72}")

        for cfg_name in prod_keys:
            cfg_entry = t_result["production"][cfg_name]
            s = cfg_entry["stats"]
            signal = cfg_entry["signal"]
            if gap_valid and abs(paired_gap) > 0.1:
                pct_str = f"{s['delta_median'] / paired_gap * 100:>7.1f}%"
            else:
                pct_str = "    n/a "
            print(
                f"  {cfg_name:<20} {s['delta_median']:>+8.2f} {s['delta_p25']:>+7.2f} "
                f"{s['delta_p75']:>+7.2f} {pct_str}  {signal:>7}"
            )

        print(f"  {'─' * 72}")
        print("  (positive = iso slower, variant saves time)")
        print(
            "  signal: stable = p25/p75 same sign, noisy = crosses zero, outlier = MAD flagged"
        )
        if gap_valid:
            print(
                f"  (~% gap uses paired gap median {paired_gap:+.2f} ms as denominator)"
            )
        else:
            print("  (% gap suppressed — gap measurement itself is contaminated)")

        results["configs_per_T"].append(t_result)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
