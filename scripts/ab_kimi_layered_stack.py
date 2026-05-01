#!/usr/bin/env python3
"""Multi-arm A/B for the Qwen3.6 Config-C stack on Kimi K2.6.

We've been retreating to "physics ceiling" on Kimi without actually applying
the stack we built for Qwen3-30B-A3B (where it gets ~9.87 tok/s). This
harness measures each layer of the stack independently so gains are
attributable, not bundled.

Arms (in order):
  L0  default cache,   NPT16=0,  observer=F, predictor=F, cliques=None
  L1  isoquant cache,  NPT16=0,  observer=F, predictor=F, cliques=None
  L2  isoquant cache,  NPT16=1,  observer=F, predictor=F, cliques=None
  L3  isoquant cache,  NPT16=1,  observer=T, predictor=F, cliques=None     [reload]
  L4  isoquant cache,  NPT16=1,  observer=T, predictor=F, cliques=file     [reload, opt]
  L5  isoquant cache,  NPT16=1,  observer=T, predictor=T, cliques=file     [reload, opt]
  L6  isoquant cache,  NPT16=1,  observer=T, predictor=T, cliques=None     [reload]

L0/L1/L2 reuse a single base model load. L3+ require reloads because
use_dedekimi_observer / use_predictor / task_expert_cliques_file are
consumed in mlx_lm/utils.py:load() (around lines 647-669) at model
construction time.

Cache type and ISOQUANT_USE_NPT16_FUSED env are per-arm — applied at
cache-construction / attention-call time, no reload needed.

Reference wiring (already verified):
  - mlx-lm/mlx_lm/models/cache.py:107-117  -> isoquant + MLA -> KimiMLAIsoQuantCache
  - mlx-lm/mlx_lm/models/mlx_isoquant.py:1248 -> reads ISOQUANT_USE_NPT16_FUSED
  - mlx-lm/mlx_lm/utils.py:647-669 -> consumes observer/predictor/cliques flags
  - scripts/ab_kimi_predictor.py -> mirror style/timing/cleanup discipline
"""

from __future__ import annotations

import argparse
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


# Arm definitions. Each arm specifies its config so we can iterate cleanly.
# `reload` = True means we tear down model and re-load with new model_config.
# `reload` = False means we reuse the currently-loaded model (only the cache
# factory and per-call env change).
ARM_DEFS = {
    "L0": {
        "cache_type": "default",
        "npt16": "0",
        "use_dedekimi_observer": False,
        "use_predictor": False,
        "use_cliques": False,
        "reload": True,  # first arm always loads
    },
    "L1": {
        "cache_type": "isoquant",
        "npt16": "0",
        "use_dedekimi_observer": False,
        "use_predictor": False,
        "use_cliques": False,
        "reload": False,
    },
    "L2": {
        "cache_type": "isoquant",
        "npt16": "1",
        "use_dedekimi_observer": False,
        "use_predictor": False,
        "use_cliques": False,
        "reload": False,
    },
    # NOTE: L3-L6 use cache_type=default. Empirical finding (today's L0/L1
    # + commit 16a6b0c artifacts): IsoQuant on Kimi MLA is a regression vs
    # default cache (default 0.96 vs IsoQuant 0.65 vs IsoQuant+NPT16 0.42).
    # Kimi MLA already compresses K/V to a 512-dim latent; layering IsoQuant
    # on top adds overhead that doesn't pay back. The actual Kimi gain is on
    # the EXPERT side (91% of step time is expert I/O), so observer/cliques/
    # predictor are the levers worth measuring on top of the default-cache
    # baseline.
    "L3": {
        "cache_type": "default",
        "npt16": "0",
        "use_dedekimi_observer": True,
        "use_predictor": False,
        "use_cliques": False,
        "reload": True,
    },
    "L4": {
        "cache_type": "default",
        "npt16": "0",
        "use_dedekimi_observer": True,
        "use_predictor": False,
        "use_cliques": True,
        "reload": True,
    },
    "L5": {
        "cache_type": "default",
        "npt16": "0",
        "use_dedekimi_observer": True,
        "use_predictor": True,
        "use_cliques": True,
        "reload": True,
    },
    "L6": {
        "cache_type": "default",
        "npt16": "0",
        "use_dedekimi_observer": True,
        "use_predictor": True,
        "use_cliques": False,
        "reload": True,
    },
}


# Headline deltas we care about for the summary table.
HEADLINE_DELTAS = [
    ("L1", "L0"),
    ("L2", "L1"),
    ("L3", "L2"),
    ("L4", "L3"),
    ("L5", "L4"),
    ("L6", "L5"),
    ("L6", "L0"),  # full-stack vs unrotated baseline
]


def build_model_config(arm: dict, max_resident: int, cliques_file: str | None) -> dict:
    """Construct the model_config kwargs for a given arm.

    Mirrors scripts/ab_kimi_predictor.py:54-59 baseline plus arm-specific flags.
    Cache type and NPT16 env are NOT in here — they're applied per-call.
    """
    cfg: dict = {
        "expert_offload": True,
        "max_resident_experts": max_resident,
    }
    if arm["use_dedekimi_observer"]:
        cfg["use_dedekimi_observer"] = True
    if arm["use_predictor"]:
        cfg["use_predictor"] = True
    if arm["use_cliques"] and cliques_file:
        # Loader expects the dict, not a file path (utils.py:672). Server CLI
        # reads the JSON itself before passing through (server.py:820-827) —
        # we do the same inline here so model_config matches what the loader
        # actually consumes.
        cfg["task_expert_cliques"] = json.loads(
            Path(cliques_file).read_text(encoding="utf-8")
        )
    return cfg


def load_model(model_path: str, model_config: dict):
    """Load model + tokenizer, mirroring ab_kimi_predictor.py."""
    from mlx_lm import load

    print(f"  loading {model_path}", flush=True)
    print(f"  model_config={model_config}", flush=True)
    t0 = time.time()
    model, tokenizer = load(model_path, model_config=model_config)
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)
    return model, tokenizer


def teardown(model, tokenizer, *caches):
    """Drop refs and clear caches between reloads."""
    import mlx.core as mx

    for c in caches:
        try:
            del c
        except Exception:
            pass
    try:
        del model
    except Exception:
        pass
    try:
        del tokenizer
    except Exception:
        pass
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass


def make_cache_for_arm(model, cache_type: str):
    """Build a fresh prompt cache for this arm.

    cache_type='default' -> let mlx-lm pick the model's default cache.
    cache_type='isoquant' -> force IsoQuant; for Kimi MLA this resolves
    to KimiMLAIsoQuantCache (cache.py:107-117).
    """
    from mlx_lm.models.cache import make_prompt_cache

    if cache_type == "default":
        return make_prompt_cache(model)
    return make_prompt_cache(model, kv_cache_type=cache_type)


def run_arm(
    model,
    arm_name: str,
    arm: dict,
    prompt_tokens: int,
    warmup: int,
    decode_steps: int,
) -> dict:
    """Per-arm protocol: fresh cache, prefill, warmup, measure."""
    import mlx.core as mx
    from mlx_lm.models.cache import finalize_deferred_kv_caches

    # --- env: per-call, set before any attention call in this arm ---
    os.environ["ISOQUANT_USE_NPT16_FUSED"] = arm["npt16"]
    print(
        f"  [{arm_name}] cache={arm['cache_type']} "
        f"NPT16={os.environ['ISOQUANT_USE_NPT16_FUSED']} "
        f"observer={arm['use_dedekimi_observer']} "
        f"predictor={arm['use_predictor']} "
        f"cliques={arm['use_cliques']}",
        flush=True,
    )

    # --- reset expert manager metrics before this arm ---
    expert_mgr = getattr(model, "expert_offload_manager", None)
    if expert_mgr is not None and hasattr(expert_mgr, "reset_metrics"):
        expert_mgr.reset_metrics()

    # --- fresh cache + prefill ---
    cache = make_cache_for_arm(model, arm["cache_type"])
    prompt = mx.array([1] * prompt_tokens)
    mx.eval(model(prompt[None, :], cache=cache))
    finalize_deferred_kv_caches(cache)
    mx.synchronize()

    y = mx.array([[42]])

    # --- warmup: lets predictor learn affinity, lets LRU stabilize ---
    print(f"  [{arm_name}] warmup {warmup} steps", flush=True)
    for _ in range(warmup):
        out = model(y, cache=cache)
        mx.eval(out)
    mx.synchronize()

    # Reset metrics after warmup, set decode phase, then measure steady state.
    if expert_mgr is not None:
        if hasattr(expert_mgr, "set_phase"):
            try:
                expert_mgr.set_phase("decode")
            except Exception:
                pass
        if hasattr(expert_mgr, "reset_metrics"):
            expert_mgr.reset_metrics()

    # --- measurement window ---
    print(f"  [{arm_name}] measuring {decode_steps} steps", flush=True)
    times = []
    for _ in range(decode_steps):
        mx.synchronize()
        t = time.perf_counter()
        out = model(y, cache=cache)
        mx.eval(out)
        mx.synchronize()
        times.append((time.perf_counter() - t) * 1000.0)

    stats = {}
    if expert_mgr is not None and hasattr(expert_mgr, "stats_summary"):
        try:
            stats = expert_mgr.stats_summary()
        except Exception:
            stats = {}

    arr = np.asarray(times)
    result = {
        "config": {
            "cache_type": arm["cache_type"],
            "npt16": arm["npt16"],
            "use_dedekimi_observer": arm["use_dedekimi_observer"],
            "use_predictor": arm["use_predictor"],
            "use_cliques": arm["use_cliques"],
        },
        "step_times_ms": times,
        "median_ms": float(np.median(arr)),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "tok_per_s": float(1000.0 / np.median(arr)),
        "expert_stats": stats,
    }

    # Drop the cache; model is kept by caller (decides reload or reuse).
    del cache
    gc.collect()

    # Reset NPT16 env so subsequent arms don't inherit our setting.
    os.environ["ISOQUANT_USE_NPT16_FUSED"] = "0"

    return result


def print_summary(label: str, r: dict):
    """Mirror scripts/ab_kimi_predictor.py:print_summary format."""
    s = r.get("expert_stats", {}) or {}
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
        print(
            f"  prefetch_useful: {pr - pac} of {pr} ({(pr - pac) / pr:.1%})",
            flush=True,
        )


def compute_delta(a: dict, b: dict) -> dict:
    """Delta of arm a vs arm b (a - b). Returns the canonical headline shape."""
    asa = a.get("expert_stats", {}) or {}
    bsa = b.get("expert_stats", {}) or {}
    base_med = max(b["median_ms"], 1e-9)
    return {
        "step_ms_delta": a["median_ms"] - b["median_ms"],
        "median_ms_pct_delta": (a["median_ms"] - b["median_ms"]) / base_med * 100.0,
        "tok_per_s_delta": a["tok_per_s"] - b["tok_per_s"],
        "decode_hit_rate_delta": (
            asa.get("decode_hit_rate", 0) - bsa.get("decode_hit_rate", 0)
        ),
        "load_count_delta": asa.get("load_count", 0) - bsa.get("load_count", 0),
        "load_time_ms_total_delta": (
            asa.get("load_time_ms_total", 0) - bsa.get("load_time_ms_total", 0)
        ),
    }


def print_delta(label: str, d: dict):
    print(f"\n========== DELTA {label} ==========", flush=True)
    print(
        f"  step_median:     {d['step_ms_delta']:+.1f} ms  "
        f"({d['median_ms_pct_delta']:+.1f}%)",
        flush=True,
    )
    print(f"  tok/s:           {d['tok_per_s_delta']:+.3f}", flush=True)
    print(f"  decode_hit_rate: {d['decode_hit_rate_delta']:+.3f}", flush=True)
    print(
        f"  load_count:      {d['load_count_delta']:+d} (fewer is better)",
        flush=True,
    )
    print(
        f"  load_time_ms:    {d['load_time_ms_total_delta']:+.1f} (fewer is better)",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/Volumes/Samsung9904tb/Kimi-K2.6")
    p.add_argument(
        "--max-resident",
        type=int,
        default=200,
        help="Max resident experts. 200 default — 2000 has OOM'd this system "
        "under memory pressure. Override on a clean boot.",
    )
    p.add_argument("--prompt-tokens", type=int, default=32)
    p.add_argument(
        "--warmup",
        type=int,
        default=30,
        help="Warmup steps (lets predictor learn affinity, LRU stabilize)",
    )
    p.add_argument("--decode-steps", type=int, default=30)
    p.add_argument(
        "--cliques-file",
        default=None,
        help="Optional absolute path to task_expert_cliques JSON. "
        "If omitted, L4/L5 are skipped.",
    )
    p.add_argument(
        "--arms",
        default="L0,L1,L2,L3,L4,L5,L6",
        help="CSV of arms to run. L4/L5 only run if --cliques-file is given.",
    )
    p.add_argument(
        "--output",
        default="artifacts/kimi_k26_layered/layered_ab.json",
    )
    args = p.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    requested = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in requested if a not in ARM_DEFS]
    if unknown:
        print(f"ERROR: unknown arms requested: {unknown}", flush=True)
        print(f"  known: {sorted(ARM_DEFS.keys())}", flush=True)
        sys.exit(2)

    # Skip cliques-dependent arms if no cliques file provided.
    if args.cliques_file is None:
        skipped = [a for a in requested if ARM_DEFS[a]["use_cliques"]]
        if skipped:
            print(
                f"NOTE: --cliques-file not given; skipping arms: {skipped}",
                flush=True,
            )
        requested = [a for a in requested if not ARM_DEFS[a]["use_cliques"]]
    elif not Path(args.cliques_file).is_file():
        print(
            f"ERROR: --cliques-file does not exist: {args.cliques_file}",
            flush=True,
        )
        sys.exit(2)

    if not requested:
        print("ERROR: no arms left to run", flush=True)
        sys.exit(2)

    print(f"\nArms to run: {requested}", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure NPT16 starts off — first arm sets it explicitly anyway.
    os.environ["ISOQUANT_USE_NPT16_FUSED"] = "0"

    results: dict = {}
    model = None
    tokenizer = None
    current_cfg_key = None  # tracks which model_config the current model was built with

    for arm_name in requested:
        arm = ARM_DEFS[arm_name]
        cfg = build_model_config(arm, args.max_resident, args.cliques_file)
        # Identify model_config uniqueness so we know when to reload.
        cfg_key = json.dumps(cfg, sort_keys=True)

        need_reload = (model is None) or arm["reload"] or (cfg_key != current_cfg_key)

        print(f"\n========== ARM {arm_name} ==========", flush=True)

        if need_reload:
            if model is not None:
                print("  tearing down previous model", flush=True)
                teardown(model, tokenizer)
                model, tokenizer = None, None
            model, tokenizer = load_model(args.model, cfg)
            current_cfg_key = cfg_key
        else:
            print("  reusing current model load", flush=True)

        try:
            r = run_arm(
                model,
                arm_name,
                arm,
                prompt_tokens=args.prompt_tokens,
                warmup=args.warmup,
                decode_steps=args.decode_steps,
            )
        except Exception as e:
            print(f"  ARM {arm_name} FAILED: {type(e).__name__}: {e}", flush=True)
            results[arm_name] = {
                "config": {
                    "cache_type": arm["cache_type"],
                    "npt16": arm["npt16"],
                    "use_dedekimi_observer": arm["use_dedekimi_observer"],
                    "use_predictor": arm["use_predictor"],
                    "use_cliques": arm["use_cliques"],
                },
                "error": f"{type(e).__name__}: {e}",
            }
            # On failure, force a reload for the next arm to avoid corrupted state.
            teardown(model, tokenizer)
            model, tokenizer = None, None
            current_cfg_key = None
            continue

        results[arm_name] = r
        print_summary(arm_name, r)

    # Final teardown.
    if model is not None:
        teardown(model, tokenizer)
        model, tokenizer = None, None

    # Compute deltas for every headline pair where both arms ran successfully.
    deltas: dict = {}
    for a, b in HEADLINE_DELTAS:
        if a in results and b in results:
            ra = results[a]
            rb = results[b]
            if "error" in ra or "error" in rb:
                continue
            deltas[f"{a}_vs_{b}"] = compute_delta(ra, rb)

    for key, d in deltas.items():
        print_delta(key, d)

    payload = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": args.model,
        "max_resident_experts": args.max_resident,
        "prompt_tokens": args.prompt_tokens,
        "warmup": args.warmup,
        "decode_steps": args.decode_steps,
        "cliques_file": args.cliques_file,
        "arms_requested": requested,
        "arms": results,
        "deltas": deltas,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
