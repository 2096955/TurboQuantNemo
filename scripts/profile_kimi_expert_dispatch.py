#!/usr/bin/env python3
"""Profile expert dispatch overhead on real Kimi K2.6.

Instruments three suspect hot paths to attribute the ~2 sec/step gap that
expert I/O alone (~217 ms/step) doesn't explain:

  1) ExpertOffloadManager.prepare_gather_triple_quantized — runs per MoE
     layer per token. Builds compact (gate, up, down) tensors via Python
     loops + mx.stack under a mutex. Suspected major Python overhead.

  2) OffloadQuantizedSwitchGLU.__call__ — the per-layer MoE forward path.
     Calls prepare_gather_triple_quantized then 3x mx.gather_qmm.

  3) DeepseekV3DecoderLayer.__call__ — wraps attention + MLP/MoE per
     layer; gives total per-layer step time for cross-check against the
     two above.

All three are wrapped with mx.synchronize fences and time.perf_counter
for honest GPU + CPU timing. Fences add overhead vs the real run, so
absolute numbers are inflated, but relative attribution between paths
is accurate.

Output: per-call median/p95/total, plus a summary that breaks the
2369 ms/step number down into expert-prepare / gather_qmm / attention /
fence-overhead buckets.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MLX_LM_ROOT = REPO_ROOT / "mlx-lm"
if MLX_LM_ROOT.exists() and str(MLX_LM_ROOT) not in sys.path:
    sys.path.insert(0, str(MLX_LM_ROOT))


def _stats(times_ms: list[float]) -> dict:
    if not times_ms:
        return {"count": 0}
    a = np.asarray(times_ms)
    return {
        "count": len(times_ms),
        "total_ms": float(a.sum()),
        "median_ms": float(np.median(a)),
        "mean_ms": float(np.mean(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/Volumes/Samsung9904tb/Kimi-K2.6")
    p.add_argument("--max-resident-experts", type=int, default=2000)
    p.add_argument("--prompt-tokens", type=int, default=32)
    p.add_argument("--decode-steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument(
        "--output",
        default="artifacts/kimi_k26_profiling/expert_dispatch_profile.json",
    )
    args = p.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import finalize_deferred_kv_caches, make_prompt_cache

    print(f"Loading model from {args.model}...")
    t0 = time.time()
    model, tokenizer = load(
        args.model,
        model_config={
            "expert_offload": True,
            "max_resident_experts": args.max_resident_experts,
        },
    )
    print(f"  loaded in {time.time() - t0:.1f}s")

    # === Instrumentation: wrap the three hot paths ===
    from mlx_lm import expert_offload as eo
    from mlx_lm.models import switch_layers as sl
    from mlx_lm.models import deepseek_v3 as dv3

    times = defaultdict(list)
    recording = {"on": False}

    # 1) prepare_gather_triple_quantized
    _orig_prep = eo.ExpertOffloadManager.prepare_gather_triple_quantized

    def _patched_prep(self, layer_idx, indices, *, mode="gather"):
        if not recording["on"]:
            return _orig_prep(self, layer_idx, indices, mode=mode)
        mx.synchronize()
        t = time.perf_counter()
        result = _orig_prep(self, layer_idx, indices, mode=mode)
        mx.eval(result[0][0], result[1][0], result[2][0], result[3])
        mx.synchronize()
        times["prepare_gather_triple_quantized"].append(
            (time.perf_counter() - t) * 1000.0
        )
        return result

    eo.ExpertOffloadManager.prepare_gather_triple_quantized = _patched_prep

    # 2) OffloadQuantizedSwitchGLU.__call__
    _orig_glu = sl.OffloadQuantizedSwitchGLU.__call__

    def _patched_glu(self, x, indices):
        if not recording["on"]:
            return _orig_glu(self, x, indices)
        mx.synchronize()
        t = time.perf_counter()
        result = _orig_glu(self, x, indices)
        mx.eval(result)
        mx.synchronize()
        times["offload_quantized_switch_glu_call"].append(
            (time.perf_counter() - t) * 1000.0
        )
        return result

    sl.OffloadQuantizedSwitchGLU.__call__ = _patched_glu

    # 3) DeepseekV3DecoderLayer.__call__
    _orig_layer = dv3.DeepseekV3DecoderLayer.__call__

    def _patched_layer(self, x, mask=None, cache=None):
        if not recording["on"]:
            return _orig_layer(self, x, mask, cache)
        mx.synchronize()
        t = time.perf_counter()
        result = _orig_layer(self, x, mask, cache)
        mx.eval(result)
        mx.synchronize()
        times["decoder_layer_call"].append((time.perf_counter() - t) * 1000.0)
        return result

    dv3.DeepseekV3DecoderLayer.__call__ = _patched_layer

    try:
        # Build cache + prefill
        cache = make_prompt_cache(model, kv_cache_type="isoquant")
        prompt = mx.array([1] * args.prompt_tokens)
        mx.eval(model(prompt[None, :], cache=cache))
        finalize_deferred_kv_caches(cache)
        mx.synchronize()

        y = mx.array([[42]])

        # Warmup (uninstrumented)
        for _ in range(args.warmup):
            out = model(y, cache=cache)
            mx.eval(out)
        mx.synchronize()

        expert_mgr = getattr(model, "expert_offload_manager", None)
        if expert_mgr is not None:
            expert_mgr.set_phase("decode")
            expert_mgr.reset_metrics()

        # Instrumented decode steps
        recording["on"] = True
        step_times = []
        for _ in range(args.decode_steps):
            mx.synchronize()
            t = time.perf_counter()
            out = model(y, cache=cache)
            mx.eval(out)
            mx.synchronize()
            step_times.append((time.perf_counter() - t) * 1000.0)
        recording["on"] = False

        expert_stats = expert_mgr.stats_summary() if expert_mgr is not None else {}

    finally:
        eo.ExpertOffloadManager.prepare_gather_triple_quantized = _orig_prep
        sl.OffloadQuantizedSwitchGLU.__call__ = _orig_glu
        dv3.DeepseekV3DecoderLayer.__call__ = _orig_layer

    # === Analysis ===
    n_steps = len(step_times)
    n_layers = len(model.model.layers) if hasattr(model, "model") else len(model.layers)

    prep_per_step = (
        sum(times["prepare_gather_triple_quantized"]) / n_steps if n_steps else 0
    )
    glu_per_step = (
        sum(times["offload_quantized_switch_glu_call"]) / n_steps if n_steps else 0
    )
    layer_per_step = sum(times["decoder_layer_call"]) / n_steps if n_steps else 0
    step_median = float(np.median(step_times)) if step_times else 0
    step_mean = float(np.mean(step_times)) if step_times else 0

    # Decompose: GLU_call = prepare + gather_qmm work
    qmm_per_step = max(0.0, glu_per_step - prep_per_step)

    # Layer time = attention + MLP + small Python work
    # MLP/MoE block ≈ glu_per_step on routed-expert layers
    # We don't separately measure attention here, but layer_per_step - glu_per_step is a proxy

    print("\n" + "=" * 70)
    print("Expert Dispatch Profile (instrumented)")
    print("=" * 70)
    print(f"Decode steps: {n_steps}, layers: {n_layers}")
    print(
        f"\nStep time (with instrumentation fences):"
        f" median={step_median:.1f} ms, mean={step_mean:.1f} ms,"
        f" min={min(step_times):.1f}, max={max(step_times):.1f}"
    )

    print("\nPer-call statistics:")
    for k, v in times.items():
        s = _stats(v)
        per_step_total = s.get("total_ms", 0) / n_steps if n_steps else 0
        per_call = s.get("median_ms", 0)
        print(
            f"  {k}: {s['count']} calls,"
            f" median={per_call:.3f} ms/call,"
            f" total/step={per_step_total:.1f} ms,"
            f" p95={s.get('p95_ms', 0):.3f} ms"
        )

    print("\nDecomposition (per step):")
    print(f"  prepare_gather_triple_quantized: {prep_per_step:.1f} ms")
    print(f"  gather_qmm + activation (GLU - prep): {qmm_per_step:.1f} ms")
    print(f"  All MLP/MoE (GLU calls):             {glu_per_step:.1f} ms")
    print(f"  All decoder layers (incl attention): {layer_per_step:.1f} ms")
    print(f"  Step median (instrumented):          {step_median:.1f} ms")
    print(
        f"  Layer-call gap (overhead/dispatch outside decoder layer):"
        f" {step_median - layer_per_step:.1f} ms"
    )

    print("\nExpert cache stats:")
    for k, v in expert_stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    payload = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": args.model,
        "max_resident_experts": args.max_resident_experts,
        "prompt_tokens": args.prompt_tokens,
        "decode_steps": n_steps,
        "n_layers": n_layers,
        "step_times_ms": step_times,
        "step_summary": {
            "median_ms": step_median,
            "mean_ms": step_mean,
            "min_ms": float(min(step_times)) if step_times else 0,
            "max_ms": float(max(step_times)) if step_times else 0,
        },
        "per_call_stats": {k: _stats(v) for k, v in times.items()},
        "per_step_decomposition_ms": {
            "prepare_gather_triple_quantized": prep_per_step,
            "gather_qmm_plus_activation": qmm_per_step,
            "all_offload_glu_calls": glu_per_step,
            "all_decoder_layers": layer_per_step,
            "above_decoder_layer_overhead": step_median - layer_per_step,
        },
        "expert_cache_stats": expert_stats,
        "interpretation_note": (
            "All times inflated by mx.synchronize fences. Use relative "
            "attribution, not absolute. The prepare/GLU/layer numbers "
            "reflect Python+MLX-dispatch+small-op overhead per layer per "
            "step. Attention path (deepseek_v3 self_attn) is NOT separately "
            "timed here; subtract gather_qmm from layer time to bound it."
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
