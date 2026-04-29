#!/usr/bin/env python3
"""Micro-profile inside ExpertOffloadManager.prepare_gather_triple_quantized.

Replaces prepare_gather_triple_quantized with an instrumented version that
times each internal phase:
  - _plan_expert_loads (Python: which experts are missing?)
  - synchronous disk loads (only when there are misses)
  - per-projection mx.stack of weights / scales / biases (×3 projs)
  - LUT build (mx.zeros + scatter assign + gather)

Times are wrapped with mx.synchronize fences so MLX work completes within
each phase. This adds overhead vs the real run, but the relative attribution
between phases is what we care about: where in those 41ms does time actually
go on Kimi K2.6?
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

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
        default="artifacts/kimi_k26_profiling/prepare_gather_microprofile.json",
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

    from mlx_lm import expert_offload as eo

    times = defaultdict(list)
    counters = defaultdict(int)
    recording = {"on": False}

    _orig = eo.ExpertOffloadManager.prepare_gather_triple_quantized

    def _instrumented(self, layer_idx, indices, *, mode="gather"):
        if not recording["on"]:
            return _orig(self, layer_idx, indices, mode=mode)

        if mode not in ("gather", "loop"):
            mode = "gather"

        # Phase 1: _plan_expert_loads (Python; minor MLX work)
        mx.synchronize()
        t = time.perf_counter()
        flat_mx, unique_eids, to_load, _ = self._plan_expert_loads(layer_idx, indices)
        mx.eval(flat_mx)
        mx.synchronize()
        times["1_plan_expert_loads"].append((time.perf_counter() - t) * 1000)
        counters["unique_eids_total"] += len(unique_eids)
        counters["to_load_total"] += len(to_load)

        # Phase 2: synchronous disk loads (real I/O)
        mx.synchronize()
        t = time.perf_counter()
        for key, spec in to_load:
            tensors = None
            try:
                tensors = self._load_expert_pair_tensors(spec)
            finally:
                self._complete_reserved_load(key, tensors)
        mx.synchronize()
        times["2_disk_loads"].append((time.perf_counter() - t) * 1000)

        # Phase 3 + 4: under the lock — stack 3 projs and build LUT
        with self._lock:
            compacts = {}

            # Phase 3a: per-proj loop (Python dict lookups + mx.stack)
            for proj in ("gate", "up", "down"):
                mx.synchronize()
                t = time.perf_counter()
                ws, ss, bs = [], [], []
                for e in unique_eids:
                    entry = self._cache[(layer_idx, int(e))]
                    ws.append(entry[f"{proj}_weight"])
                    ss.append(entry[f"{proj}_scales"])
                    bs.append(entry.get(f"{proj}_biases"))
                times[f"3a_python_lookup_{proj}"].append(
                    (time.perf_counter() - t) * 1000
                )

                # Phase 3b: mx.stack for this proj
                mx.synchronize()
                t = time.perf_counter()
                compact_w = mx.stack(ws, axis=0)
                compact_s = mx.stack(ss, axis=0)
                if all(b is not None for b in bs):
                    compact_b = mx.stack(bs, axis=0)
                elif any(b is not None for b in bs):
                    raise RuntimeError(
                        f"Mixed presence of biases across experts in layer {layer_idx} proj {proj}"
                    )
                else:
                    compact_b = None
                mx.eval(compact_w, compact_s)
                if compact_b is not None:
                    mx.eval(compact_b)
                mx.synchronize()
                times[f"3b_mx_stack_{proj}"].append((time.perf_counter() - t) * 1000)
                compacts[proj] = (compact_w, compact_s, compact_b)

            # Phase 4: LUT build (mx.zeros + scatter + gather + reshape)
            mx.synchronize()
            t = time.perf_counter()
            max_eid = max(unique_eids) + 1
            lut = mx.zeros((max_eid,), dtype=mx.int32)
            unique_arr = mx.array(unique_eids, dtype=mx.int32)
            lut[unique_arr] = mx.arange(len(unique_eids), dtype=mx.int32)
            remapped = lut[flat_mx].reshape(indices.shape)
            mx.eval(remapped)
            mx.synchronize()
            times["4_lut_build"].append((time.perf_counter() - t) * 1000)

        return compacts["gate"], compacts["up"], compacts["down"], remapped

    eo.ExpertOffloadManager.prepare_gather_triple_quantized = _instrumented

    try:
        cache = make_prompt_cache(model, kv_cache_type="isoquant")
        prompt = mx.array([1] * args.prompt_tokens)
        mx.eval(model(prompt[None, :], cache=cache))
        finalize_deferred_kv_caches(cache)
        mx.synchronize()

        y = mx.array([[42]])

        for _ in range(args.warmup):
            out = model(y, cache=cache)
            mx.eval(out)
        mx.synchronize()

        expert_mgr = getattr(model, "expert_offload_manager", None)
        if expert_mgr is not None:
            expert_mgr.set_phase("decode")
            expert_mgr.reset_metrics()

        recording["on"] = True
        for _ in range(args.decode_steps):
            mx.synchronize()
            out = model(y, cache=cache)
            mx.eval(out)
            mx.synchronize()
        recording["on"] = False

        expert_stats = expert_mgr.stats_summary() if expert_mgr is not None else {}
    finally:
        eo.ExpertOffloadManager.prepare_gather_triple_quantized = _orig

    # === Analysis ===
    n_steps = args.decode_steps
    n_calls = len(times["1_plan_expert_loads"])

    print(
        f"\n{'=' * 78}\nMicroprofile prepare_gather_triple_quantized "
        f"({n_calls} calls / {n_steps} steps)\n{'=' * 78}"
    )

    grand_total = 0.0
    for phase in [
        "1_plan_expert_loads",
        "2_disk_loads",
        "3a_python_lookup_gate",
        "3b_mx_stack_gate",
        "3a_python_lookup_up",
        "3b_mx_stack_up",
        "3a_python_lookup_down",
        "3b_mx_stack_down",
        "4_lut_build",
    ]:
        v = times[phase]
        if not v:
            continue
        s = _stats(v)
        per_step = s["total_ms"] / n_steps
        grand_total += per_step
        print(
            f"  {phase:30s}"
            f" median={s['median_ms']:6.3f} ms,"
            f" total/step={per_step:7.2f} ms,"
            f" p95={s['p95_ms']:6.3f} ms"
        )
    print(f"  {'-' * 78}")
    print(f"  {'TOTAL accounted/step':30s} {grand_total:7.2f} ms")

    # Stack vs Python lookup roll-up
    stack_total = (
        sum(sum(times[k]) for k in times if k.startswith("3b_mx_stack")) / n_steps
    )
    lookup_total = (
        sum(sum(times[k]) for k in times if k.startswith("3a_python_lookup")) / n_steps
    )
    print("\n  ROLL-UP per step:")
    print(
        f"    Phase 1 (plan):                {sum(times['1_plan_expert_loads']) / n_steps:.2f} ms"
    )
    print(
        f"    Phase 2 (disk loads):          {sum(times['2_disk_loads']) / n_steps:.2f} ms"
    )
    print(f"    Phase 3a (Python lookups, x3): {lookup_total:.2f} ms")
    print(f"    Phase 3b (mx.stack, x3):       {stack_total:.2f} ms")
    print(
        f"    Phase 4 (LUT build):           {sum(times['4_lut_build']) / n_steps:.2f} ms"
    )

    print("\n  Counters:")
    for k, v in counters.items():
        per_call = v / n_calls if n_calls else 0
        print(f"    {k}: {v} ({per_call:.2f}/call)")

    payload = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": args.model,
        "max_resident_experts": args.max_resident_experts,
        "decode_steps": n_steps,
        "n_prepare_calls": n_calls,
        "phase_stats_ms": {k: _stats(v) for k, v in times.items()},
        "per_step_total_ms": {
            "phase_1_plan": sum(times["1_plan_expert_loads"]) / n_steps,
            "phase_2_disk_loads": sum(times["2_disk_loads"]) / n_steps,
            "phase_3a_python_lookups": lookup_total,
            "phase_3b_mx_stacks": stack_total,
            "phase_4_lut_build": sum(times["4_lut_build"]) / n_steps,
            "grand_total_accounted": grand_total,
        },
        "counters": dict(counters),
        "expert_cache_stats": expert_stats,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n  Results: {out_path}")


if __name__ == "__main__":
    main()
