#!/usr/bin/env python3
"""Branch C profiling: component decomposition of fused NPT=8 IsoQuant decode.

Measures where the 1.91x gap (iso vs default KV at 8K) comes from by
inserting mx.synchronize() fences around each component of the tiled
NPT=8 attention path.

Components:
  query_rotation        - forward SO(4)+WHT rotation of the query vector
  metal_kernel          - tiled NPT=8 Metal dispatch (QK+softmax+V per tile)
  fa2_merge             - Python-side FA2 tile merge (log-sum-exp)
  inverse_rotation      - post-merge inverse SO(4)+WHT rotation
  compress_batch        - key/value _compress_batch
  pack_indices_3bit     - 3-bit index packing

Both T=4096 and T=8192 use the tiled path (_NPT8_TILED_T_THRESHOLD=512).
"""

import os
import sys
import time
import json
import argparse
from collections import defaultdict

# --- Set environment variables BEFORE importing MLX/model code ---
os.environ["ISOQUANT_USE_NPT8_FUSED"] = "1"
os.environ["ISOQUANT_BITS"] = "3"
os.environ.setdefault("ISOQUANT_CACHE_MODE", "concat_append")
# Pin fused-encode OFF as the script baseline so Phase B "IsoQuant unpatched"
# does not silently inherit the post-§3.4 runtime default of FUSED_ENCODE=1
# (added 2026-05-06 after §3.4 graduated FUSED_ENCODE=1 globally).
# Phase B2 explicitly sets ISOQUANT_FUSED_ENCODE=1 and its `finally` restores
# it to "0" (NOT pop) so subsequent T-loop iterations remain unpatched.
os.environ.setdefault("ISOQUANT_FUSED_ENCODE", "0")

mx = None
load = None
finalize_deferred_kv_caches = None
make_prompt_cache = None
mlx_isoquant = None
fkdt = None

# ---- Timing infrastructure ----

component_times = defaultdict(list)

_originals = {}
_patched = False


def import_mlx_modules():
    """Import MLX after argument parsing so --help does not touch Metal."""
    global mx
    global load
    global finalize_deferred_kv_caches
    global make_prompt_cache
    global mlx_isoquant
    global fkdt

    import mlx.core as _mx
    from mlx_lm import load as _load
    from mlx_lm.models.cache import (
        finalize_deferred_kv_caches as _finalize_deferred_kv_caches,
        make_prompt_cache as _make_prompt_cache,
    )
    from mlx_lm.models import fused_kv_decode_npt8_tiled as _fkdt
    from mlx_lm.models import mlx_isoquant as _mlx_isoquant

    mx = _mx
    load = _load
    finalize_deferred_kv_caches = _finalize_deferred_kv_caches
    make_prompt_cache = _make_prompt_cache
    mlx_isoquant = _mlx_isoquant
    fkdt = _fkdt


def sync_time():
    mx.synchronize()
    return time.perf_counter()


def record_time(name, t0):
    mx.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    component_times[name].append(elapsed_ms)


# ---- Monkeypatching (idempotent; safe to call multiple times) ----


def patch_instrumentation():
    """Patch once, idempotently. Saves originals for restore."""
    global _patched
    if _patched:
        return
    _patched = True

    # 1. Patch IsoQuantKVCache.fused_attention to time query_rotation
    _originals["fused_attention"] = mlx_isoquant.IsoQuantKVCache.fused_attention

    def timed_fused_attention(self, queries, scale, mask=None):
        if self.compressed_keys is None or self._seq_len == 0:
            return mx.zeros_like(queries)

        B, H_q, _, D = queries.shape
        H_kv = self.num_heads
        if self._cache_mode == "prealloc":
            T = self.offset
        else:
            T = self.compressed_keys["indices"].shape[1]
        repeats = H_q // H_kv if H_q != H_kv else 1

        if not self.supports_fused_attention:
            return _originals["fused_attention"](self, queries, scale, mask)

        # --- Time query_rotation ---
        t0 = sync_time()
        R_T = mx.swapaxes(self.rotation_matrices, -2, -1)
        q_flat = queries[0, :, 0, :]
        R_T_exp = mx.repeat(R_T, repeats, axis=0) if repeats > 1 else R_T
        q_rot = mx.squeeze(mx.matmul(q_flat[:, None, :], R_T_exp), axis=1)
        mx.eval(q_rot)
        record_time("query_rotation", t0)

        # --- Dispatch to Metal fused path ---
        if self._fused_metal_ok is not False:
            try:
                out = self._fused_attention_metal(
                    q_rot, scale, mask, H_q, H_kv, T, D, repeats
                )
                self._fused_metal_ok = True
                return out[None, :, None, :].astype(queries.dtype)
            except Exception:
                self._fused_metal_ok = False

        out = self._fused_attention_mlx(q_rot, scale, mask, H_q, H_kv, T, D, repeats)
        return out[None, :, None, :].astype(queries.dtype)

    mlx_isoquant.IsoQuantKVCache.fused_attention = timed_fused_attention

    # 2. Patch the tiled kernel dispatch function (fused_attention_npt8_tiled)
    #    to time metal_kernel, fa2_merge, and inverse_rotation separately.
    #    We patch the three internal functions that the dispatch calls:
    #      - The kernel object returned by _get_tiled_kernel() is cached in
    #        _tiled_kernel_cache, so we patch the cache entry directly.
    #      - _fa2_merge is a module-level function
    #      - structured_rotate_inverse is imported from mlx_isoquant

    # 2a. Patch the cached kernel object (not the factory)
    #     Force kernel creation first, then wrap the cached object.
    _originals["_get_tiled_kernel"] = fkdt._get_tiled_kernel
    _originals["_tiled_kernel_cache_had_key"] = "npt8_tiled" in fkdt._tiled_kernel_cache
    _originals["_tiled_kernel_cache_value"] = fkdt._tiled_kernel_cache.get("npt8_tiled")
    original_kernel = fkdt._get_tiled_kernel()  # force creation + cache

    class KernelTimingWrapper:
        def __init__(self, real_kernel):
            self._real = real_kernel

        def __call__(self, *args, **kwargs):
            t0 = sync_time()
            res = self._real(*args, **kwargs)
            if isinstance(res, (tuple, list)):
                mx.eval(*res)
            else:
                mx.eval(res)
            record_time("metal_kernel", t0)
            return res

    fkdt._tiled_kernel_cache["npt8_tiled"] = KernelTimingWrapper(original_kernel)

    # 2b. Patch _fa2_merge
    _originals["_fa2_merge"] = fkdt._fa2_merge

    def timed_fa2_merge(*args, **kwargs):
        t0 = sync_time()
        res = _originals["_fa2_merge"](*args, **kwargs)
        mx.eval(res)
        record_time("fa2_merge", t0)
        return res

    fkdt._fa2_merge = timed_fa2_merge

    # 2c. Patch structured_rotate_inverse (used by tiled kernel's final step)
    _originals["structured_rotate_inverse"] = mlx_isoquant.structured_rotate_inverse

    def timed_inverse(*args, **kwargs):
        t0 = sync_time()
        res = _originals["structured_rotate_inverse"](*args, **kwargs)
        mx.eval(res)
        record_time("inverse_rotation", t0)
        return res

    mlx_isoquant.structured_rotate_inverse = timed_inverse

    # 3. Patch _compress_batch and pack_indices_3bit separately
    _originals["_compress_batch"] = mlx_isoquant.IsoQuantKVCache._compress_batch

    def timed_compress_batch(self, x):
        t0 = sync_time()
        res = _originals["_compress_batch"](self, x)
        mx.eval(res["indices"], res["x_norm"])
        record_time("compress_batch", t0)
        return res

    mlx_isoquant.IsoQuantKVCache._compress_batch = timed_compress_batch

    from mlx_lm.models.fused_kv_decode_kernels import pack_indices_3bit as original_pack

    _originals["pack_indices_3bit"] = original_pack

    def timed_pack(*args, **kwargs):
        t0 = sync_time()
        res = _originals["pack_indices_3bit"](*args, **kwargs)
        mx.eval(res)
        record_time("pack_indices_3bit", t0)
        return res

    import mlx_lm.models.fused_kv_decode_kernels as fkdk

    fkdk.pack_indices_3bit = timed_pack


def unpatch_instrumentation():
    """Restore all originals."""
    global _patched
    if not _patched:
        return
    _patched = False

    mlx_isoquant.IsoQuantKVCache.fused_attention = _originals["fused_attention"]
    fkdt._get_tiled_kernel = _originals["_get_tiled_kernel"]
    if _originals["_tiled_kernel_cache_had_key"]:
        fkdt._tiled_kernel_cache["npt8_tiled"] = _originals["_tiled_kernel_cache_value"]
    else:
        fkdt._tiled_kernel_cache.pop("npt8_tiled", None)
    fkdt._fa2_merge = _originals["_fa2_merge"]
    mlx_isoquant.structured_rotate_inverse = _originals["structured_rotate_inverse"]
    mlx_isoquant.IsoQuantKVCache._compress_batch = _originals["_compress_batch"]
    import mlx_lm.models.fused_kv_decode_kernels as fkdk

    fkdk.pack_indices_3bit = _originals["pack_indices_3bit"]
    _originals.clear()


# ---- Benchmark phases ----


def make_fresh_cache(model, cache_type, T):
    """Create a fresh cache, prefill T tokens, finalize."""
    cache = make_prompt_cache(model, kv_cache_type=cache_type)
    prompt = mx.array([1] * T)
    mx.eval(model(prompt[None, :], cache=cache))
    finalize_deferred_kv_caches(cache)
    mx.synchronize()
    return cache


def run_decode(model, cache, steps, clear_components=False):
    """Run decode steps, return mean ms/step."""
    y = mx.array([[42]])
    # Warmup
    out = model(y, cache=cache)
    mx.eval(out)
    mx.synchronize()
    if clear_components:
        component_times.clear()

    t0 = time.perf_counter()
    for _ in range(steps):
        out = model(y, cache=cache)
        mx.eval(out)
    mx.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000.0
    return total_ms / steps


def validate_roofline(path):
    """Load and validate roofline calibration JSON. Returns dict or None."""
    if not os.path.exists(path):
        print(f"WARNING: roofline file not found: {path}")
        return None
    with open(path) as f:
        data = json.load(f)
    eff = data.get("bw_efficiency")
    if eff is None:
        eff = data.get("bandwidth_efficiency_ratio")
    if eff is None:
        eff = data.get("calibration", {}).get("bandwidth_efficiency_ratio")
    if eff is None:
        print("WARNING: roofline JSON missing bw_efficiency field")
        return data
    if eff < 0.3:
        print(
            f"FATAL: roofline BW efficiency {eff:.3f} < 0.3; GPU unhealthy. Aborting."
        )
        sys.exit(1)
    if eff < 0.50:
        print(
            f"WARNING: roofline BW efficiency {eff:.3f} < 0.73 (plan expected 0.73-0.85)."
        )
        print("         Results are usable but may understate peak GPU capability.")
    else:
        print(f"Roofline BW efficiency: {eff:.3f}; OK")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Branch C: NPT=8 fused kernel profiling"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--roofline", required=True, help="Path to roofline calibration JSON"
    )
    parser.add_argument("--decode-steps", type=int, default=100)
    parser.add_argument(
        "--capture-traces",
        action="store_true",
        help="Attempt mx.metal.start_capture for GPU traces",
    )
    args = parser.parse_args()

    # Validate roofline first
    print("Validating roofline calibration...")
    roofline_data = validate_roofline(args.roofline)

    import_mlx_modules()

    results = {
        "model": args.model,
        "decode_steps": args.decode_steps,
        "roofline": roofline_data,
        "configs": [],
    }

    print(f"Loading model {args.model}...")
    model, _ = load(args.model)

    for T in [4096, 8192]:
        print(f"\n{'=' * 60}")
        print(f"Context T={T} (tiled path, threshold=512)")
        print(f"{'=' * 60}")
        config_results = {"T": T}

        # Phase A: Default KV baseline
        print("\n  Phase A: Default KV baseline...")
        cache_default = make_fresh_cache(model, "default", T)
        default_ms = run_decode(model, cache_default, args.decode_steps)
        config_results["default_total_ms_per_step"] = default_ms
        print(f"    default: {default_ms:.2f} ms/step")
        del cache_default

        # Phase B: IsoQuant unpatched (true production latency)
        print("\n  Phase B: IsoQuant unpatched...")
        cache_iso_b = make_fresh_cache(model, "isoquant", T)
        iso_unpatched_ms = run_decode(model, cache_iso_b, args.decode_steps)
        config_results["iso_unpatched_ms_per_step"] = iso_unpatched_ms
        print(f"    iso unpatched: {iso_unpatched_ms:.2f} ms/step")
        del cache_iso_b

        # Phase B2: IsoQuant with fused encode (write-path Metal kernel)
        print("\n  Phase B2: IsoQuant with fused encode...")
        os.environ["ISOQUANT_FUSED_ENCODE"] = "1"
        try:
            # Clear kernel cache so fresh cache picks up the env var
            from mlx_lm.models import fused_kv_compress

            fused_kv_compress._kernel_cache.clear()
            cache_iso_b2 = make_fresh_cache(model, "isoquant", T)
            # Verify fused encode is active
            iso_caches_b2 = [
                c
                for layer_cache in cache_iso_b2
                for c in (
                    [layer_cache] if hasattr(layer_cache, "_use_fused_encode") else []
                )
            ]
            fused_active = any(c._use_fused_encode for c in iso_caches_b2)
            n_fused = sum(1 for c in iso_caches_b2 if c._use_fused_encode)
            if iso_caches_b2:
                dims = set(c.head_dim for c in iso_caches_b2)
                print(
                    f"    IsoQuant layers: {len(iso_caches_b2)}, fused: {n_fused}, head_dims: {dims}"
                )
            if not fused_active:
                print("    WARNING: fused encode not active on any cache layer")
            iso_fused_ms = run_decode(model, cache_iso_b2, args.decode_steps)
            fused_after = any(c._use_fused_encode for c in iso_caches_b2)
            config_results["iso_fused_ms_per_step"] = iso_fused_ms
            config_results["fused_encode_active"] = fused_active
            config_results["fused_encode_active_after_decode"] = fused_after
            if fused_active and not fused_after:
                print(
                    "    WARNING: fused encode latched off during decode (fallback triggered)"
                )
            speedup = iso_unpatched_ms / iso_fused_ms if iso_fused_ms > 0 else 0
            write_path_saved = iso_unpatched_ms - iso_fused_ms
            print(f"    iso fused encode: {iso_fused_ms:.2f} ms/step")
            print(f"    vs unfused: {iso_unpatched_ms:.2f} ms/step")
            print(f"    write-path saved: {write_path_saved:.2f} ms")
            print(f"    speedup: {speedup:.2f}x")
            del cache_iso_b2
        finally:
            # Restore script-baseline (FUSED_ENCODE=0) instead of popping —
            # popping would leave the next iteration's Phase B "unpatched"
            # vulnerable to inheriting the post-§3.4 runtime default of
            # FUSED_ENCODE=1. The setdefault at the top of the file pinned
            # this; restore explicitly here too.
            os.environ["ISOQUANT_FUSED_ENCODE"] = "0"

        # Phase C: IsoQuant instrumented (fresh cache, patched)
        print("\n  Phase C: IsoQuant instrumented (6-component decomposition)...")
        cache_iso_c = make_fresh_cache(model, "isoquant", T)
        patch_instrumentation()
        try:
            iso_instrumented_ms = run_decode(
                model, cache_iso_c, args.decode_steps, clear_components=True
            )
        finally:
            unpatch_instrumentation()
        config_results["iso_instrumented_ms_per_step"] = iso_instrumented_ms
        print(f"    iso instrumented: {iso_instrumented_ms:.2f} ms/step")
        del cache_iso_c

        # Compute overhead and attribution
        overhead_pct = (
            (iso_instrumented_ms - iso_unpatched_ms) / iso_unpatched_ms * 100.0
        )
        config_results["instrumentation_overhead_pct"] = overhead_pct
        print(f"    instrumentation overhead: {overhead_pct:.1f}%")

        gap = iso_unpatched_ms - default_ms
        config_results["gap_ms"] = gap

        scale_factor = (
            iso_unpatched_ms / iso_instrumented_ms if iso_instrumented_ms > 0 else 1.0
        )

        components = []
        sum_component_ms = 0.0
        for name, times in component_times.items():
            avg_ms = sum(times) / args.decode_steps
            sum_component_ms += avg_ms
            components.append(
                {
                    "name": name,
                    "call_count": len(times),
                    "avg_ms_per_step": avg_ms,
                    "pct_of_instrumented_total": avg_ms / iso_instrumented_ms * 100.0,
                    "gap_attribution_pct": (avg_ms * scale_factor) / gap * 100.0
                    if gap > 0
                    else 0,
                }
            )

        components.sort(key=lambda c: c["avg_ms_per_step"], reverse=True)
        config_results["components"] = components
        config_results["sum_components_ms"] = sum_component_ms
        config_results["residual_ms"] = iso_instrumented_ms - sum_component_ms
        config_results["residual_pct"] = (
            (iso_instrumented_ms - sum_component_ms) / iso_instrumented_ms * 100.0
        )
        observed = {c["name"] for c in components}
        expected = {
            "query_rotation",
            "metal_kernel",
            "fa2_merge",
            "inverse_rotation",
            "compress_batch",
            "pack_indices_3bit",
        }
        config_results["missing_components"] = sorted(expected - observed)
        config_results["tiled_kernel_observed"] = "metal_kernel" in observed
        if config_results["missing_components"]:
            print(
                "    WARNING: missing components: "
                + ", ".join(config_results["missing_components"])
            )

        print("\n    Component breakdown (instrumented):")
        print(f"    {'Component':<30s} {'ms/step':>8s} {'% instr':>8s} {'% gap':>8s}")
        print(f"    {'-' * 30} {'-' * 8} {'-' * 8} {'-' * 8}")
        for c in components:
            print(
                f"    {c['name']:<30s} {c['avg_ms_per_step']:>8.2f} {c['pct_of_instrumented_total']:>7.1f}% {c['gap_attribution_pct']:>7.1f}%"
            )
        print(
            f"    {'residual':<30s} {config_results['residual_ms']:>8.2f} {config_results['residual_pct']:>7.1f}%"
        )

        results["configs"].append(config_results)

    # Phase D: GPU trace capture (optional)
    if args.capture_traces:
        print("\n\nPhase D: GPU trace capture...")
        trace_dir = os.path.join(os.path.dirname(args.output), "traces")
        os.makedirs(trace_dir, exist_ok=True)

        for label, cache_type in [("default", "default"), ("isoquant", "isoquant")]:
            trace_path = os.path.join(trace_dir, f"{label}_T8192.gputrace")
            print(f"  Attempting trace for {label} at T=8192...")
            try:
                cache_trace = make_fresh_cache(model, cache_type, 8192)
                mx.metal.start_capture(trace_path)
                run_decode(model, cache_trace, 10)
                mx.metal.stop_capture()
                print(f"    Trace saved: {trace_path}")
                del cache_trace
            except Exception as e:
                print(
                    f"    Trace capture failed ({e}). Use manual Instruments attach instead."
                )
                print("    See docs/gpu-profiling-protocol.md sections 3.3-3.4.")

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
