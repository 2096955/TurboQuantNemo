#!/usr/bin/env python3
import os
import time
import json
import argparse
from collections import defaultdict

# --- Set environment variables BEFORE importing MLX/model code ---
os.environ["ISOQUANT_USE_NPT8_FUSED"] = "1"
os.environ["ISOQUANT_BITS"] = "3"
os.environ.setdefault("ISOQUANT_CACHE_MODE", "concat_append")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import finalize_deferred_kv_caches, make_prompt_cache
from mlx_lm.models import mlx_isoquant
from mlx_lm.models import fused_kv_decode_npt8_tiled as fkdt

# Component timing storage
component_times = defaultdict(list)

def sync_time():
    mx.synchronize()
    return time.perf_counter()

def record_time(name, t0):
    mx.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    component_times[name].append(elapsed_ms)

# --- Monkeypatching Strategy ---

def patch_instrumentation():
    # 1. Patch Query Rotation (Inline in fused_attention)
    original_fused_attention = mlx_isoquant.IsoQuantKVCache.fused_attention

    def timed_fused_attention(self, queries, scale, mask=None):
        if self.compressed_keys is None or self._seq_len == 0:
            return mx.zeros_like(queries)
        
        B, H_q, _, D = queries.shape
        H_kv = self.num_heads
        T = self.offset if self._cache_mode == "prealloc" else self.compressed_keys["indices"].shape[1]
        repeats = H_q // H_kv if H_q != H_kv else 1

        if not self.supports_fused_attention:
            return original_fused_attention(self, queries, scale, mask)

        # --- Time query_rotation ---
        t0 = sync_time()
        R_T = mx.swapaxes(self.rotation_matrices, -2, -1)
        q_flat = queries[0, :, 0, :]
        R_T_exp = mx.repeat(R_T, repeats, axis=0) if repeats > 1 else R_T
        q_rot = mx.squeeze(mx.matmul(q_flat[:, None, :], R_T_exp), axis=1)
        mx.eval(q_rot)
        record_time("query_rotation", t0)

        # --- Try Metal-fused kernels ---
        if self._fused_metal_ok is not False:
            try:
                # We time the internals of _fused_attention_metal separately
                out = self._fused_attention_metal(q_rot, scale, mask, H_q, H_kv, T, D, repeats)
                self._fused_metal_ok = True
                return out[None, :, None, :].astype(queries.dtype)
            except Exception:
                self._fused_metal_ok = False

        return self._fused_attention_mlx(q_rot, scale, mask, H_q, H_kv, T, D, repeats)[None, :, None, :].astype(queries.dtype)

    mlx_isoquant.IsoQuantKVCache.fused_attention = timed_fused_attention

    # 2. Patch fused_attention_npt8_tiled (Kernel, Merge, Inverse)
    original_kernel_call = fkdt._get_tiled_kernel
    def wrapped_get_kernel():
        k = original_kernel_call()
        class KernelWrapper:
            def __call__(self, *a, **kw):
                t0 = sync_time()
                res = k(*a, **kw)
                mx.eval(res)
                record_time("metal_kernel", t0)
                return res
        return KernelWrapper()
    fkdt._get_tiled_kernel = wrapped_get_kernel

    original_merge = fkdt._fa2_merge
    def timed_merge(*a, **kw):
        t0 = sync_time()
        res = original_merge(*a, **kw)
        mx.eval(res)
        record_time("fa2_merge", t0)
        return res
    fkdt._fa2_merge = timed_merge

    original_inv = mlx_isoquant.structured_rotate_inverse
    def timed_inv(*a, **kw):
        t0 = sync_time()
        res = original_inv(*a, **kw)
        mx.eval(res)
        record_time("inverse_rotation", t0)
        return res
    mlx_isoquant.structured_rotate_inverse = timed_inv

    # 3. Patch Compression & Packing
    from mlx_lm.models.fused_kv_decode_kernels import pack_indices_3bit
    original_pack = pack_indices_3bit
    def timed_pack(*a, **kw):
        t0 = sync_time()
        res = original_pack(*a, **kw)
        mx.eval(res)
        record_time("compression_and_packing", t0)
        return res
    import mlx_lm.models.fused_kv_decode_kernels as fkdk
    fkdk.pack_indices_3bit = timed_pack

def run_phase(model, cache, steps, name):
    print(f"  Running {name} phase...")
    y = mx.array([[42]])
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        out = model(y, cache=cache)
        mx.eval(out)
    mx.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000.0
    return total_ms / steps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--roofline", required=True)
    parser.add_argument("--decode-steps", type=int, default=100)
    parser.add_argument("--capture-traces", action="store_true")
    args = parser.parse_args()

    results = {
        "model": args.model,
        "decode_steps": args.decode_steps,
        "configs": []
    }

    print(f"Loading model {args.model}...")
    model, _ = load(args.model)

    for T in [4096, 8192]:
        print(f"\n--- Context T={T} ---")
        config_results = {"T": T}
        
        # Phase A: Default KV baseline
        print("  Creating default KV cache...")
        cache_default = make_prompt_cache(model, kv_cache_type="default")
        prompt = mx.array([1] * T)
        mx.eval(model(prompt[None, :], cache=cache_default))
        finalize_deferred_kv_caches(cache_default)
        
        default_ms = run_phase(model, cache_default, args.decode_steps, "Baseline (Default)")
        config_results["default_total_ms_per_step"] = default_ms

        # Phase B: IsoQuant Unpatched
        print("  Creating IsoQuant cache...")
        cache_iso = make_prompt_cache(model, kv_cache_type="isoquant")
        mx.eval(model(prompt[None, :], cache=cache_iso))
        finalize_deferred_kv_caches(cache_iso)
        
        iso_unpatched_ms = run_phase(model, cache_iso, args.decode_steps, "IsoQuant Unpatched")
        config_results["iso_unpatched_ms_per_step"] = iso_unpatched_ms

        # Phase C: IsoQuant Instrumented
        component_times.clear()
        patch_instrumentation()
        iso_instrumented_ms = run_phase(model, cache_iso, args.decode_steps, "IsoQuant Instrumented")
        config_results["iso_instrumented_ms_per_step"] = iso_instrumented_ms
        
        # Calculate decomposition
        config_results["instrumentation_overhead_pct"] = (iso_instrumented_ms - iso_unpatched_ms) / iso_unpatched_ms * 100.0
        
        components = []
        sum_components = 0
        gap = iso_unpatched_ms - default_ms
        
        scale_factor = iso_unpatched_ms / iso_instrumented_ms if iso_instrumented_ms > 0 else 1.0

        for name, times in component_times.items():
            total_avg_ms_per_step = sum(times) / args.decode_steps
            sum_components += total_avg_ms_per_step
            
            components.append({
                "name": name,
                "avg_ms_per_step": total_avg_ms_per_step,
                "pct_of_instrumented_total": total_avg_ms_per_step / iso_instrumented_ms * 100.0,
                "gap_attribution_pct": (total_avg_ms_per_step * scale_factor) / gap * 100.0 if gap > 0 else 0
            })
        
        config_results["components"] = components
        config_results["residual_ms"] = iso_instrumented_ms - sum_components
        
        results["configs"].append(config_results)

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    if args.capture_traces:
        print("\nAttempting GPU trace capture...")
        try:
            mx.metal.start_capture(f"artifacts/branch_c_profiling/iso_T8192.gputrace")
            mx.metal.stop_capture()
            print("Trace saved.")
        except Exception as e:
            print(f"Trace capture failed: {e}")

if __name__ == "__main__":
    main()
