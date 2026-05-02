#!/usr/bin/env python3
"""Low-overhead Metal profiling for IsoQuant decode gap attribution.

The script has three phases:

1. Synthetic component isolation. No model is loaded. Each read-path and
   write-path component is timed by itself with one fence per iteration,
   and with a serial 10x batch to estimate the cost of ten IsoQuant layers
   without ten synchronization fences.
2. Minimal-fence end-to-end decode timing. The model is loaded once and each
   cache type is timed across many decode steps with only an outer fence pair.
3. Optional GPU trace capture for Xcode Instruments.

The synthetic path targets the Qwen3.6-35B-A3B-nvfp4 IsoQuant shape:
head_dim=256, H_kv=2, H_q=16, and ten IsoQuant layers.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MLX_LM_ROOT = REPO_ROOT / "mlx-lm"

HEAD_DIM = 256
H_KV = 2
H_Q = 16
ISOQUANT_LAYERS = 10
TILE_SIZE = 256

ENV_DEFAULTS = {
    "ISOQUANT_BITS": "3",
    "ISOQUANT_USE_NPT8_FUSED": "1",
    "ISOQUANT_FUSED_ENCODE": "0",
    "ISOQUANT_CACHE_MODE": "concat_append",
    "ISOQUANT_USE_METAL": "0",
}

mx = None
_load_model = None
_make_prompt_cache = None
_finalize_deferred_kv_caches = None
_get_tiled_kernel = None
_fa2_merge = None
_pack_indices_3bit = None
_build_isoquant_rotation_components = None
_structured_rotate_inverse = None
_structured_rotate_forward = None
_quantize_scalar = None
_fused_compress_and_pack = None
_load_codebook = None
_get_default_codebook_dir = None


def _ensure_local_mlx_lm() -> None:
    if MLX_LM_ROOT.exists():
        path = str(MLX_LM_ROOT)
        if path not in sys.path:
            sys.path.insert(0, path)


def import_mlx_modules() -> None:
    """Import MLX lazily so --help and py_compile do not touch Metal."""

    global mx
    global _load_model
    global _make_prompt_cache
    global _finalize_deferred_kv_caches
    global _get_tiled_kernel
    global _fa2_merge
    global _pack_indices_3bit
    global _build_isoquant_rotation_components
    global _structured_rotate_inverse
    global _structured_rotate_forward
    global _quantize_scalar
    global _fused_compress_and_pack
    global _load_codebook
    global _get_default_codebook_dir

    _ensure_local_mlx_lm()

    import mlx.core as _mx
    from mlx_lm import load as load_model
    from mlx_lm.models.cache import finalize_deferred_kv_caches, make_prompt_cache
    from mlx_lm.models.fused_kv_decode_kernels import pack_indices_3bit
    from mlx_lm.models.fused_kv_decode_npt8_tiled import _fa2_merge as fa2_merge
    from mlx_lm.models.fused_kv_decode_npt8_tiled import (
        _get_tiled_kernel as get_tiled_kernel,
    )
    from mlx_lm.models.mlx_isoquant import (
        build_isoquant_rotation_components,
        structured_rotate_forward,
        structured_rotate_inverse,
    )
    from mlx_lm.models.mlx_turboquant import (
        get_default_codebook_dir,  # noqa: F811
        load_codebook,
        quantize_scalar,
    )

    try:
        from mlx_lm.models.fused_kv_compress import (
            fused_compress_and_pack as _fused_cap,
        )
    except Exception:
        _fused_cap = None

    mx = _mx
    _load_model = load_model
    _make_prompt_cache = make_prompt_cache
    _finalize_deferred_kv_caches = finalize_deferred_kv_caches
    _get_tiled_kernel = get_tiled_kernel
    _fa2_merge = fa2_merge
    _pack_indices_3bit = pack_indices_3bit
    _build_isoquant_rotation_components = build_isoquant_rotation_components
    _structured_rotate_inverse = structured_rotate_inverse
    _structured_rotate_forward = structured_rotate_forward
    _quantize_scalar = quantize_scalar
    _fused_compress_and_pack = _fused_cap
    _load_codebook = load_codebook
    _get_default_codebook_dir = get_default_codebook_dir


@contextmanager
def env_overrides(overrides: dict[str, str]):
    originals = {}
    merged = {**ENV_DEFAULTS, **overrides}
    for key, value in merged.items():
        originals[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, value in originals.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def clear_kernel_caches() -> None:
    """Clear cached Metal kernels that are sensitive to env-var toggles."""

    import importlib

    for mod_path, attr in [
        ("mlx_lm.models.fused_kv_compress", "_kernel_cache"),
        ("mlx_lm.models.fused_kv_decode_npt8_tiled", "_tiled_kernel_cache"),
        ("mlx_lm.models.fused_kv_decode_kernels", "_fused_kernel_cache"),
        ("mlx_lm.models.isoquant_metal_kernels", "_kernel_cache"),
    ]:
        try:
            mod = importlib.import_module(mod_path)
            getattr(mod, attr, {}).clear()
        except Exception:
            pass


def eval_result(result: Any) -> Any:
    if isinstance(result, (tuple, list)):
        mx.eval(*result)
    else:
        mx.eval(result)
    return result


def eval_many_results(results: list[Any]) -> list[Any]:
    arrays = []
    for result in results:
        if isinstance(result, (tuple, list)):
            arrays.extend(result)
        else:
            arrays.append(result)
    if arrays:
        mx.eval(*arrays)
    return results


def summarize_times(times_ms: list[float]) -> dict[str, Any]:
    arr = np.array(times_ms, dtype=np.float64)
    return {
        "median_ms": float(np.median(arr)),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "p25_ms": float(np.percentile(arr, 25)),
        "p75_ms": float(np.percentile(arr, 75)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "samples_ms": [float(x) for x in arr],
    }


def bench_component(
    fn: Callable[[], Any],
    *,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        eval_result(fn())
    mx.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        eval_result(fn())
        mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return summarize_times(times)


def bench_serial_10x(
    fns: list[Callable[[], Any]],
    *,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    if len(fns) != 10:
        raise ValueError(
            f"serial 10x benchmark requires exactly 10 functions, got {len(fns)}"
        )
    for _ in range(warmup):
        results = [fn() for fn in fns]
        eval_many_results(results)
    mx.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        results = [fn() for fn in fns]
        eval_many_results(results)
        mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return summarize_times(times)


def with_fence_correction(stats: dict[str, Any], fence_ms: float) -> dict[str, Any]:
    corrected = dict(stats)
    corrected["fence_corrected_ms"] = max(0.0, stats["median_ms"] - fence_ms)
    corrected["fence_corrected_p25_ms"] = max(0.0, stats["p25_ms"] - fence_ms)
    corrected["fence_corrected_p75_ms"] = max(0.0, stats["p75_ms"] - fence_ms)
    return corrected


def measure_fence_overhead(warmup: int, iters: int) -> dict[str, Any]:
    def null_eval():
        return mx.zeros((1,), dtype=mx.float32)

    stats = bench_component(null_eval, warmup=warmup, iters=iters)
    stats["method"] = "mx.eval(mx.zeros(1)); mx.synchronize()"
    return stats


def synthetic_d256(T: int, seed: int) -> tuple[Any, ...]:
    """Synthetic packed 3-bit KV data matching tests/conftest_npt8.py."""

    rng = np.random.default_rng(seed)
    indices_k = mx.array(rng.integers(0, 8, (H_KV, T, HEAD_DIM), dtype=np.uint8))
    indices_v = mx.array(rng.integers(0, 8, (H_KV, T, HEAD_DIM), dtype=np.uint8))
    norms_k = mx.array(rng.standard_normal((H_KV, T)).astype(np.float32))
    norms_v = mx.array(rng.standard_normal((H_KV, T)).astype(np.float32))
    centroids = mx.array(np.linspace(-1.5, 1.5, 8, dtype=np.float32))
    q_rot = mx.array(rng.standard_normal((H_Q, HEAD_DIM)).astype(np.float32))
    kv_head_map = mx.arange(H_Q, dtype=mx.uint32) // (H_Q // H_KV)
    k_packed = _pack_indices_3bit(indices_k)
    v_packed = _pack_indices_3bit(indices_v)
    mx.eval(k_packed, v_packed, norms_k, norms_v, centroids, q_rot, kv_head_map)
    mx.synchronize()
    return k_packed, v_packed, centroids, norms_k, norms_v, q_rot, kv_head_map


def build_synthetic_fixture(T: int, layer_seed: int) -> dict[str, Any]:
    k_packed, v_packed, centroids, norms_k, norms_v, q_rot, kv_head_map = (
        synthetic_d256(T, seed=11 + layer_seed)
    )
    comps = _build_isoquant_rotation_components(
        H_KV,
        HEAD_DIM,
        seed=42,
        layer_idx=layer_seed,
        apply_global_mix=True,
    )
    rotation_matrices = comps["rotation_matrices"]
    block_matrices = comps["block_matrices"]
    use_hadamard = bool(comps["use_hadamard"])

    repeats = H_Q // H_KV
    rotation_t = mx.swapaxes(rotation_matrices, -2, -1)
    rotation_t_expanded = mx.repeat(rotation_t, repeats, axis=0)
    expanded_blocks = mx.take(block_matrices, kv_head_map, axis=0)

    rng = np.random.default_rng(23 + layer_seed)
    q = mx.array(rng.standard_normal((H_Q, HEAD_DIM)).astype(np.float32))

    num_tiles = (T + TILE_SIZE - 1) // TILE_SIZE
    kernel = _get_tiled_kernel()
    scale_arr = mx.array([1.0 / math.sqrt(HEAD_DIM)], dtype=mx.float32)
    seq_len_arr = mx.array([T], dtype=mx.uint32)
    stride_arr = mx.array([T], dtype=mx.uint32)
    tile_arr = mx.array([TILE_SIZE], dtype=mx.uint32)
    heads_arr = mx.array([H_Q], dtype=mx.uint32)
    has_mask_arr = mx.array([0], dtype=mx.uint32)
    mask_flat = mx.zeros((1,), dtype=mx.float32)

    def tiled_kernel_call():
        return kernel(
            inputs=[
                k_packed.reshape(-1),
                v_packed.reshape(-1),
                centroids.reshape(-1),
                norms_k.reshape(-1),
                norms_v.reshape(-1),
                q_rot.reshape(-1),
                kv_head_map.reshape(-1),
                scale_arr,
                seq_len_arr,
                stride_arr,
                tile_arr,
                heads_arr,
                has_mask_arr,
                mask_flat,
            ],
            output_shapes=[
                (num_tiles * H_Q * HEAD_DIM,),
                (num_tiles * H_Q * 2,),
            ],
            output_dtypes=[mx.float32, mx.float32],
            grid=(num_tiles * 32, H_Q, 1),
            threadgroup=(32, 1, 1),
        )

    o_partials, ml_partials = tiled_kernel_call()
    mx.eval(o_partials, ml_partials)
    merged = _fa2_merge(o_partials, ml_partials, num_tiles, H_Q, HEAD_DIM)
    mx.eval(merged)
    mx.synchronize()

    # --- Write-path synthetic data ---
    # Single decode token: (H_KV, 1, HEAD_DIM) FP32 KV input
    rng2 = np.random.default_rng(99 + layer_seed)
    kv_new = mx.array(rng2.standard_normal((H_KV, 1, HEAD_DIM)).astype(np.float32))
    # block_matrices_t for forward rotation (transpose of block_matrices)
    block_matrices_t = comps.get("block_matrices_t")
    if block_matrices_t is None:
        block_matrices_t = mx.swapaxes(block_matrices, -2, -1)

    # Load codebook for quantize_scalar
    centroids_full, boundaries = _load_codebook(
        HEAD_DIM, 3, _get_default_codebook_dir()
    )
    mx.eval(centroids_full, boundaries, kv_new, block_matrices_t)

    # Pre-existing cache buffers for concat simulation (shape matches T tokens)
    existing_indices = mx.zeros((H_KV, T, HEAD_DIM), dtype=mx.uint8)
    existing_norms = mx.zeros((H_KV, T, 1), dtype=mx.float16)
    existing_packed = mx.zeros((H_KV, T, k_packed.shape[-1]), dtype=mx.uint8)
    new_indices_1 = mx.array(rng2.integers(0, 8, (H_KV, 1, HEAD_DIM), dtype=np.uint8))
    new_norms_1 = mx.array(rng2.standard_normal((H_KV, 1, 1)).astype(np.float16))
    new_packed_1 = _pack_indices_3bit(new_indices_1)

    # Prealloc buffers: oversize by 256 so slice-assign works
    prealloc_indices = mx.zeros((H_KV, T + 256, HEAD_DIM), dtype=mx.uint8)
    prealloc_norms = mx.zeros((H_KV, T + 256, 1), dtype=mx.float16)
    prealloc_packed = mx.zeros((H_KV, T + 256, k_packed.shape[-1]), dtype=mx.uint8)
    mx.eval(
        existing_indices,
        existing_norms,
        existing_packed,
        new_indices_1,
        new_norms_1,
        new_packed_1,
        prealloc_indices,
        prealloc_norms,
        prealloc_packed,
    )
    mx.synchronize()

    return {
        "T": T,
        "num_tiles": num_tiles,
        "q": q,
        "rotation_t_expanded": rotation_t_expanded,
        "k_packed": k_packed,
        "v_packed": v_packed,
        "centroids": centroids,
        "norms_k": norms_k,
        "norms_v": norms_v,
        "q_rot": q_rot,
        "kv_head_map": kv_head_map,
        "tiled_kernel_call": tiled_kernel_call,
        "o_partials": o_partials,
        "ml_partials": ml_partials,
        "merged": merged,
        "expanded_blocks": expanded_blocks,
        "use_hadamard": use_hadamard,
        # Write-path data
        "kv_new": kv_new,
        "block_matrices": block_matrices,
        "block_matrices_t": block_matrices_t,
        "centroids_full": centroids_full,
        "boundaries": boundaries,
        "existing_indices": existing_indices,
        "existing_norms": existing_norms,
        "existing_packed": existing_packed,
        "new_indices_1": new_indices_1,
        "new_norms_1": new_norms_1,
        "new_packed_1": new_packed_1,
        "prealloc_indices": prealloc_indices,
        "prealloc_norms": prealloc_norms,
        "prealloc_packed": prealloc_packed,
        "shapes": {
            "q": list(q.shape),
            "rotation_t_expanded": list(rotation_t_expanded.shape),
            "k_packed": list(k_packed.shape),
            "v_packed": list(v_packed.shape),
            "o_partials": list(o_partials.shape),
            "ml_partials": list(ml_partials.shape),
            "expanded_blocks": list(expanded_blocks.shape),
            "kv_new": list(kv_new.shape),
            "block_matrices": list(block_matrices.shape),
        },
    }


def component_fns(fixture: dict[str, Any]) -> dict[str, Callable[[], Any]]:
    def query_rotation():
        return mx.matmul(
            fixture["q"][:, None, :],
            fixture["rotation_t_expanded"],
        ).squeeze(1)

    def tiled_kernel():
        return fixture["tiled_kernel_call"]()

    def fa2_merge():
        return _fa2_merge(
            fixture["o_partials"],
            fixture["ml_partials"],
            fixture["num_tiles"],
            H_Q,
            HEAD_DIM,
        )

    def inverse_rotation():
        return _structured_rotate_inverse(
            fixture["merged"],
            fixture["expanded_blocks"],
            fixture["use_hadamard"],
        )

    return {
        "query_rotation": query_rotation,
        "tiled_kernel": tiled_kernel,
        "fa2_merge": fa2_merge,
        "inverse_rotation": inverse_rotation,
    }


def write_component_fns(fixture: dict[str, Any]) -> dict[str, Callable[[], Any]]:
    """Write-path components: compress, pack, cache append (×2 for K+V)."""
    kv_new = fixture["kv_new"]
    block_matrices_t = fixture["block_matrices_t"]
    block_matrices = fixture["block_matrices"]
    centroids_full = fixture["centroids_full"]
    boundaries = fixture["boundaries"]
    use_hadamard = fixture["use_hadamard"]

    def compress_python():
        x_f32 = kv_new.astype(mx.float32)
        x_norm = mx.linalg.norm(x_f32, axis=-1, keepdims=True)
        x_unit = x_f32 / mx.maximum(x_norm, mx.array(1e-8, dtype=mx.float32))
        x_rot = _structured_rotate_forward(x_unit, block_matrices_t, use_hadamard)
        indices, _ = _quantize_scalar(x_rot, centroids_full, boundaries)
        return indices.astype(mx.uint8), x_norm.astype(mx.float16)

    def pack_3bit():
        return _pack_indices_3bit(fixture["new_indices_1"])

    T = fixture["T"]

    def cache_concat():
        idx_cat = mx.concatenate(
            [fixture["existing_indices"], fixture["new_indices_1"]], axis=1
        )
        norm_cat = mx.concatenate(
            [fixture["existing_norms"], fixture["new_norms_1"]], axis=1
        )
        pk_cat = mx.concatenate(
            [fixture["existing_packed"], fixture["new_packed_1"]], axis=1
        )
        return idx_cat, norm_cat, pk_cat

    def cache_prealloc():
        pos = T
        end = pos + 1
        fixture["prealloc_indices"][:, pos:end, :] = fixture["new_indices_1"]
        fixture["prealloc_norms"][:, pos:end, :] = fixture["new_norms_1"]
        fixture["prealloc_packed"][:, pos:end, :] = fixture["new_packed_1"]
        return (
            fixture["prealloc_indices"],
            fixture["prealloc_norms"],
            fixture["prealloc_packed"],
        )

    fns: dict[str, Callable[[], Any]] = {
        "compress_python": compress_python,
        "pack_3bit": pack_3bit,
        "cache_concat": cache_concat,
        "cache_prealloc": cache_prealloc,
    }

    if _fused_compress_and_pack is not None:

        def compress_fused_metal():
            return _fused_compress_and_pack(
                kv_new, block_matrices, centroids_full, boundaries
            )

        fns["compress_fused_metal"] = compress_fused_metal

    return fns


def run_synthetic_phase(
    prefill_values: list[int],
    *,
    warmup: int,
    iters: int,
    serial_iters: int,
    fence_ms: float,
) -> dict[str, Any]:
    synthetic_results = {}
    for T in prefill_values:
        print(f"\nSynthetic component isolation T={T}")
        fixtures = [build_synthetic_fixture(T, layer_seed=i) for i in range(10)]
        fixture = fixtures[0]
        components = {}
        sum_single = 0.0
        sum_serial = 0.0

        # --- Read-path components ---
        read_single_fns = component_fns(fixture)
        read_serial_fns_by_name = {
            name: [component_fns(lf)[name] for lf in fixtures]
            for name in read_single_fns
        }
        read_sum_single = 0.0
        read_sum_serial = 0.0

        for name, fn in read_single_fns.items():
            print(f"  {name} (read)...")
            one_stats = with_fence_correction(
                bench_component(fn, warmup=warmup, iters=iters),
                fence_ms,
            )
            serial_stats = with_fence_correction(
                bench_serial_10x(
                    read_serial_fns_by_name[name],
                    warmup=max(1, warmup // 2),
                    iters=serial_iters,
                ),
                fence_ms,
            )

            one_ms = one_stats["fence_corrected_ms"]
            serial_10x_ms = serial_stats["fence_corrected_ms"]
            dispatch_delta = serial_10x_ms - (10.0 * one_ms)
            component = {
                **one_stats,
                "path": "read",
                "serial_10x_ms": serial_10x_ms,
                "serial_10x_raw_median_ms": serial_stats["median_ms"],
                "serial_10x_p25_ms": serial_stats["fence_corrected_p25_ms"],
                "serial_10x_p75_ms": serial_stats["fence_corrected_p75_ms"],
                "serial_10x_samples_ms": serial_stats["samples_ms"],
                "serial_per_call_ms": serial_10x_ms / 10.0,
                "serial_minus_10x_single_ms": dispatch_delta,
                "serial_minus_10x_single_per_call_ms": dispatch_delta / 10.0,
            }
            components[name] = component
            read_sum_single += one_ms
            read_sum_serial += serial_10x_ms

            print(
                f"    median={component['median_ms']:.3f} ms, "
                f"corrected={one_ms:.3f} ms, "
                f"serial10={serial_10x_ms:.3f} ms"
            )

        # --- Write-path components ---
        write_single_fns = write_component_fns(fixture)
        write_serial_fns_by_name = {
            name: [write_component_fns(lf)[name] for lf in fixtures]
            for name in write_single_fns
        }
        write_sum_single = 0.0
        write_sum_serial = 0.0
        active_write_serial = 0.0
        active_write_single = 0.0

        active_compress = (
            "compress_fused_metal"
            if ENV_DEFAULTS.get("ISOQUANT_FUSED_ENCODE") == "1"
            else "compress_python"
        )
        active_cache = (
            "cache_prealloc"
            if ENV_DEFAULTS.get("ISOQUANT_CACHE_MODE") == "prealloc"
            else "cache_concat"
        )
        active_write_names = {active_compress, "pack_3bit", active_cache}

        for name, fn in write_single_fns.items():
            is_active = name in active_write_names
            tag = "" if is_active else " [inactive]"
            print(f"  {name} (write){tag}...")
            one_stats = with_fence_correction(
                bench_component(fn, warmup=warmup, iters=iters),
                fence_ms,
            )
            serial_stats = with_fence_correction(
                bench_serial_10x(
                    write_serial_fns_by_name[name],
                    warmup=max(1, warmup // 2),
                    iters=serial_iters,
                ),
                fence_ms,
            )

            one_ms = one_stats["fence_corrected_ms"]
            serial_10x_ms = serial_stats["fence_corrected_ms"]
            dispatch_delta = serial_10x_ms - (10.0 * one_ms)
            component = {
                **one_stats,
                "path": "write",
                "active": is_active,
                "serial_10x_ms": serial_10x_ms,
                "serial_10x_raw_median_ms": serial_stats["median_ms"],
                "serial_10x_p25_ms": serial_stats["fence_corrected_p25_ms"],
                "serial_10x_p75_ms": serial_stats["fence_corrected_p75_ms"],
                "serial_10x_samples_ms": serial_stats["samples_ms"],
                "serial_per_call_ms": serial_10x_ms / 10.0,
                "serial_minus_10x_single_ms": dispatch_delta,
                "serial_minus_10x_single_per_call_ms": dispatch_delta / 10.0,
            }
            components[name] = component
            write_sum_single += one_ms
            write_sum_serial += serial_10x_ms
            if is_active:
                active_write_single += one_ms
                active_write_serial += serial_10x_ms

            print(
                f"    median={component['median_ms']:.3f} ms, "
                f"corrected={one_ms:.3f} ms, "
                f"serial10={serial_10x_ms:.3f} ms{tag}"
            )

        # Write-path runs 2x per decode step (once for K, once for V)
        # Prediction uses only active write components
        predicted_10layer = read_sum_serial + active_write_serial * 2.0

        synthetic_results[f"T{T}"] = {
            **components,
            "read_sum_single_ms": read_sum_single,
            "read_sum_serial_ms": read_sum_serial,
            "write_sum_serial_all_ms": write_sum_serial,
            "write_sum_serial_active_ms": active_write_serial,
            "active_write_components": sorted(active_write_names),
            "predicted_10layer_ms": predicted_10layer,
            "num_tiles": fixture["num_tiles"],
            "shapes": fixture["shapes"],
            "method": (
                "predicted_10layer_ms = read_serial_sum + 2 * active_write_serial_sum. "
                "Only mutually-exclusive active write components are summed. "
                "Write runs 2x per step (K + V)."
            ),
        }
        print(
            f"  read serial sum: {read_sum_serial:.3f} ms; "
            f"active write serial sum: {active_write_serial:.3f} ms "
            f"(×2 = {active_write_serial * 2:.3f}); "
            f"predicted 10-layer: {predicted_10layer:.3f} ms"
        )
        gc.collect()
    return synthetic_results


def cache_activation_state(cache: list[Any]) -> dict[str, Any]:
    from mlx_lm.models.mlx_isoquant import IsoQuantKVCache

    info = {"iso_layers": 0, "total_layers": len(cache), "flags": {}}
    flag_sets: dict[str, set[Any]] = {
        "head_dim": set(),
        "bit_width": set(),
        "fused_encode": set(),
        "cache_mode": set(),
        "use_metal_runtime": set(),
        "supports_fused": set(),
    }
    for entry in cache:
        if isinstance(entry, IsoQuantKVCache):
            info["iso_layers"] += 1
            flag_sets["head_dim"].add(entry.head_dim)
            flag_sets["bit_width"].add(entry.bit_width)
            flag_sets["fused_encode"].add(getattr(entry, "_use_fused_encode", False))
            flag_sets["cache_mode"].add(getattr(entry, "_cache_mode", "unknown"))
            flag_sets["use_metal_runtime"].add(
                getattr(entry, "_use_metal_runtime", False)
            )
            flag_sets["supports_fused"].add(
                getattr(entry, "supports_fused_attention", False)
            )

    info["flags"] = {
        key: sorted(values, key=lambda v: str(v)) for key, values in flag_sets.items()
    }
    return info


def make_fresh_cache(model: Any, cache_type: str, T: int) -> list[Any]:
    cache = _make_prompt_cache(model, kv_cache_type=cache_type)
    prompt = mx.array([1] * T)
    mx.eval(model(prompt[None, :], cache=cache))
    _finalize_deferred_kv_caches(cache)
    mx.synchronize()
    return cache


def run_decode_min_fence(model: Any, cache: list[Any], steps: int) -> float:
    y = mx.array([[42]])
    out = model(y, cache=cache)
    mx.eval(out)
    mx.synchronize()

    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        out = model(y, cache=cache)
        mx.eval(out)
    mx.synchronize()
    return (time.perf_counter() - t0) / steps * 1000.0


def run_e2e_phase(
    model: Any, prefill_values: list[int], decode_steps: int
) -> dict[str, Any]:
    results = {}
    for T in prefill_values:
        print(f"\nEnd-to-end minimal-fence T={T}")
        entry: dict[str, Any] = {}

        for cache_type in ["default", "isoquant"]:
            print(f"  {cache_type}...")
            with env_overrides({}):
                clear_kernel_caches()
                cache = make_fresh_cache(model, cache_type, T)
                entry[f"{cache_type}_activation"] = cache_activation_state(cache)
                ms = run_decode_min_fence(model, cache, decode_steps)
                entry[f"{cache_type}_ms"] = ms
                del cache
                gc.collect()
            print(f"    {ms:.3f} ms/step")

        entry["gap_ms"] = entry["isoquant_ms"] - entry["default_ms"]
        results[f"T{T}"] = entry
        print(f"  gap: {entry['gap_ms']:+.3f} ms/step")
    return results


def capture_trace(
    model: Any,
    *,
    cache_type: str,
    T: int,
    steps: int,
    trace_path: Path,
) -> dict[str, Any]:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  trace {cache_type} T={T}: {trace_path}")

    try:
        with env_overrides({}):
            clear_kernel_caches()
            cache = make_fresh_cache(model, cache_type, T)
            y = mx.array([[42]])
            warm = model(y, cache=cache)
            mx.eval(warm)
            mx.synchronize()

            mx.metal.start_capture(str(trace_path))
            try:
                for _ in range(steps):
                    out = model(y, cache=cache)
                    mx.eval(out)
                mx.synchronize()
            finally:
                mx.metal.stop_capture()
            del cache
            gc.collect()
        return {
            "path": str(trace_path),
            "status": "ok",
            "cache_type": cache_type,
            "T": T,
            "steps": steps,
        }
    except Exception as exc:
        return {
            "path": str(trace_path),
            "status": "failed",
            "error": repr(exc),
            "cache_type": cache_type,
            "T": T,
            "steps": steps,
        }


def run_trace_phase(model: Any, output_path: Path, trace_steps: int) -> dict[str, Any]:
    trace_dir = output_path.parent / "traces"
    traces = {}
    print("\nGPU trace capture T=8192")
    for cache_type in ["isoquant", "default"]:
        trace_name = f"{cache_type}_T8192.gputrace"
        traces[f"{cache_type}_T8192"] = capture_trace(
            model,
            cache_type=cache_type,
            T=8192,
            steps=trace_steps,
            trace_path=trace_dir / trace_name,
        )
        status = traces[f"{cache_type}_T8192"]["status"]
        print(f"    {cache_type}: {status}")
    return traces


def compare_prediction_to_e2e(
    synthetic: dict[str, Any], e2e: dict[str, Any]
) -> dict[str, Any]:
    comparison = {}
    for key, syn in synthetic.items():
        if key not in e2e:
            continue
        gap = e2e[key].get("gap_ms")
        pred = syn.get("predicted_10layer_ms")
        if gap is None or pred is None:
            continue
        ratio = pred / gap if abs(gap) > 1e-9 else None
        comparison[key] = {
            "predicted_10layer_ms": pred,
            "e2e_gap_ms": gap,
            "residual_ms": gap - pred,
            "prediction_to_gap_ratio": ratio,
            "within_2x_sanity": bool(ratio is not None and 0.5 <= ratio <= 2.0),
        }
    return comparison


def print_attribution_table(
    synthetic: dict[str, Any], e2e: dict[str, Any], comparison: dict[str, Any]
) -> None:
    read_names = ["query_rotation", "tiled_kernel", "fa2_merge", "inverse_rotation"]
    write_names = [
        "compress_python",
        "compress_fused_metal",
        "pack_3bit",
        "cache_concat",
        "cache_prealloc",
    ]

    print("\nAttribution summary (inactive components marked with *)")
    print(
        "T      path  component            one-layer ms   serial10 ms   "
        "pct of pred   pct of gap"
    )
    print("-" * 92)
    for key in sorted(synthetic):
        syn = synthetic[key]
        pred = syn["predicted_10layer_ms"]
        gap = e2e.get(key, {}).get("gap_ms")

        for path_label, names, multiplier in [
            ("read", read_names, 1),
            ("write", write_names, 2),
        ]:
            for name in names:
                if name not in syn:
                    continue
                comp = syn[name]
                is_active = comp.get("active", True)
                serial = comp["serial_10x_ms"]
                effective = serial * multiplier
                if is_active:
                    pct_pred = effective / pred * 100.0 if pred > 0 else 0.0
                    pct_gap = effective / gap * 100.0 if gap else None
                else:
                    pct_pred = 0.0
                    pct_gap = None
                pct_gap_str = f"{pct_gap:9.1f}%" if pct_gap is not None else "      n/a"
                mult_str = f" (×{multiplier})" if multiplier > 1 else ""
                inactive_mark = " *" if not is_active else ""
                print(
                    f"{key:<6} {path_label:<5} {name:<20}{inactive_mark:2} "
                    f"{comp['fence_corrected_ms']:>11.3f} "
                    f"{serial:>11.3f}{mult_str} "
                    f"{pct_pred:>10.1f}% {pct_gap_str}"
                )

        comp_entry = comparison.get(key)
        if comp_entry:
            print(
                f"{key:<6} {'':5} {'predicted total':<20} {'':>11} "
                f"{pred:>11.3f} {'100.0%':>11} "
                f"{pred / comp_entry['e2e_gap_ms'] * 100.0:>8.1f}%"
            )
            print(
                f"{key:<6} {'':5} {'e2e gap':<20} {'':>11} "
                f"{comp_entry['e2e_gap_ms']:>11.3f}"
            )
        print("-" * 88)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IsoQuant Metal counter-style decode gap attribution"
    )
    parser.add_argument("--model", default=None, help="Path to MLX model")
    parser.add_argument(
        "--output",
        default="artifacts/metal-counters/profile.json",
        help="Output JSON path",
    )
    parser.add_argument("--prefill", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--serial-iters",
        type=int,
        default=None,
        help="Iterations for serial 10x component timing (default: --iters)",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--decode-steps", type=int, default=100)
    parser.add_argument("--trace-steps", type=int, default=10)
    parser.add_argument("--skip-e2e", action="store_true")
    parser.add_argument("--skip-traces", action="store_true")
    parser.add_argument(
        "--fused-encode",
        action="store_true",
        help="Enable ISOQUANT_FUSED_ENCODE=1 for E2E and synthetic phases",
    )
    parser.add_argument(
        "--prealloc",
        action="store_true",
        help="Enable ISOQUANT_CACHE_MODE=prealloc for E2E and synthetic phases",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    if (not args.skip_e2e or not args.skip_traces) and not args.model:
        raise SystemExit(
            "--model is required unless both --skip-e2e and --skip-traces are set"
        )

    if args.fused_encode:
        ENV_DEFAULTS["ISOQUANT_FUSED_ENCODE"] = "1"
    if args.prealloc:
        ENV_DEFAULTS["ISOQUANT_CACHE_MODE"] = "prealloc"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with env_overrides({}):
        import_mlx_modules()

        if args.fused_encode and _fused_compress_and_pack is None:
            raise SystemExit(
                "--fused-encode requires the Metal fused compress kernel "
                "(mlx_lm.models.fused_kv_compress) which failed to import"
            )
        clear_kernel_caches()

        print("Measuring fence overhead...")
        fence_stats = measure_fence_overhead(args.warmup, args.iters)
        fence_ms = fence_stats["median_ms"]
        print(f"  fence median: {fence_ms:.3f} ms")

        serial_iters = (
            args.serial_iters if args.serial_iters is not None else args.iters
        )
        synthetic = run_synthetic_phase(
            args.prefill,
            warmup=args.warmup,
            iters=args.iters,
            serial_iters=serial_iters,
            fence_ms=fence_ms,
        )

        model = None
        e2e = {}
        gpu_traces: dict[str, Any] = {}
        if not args.skip_e2e or not args.skip_traces:
            print(f"\nLoading model {args.model}...")
            model, _ = _load_model(args.model)

        if not args.skip_e2e:
            e2e = run_e2e_phase(model, args.prefill, args.decode_steps)

        if not args.skip_traces:
            gpu_traces = run_trace_phase(model, output_path, args.trace_steps)
        else:
            gpu_traces = {"status": "skipped"}

        comparison = compare_prediction_to_e2e(synthetic, e2e)
        if e2e:
            print_attribution_table(synthetic, e2e, comparison)

        results = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model": args.model,
            "env_defaults": ENV_DEFAULTS,
            "shape": {
                "head_dim": HEAD_DIM,
                "h_kv": H_KV,
                "h_q": H_Q,
                "isoquant_layers": ISOQUANT_LAYERS,
                "tile_size": TILE_SIZE,
            },
            "method": {
                "fence_overhead": "median null mx.eval + mx.synchronize",
                "component_timing": "one fence per isolated iteration",
                "serial_10x": "ten component calls with one final eval/synchronize",
                "e2e": "outer fence pair around decode loop",
            },
            "iters": args.iters,
            "serial_iters": serial_iters,
            "warmup": args.warmup,
            "decode_steps": args.decode_steps,
            "fence_overhead_ms": fence_ms,
            "fence_overhead": fence_stats,
            "synthetic": synthetic,
            "e2e": e2e,
            "comparison": comparison,
            "gpu_traces": gpu_traces,
        }

        with output_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
