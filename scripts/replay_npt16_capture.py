"""Offline replay of a captured Kimi K2.6 decode-step.

Loads a .npz produced by mlx_isoquant._capture_decode_state (triggered by
ISOQUANT_CAPTURE_PATH env during a real validation run), reconstructs the
exact decode-step state, runs the inputs through both _fused_attention_metal
and _fused_attention_mlx on the same materialized cache, and reports
per-stage divergence.

Usage:
  python3 scripts/replay_npt16_capture.py <path/to/capture.npz>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mlx-lm"))

import mlx.core as mx  # noqa: E402

from mlx_lm.models.mlx_isoquant import IsoQuantKVCache  # noqa: E402


def _l_inf(a, b):
    return float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max())


def _cosine(a, b):
    af = a.astype(mx.float32).reshape(-1)
    bf = b.astype(mx.float32).reshape(-1)
    n = mx.sqrt((af * af).sum() * (bf * bf).sum())
    return float((af * bf).sum() / (n + 1e-30))


def _mse(a, b):
    diff = (a.astype(mx.float32) - b.astype(mx.float32)).reshape(-1)
    return float((diff * diff).mean())


def _stats(label, a, b):
    return {
        "stage": label,
        "shape": list(a.shape),
        "linf": _l_inf(a, b),
        "mse": _mse(a, b),
        "cosine": _cosine(a, b),
    }


def reconstruct_cache_from_capture(cap: dict) -> IsoQuantKVCache:
    """Build an IsoQuantKVCache and patch its state to mirror what the live
    cache contained at capture time. Avoids re-running update_and_fetch /
    finalize since we already have the post-finalize compressed state."""
    H_kv = int(cap["meta_H_kv"][0])
    D = int(cap["meta_D"][0])
    bit_width = int(cap["meta_bit_width"][0])

    cache = IsoQuantKVCache(
        num_heads=H_kv,
        head_dim=D,
        bit_width=bit_width,
        max_seq_len=512,
        layer_idx=int(cap["meta_layer_idx"][0]),
        codebook_dir=str(
            Path(__file__).resolve().parent.parent
            / "mlx-lm/mlx_lm/models/turboquant_codebooks"
        ),
        seed=42,
    )

    T = int(cap["meta_T"][0])

    # Patch in the captured rotation matrices and centroids so the cache uses
    # exactly what the live system had — not freshly-generated ones.
    cache.rotation_matrices = mx.array(cap["rotation_matrices"])
    cache.compressor.centroids = mx.array(cap["centroids"])

    # Reconstruct compressed_keys / compressed_values from capture
    cache.compressed_keys = {
        "indices": mx.array(cap["k_indices"]),
        "x_norm": mx.array(cap["k_xnorm"]),
    }
    cache.compressed_values = {
        "indices": mx.array(cap["v_indices"]),
        "x_norm": mx.array(cap["v_xnorm"]),
    }
    cache._seq_len = T
    cache.offset = T
    cache._deferred = False

    # Cache mode reflection
    cache_mode = bytes(cap["meta_cache_mode"][0]).decode("utf-8")
    cache._cache_mode = cache_mode

    return cache


def run_paths(cache, q_rot, mask, scale, T, H_q, H_kv, D, repeats):
    """Call both fused paths with explicit q_rot (skip the rotation step
    since q_rot is already from the capture)."""
    # Reset failure latch so kernel is re-attempted
    cache._fused_metal_ok = None

    out_kernel = cache._fused_attention_metal(
        q_rot, scale, mask, H_q, H_kv, T, D, repeats
    )
    out_fb = cache._fused_attention_mlx(q_rot, scale, mask, H_q, H_kv, T, D, repeats)
    mx.eval(out_kernel, out_fb)
    return out_kernel, out_fb


def fallback_with_intermediates(cache, q_rot, mask, scale, T, H_q, H_kv, D, repeats):
    """Step-by-step reproduction of _fused_attention_mlx, returning per-stage
    intermediates so we can localize divergence."""
    if cache._cache_mode == "prealloc":
        k_indices = cache.compressed_keys["indices"][:, :T, :]
        k_xnorm = cache.compressed_keys["x_norm"][:, :T, :]
        v_indices = cache.compressed_values["indices"][:, :T, :]
        v_xnorm = cache.compressed_values["x_norm"][:, :T, :]
    else:
        k_indices = cache.compressed_keys["indices"]
        k_xnorm = cache.compressed_keys["x_norm"]
        v_indices = cache.compressed_values["indices"]
        v_xnorm = cache.compressed_values["x_norm"]

    k_rot_quant = cache.compressor.centroids[k_indices]
    k_rot_scaled = k_rot_quant * k_xnorm.astype(mx.float32)
    v_rot_quant = cache.compressor.centroids[v_indices]
    v_rot_scaled = v_rot_quant * v_xnorm.astype(mx.float32)

    if repeats > 1:
        k_rot_scaled = mx.repeat(k_rot_scaled, repeats, axis=0)
        v_rot_scaled = mx.repeat(v_rot_scaled, repeats, axis=0)

    scores = mx.matmul(q_rot[:, None, :], mx.swapaxes(k_rot_scaled, -2, -1)).squeeze(1)
    scores = scores * scale

    if mask is not None:
        m = mask
        target_ndim = scores.ndim
        while m.ndim > target_ndim:
            sz1 = next((i for i, s in enumerate(m.shape) if s == 1), None)
            if sz1 is None:
                break
            m = m.squeeze(sz1)
        scores_post_mask = scores + m
    else:
        scores_post_mask = scores

    attn_weights = mx.softmax(scores_post_mask, axis=-1)
    output_rot = mx.matmul(attn_weights[:, None, :], v_rot_scaled).squeeze(
        1
    )  # (H_q, D)

    return {
        "k_rot_scaled": k_rot_scaled,
        "v_rot_scaled": v_rot_scaled,
        "scores_pre_mask": scores,
        "scores_post_mask": scores_post_mask,
        "attn_weights": attn_weights,
        "output_rot_pre_inverse": output_rot,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: replay_npt16_capture.py <capture.npz> [--out replay.json]")
        sys.exit(2)

    cap_path = sys.argv[1]
    out_path = None
    if "--out" in sys.argv:
        out_path = sys.argv[sys.argv.index("--out") + 1]
    else:
        out_path = Path(cap_path).with_suffix("").as_posix() + "_replay.json"

    print("=== NPT=16 capture replay ===")
    print(f"Capture: {cap_path}")
    print()

    cap = np.load(cap_path, allow_pickle=True)
    print("Capture contents:")
    for k in cap.files:
        a = cap[k]
        try:
            print(f"  {k:>20s}: shape={a.shape} dtype={a.dtype}")
        except Exception:
            print(f"  {k:>20s}: <opaque>")
    print()

    T = int(cap["meta_T"][0])
    H_q = int(cap["meta_H_q"][0])
    H_kv = int(cap["meta_H_kv"][0])
    D = int(cap["meta_D"][0])
    repeats = int(cap["meta_repeats"][0])
    scale = float(cap["scale"][0])
    layer_idx = int(cap["meta_layer_idx"][0])
    bit_width = int(cap["meta_bit_width"][0])
    mask_present = bool(cap["mask_present"][0])

    print(
        f"T={T} H_q={H_q} H_kv={H_kv} D={D} repeats={repeats} "
        f"scale={scale:.6f} layer={layer_idx} bits={bit_width} "
        f"mask_present={mask_present}"
    )
    print()

    t0 = time.time()
    cache = reconstruct_cache_from_capture(cap)
    print(f"Cache reconstruct: {time.time() - t0:.2f}s")
    print(f"  cache supports_fused_attention: {cache.supports_fused_attention}")
    print(f"  cache._fused_metal_ok: {cache._fused_metal_ok}")

    q_rot = mx.array(cap["q_rot"])
    mask = mx.array(cap["mask"]) if mask_present else None
    if mask is not None:
        print(
            f"  mask shape={tuple(mask.shape)} dtype={mask.dtype} "
            f"min={float(mask.min()):.4f} max={float(mask.max()):.4f}"
        )
    print(f"  q_rot |max|={float(mx.abs(q_rot).max()):.4f}")
    print()

    # Force NPT16 path
    import os

    os.environ["ISOQUANT_USE_NPT16_FUSED"] = "1"

    t0 = time.time()
    out_kernel, out_fb = run_paths(cache, q_rot, mask, scale, T, H_q, H_kv, D, repeats)
    print(f"Both paths: {time.time() - t0:.2f}s")
    print()

    # Path 3: reconstruct + SDPA — what the pre-state-mirror-fix code took
    # when supports_fused_attention=False. This is what produced the original
    # 0.8727 baseline. Hypothesis 2 says this differs from _fused_attention_mlx
    # numerically even though both are "correct" — different rotation orders
    # and FP precision compounding.
    print("=== Path 3: reconstruct + scaled_dot_product_attention ===")
    queries = mx.array(cap["queries"])
    t0 = time.time()
    keys_recon = cache.reconstruct_keys()  # (1, H_kv=1, T, D), already rank-4
    values_recon = cache.get_values()  # (1, H_kv=1, T, D)
    from mlx_lm.models.base import scaled_dot_product_attention  # noqa: E402

    # queries shape is (1, H_q=64, 1, D=512); keys/values are (1, H_kv=1, T, D).
    # SDPA broadcasts H_kv → H_q via repeats internally.
    out_recon = scaled_dot_product_attention(
        queries, keys_recon, values_recon, cache=None, scale=scale, mask=mask
    )
    mx.eval(out_recon, keys_recon, values_recon)
    print(f"reconstruct+SDPA: {time.time() - t0:.2f}s")
    print(f"  out_recon shape: {tuple(out_recon.shape)}")
    print()

    # Three-way comparison
    print("=== Three-way path comparison ===")
    e2e_kernel_vs_fb = _stats("kernel_vs_fb", out_kernel, out_fb)
    e2e_kernel_vs_recon = _stats("kernel_vs_recon", out_kernel, out_recon)
    e2e_fb_vs_recon = _stats("fb_vs_recon", out_fb, out_recon)
    print(
        f"  kernel vs fb:    linf={e2e_kernel_vs_fb['linf']:.6e}  "
        f"cos={e2e_kernel_vs_fb['cosine']:.6f}"
    )
    print(
        f"  kernel vs recon: linf={e2e_kernel_vs_recon['linf']:.6e}  "
        f"cos={e2e_kernel_vs_recon['cosine']:.6f}"
    )
    print(
        f"  fb vs recon:     linf={e2e_fb_vs_recon['linf']:.6e}  "
        f"cos={e2e_fb_vs_recon['cosine']:.6f}"
    )
    print()

    # End-to-end (post-inverse-rotation outputs from each path)
    e2e = _stats("end_to_end_post_inverse", out_kernel, out_fb)
    print(
        f"END-TO-END (kernel vs fb)  linf={e2e['linf']:.6e}  mse={e2e['mse']:.6e}  "
        f"cos={e2e['cosine']:.6f}"
    )
    print()

    # Per-stage MLX intermediates (truth side; kernel intermediates not exposed)
    t0 = time.time()
    fb_inter = fallback_with_intermediates(
        cache, q_rot, mask, scale, T, H_q, H_kv, D, repeats
    )
    mx.eval(*fb_inter.values())
    print(f"Fallback intermediates: {time.time() - t0:.2f}s")
    print()

    # Stat-summary of each intermediate
    print("=== Fallback intermediate stats (sanity) ===")
    rows_intermediate = []
    for name, a in fb_inter.items():
        stat = {
            "stage": name,
            "shape": list(a.shape),
            "max_abs": float(mx.abs(a).max()),
            "mean": float(a.mean()),
            "std": float(a.std()),
        }
        rows_intermediate.append(stat)
        print(
            f"  {name:>26s}  shape={tuple(a.shape)!s:>22s}  "
            f"|max|={stat['max_abs']:8.4f}  mean={stat['mean']:+.4e}  "
            f"std={stat['std']:.4e}"
        )
    print()

    # Attention-sharpness diagnostic
    aw = fb_inter["attn_weights"]  # (H_q, T)
    mean_max_aw = float(aw.max(axis=-1).mean())
    p99 = float(mx.sort(aw.reshape(-1))[-(aw.size // 100)])
    print(
        f"Attention sharpness: mean_max_attn_weight={mean_max_aw:.4f}  "
        f"99th-percentile-aw={p99:.4f}"
    )
    print()

    # Bisect: which stage of the kernel could deviate?
    # Since we can't get kernel intermediates, instead test variants of
    # the fallback that mimic kernel-style decisions:
    #
    # B1. Recompute output_rot using a per-token loop instead of matmul
    #     (mimics kernel's serial accumulation order).
    print("=== Bisection: per-token serial vs batched matmul ===")
    k_rs = fb_inter["k_rot_scaled"]
    v_rs = fb_inter["v_rot_scaled"]
    aw_pre = fb_inter["attn_weights"]
    # Batched matmul (same as fallback)
    out_rot_batched = mx.matmul(aw_pre[:, None, :], v_rs).squeeze(1)
    # Per-token serial (mimics kernel's online accumulation, but in fp32 here)
    out_rot_serial = mx.zeros_like(out_rot_batched)
    # Use vectorized accumulation per t (still via MLX, just changes summation order)
    for t in range(T):
        out_rot_serial = out_rot_serial + (aw_pre[:, t : t + 1] * v_rs[:, t, :])
    mx.eval(out_rot_batched, out_rot_serial)
    bisect_b1 = _stats("output_rot_serial_vs_matmul", out_rot_serial, out_rot_batched)
    print(
        f"  matmul vs serial accumulation  linf={bisect_b1['linf']:.6e}  "
        f"cos={bisect_b1['cosine']:.6f}"
    )
    print()

    # B2. Recompute scores via per-token QK dot vs matmul
    print("=== Bisection: per-token QK dot vs matmul ===")
    s_matmul = mx.matmul(q_rot[:, None, :], mx.swapaxes(k_rs, -2, -1)).squeeze(1)
    s_serial = mx.zeros_like(s_matmul)
    for t in range(T):
        s_serial = mx.concatenate(
            [
                s_serial[:, :t],
                (q_rot * k_rs[:, t, :]).sum(axis=-1, keepdims=True),
                s_serial[:, t + 1 :],
            ],
            axis=-1,
        )
    # Easier: vectorized over t with explicit broadcast (skip the slicing)
    s_serial2 = (q_rot[:, None, :] * k_rs).sum(axis=-1)  # (H_q, T)
    mx.eval(s_matmul, s_serial2)
    bisect_b2 = _stats("scores_serial_vs_matmul", s_serial2, s_matmul)
    print(
        f"  QK matmul vs broadcast-sum  linf={bisect_b2['linf']:.6e}  "
        f"cos={bisect_b2['cosine']:.6f}"
    )
    print()

    payload = {
        "capture_path": cap_path,
        "shapes": {
            "T": T,
            "H_q": H_q,
            "H_kv": H_kv,
            "D": D,
            "repeats": repeats,
            "bit_width": bit_width,
            "layer_idx": layer_idx,
            "mask_present": mask_present,
        },
        "scale": scale,
        "end_to_end": e2e,
        "fallback_intermediates": rows_intermediate,
        "attention_sharpness": {
            "mean_max_attn_weight": mean_max_aw,
            "p99_attn_weight": p99,
        },
        "bisect": {
            "output_rot_serial_vs_matmul": bisect_b1,
            "scores_serial_vs_matmul": bisect_b2,
        },
    }
    if mask is not None:
        payload["mask_stats"] = {
            "shape": list(mask.shape),
            "min": float(mask.min()),
            "max": float(mask.max()),
            "mean": float(mask.mean()),
            "abs_max": float(mx.abs(mask).max()),
        }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved replay artifact: {out_path}")


if __name__ == "__main__":
    main()
