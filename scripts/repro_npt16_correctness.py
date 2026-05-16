"""Stage-by-stage correctness diagnostic: NPT=16 fused kernel vs MLX-ops fallback.

Question this answers
---------------------
The first-ever real fused NPT=16 validation run produced top1=0.8000 against the
mlx_ops_fallback's 0.8727 baseline at the same context. The kernel runs end-to-end
(3304/3304 attempts, 0 failures), but produces materially different numerics from
the fallback on the same compressed inputs. This script identifies WHICH stage
of the attention computation hosts the divergence.

Approach
--------
Both `_fused_attention_metal` and `_fused_attention_mlx` are sibling methods on
the SAME IsoQuantKVCache object — they read the same compressed K/V state, take
the same (q_rot, scale, mask) inputs, and should produce algorithmically
identical outputs. Cleanest possible cut:

  1. Build an IsoQuantKVCache, populate it with random K/V via update_and_fetch,
     finalize it.
  2. Synthesize a Kimi-realistic (q, pe_scores).
  3. Call cache._fused_attention_metal(...) → out_kernel
  4. Call cache._fused_attention_mlx(...)   → out_fallback
  5. Compare end-to-end. If divergent, drill into per-stage MLX intermediates.

Hypothesis ladder (refined after reading kernel C source)
---------------------------------------------------------
  H2 (refuted-as-stated, but a refined variant remains):
       Kernel C uses online softmax with max-subtract — naive sum-of-exp is OFF
       the table. BUT the QK reduction uses simd_sum() across 32 threads each
       holding 16 partial sums; MLX matmul uses a different blocking. fp32
       reductions are not associative, and the difference compounds across T
       softmax terms.

  H1: pe_scores additive-mask precision/shape after the squeeze fix.

  H3: Inverse SO(4) rotation order — kernel does it inline per-thread (4 blocks
      per lane), fallback uses _apply_inverse_rotation. Block index mapping
      could differ.

Output
------
artifacts/kimi_k26_profiling/npt16_correctness_repro.json + console table.
Runtime ~30s, no model weights needed.
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

# Shape constants matching real Kimi K2.6 MLA at decode
H_Q = 64
H_KV = 1
D = 512
T = 64
BIT_WIDTH = 3


def _l_inf(a: mx.array, b: mx.array) -> float:
    return float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max())


def _cosine(a: mx.array, b: mx.array) -> float:
    af = a.astype(mx.float32).reshape(-1)
    bf = b.astype(mx.float32).reshape(-1)
    n = mx.sqrt((af * af).sum() * (bf * bf).sum())
    return float((af * bf).sum() / (n + 1e-30))


def _mse(a: mx.array, b: mx.array) -> float:
    diff = (a.astype(mx.float32) - b.astype(mx.float32)).reshape(-1)
    return float((diff * diff).mean())


def _stat_row(label: str, a: mx.array, b: mx.array) -> dict:
    return {
        "stage": label,
        "shape": list(a.shape),
        "linf": _l_inf(a, b),
        "mse": _mse(a, b),
        "cosine": _cosine(a, b),
    }


def build_cache_with_random_kv(seed: int = 42) -> IsoQuantKVCache:
    """Construct an IsoQuantKVCache, write T random K/V tokens, finalize."""
    rng = np.random.default_rng(seed)

    cache = IsoQuantKVCache(
        num_heads=H_KV,
        head_dim=D,
        bit_width=BIT_WIDTH,
        max_seq_len=512,
        layer_idx=0,
        codebook_dir=str(
            Path(__file__).resolve().parent.parent
            / "mlx-lm/mlx_lm/models/turboquant_codebooks"
        ),
        seed=seed,
    )

    # Synthesize random K and V latents (post-RMSNorm-like distribution)
    # Shape conventions: update_and_fetch wants (B=1, num_heads, T, head_dim)
    K = mx.array(rng.standard_normal((1, H_KV, T, D)).astype(np.float32))
    V = mx.array(rng.standard_normal((1, H_KV, T, D)).astype(np.float32))

    # Drive update_and_fetch through prefill (deferred) → finalize_deferred_prefill
    cache.update_and_fetch(K, V)
    cache.finalize_deferred_prefill()

    return cache


def synthesize_query_and_mask(seed: int = 42):
    """Build a Kimi-realistic (queries, mask) for cache.fused_attention."""
    rng = np.random.default_rng(seed + 1)
    # queries: (B=1, H_q=64, 1, D=512) — same shape as Kimi MLA decode-step query
    queries = mx.array(rng.standard_normal((1, H_Q, 1, D)).astype(np.float32))
    # pe_scores additive mask: (1, H_q, 1, T) — what deepseek_v3.py:146 produces
    pe_scores = mx.array((rng.standard_normal((1, H_Q, 1, T)) * 0.5).astype(np.float32))
    scale = 1.0 / np.sqrt(64.0)  # Kimi MLA scale = 1/sqrt(qk_nope_dim=64)
    return queries, pe_scores, scale


def call_kernel_path(cache, queries, pe_scores, scale):
    """Mirror cache.fused_attention's setup, then explicitly route to the
    Metal kernel only. Returns final output (B, H_q, 1, D)."""
    B, H_q, _, D_ = queries.shape
    H_kv = cache.num_heads
    T_ = (
        cache.offset
        if cache._cache_mode == "prealloc"
        else (cache.compressed_keys["indices"].shape[1])
    )
    repeats = H_q // H_kv if H_q != H_kv else 1

    R_T = mx.swapaxes(cache.rotation_matrices, -2, -1)
    q_flat = queries[0, :, 0, :]
    R_T_exp = mx.repeat(R_T, repeats, axis=0) if repeats > 1 else R_T
    q_rot = mx.squeeze(mx.matmul(q_flat[:, None, :], R_T_exp), axis=1)  # (H_q, D)

    out_2d = cache._fused_attention_metal(
        q_rot, scale, pe_scores, H_q, H_kv, T_, D_, repeats
    )
    return out_2d[None, :, None, :].astype(queries.dtype), q_rot, T_, repeats


def call_fallback_path(cache, queries, pe_scores, scale, q_rot=None):
    """Mirror cache.fused_attention's setup, route to MLX-ops fallback."""
    B, H_q, _, D_ = queries.shape
    H_kv = cache.num_heads
    T_ = (
        cache.offset
        if cache._cache_mode == "prealloc"
        else (cache.compressed_keys["indices"].shape[1])
    )
    repeats = H_q // H_kv if H_q != H_kv else 1

    if q_rot is None:
        R_T = mx.swapaxes(cache.rotation_matrices, -2, -1)
        q_flat = queries[0, :, 0, :]
        R_T_exp = mx.repeat(R_T, repeats, axis=0) if repeats > 1 else R_T
        q_rot = mx.squeeze(mx.matmul(q_flat[:, None, :], R_T_exp), axis=1)

    out_2d = cache._fused_attention_mlx(
        q_rot, scale, pe_scores, H_q, H_kv, T_, D_, repeats
    )
    return out_2d[None, :, None, :].astype(queries.dtype)


def fallback_with_intermediates(cache, q_rot, pe_scores, scale, T_, repeats):
    """Reproduce _fused_attention_mlx step by step, returning every stage.
    These are the reference values for the MLX-ops path; the kernel doesn't
    expose intermediates so we use these as the truth and bisect by checking
    where final outputs diverge under controlled perturbations."""
    if cache._cache_mode == "prealloc":
        k_indices = cache.compressed_keys["indices"][:, :T_, :]
        k_xnorm = cache.compressed_keys["x_norm"][:, :T_, :]
        v_indices = cache.compressed_values["indices"][:, :T_, :]
        v_xnorm = cache.compressed_values["x_norm"][:, :T_, :]
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

    scores = mx.matmul(
        q_rot[:, None, :], mx.swapaxes(k_rot_scaled, -2, -1)
    )  # (H_q, 1, T)
    scores_scaled = scores * scale

    # Mask reduction: pe_scores (1, H_q, 1, T) → broadcastable to (H_q, 1, T)
    m = pe_scores
    while m.ndim > scores_scaled.ndim:
        m = m.squeeze(0)
    scores_post_mask = scores_scaled + m

    attn_weights = mx.softmax(scores_post_mask, axis=-1)
    output_rot = mx.matmul(attn_weights, v_rot_scaled)[:, 0, :]  # (H_q, D)

    return {
        "k_rot_scaled": k_rot_scaled,
        "v_rot_scaled": v_rot_scaled,
        "scores_scaled_no_mask": scores_scaled.squeeze(1),  # (H_q, T)
        "scores_post_mask": scores_post_mask.squeeze(1),
        "attn_weights": attn_weights.squeeze(1),
        "output_rot": output_rot,
    }


def main():
    print("=== NPT=16 correctness repro (kernel vs MLX-ops fallback) ===")
    print(f"Shapes: H_q={H_Q} H_kv={H_KV} D={D} T={T} bit_width={BIT_WIDTH}")
    print()

    t0 = time.time()
    cache = build_cache_with_random_kv(seed=42)
    print(f"Cache build + finalize: {time.time() - t0:.2f}s")

    queries, pe_scores, scale = synthesize_query_and_mask(seed=42)
    print(
        f"queries={tuple(queries.shape)} pe_scores={tuple(pe_scores.shape)} "
        f"scale={scale:.4f}"
    )
    print(f"supports_fused_attention: {cache.supports_fused_attention}")
    print()

    # === Stage 1: end-to-end head-to-head ===
    # Force NPT16 path
    import os

    os.environ["ISOQUANT_USE_NPT16_FUSED"] = "1"

    # Reset _fused_metal_ok so kernel is re-attempted
    cache._fused_metal_ok = None

    t0 = time.time()
    out_kernel, q_rot, T_, repeats = call_kernel_path(cache, queries, pe_scores, scale)
    mx.eval(out_kernel)
    t_kernel = time.time() - t0

    t0 = time.time()
    out_fb = call_fallback_path(cache, queries, pe_scores, scale, q_rot=q_rot)
    mx.eval(out_fb)
    t_fb = time.time() - t0

    print(f"Kernel  ({t_kernel * 1000:.1f} ms): {tuple(out_kernel.shape)}")
    print(f"Fallback ({t_fb * 1000:.1f} ms): {tuple(out_fb.shape)}")
    print()

    end_to_end = _stat_row("end_to_end (post-inverse-rotation)", out_kernel, out_fb)
    print(
        f"END-TO-END:  linf={end_to_end['linf']:.6e}  "
        f"mse={end_to_end['mse']:.6e}  cos={end_to_end['cosine']:.6f}"
    )

    if end_to_end["linf"] < 1e-4:
        print()
        print("Kernel and fallback agree to ~1e-4 fp32 on synthetic inputs.")
        print("The real-weight discrepancy must come from value distribution,")
        print("not the algorithm. Re-run with K2.6-derived K/V (saved from a")
        print("real decode step) to reproduce the divergence.")
        payload = {
            "shapes": {"H_q": H_Q, "H_kv": H_KV, "D": D, "T": T},
            "end_to_end": end_to_end,
            "verdict": "AGREE_ON_SYNTHETIC",
        }
    else:
        print()
        print(f"⚠ Significant divergence ({end_to_end['linf']:.2e} > 1e-4).")
        print("Drilling into intermediate stages and bisection tests...")
        print()

        # === Stage 2: refined H2 (reduction-order) ===
        # The kernel C does QK via simd_sum over 32 lanes × 16 dims; MLX matmul
        # uses a different blocking. Test: does explicit lane-chunked sum
        # diverge from native matmul on these same inputs?
        print("=== H2 refined: QK reduction-order (matmul vs lane-chunked sum) ===")
        intermediates = fallback_with_intermediates(
            cache, q_rot, pe_scores, scale, T_, repeats
        )
        k_rs = intermediates["k_rot_scaled"]  # (H_q, T, D)
        s_matmul = mx.matmul(q_rot[:, None, :], mx.swapaxes(k_rs, -2, -1)).squeeze(1)
        n_lanes = D // 16
        partials = mx.zeros((H_Q, T))
        for lane in range(n_lanes):
            d_lo = lane * 16
            d_hi = (lane + 1) * 16
            partials = partials + (
                q_rot[:, None, d_lo:d_hi] * k_rs[:, :, d_lo:d_hi]
            ).sum(axis=-1)
        mx.eval(s_matmul, partials)
        h2_linf = _l_inf(s_matmul, partials)
        print(f"  matmul-vs-lane-chunked QK  linf={h2_linf:.6e}")

        # === H1: mask precision and shape ===
        print()
        print("=== H1: mask shape after squeeze loop ===")
        m = pe_scores
        path = [tuple(m.shape)]
        while m.ndim > 3:  # SDPA-shape (H_q, 1, T)
            sz1 = next((i for i, s in enumerate(m.shape) if s == 1), None)
            m = m.squeeze(sz1)
            path.append(tuple(m.shape))
        print(f"  pe_scores reduction path: {' -> '.join(str(s) for s in path)}")
        print(f"  final mask shape: {tuple(m.shape)} dtype={m.dtype}")
        print(f"  mask range: [{float(m.min()):.4f}, {float(m.max()):.4f}]")

        # === H3: inverse rotation sanity ===
        # Build the kernel-side "expected pre-inverse-rotation output" by
        # running the inverse rotation on the fallback's output_rot, then
        # check whether kernel's final out matches.
        print()
        print("=== H3: inverse rotation sanity ===")
        out_rot = intermediates["output_rot"]  # (H_q, D)
        # Apply inverse rotation via the cache's helper to match what the
        # fallback produces in its final return.
        out_fb_via_helper = cache._apply_inverse_rotation(
            out_rot, H_Q, H_KV, D, repeats, use_metal_kernel=False
        )
        mx.eval(out_fb_via_helper)
        # Compare to kernel's output at the post-inverse stage
        # (kernel returns post-inverse already; we ran call_kernel_path above)
        out_kernel_2d = out_kernel[0, :, 0, :]  # (H_q, D)
        h3 = _stat_row("inverse_rotation", out_kernel_2d, out_fb_via_helper)
        print(
            f"  kernel(post-inv) vs fallback(post-inv via helper)  "
            f"linf={h3['linf']:.6e}  cos={h3['cosine']:.6f}"
        )

        payload = {
            "shapes": {"H_q": H_Q, "H_kv": H_KV, "D": D, "T": T},
            "end_to_end": end_to_end,
            "h2_qk_reduction_order": {"linf": h2_linf},
            "h1_mask_reduction": {
                "path": [list(s) for s in path],
                "final_shape": list(m.shape),
                "min": float(m.min()),
                "max": float(m.max()),
            },
            "h3_inverse_rotation": h3,
            "verdict": "DIVERGENT_ON_SYNTHETIC",
        }

    artifact_dir = (
        Path(__file__).resolve().parent.parent / "artifacts/kimi_k26_profiling"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifact_dir / "npt16_correctness_repro.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
