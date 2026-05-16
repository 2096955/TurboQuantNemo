"""Distribution sweep extending repro_npt16_correctness.py.

The benign-Gaussian repro showed kernel-vs-fallback linf=1.49e-06 (essentially
fp32 noise). The real-K2.6 run showed top1 dropping from 0.8727 (fallback) to
0.8000 (npt16 fused). The bug is data-dependent.

Sweep variants (per user request, in priority order):
  V0  baseline           — pe ~ N(0, 0.5),  K,V ~ N(0, 1) (the existing repro)
  V1  sharp pe_scores    — pe ~ N(0, 5)
  V2  causal-mask sentinels — pe with finfo.min in ~half positions
  V3  heavy-tailed K,V   — Student-t(df=3) RMSNorm-rescaled
  V4  V1 + V3 combined   — sharp pe AND heavy-tailed K,V

For each variant: build cache, populate K/V, finalize, run both paths on the
same q_rot/pe_scores, compare end-to-end.

Goal: find the variant that produces linf > 1e-3 (or sufficiently larger than
V0's 1.49e-06) — that's the mechanism class.
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

H_Q = 64
H_KV = 1
D = 512
T = 64
BIT_WIDTH = 3


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


def _build_cache_with_kv(K: mx.array, V: mx.array, seed: int = 42) -> IsoQuantKVCache:
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
    cache.update_and_fetch(K, V)
    cache.finalize_deferred_prefill()
    return cache


def _gen_kv_gaussian(rng):
    K = mx.array(rng.standard_normal((1, H_KV, T, D)).astype(np.float32))
    V = mx.array(rng.standard_normal((1, H_KV, T, D)).astype(np.float32))
    return K, V


def _gen_kv_student_t(rng, df: float = 3.0):
    """Heavy-tailed K, V approximating real RMSNorm'd activations."""
    K_np = rng.standard_t(df, size=(1, H_KV, T, D)).astype(np.float32)
    V_np = rng.standard_t(df, size=(1, H_KV, T, D)).astype(np.float32)

    # RMSNorm-rescale per token: divide by sqrt(mean(x^2))
    def rmsnorm(x):
        rms = np.sqrt((x * x).mean(axis=-1, keepdims=True) + 1e-6)
        return x / rms

    return mx.array(rmsnorm(K_np)), mx.array(rmsnorm(V_np))


def _gen_pe_gaussian(rng, sigma: float = 0.5):
    return mx.array((rng.standard_normal((1, H_Q, 1, T)) * sigma).astype(np.float32))


def _gen_pe_with_sentinels(rng, sigma: float = 0.5, mask_frac: float = 0.5):
    """Half the positions get finfo.min sentinel — mirrors deepseek_v3.py:148
    causal-mask substitution. Even though Codex traced that decode-step
    pe_scores doesn't actually carry sentinels, this variant tests whether
    the kernel handles them correctly when they do appear (e.g., during
    teacher-forced prefill if it ever entered the fused path)."""
    pe = (rng.standard_normal((1, H_Q, 1, T)) * sigma).astype(np.float32)
    sentinel = np.finfo(np.float32).min
    mask = rng.random((1, H_Q, 1, T)) < mask_frac
    pe = np.where(mask, sentinel, pe)
    return mx.array(pe)


def _query(rng):
    return mx.array(rng.standard_normal((1, H_Q, 1, D)).astype(np.float32))


def call_both_paths(cache, queries, pe_scores, scale):
    """Run both _fused_attention_metal and _fused_attention_mlx on the same
    (q_rot, scale, pe_scores). Returns (out_kernel_2d, out_fb_2d) shape (H_q, D)."""
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
    q_rot = mx.squeeze(mx.matmul(q_flat[:, None, :], R_T_exp), axis=1)
    cache._fused_metal_ok = None
    out_kernel = cache._fused_attention_metal(
        q_rot, scale, pe_scores, H_q, H_kv, T_, D_, repeats
    )
    out_fb = cache._fused_attention_mlx(
        q_rot, scale, pe_scores, H_q, H_kv, T_, D_, repeats
    )
    mx.eval(out_kernel, out_fb)
    return out_kernel, out_fb


def run_variant(label: str, K, V, queries, pe_scores, scale: float):
    cache = _build_cache_with_kv(K, V, seed=42)
    out_kernel, out_fb = call_both_paths(cache, queries, pe_scores, scale)
    # Also stats on attention sharpness (proxy for one-hot risk)
    H_q = queries.shape[1]
    H_kv = cache.num_heads
    T_ = cache.offset
    repeats = H_q // H_kv
    R_T = mx.swapaxes(cache.rotation_matrices, -2, -1)
    q_flat = queries[0, :, 0, :]
    R_T_exp = mx.repeat(R_T, repeats, axis=0) if repeats > 1 else R_T
    q_rot = mx.squeeze(mx.matmul(q_flat[:, None, :], R_T_exp), axis=1)
    # Build scores via the fallback path internals to measure sharpness
    if cache._cache_mode == "prealloc":
        k_idx = cache.compressed_keys["indices"][:, :T_, :]
        k_xn = cache.compressed_keys["x_norm"][:, :T_, :]
    else:
        k_idx = cache.compressed_keys["indices"]
        k_xn = cache.compressed_keys["x_norm"]
    k_rot_scaled = cache.compressor.centroids[k_idx] * k_xn.astype(mx.float32)
    if repeats > 1:
        k_rot_scaled = mx.repeat(k_rot_scaled, repeats, axis=0)
    scores = mx.matmul(q_rot[:, None, :], mx.swapaxes(k_rot_scaled, -2, -1)).squeeze(1)
    scores = scores * scale
    # scores is now (H_q, T). Reduce pe_scores (1, H_q, 1, T) → (H_q, T).
    m = pe_scores
    while m.ndim > scores.ndim:
        sz1 = next((i for i, s in enumerate(m.shape) if s == 1), None)
        if sz1 is None:
            raise ValueError(f"can't reduce mask of shape {tuple(m.shape)}")
        m = m.squeeze(sz1)
    scores = scores + m
    # Replace inf/-inf for stat purposes
    finite_scores = mx.where(
        mx.isinf(scores), mx.array(0.0, dtype=scores.dtype), scores
    )
    aw = mx.softmax(scores, axis=-1)
    max_aw_per_head = float(aw.max(axis=-1).mean())  # avg of per-head max attn weight
    score_max_abs = float(mx.abs(finite_scores).max())
    return {
        "label": label,
        "linf": _l_inf(out_kernel, out_fb),
        "mse": _mse(out_kernel, out_fb),
        "cosine": _cosine(out_kernel, out_fb),
        "score_max_abs_finite": score_max_abs,
        "mean_max_attn_weight": max_aw_per_head,
    }


def main():
    print("=== NPT=16 distribution sweep ===")
    print(f"H_q={H_Q} H_kv={H_KV} D={D} T={T} bit_width={BIT_WIDTH}")
    print()

    rng = np.random.default_rng(42)

    # Generate inputs once where possible (one query / pe per variant family)
    queries = _query(rng)
    scale = 1.0 / np.sqrt(64.0)

    K_g, V_g = _gen_kv_gaussian(np.random.default_rng(100))
    K_t, V_t = _gen_kv_student_t(np.random.default_rng(101), df=3.0)
    pe_tame = _gen_pe_gaussian(np.random.default_rng(200), sigma=0.5)
    pe_sharp = _gen_pe_gaussian(np.random.default_rng(201), sigma=5.0)
    pe_sentinel = _gen_pe_with_sentinels(
        np.random.default_rng(202), sigma=0.5, mask_frac=0.5
    )

    variants = [
        ("V0_baseline_gauss_tame_pe", K_g, V_g, queries, pe_tame, scale),
        ("V1_gauss_sharp_pe_N(0,5)", K_g, V_g, queries, pe_sharp, scale),
        ("V2_gauss_pe_with_sentinels", K_g, V_g, queries, pe_sentinel, scale),
        ("V3_studentT_kv_tame_pe", K_t, V_t, queries, pe_tame, scale),
        ("V4_studentT_kv_sharp_pe", K_t, V_t, queries, pe_sharp, scale),
    ]

    rows = []
    for label, K, V, q, pe, sc in variants:
        t0 = time.time()
        r = run_variant(label, K, V, q, pe, sc)
        elapsed = time.time() - t0
        r["wall_s"] = round(elapsed, 2)
        rows.append(r)
        print(
            f"{label:32s}  linf={r['linf']:.6e}  mse={r['mse']:.6e}  "
            f"cos={r['cosine']:.6f}  scoreM={r['score_max_abs_finite']:7.2f}  "
            f"meanMaxAW={r['mean_max_attn_weight']:.4f}  ({elapsed:.1f}s)"
        )
    print()

    artifact_dir = (
        Path(__file__).resolve().parent.parent / "artifacts/kimi_k26_profiling"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifact_dir / "npt16_distribution_sweep.json"
    with open(out_path, "w") as f:
        json.dump(
            {"variants": rows, "shapes": {"H_q": H_Q, "D": D, "T": T}}, f, indent=2
        )
    print(f"Saved: {out_path}")

    # Verdict
    macro = [r for r in rows if r["linf"] > 1e-3]
    if macro:
        print()
        print("⚠ Mechanism isolated — variants with linf > 1e-3:")
        for r in macro:
            print(f"  {r['label']}: linf={r['linf']:.6e}")
    else:
        print()
        print(
            "All variants stay near fp32 noise (~1e-6). The bug isn't in the "
            "synthetic distribution shapes tested. Real-input capture needed."
        )


if __name__ == "__main__":
    main()
