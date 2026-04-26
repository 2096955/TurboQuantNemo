"""Phase 3: fused NPT=8 T-tiled kernel must equal the stable 3-kernel pipeline.

The reference uses ``fused_value_accum`` (serial T loop) — not
``fused_value_accum_tiled`` — for primary equivalence so failures localize
to the new kernel. A secondary test cross-checks the tiled V path.

See: docs/superpowers/plans/2026-04-24-isoquant-decode-performance.md (Phase 3).
"""

import os

os.environ.setdefault("ISOQUANT_BITS", "3")

import mlx.core as mx
import numpy as np
from mlx_lm.models.fused_kv_decode_kernels import (
    fused_qk_dot,
    fused_value_accum,
    pack_indices_3bit,
)


def _ref_3kernel_stable(
    K_packed: mx.array,
    V_packed: mx.array,
    centroids: mx.array,
    k_norms: mx.array,
    v_norms: mx.array,
    q_rot: mx.array,
    kv_head_map: mx.array,
    num_heads: int,
    T: int,
    D: int,
    scale: float,
    mask: mx.array | None = None,
) -> mx.array:
    """Reference: 3-kernel pipeline; output in rotated space (no inverse SO(4)."""
    scores = (
        fused_qk_dot(
            K_packed,
            centroids,
            k_norms,
            q_rot,
            kv_head_map,
            num_heads,
            T,
            D,
        )
        * scale
    )
    if mask is not None:
        scores = scores + mask
    attn = mx.softmax(scores, axis=-1)
    return fused_value_accum(
        V_packed,
        centroids,
        v_norms,
        attn,
        kv_head_map,
        num_heads,
        T,
        D,
    )


def _synthetic_d256(T: int, h_kv: int = 2, h_q: int = 16) -> tuple[mx.array, ...]:
    rng = np.random.default_rng(11)
    indices_k = mx.array(rng.integers(0, 8, (h_kv, T, 256), dtype=np.uint8))
    indices_v = mx.array(rng.integers(0, 8, (h_kv, T, 256), dtype=np.uint8))
    norms_k = mx.array(rng.standard_normal((h_kv, T)).astype(np.float32))
    norms_v = mx.array(rng.standard_normal((h_kv, T)).astype(np.float32))
    centroids = mx.array(np.linspace(-1.5, 1.5, 8, dtype=np.float32))
    q_rot = mx.array(rng.standard_normal((h_q, 256)).astype(np.float32))
    kv_head_map = mx.arange(h_q, dtype=mx.uint32) // (h_q // h_kv)
    k_p = pack_indices_3bit(indices_k)
    v_p = pack_indices_3bit(indices_v)
    mx.eval(k_p, v_p, norms_k, norms_v, centroids, q_rot, kv_head_map)
    return k_p, v_p, centroids, norms_k, norms_v, q_rot, kv_head_map


def _identity_blocks(h_kv: int, d: int) -> mx.array:
    """Identity SO(4) blocks — inverse rotation is a no-op."""
    n_blocks = d // 4
    blocks = np.zeros((h_kv, n_blocks, 4, 4), dtype=np.float32)
    for h in range(h_kv):
        for b in range(n_blocks):
            blocks[h, b] = np.eye(4, dtype=np.float32)
    return mx.array(blocks)


def test_fused_npt8_matches_stable_3kernel_no_inverse_rotation() -> None:
    from mlx_lm.models.fused_kv_decode_npt8_tiled import fused_attention_npt8_tiled

    t, h_kv, h_q, d = 256, 2, 16, 256
    k_p, v_p, c, nk, nv, q, kv_map = _synthetic_d256(t, h_kv, h_q)
    scale = float(1.0 / np.sqrt(d))
    blocks_t = _identity_blocks(h_kv, d)
    ref = _ref_3kernel_stable(k_p, v_p, c, nk, nv, q, kv_map, h_q, t, d, scale)
    out = fused_attention_npt8_tiled(
        k_p,
        v_p,
        c,
        nk,
        nv,
        q,
        kv_map,
        blocks_t=blocks_t,
        scale=scale,
        use_hadamard=False,
        mask=None,
        tile_size=128,
        num_heads=h_q,
        seq_len=t,
        head_dim=d,
    )
    mx.eval(ref, out)
    np.testing.assert_allclose(np.asarray(ref), np.asarray(out), rtol=1e-3, atol=1e-4)


def test_fused_npt8_also_matches_tiled_v_accum_path() -> None:
    from mlx_lm.models.fused_kv_decode_npt8_tiled import fused_attention_npt8_tiled
    from mlx_lm.models.fused_kv_decode_tiled import fused_value_accum_tiled

    t, h_kv, h_q, d = 256, 2, 16, 256
    k_p, v_p, c, nk, nv, q, kv_map = _synthetic_d256(t, h_kv, h_q)
    scale = float(1.0 / np.sqrt(d))
    blocks_t = _identity_blocks(h_kv, d)
    scores = fused_qk_dot(k_p, c, nk, q, kv_map, h_q, t, d) * scale
    attn = mx.softmax(scores, axis=-1)
    ref_tiled = fused_value_accum_tiled(
        v_p, c, nv, attn, kv_map, h_q, t, d, tile_size=128
    )
    out = fused_attention_npt8_tiled(
        k_p,
        v_p,
        c,
        nk,
        nv,
        q,
        kv_map,
        blocks_t=blocks_t,
        scale=scale,
        use_hadamard=False,
        mask=None,
        tile_size=128,
        num_heads=h_q,
        seq_len=t,
        head_dim=d,
    )
    mx.eval(ref_tiled, out)
    np.testing.assert_allclose(
        np.asarray(ref_tiled), np.asarray(out), rtol=1e-3, atol=1e-4
    )


def test_fused_npt8_with_mask() -> None:
    """NPT=8 kernel handles attention mask correctly."""
    from mlx_lm.models.fused_kv_decode_npt8_tiled import fused_attention_npt8_tiled

    t, h_kv, h_q, d = 128, 2, 16, 256
    k_p, v_p, c, nk, nv, q, kv_map = _synthetic_d256(t, h_kv, h_q)
    scale = float(1.0 / np.sqrt(d))
    blocks_t = _identity_blocks(h_kv, d)

    rng = np.random.default_rng(99)
    mask = mx.array(rng.standard_normal((h_q, t)).astype(np.float32))

    ref = _ref_3kernel_stable(k_p, v_p, c, nk, nv, q, kv_map, h_q, t, d, scale, mask)
    out = fused_attention_npt8_tiled(
        k_p,
        v_p,
        c,
        nk,
        nv,
        q,
        kv_map,
        blocks_t=blocks_t,
        scale=scale,
        use_hadamard=False,
        mask=mask,
        tile_size=128,
        num_heads=h_q,
        seq_len=t,
        head_dim=d,
    )
    mx.eval(ref, out)
    np.testing.assert_allclose(np.asarray(ref), np.asarray(out), rtol=1e-3, atol=1e-4)


def test_fused_npt8_with_storage_stride() -> None:
    """NPT=8 kernel works with padded buffers (storage_stride > seq_len)."""
    from mlx_lm.models.fused_kv_decode_npt8_tiled import fused_attention_npt8_tiled

    t, h_kv, h_q, d = 128, 2, 16, 256
    k_p, v_p, c, nk, nv, q, kv_map = _synthetic_d256(t, h_kv, h_q)
    scale = float(1.0 / np.sqrt(d))
    blocks_t = _identity_blocks(h_kv, d)

    ref = _ref_3kernel_stable(k_p, v_p, c, nk, nv, q, kv_map, h_q, t, d, scale)

    padded_t = t + 64
    pad_k = mx.zeros((h_kv, 64, k_p.shape[2]), dtype=mx.uint8)
    pad_v = mx.zeros((h_kv, 64, v_p.shape[2]), dtype=mx.uint8)
    k_padded = mx.concatenate([k_p, pad_k], axis=1)
    v_padded = mx.concatenate([v_p, pad_v], axis=1)
    nk_padded = mx.concatenate([nk, mx.zeros((h_kv, 64))], axis=1)
    nv_padded = mx.concatenate([nv, mx.zeros((h_kv, 64))], axis=1)

    out = fused_attention_npt8_tiled(
        k_padded,
        v_padded,
        c,
        nk_padded,
        nv_padded,
        q,
        kv_map,
        blocks_t=blocks_t,
        scale=scale,
        use_hadamard=False,
        mask=None,
        tile_size=128,
        num_heads=h_q,
        seq_len=t,
        head_dim=d,
        storage_stride=padded_t,
    )
    mx.eval(ref, out)
    np.testing.assert_allclose(np.asarray(ref), np.asarray(out), rtol=1e-3, atol=1e-4)
