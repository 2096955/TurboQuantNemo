"""Shared test helpers for NPT=16 fused attention kernels (D=512).

Mirrors conftest_npt8.py but for head_dim=512 (Kimi MLA latent space).
"""

import mlx.core as mx
import numpy as np
from mlx_lm.models.fused_kv_decode_kernels import (
    fused_qk_dot,
    fused_value_accum,
    pack_indices_3bit,
)


def _ref_3kernel_stable_d512(
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
    """Reference: 3-kernel pipeline; output in rotated space (no inverse SO(4))."""
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


def _synthetic_d512(T: int, h_kv: int = 1, h_q: int = 64) -> tuple[mx.array, ...]:
    """Synthetic packed data for D=512 (Kimi MLA-like: 1 KV head, 64 query heads)."""
    rng = np.random.default_rng(11)
    indices_k = mx.array(rng.integers(0, 8, (h_kv, T, 512), dtype=np.uint8))
    indices_v = mx.array(rng.integers(0, 8, (h_kv, T, 512), dtype=np.uint8))
    norms_k = mx.array(rng.standard_normal((h_kv, T)).astype(np.float32))
    norms_v = mx.array(rng.standard_normal((h_kv, T)).astype(np.float32))
    centroids = mx.array(np.linspace(-1.5, 1.5, 8, dtype=np.float32))
    q_rot = mx.array(rng.standard_normal((h_q, 512)).astype(np.float32))
    kv_head_map = mx.arange(h_q, dtype=mx.uint32) // (h_q // h_kv)
    k_p = pack_indices_3bit(indices_k)
    v_p = pack_indices_3bit(indices_v)
    mx.eval(k_p, v_p, norms_k, norms_v, centroids, q_rot, kv_head_map)
    return k_p, v_p, centroids, norms_k, norms_v, q_rot, kv_head_map


def _synthetic_d512_shared_kv(
    T: int, h_kv: int = 1, h_q: int = 64
) -> tuple[mx.array, ...]:
    """Same as _synthetic_d512 but K=V (MLA-style shared latent)."""
    rng = np.random.default_rng(11)
    indices = mx.array(rng.integers(0, 8, (h_kv, T, 512), dtype=np.uint8))
    norms = mx.array(rng.standard_normal((h_kv, T)).astype(np.float32))
    centroids = mx.array(np.linspace(-1.5, 1.5, 8, dtype=np.float32))
    q_rot = mx.array(rng.standard_normal((h_q, 512)).astype(np.float32))
    kv_head_map = mx.arange(h_q, dtype=mx.uint32) // (h_q // h_kv)
    packed = pack_indices_3bit(indices)
    mx.eval(packed, norms, centroids, q_rot, kv_head_map)
    return packed, packed, centroids, norms, norms, q_rot, kv_head_map


def _identity_blocks_d512(h_kv: int) -> mx.array:
    """Identity SO(4) blocks for D=512 — inverse rotation is a no-op."""
    n_blocks = 512 // 4  # = 128
    blocks = np.zeros((h_kv, n_blocks, 4, 4), dtype=np.float32)
    for h in range(h_kv):
        for b in range(n_blocks):
            blocks[h, b] = np.eye(4, dtype=np.float32)
    return mx.array(blocks)


def _random_so4_blocks_d512(h_kv: int, seed: int = 42) -> tuple[mx.array, mx.array]:
    """Random SO(4) block matrices and their transposes for D=512."""
    from mlx_lm.models.mlx_isoquant import _build_isoclinic_block_matrices

    rng = np.random.default_rng(seed)
    blocks_list = []
    blocks_t_list = []
    for _ in range(h_kv):
        blk = _build_isoclinic_block_matrices(512, rng)
        blocks_list.append(blk)
        blocks_t_list.append(blk.transpose(0, 2, 1))
    blocks = mx.array(np.stack(blocks_list, dtype=np.float32))
    blocks_t = mx.array(np.stack(blocks_t_list, dtype=np.float32))
    return blocks, blocks_t


def _python_inverse_rotation(
    output_rot: mx.array,
    blocks: mx.array,
    use_hadamard: bool,
) -> mx.array:
    """Python reference inverse rotation: SO(4) blocks then optional WHT."""
    from mlx_lm.models.mlx_isoquant import (
        _apply_so4_blocks_last_axis,
        _fwht_last_axis,
    )

    y = _apply_so4_blocks_last_axis(output_rot, blocks)
    if use_hadamard:
        y = _fwht_last_axis(y)
    return y
