"""Phase 3b: T-tiled NPT=8 fused IsoQuant attention (head_dim=256).

This path splits the KV sequence across tiles, runs an online-softmax decode
kernel per tile, then merges the per-tile partials with FlashAttention-style
log-sum-exp formulas in MLX. Each tile writes tile-local softmax-normalized
rotated-space output plus its `(m, l)` statistics. Inverse rotation and
optional Hadamard are applied once after merge via the existing structured
Python path.
"""

from __future__ import annotations

import mlx.core as mx

__all__ = ["fused_attention_npt8_tiled"]

_tiled_kernel_cache: dict[str, mx.fast.MetalKernel] = {}


_NPT8_TILED_ATTENTION_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    uint tile_id = threadgroup_position_in_grid.x;
    uint q_head = threadgroup_position_in_grid.y;
    uint kv_head = kv_head_map[q_head];

    uint T = seq_len[0];
    uint T_stride = storage_stride[0];
    uint T_TILE = tile_size[0];
    uint H_q = num_heads_q[0];
    uint use_mask = has_mask[0];

    uint t_start = tile_id * T_TILE;
    uint t_end = min(t_start + T_TILE, T);

    uint dim_base = lane * 8;
    uint w_byte = lane * 3;

    float q_r[8];
    for (uint i = 0; i < 8; i++) {
        q_r[i] = q[q_head * 256 + dim_base + i];
    }

    float m_run = -1e38f;
    float l_run = 0.0f;
    float O_r[8];
    for (uint i = 0; i < 8; i++) O_r[i] = 0.0f;

    uint packed_words = 256 / 8;
    uint stride_bytes = packed_words * 3;
    uint kv_k_base = kv_head * T_stride * stride_bytes;
    uint kv_v_base = kv_head * T_stride * stride_bytes;

    for (uint t = t_start; t < t_end; t++) {
        uint k_off = kv_k_base + t * stride_bytes + w_byte;
        uint kw = uint(K_packed[k_off]) |
                  (uint(K_packed[k_off + 1]) << 8) |
                  (uint(K_packed[k_off + 2]) << 16);
        float k_norm = k_norms[kv_head * T_stride + t];

        float partial = 0.0f;
        for (uint i = 0; i < 8; i++) {
            float k_val = centroids[(kw >> (i * 3)) & 0x7] * k_norm;
            partial += q_r[i] * k_val;
        }

        float score = simd_sum(partial) * scale_val[0];
        if (use_mask) score += mask_data[q_head * T + t];

        float m_new = max(m_run, score);
        float corr = exp(m_run - m_new);
        float es = exp(score - m_new);

        uint v_off = kv_v_base + t * stride_bytes + w_byte;
        uint vw = uint(V_packed[v_off]) |
                  (uint(V_packed[v_off + 1]) << 8) |
                  (uint(V_packed[v_off + 2]) << 16);
        float v_norm = v_norms[kv_head * T_stride + t];

        for (uint i = 0; i < 8; i++) {
            float v_val = centroids[(vw >> (i * 3)) & 0x7] * v_norm;
            O_r[i] = O_r[i] * corr + es * v_val;
        }
        l_run = l_run * corr + es;
        m_run = m_new;
    }

    float inv_l = (l_run > 0.0f) ? (1.0f / l_run) : 0.0f;
    uint out_base = (tile_id * H_q + q_head) * 256;
    for (uint i = 0; i < 8; i++) {
        o_partials[out_base + dim_base + i] = O_r[i] * inv_l;
    }

    if (lane == 0) {
        uint ml_base = (tile_id * H_q + q_head) * 2;
        ml_partials[ml_base + 0] = m_run;
        ml_partials[ml_base + 1] = l_run;
    }
"""


def _get_tiled_kernel() -> mx.fast.MetalKernel:
    key = "npt8_tiled"
    if key not in _tiled_kernel_cache:
        _tiled_kernel_cache[key] = mx.fast.metal_kernel(
            name="fused_attention_npt8_tiled",
            input_names=[
                "K_packed",
                "V_packed",
                "centroids",
                "k_norms",
                "v_norms",
                "q",
                "kv_head_map",
                "scale_val",
                "seq_len",
                "storage_stride",
                "tile_size",
                "num_heads_q",
                "has_mask",
                "mask_data",
            ],
            output_names=["o_partials", "ml_partials"],
            source=_NPT8_TILED_ATTENTION_SOURCE,
        )
    return _tiled_kernel_cache[key]


def _fa2_merge(
    o_partials: mx.array,
    ml_partials: mx.array,
    num_tiles: int,
    num_heads: int,
    head_dim: int,
) -> mx.array:
    """Merge tiled online-softmax partials into a final rotated-space output."""

    o_all = o_partials.reshape(num_tiles, num_heads, head_dim)
    ml_all = ml_partials.reshape(num_tiles, num_heads, 2)

    m_all = ml_all[:, :, 0]
    l_all = ml_all[:, :, 1]

    m_max = mx.max(m_all, axis=0)
    weights = mx.exp(m_all - m_max[None, :]) * l_all

    numerator = mx.sum(o_all * weights[:, :, None], axis=0)
    denominator = mx.sum(weights, axis=0)

    denominator_safe = mx.where(denominator > 0, denominator, mx.ones_like(denominator))
    merged = numerator / denominator_safe[:, None]
    return mx.where(denominator[:, None] > 0, merged, mx.zeros_like(merged))


def fused_attention_npt8_tiled(
    K_packed: mx.array,
    V_packed: mx.array,
    centroids: mx.array,
    k_norms: mx.array,
    v_norms: mx.array,
    q_rot: mx.array,
    kv_head_map: mx.array,
    *,
    block_matrices: mx.array,
    scale: float,
    use_hadamard: bool,
    mask: mx.array | None,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    tile_size: int = 256,
    storage_stride: int | None = None,
) -> mx.array:
    """T-tiled NPT=8 attention for head_dim=256 with FA2-style merge."""

    from .mlx_isoquant import structured_rotate_inverse

    if head_dim != 256:
        raise ValueError(f"NPT=8 tiled kernel requires head_dim=256, got {head_dim}")
    if tile_size < 1:
        raise ValueError(f"tile_size must be >= 1, got {tile_size}")

    num_tiles = (seq_len + tile_size - 1) // tile_size
    kernel = _get_tiled_kernel()

    if mask is not None:
        m = mask
        while m.ndim > 2:
            m = m.squeeze(0)
        if m.shape[0] == 1 and num_heads > 1:
            m = mx.broadcast_to(m, (num_heads, seq_len))
        mask_flat = m.reshape(-1).astype(mx.float32)
        has_mask_val = mx.array([1], dtype=mx.uint32)
    else:
        mask_flat = mx.zeros((1,), dtype=mx.float32)
        has_mask_val = mx.array([0], dtype=mx.uint32)

    scale_arr = mx.array([scale], dtype=mx.float32)
    sl = mx.array([seq_len], dtype=mx.uint32)
    ss = mx.array(
        [storage_stride if storage_stride is not None else seq_len], dtype=mx.uint32
    )
    ts = mx.array([tile_size], dtype=mx.uint32)
    nh = mx.array([num_heads], dtype=mx.uint32)

    o_partials, ml_partials = kernel(
        inputs=[
            K_packed.reshape(-1),
            V_packed.reshape(-1),
            centroids.reshape(-1),
            k_norms.reshape(-1),
            v_norms.reshape(-1),
            q_rot.reshape(-1),
            kv_head_map.reshape(-1),
            scale_arr,
            sl,
            ss,
            ts,
            nh,
            has_mask_val,
            mask_flat,
        ],
        output_shapes=[
            (num_tiles * num_heads * head_dim,),
            (num_tiles * num_heads * 2,),
        ],
        output_dtypes=[mx.float32, mx.float32],
        grid=(num_tiles * 32, num_heads, 1),
        threadgroup=(32, 1, 1),
    )

    merged = _fa2_merge(o_partials, ml_partials, num_tiles, num_heads, head_dim)

    expanded_blocks = mx.take(block_matrices, kv_head_map, axis=0)
    return structured_rotate_inverse(merged, expanded_blocks, use_hadamard)
