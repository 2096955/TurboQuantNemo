"""T-parallel-tiled fused value accumulation for IsoQuant decode.

The existing ``fused_value_accum`` (in fused_kv_decode_kernels.py) is the
hot path of IsoQuant decode at long context. This module provides a T-parallel
drop-in replacement that splits the sequence dimension across threadgroups and
reduces per-tile partials with ``mx.sum``.
"""

from __future__ import annotations

import mlx.core as mx

_FUSED_VALUE_ACCUM_TILED_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    uint tile_id = threadgroup_position_in_grid.x;
    uint q_head_idx = threadgroup_position_in_grid.y;
    uint kv_head_idx = kv_head_map[q_head_idx];

    uint D = head_dim[0];
    uint T = seq_len[0];
    uint T_TILE = tile_size[0];
    uint VALS_PER_WORD = 8;
    uint PACKED_WORDS = D / VALS_PER_WORD;
    uint TG_SIZE = threads_per_threadgroup.x;

    uint t_start = tile_id * T_TILE;
    uint t_end = min(t_start + T_TILE, T);

    for (uint d = lane; d < D; d += TG_SIZE) {
        float sum = 0.0f;
        uint w_idx = d / VALS_PER_WORD;
        uint bit_pos = d % VALS_PER_WORD;

        for (uint t = t_start; t < t_end; t++) {
            float attn_w = attn_weights[q_head_idx * T + t];
            if (attn_w == 0.0f) continue;

            float norm_val = norms[kv_head_idx * T + t];
            const device uint8_t* packed_base =
                V_packed + (kv_head_idx * T + t) * PACKED_WORDS * 3;

            uint byte0 = packed_base[w_idx * 3 + 0];
            uint byte1 = packed_base[w_idx * 3 + 1];
            uint byte2 = packed_base[w_idx * 3 + 2];
            uint word = byte0 | (byte1 << 8) | (byte2 << 16);

            uint idx = (word >> (bit_pos * 3)) & 0x7;
            float v_val = centroids[idx] * norm_val;
            sum += attn_w * v_val;
        }

        partials[(tile_id * num_heads_q[0] + q_head_idx) * D + d] = sum;
    }
"""


_kernel_cache: dict[str, object] = {}


def _get_tiled_kernel():
    if "tiled_v" not in _kernel_cache:
        _kernel_cache["tiled_v"] = mx.fast.metal_kernel(
            name="fused_value_accum_3bit_tiled",
            input_names=[
                "V_packed",
                "centroids",
                "norms",
                "attn_weights",
                "kv_head_map",
                "head_dim",
                "seq_len",
                "tile_size",
                "num_heads_q",
            ],
            output_names=["partials"],
            source=_FUSED_VALUE_ACCUM_TILED_SOURCE,
        )
    return _kernel_cache["tiled_v"]


def fused_value_accum_tiled(
    V_packed: mx.array,
    centroids: mx.array,
    norms: mx.array,
    attn_weights: mx.array,
    kv_head_map: mx.array,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    tile_size: int = 128,
) -> mx.array:
    """T-parallel fused value accumulation."""
    if head_dim % 8 != 0:
        raise ValueError(f"head_dim {head_dim} must be divisible by 8")
    if tile_size < 1:
        raise ValueError(f"tile_size {tile_size} must be >= 1")

    num_tiles = (seq_len + tile_size - 1) // tile_size
    kernel = _get_tiled_kernel()

    hd = mx.array([head_dim], dtype=mx.uint32)
    sl = mx.array([seq_len], dtype=mx.uint32)
    ts = mx.array([tile_size], dtype=mx.uint32)
    nh = mx.array([num_heads], dtype=mx.uint32)

    (partials,) = kernel(
        inputs=[
            V_packed.reshape(-1),
            centroids.reshape(-1),
            norms.reshape(-1),
            attn_weights.reshape(-1),
            kv_head_map.reshape(-1),
            hd,
            sl,
            ts,
            nh,
        ],
        output_shapes=[(num_tiles * num_heads * head_dim,)],
        output_dtypes=[mx.float32],
        grid=(num_tiles * 32, num_heads, 1),
        threadgroup=(32, 1, 1),
    )
    return partials.reshape(num_tiles, num_heads, head_dim).sum(axis=0)
