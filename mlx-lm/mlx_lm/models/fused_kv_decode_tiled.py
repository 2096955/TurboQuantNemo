"""T-parallel-tiled fused value accumulation for IsoQuant decode.

The existing ``fused_value_accum`` (in fused_kv_decode_kernels.py) is the
hot path of IsoQuant decode at long context — instrumentation showed it
accounts for ~39% of decode wall time. The bottleneck is its inner T-loop:
each threadgroup iterates the full stored cache serially, so the kernel
parallelizes over D dims but not T tokens.

This module provides a T-parallel version. The compute is split across
``num_tiles`` threadgroups, each handling a contiguous T-slice. Per-tile
partial sums are written to global memory and summed via a single ``mx.sum``
reduction. No FA2 merge formula is needed because the softmax has already
been applied upstream — the merge is just elementwise addition.

Drop-in replacement for ``fused_value_accum``: same inputs/outputs, same
numerical contract.
"""

from __future__ import annotations

import mlx.core as mx

# Per-tile fused value accumulation kernel.
#
# Launch:      grid=(num_tiles * 32, H_q, 1), threadgroup=(32, 1, 1)
#              → num_tiles TGs in x, H_q TGs in y; 32 threads per TG
# Per TG:      one threadgroup per (tile, query head); 32 threads cooperate
#              over D dims via lane-stride loop
# Each TG:     iterates [t_start, t_end) instead of [0, T), writing a partial
#              sum to ``partials[tile_id, head, d]``.
_FUSED_VALUE_ACCUM_TILED_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    uint tile_id = threadgroup_position_in_grid.x;
    uint q_head_idx = threadgroup_position_in_grid.y;
    uint kv_head_idx = kv_head_map[q_head_idx];

    uint D = head_dim[0];
    uint T = seq_len[0];
    uint T_stride = storage_stride[0];
    uint T_TILE = tile_size[0];
    uint VALS_PER_WORD = 8;
    uint PACKED_WORDS = D / VALS_PER_WORD;
    uint TG_SIZE = threads_per_threadgroup.x;

    uint t_start = tile_id * T_TILE;
    uint t_end = min(t_start + T_TILE, T);

    // Each thread accumulates partial sums for its assigned dimensions
    // over the tile's [t_start, t_end) token range.
    for (uint d = lane; d < D; d += TG_SIZE) {
        float sum = 0.0f;
        uint w_idx = d / VALS_PER_WORD;
        uint bit_pos = d % VALS_PER_WORD;

        for (uint t = t_start; t < t_end; t++) {
            float attn_w = attn_weights[q_head_idx * T + t];
            if (attn_w == 0.0f) continue;

            float norm_val = norms[kv_head_idx * T_stride + t];
            const device uint8_t* packed_base =
                V_packed + (kv_head_idx * T_stride + t) * PACKED_WORDS * 3;

            uint byte0 = packed_base[w_idx * 3 + 0];
            uint byte1 = packed_base[w_idx * 3 + 1];
            uint byte2 = packed_base[w_idx * 3 + 2];
            uint word = byte0 | (byte1 << 8) | (byte2 << 16);

            uint idx = (word >> (bit_pos * 3)) & 0x7;
            float v_val = centroids[idx] * norm_val;
            sum += attn_w * v_val;
        }

        // Write per-tile partial: partials[tile_id, head, d]
        uint num_tiles = (T + T_TILE - 1) / T_TILE;
        partials[(tile_id * num_heads_q[0] + q_head_idx) * D + d] = sum;
    }
"""


_kernel_cache: dict[str, object] = {}


def _get_tiled_kernel():
    key = "tiled_v_v2"
    if key not in _kernel_cache:
        _kernel_cache[key] = mx.fast.metal_kernel(
            name="fused_value_accum_3bit_tiled",
            input_names=[
                "V_packed",
                "centroids",
                "norms",
                "attn_weights",
                "kv_head_map",
                "head_dim",
                "seq_len",
                "storage_stride",
                "tile_size",
                "num_heads_q",
            ],
            output_names=["partials"],
            source=_FUSED_VALUE_ACCUM_TILED_SOURCE,
        )
    return _kernel_cache[key]


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
    storage_stride: int | None = None,
) -> mx.array:
    """T-parallel fused value accumulation.

    Drop-in replacement for ``fused_value_accum``. Splits the T-loop across
    ``ceil(seq_len / tile_size)`` threadgroups and reduces per-tile partials
    via ``mx.sum``.

    Args:
        V_packed:    (H_kv, T, packed_bytes) uint8 — 3-bit packed value indices
        centroids:   (num_centroids,) float32 — Lloyd-Max centroids
        norms:       (H_kv, T) float32 — value norms
        attn_weights: (H_q, T) float32 — softmax-normalized attention weights
        kv_head_map: (H_q,) uint32 — query head → KV head mapping (for GQA)
        num_heads:   H_q
        seq_len:     T
        head_dim:    D (must be divisible by 8)
        tile_size:   T_TILE (default 128). Auto-falls-through to single-tile
                     when seq_len <= tile_size.
        storage_stride: buffer stride along T axis (defaults to seq_len)

    Returns:
        output: (H_q, D) float32 — same contract as fused_value_accum
    """
    if head_dim % 8 != 0:
        raise ValueError(f"head_dim {head_dim} must be divisible by 8")
    if tile_size < 1:
        raise ValueError(f"tile_size {tile_size} must be >= 1")

    num_tiles = (seq_len + tile_size - 1) // tile_size
    kernel = _get_tiled_kernel()

    hd = mx.array([head_dim], dtype=mx.uint32)
    sl = mx.array([seq_len], dtype=mx.uint32)
    ss = mx.array(
        [storage_stride if storage_stride is not None else seq_len], dtype=mx.uint32
    )
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
            ss,
            ts,
            nh,
        ],
        output_shapes=[(num_tiles * num_heads * head_dim,)],
        output_dtypes=[mx.float32],
        # grid is in *threads*, not threadgroups: x = num_tiles TGs * 32 threads/TG,
        # y = num_heads TGs * 1 thread/TG. Mirrors fused_qk_dot's grid=(32*seq_len, H, 1).
        grid=(num_tiles * 32, num_heads, 1),
        threadgroup=(32, 1, 1),
    )
    # Shape: (num_tiles, H_q, D); reduce over tiles.
    partials = partials.reshape(num_tiles, num_heads, head_dim)
    return partials.sum(axis=0)
