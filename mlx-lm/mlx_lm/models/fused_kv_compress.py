"""Fused Metal kernel: normalize -> FWHT -> SO(4) -> quantize -> 3-bit pack.

Replaces the Python-side _compress_batch + pack_indices_3bit pipeline
with a single Metal dispatch.

Supports D=128 (NPT=4, BPT=1) and D=256 (NPT=8, BPT=2).
Grid: (32, 1, n_vectors), threadgroup: (32, 1, 1).

For D=128 (NPT=4): each lane quantizes 4 elements, writes them to shared
memory, then even lanes pack 8 consecutive indices into 3 bytes.
For D=256 (NPT=8): each lane quantizes 8 elements and packs them directly.
"""

from __future__ import annotations

import mlx.core as mx

_FUSED_COMPRESS_PACK_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    uint vec_idx = thread_position_in_grid.z;
    uint head_idx = vec_idx / seq_len[0];
    uint in_offset = vec_idx * D;
    uint block_base = head_idx * (D / 4);

    threadgroup float shared_a[D];
    threadgroup float shared_b[D];

    // --- Stage 1: Load + L2 normalize ---
    float v[NPT];
    for (uint i = 0; i < NPT; ++i) {
        v[i] = x_in[in_offset + lane * NPT + i];
    }

    float local_sq = 0.0f;
    for (uint i = 0; i < NPT; ++i) local_sq += v[i] * v[i];
    shared_a[lane] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 16; s >= 1; s >>= 1) {
        if (lane < s) shared_a[lane] += shared_a[lane + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        float total_sq = shared_a[0];
        float norm_val = sqrt(total_sq);
        norms_out[vec_idx] = (half)norm_val;
        shared_a[0] = 1.0f / max(norm_val, 1e-8f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_norm = shared_a[0];
    for (uint i = 0; i < NPT; ++i) v[i] *= inv_norm;

    // --- Stage 2: FWHT (in-place via shared memory) ---
    for (uint i = 0; i < NPT; ++i) shared_a[lane * NPT + i] = v[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float* src = shared_a;
    threadgroup float* dst = shared_b;

    for (uint stride = 1; stride < D; stride <<= 1) {
        for (uint i = 0; i < NPT; ++i) {
            uint idx = lane * NPT + i;
            uint partner = idx ^ stride;
            float self_val = src[idx];
            float other_val = src[partner];
            dst[idx] = ((idx & stride) == 0)
                ? (self_val + other_val)
                : (other_val - self_val);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* tmp = src;
        src = dst;
        dst = tmp;
    }

    // --- Stage 3: SO(4) block multiply (column-vector: M @ v) ---
    float fwht_scale = 1.0f / sqrt((float)D);
    float rotated[NPT];
    for (uint bi = 0; bi < BPT; ++bi) {
        uint block_idx = lane * BPT + bi;
        uint base = block_idx * 4;
        uint blk_offset = (block_base + block_idx) * 16;

        float v0 = src[base + 0] * fwht_scale;
        float v1 = src[base + 1] * fwht_scale;
        float v2 = src[base + 2] * fwht_scale;
        float v3 = src[base + 3] * fwht_scale;

        rotated[bi * 4 + 0] =
            blocks[blk_offset + 0] * v0 + blocks[blk_offset + 1] * v1 +
            blocks[blk_offset + 2] * v2 + blocks[blk_offset + 3] * v3;
        rotated[bi * 4 + 1] =
            blocks[blk_offset + 4] * v0 + blocks[blk_offset + 5] * v1 +
            blocks[blk_offset + 6] * v2 + blocks[blk_offset + 7] * v3;
        rotated[bi * 4 + 2] =
            blocks[blk_offset + 8] * v0 + blocks[blk_offset + 9] * v1 +
            blocks[blk_offset + 10] * v2 + blocks[blk_offset + 11] * v3;
        rotated[bi * 4 + 3] =
            blocks[blk_offset + 12] * v0 + blocks[blk_offset + 13] * v1 +
            blocks[blk_offset + 14] * v2 + blocks[blk_offset + 15] * v3;
    }

    // --- Stage 4: Scalar quantize (count boundaries exceeded) ---
    uint8_t idx_arr[NPT];
    for (uint i = 0; i < NPT; ++i) {
        float val = rotated[i];
        uint8_t count = 0;
        for (uint b = 0; b < N_BOUNDARIES; ++b) {
            if (val > boundaries[b]) count++;
        }
        idx_arr[i] = count;
        indices_out[in_offset + lane * NPT + i] = count;
    }

    // --- Stage 5: 3-bit pack (8 indices -> 3 bytes) ---
    if (COOPERATIVE_PACK) {
        // D=128 path: NPT=4, so each lane has only 4 indices.
        // Write to shared memory, barrier, then even lanes pack groups of 8.
        threadgroup uint8_t qidx[D];
        for (uint i = 0; i < NPT; ++i) {
            qidx[lane * NPT + i] = idx_arr[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if ((lane & 1) == 0) {
            uint pack_lane = lane / 2;
            uint qbase = lane * NPT;  // = pack_lane * 8
            uint word = 0;
            for (uint i = 0; i < 8; ++i) {
                word |= ((uint)qidx[qbase + i]) << (i * 3);
            }
            uint pack_offset = vec_idx * PACKED_BYTES + pack_lane * 3;
            packed_out[pack_offset + 0] = (uint8_t)(word & 0xFF);
            packed_out[pack_offset + 1] = (uint8_t)((word >> 8) & 0xFF);
            packed_out[pack_offset + 2] = (uint8_t)((word >> 16) & 0xFF);
        }
    } else {
        // D=256 path: NPT=8, each lane packs its own 8 indices directly.
        uint word = 0;
        for (uint i = 0; i < NPT; ++i) {
            word |= ((uint)idx_arr[i]) << (i * 3);
        }
        uint pack_offset = vec_idx * PACKED_BYTES + lane * 3;
        packed_out[pack_offset + 0] = (uint8_t)(word & 0xFF);
        packed_out[pack_offset + 1] = (uint8_t)((word >> 8) & 0xFF);
        packed_out[pack_offset + 2] = (uint8_t)((word >> 16) & 0xFF);
    }
"""

_kernel_cache: dict[str, any] = {}


def _get_fused_compress_kernel(head_dim: int):
    key = f"fused_compress_pack_d{head_dim}"
    if key not in _kernel_cache:
        _kernel_cache[key] = mx.fast.metal_kernel(
            name=f"isoquant_fused_compress_pack_d{head_dim}",
            input_names=["x_in", "blocks", "boundaries", "seq_len"],
            output_names=["packed_out", "norms_out", "indices_out"],
            source=_FUSED_COMPRESS_PACK_SOURCE,
        )
    return _kernel_cache[key]


def fused_compress_and_pack(
    x: mx.array,
    block_matrices: mx.array,
    centroids: mx.array,
    boundaries: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Fused Metal encode: normalize -> FWHT -> SO(4) -> quantize -> 3-bit pack.

    Args:
        x: (num_heads, seq_len, head_dim) float32
        block_matrices: (num_heads, num_blocks, 4, 4) float32 — forward rotation
        centroids: (N_CENTROIDS,) float32 — unused by kernel, kept for API compat
        boundaries: (N_BOUNDARIES,) float32 — decision boundaries

    Returns:
        packed: (num_heads, seq_len, packed_bytes) uint8 — 3-bit packed indices
        norms: (num_heads, seq_len, 1) float16 — L2 norms
        indices: (num_heads, seq_len, head_dim) uint8 — raw quantization indices
    """
    num_heads, seq_len, head_dim = x.shape
    assert head_dim in (128, 256), (
        f"Fused compress requires D in {{128, 256}}, got {head_dim}"
    )
    n_vectors = num_heads * seq_len
    npt = head_dim // 32
    bpt = head_dim // 128
    packed_bytes_per_vec = head_dim // 8 * 3
    cooperative = 1 if head_dim == 128 else 0

    blocks_flat = block_matrices.reshape(-1)
    sl = mx.array([seq_len], dtype=mx.uint32)

    kernel = _get_fused_compress_kernel(head_dim)
    packed_flat, norms_flat, indices_flat = kernel(
        inputs=[x.reshape(-1).astype(mx.float32), blocks_flat, boundaries, sl],
        template=[
            ("D", head_dim),
            ("NPT", npt),
            ("BPT", bpt),
            ("N_BOUNDARIES", boundaries.shape[0]),
            ("PACKED_BYTES", packed_bytes_per_vec),
            ("COOPERATIVE_PACK", cooperative),
        ],
        output_shapes=[
            (n_vectors * packed_bytes_per_vec,),
            (n_vectors,),
            (n_vectors * head_dim,),
        ],
        output_dtypes=[mx.uint8, mx.float16, mx.uint8],
        grid=(32, 1, n_vectors),
        threadgroup=(32, 1, 1),
    )

    packed = packed_flat.reshape(num_heads, seq_len, packed_bytes_per_vec)
    norms = norms_flat.reshape(num_heads, seq_len, 1)
    indices = indices_flat.reshape(num_heads, seq_len, head_dim)
    return packed, norms, indices
