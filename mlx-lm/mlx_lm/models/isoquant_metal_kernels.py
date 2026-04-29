"""Metal kernels for IsoQuant structured rotation: FWHT + block-diagonal SO(4).

Replaces the dense d×d matmul with two structured passes:
1. Fast Walsh-Hadamard Transform (FWHT): O(d log d) global decorrelation
2. Block-diagonal 4×4 SO(4) multiply: O(d) per-block rotation

Combined cost: ~1,408 FMAs for d=128 vs 16,384 for dense matmul.

Usage:
    from .isoquant_metal_kernels import metal_rotate_forward, metal_rotate_inverse
    x_rot = metal_rotate_forward(x, block_matrices_t, use_hadamard=True)
    x_hat = metal_rotate_inverse(x_rot, block_matrices, use_hadamard=True)
"""

import mlx.core as mx

# ---------------------------------------------------------------------------
# Metal shader sources
# ---------------------------------------------------------------------------

# In-place FWHT on the last dimension of a (H, S, D) tensor.
# Each thread handles one (head, seq) position.
# D must be a power of 2.  Normalized by 1/sqrt(D) at the end.
_FWHT_KERNEL_SOURCE = """
    // grid: (num_heads * seq_len, 1, 1)
    // Each thread processes one D-dimensional vector in-place.
    uint tid = thread_position_in_grid.x;
    uint D = head_dim[0];
    uint offset = tid * D;

    // Butterfly stages
    for (uint stride = 1; stride < D; stride <<= 1) {
        for (uint j = 0; j < D; j += stride * 2) {
            for (uint k = 0; k < stride; k++) {
                uint i0 = offset + j + k;
                uint i1 = i0 + stride;
                float a = out[i0];
                float b = out[i1];
                out[i0] = a + b;
                out[i1] = a - b;
            }
        }
    }

    // Normalize by 1/sqrt(D)
    float norm = 1.0f / sqrt((float)D);
    for (uint j = 0; j < D; j++) {
        out[offset + j] *= norm;
    }
"""

# Block-diagonal 4×4 matmul on the last dimension.
# blocks: (num_heads, num_blocks, 4, 4)
# x: (num_heads, seq_len, head_dim)  -- contiguous
# out: (num_heads, seq_len, head_dim)
# Each thread handles one (head, seq) position.
_BLOCK4x4_KERNEL_SOURCE = """
    uint tid = thread_position_in_grid.x;
    uint D = head_dim[0];
    uint n_blocks = D / 4;
    uint x_offset = tid * D;

    for (uint b = 0; b < n_blocks; b++) {
        uint head_idx = tid / seq_len[0];
        uint blk_offset = (head_idx * n_blocks + b) * 16;  // 4*4 = 16 per block
        uint v_offset = x_offset + b * 4;

        float v0 = x[v_offset + 0];
        float v1 = x[v_offset + 1];
        float v2 = x[v_offset + 2];
        float v3 = x[v_offset + 3];

        out[v_offset + 0] = blocks[blk_offset + 0] * v0 + blocks[blk_offset + 1] * v1 + blocks[blk_offset + 2] * v2 + blocks[blk_offset + 3] * v3;
        out[v_offset + 1] = blocks[blk_offset + 4] * v0 + blocks[blk_offset + 5] * v1 + blocks[blk_offset + 6] * v2 + blocks[blk_offset + 7] * v3;
        out[v_offset + 2] = blocks[blk_offset + 8] * v0 + blocks[blk_offset + 9] * v1 + blocks[blk_offset + 10] * v2 + blocks[blk_offset + 11] * v3;
        out[v_offset + 3] = blocks[blk_offset + 12] * v0 + blocks[blk_offset + 13] * v1 + blocks[blk_offset + 14] * v2 + blocks[blk_offset + 15] * v3;
    }
"""

# Fused forward path matching row-vector semantics in mlx_isoquant.py:
# x @ R.T = x @ H @ R_block.T
_FUSED_FORWARD_SOURCE = """
    uint tid = thread_position_in_grid.x;
    uint D = head_dim[0];
    uint n_blocks = D / 4;
    uint offset = tid * D;

    // Forward row-vector rotation:
    // x @ R.T = x @ H @ R_block.T
    // Step 1: copy input to output buffer, then apply FWHT in-place.
    for (uint j = 0; j < D; j++) {
        out[offset + j] = x[offset + j];
    }

    for (uint stride = 1; stride < D; stride <<= 1) {
        for (uint j = 0; j < D; j += stride * 2) {
            for (uint k = 0; k < stride; k++) {
                uint i0 = offset + j + k;
                uint i1 = i0 + stride;
                float a = out[i0];
                float b = out[i1];
                out[i0] = a + b;
                out[i1] = a - b;
            }
        }
    }
    float norm = 1.0f / sqrt((float)D);
    for (uint j = 0; j < D; j++) {
        out[offset + j] *= norm;
    }

    // Step 2: block-diagonal 4x4 multiply with transposed blocks.
    uint head_idx = tid / seq_len[0];
    for (uint b = 0; b < n_blocks; b++) {
        uint blk_offset = (head_idx * n_blocks + b) * 16;
        uint v_off = offset + b * 4;

        float v0 = out[v_off + 0];
        float v1 = out[v_off + 1];
        float v2 = out[v_off + 2];
        float v3 = out[v_off + 3];

        float r0 = blocks[blk_offset + 0] * v0 + blocks[blk_offset + 1] * v1 + blocks[blk_offset + 2] * v2 + blocks[blk_offset + 3] * v3;
        float r1 = blocks[blk_offset + 4] * v0 + blocks[blk_offset + 5] * v1 + blocks[blk_offset + 6] * v2 + blocks[blk_offset + 7] * v3;
        float r2 = blocks[blk_offset + 8] * v0 + blocks[blk_offset + 9] * v1 + blocks[blk_offset + 10] * v2 + blocks[blk_offset + 11] * v3;
        float r3 = blocks[blk_offset + 12] * v0 + blocks[blk_offset + 13] * v1 + blocks[blk_offset + 14] * v2 + blocks[blk_offset + 15] * v3;

        out[v_off + 0] = r0;
        out[v_off + 1] = r1;
        out[v_off + 2] = r2;
        out[v_off + 3] = r3;
    }
"""

# Fused inverse path matching row-vector semantics in mlx_isoquant.py:
# x_rot @ R = (x_rot @ R_block) @ H
_FUSED_INVERSE_SOURCE = """
    uint tid = thread_position_in_grid.x;
    uint D = head_dim[0];
    uint n_blocks = D / 4;
    uint offset = tid * D;

    // Inverse row-vector rotation:
    // x_rot @ R = (x_rot @ R_block) @ H
    // Step 1: block-diagonal 4x4 multiply with forward blocks.
    uint head_idx = tid / seq_len[0];
    for (uint b = 0; b < n_blocks; b++) {
        uint blk_offset = (head_idx * n_blocks + b) * 16;
        uint v_off = offset + b * 4;

        float v0 = x[v_off + 0];
        float v1 = x[v_off + 1];
        float v2 = x[v_off + 2];
        float v3 = x[v_off + 3];

        float r0 = blocks[blk_offset + 0] * v0 + blocks[blk_offset + 1] * v1 + blocks[blk_offset + 2] * v2 + blocks[blk_offset + 3] * v3;
        float r1 = blocks[blk_offset + 4] * v0 + blocks[blk_offset + 5] * v1 + blocks[blk_offset + 6] * v2 + blocks[blk_offset + 7] * v3;
        float r2 = blocks[blk_offset + 8] * v0 + blocks[blk_offset + 9] * v1 + blocks[blk_offset + 10] * v2 + blocks[blk_offset + 11] * v3;
        float r3 = blocks[blk_offset + 12] * v0 + blocks[blk_offset + 13] * v1 + blocks[blk_offset + 14] * v2 + blocks[blk_offset + 15] * v3;

        out[v_off + 0] = r0;
        out[v_off + 1] = r1;
        out[v_off + 2] = r2;
        out[v_off + 3] = r3;
    }

    // Step 2: in-place FWHT on the block-rotated output.
    for (uint stride = 1; stride < D; stride <<= 1) {
        for (uint j = 0; j < D; j += stride * 2) {
            for (uint k = 0; k < stride; k++) {
                uint i0 = offset + j + k;
                uint i1 = i0 + stride;
                float a = out[i0];
                float b = out[i1];
                out[i0] = a + b;
                out[i1] = a - b;
            }
        }
    }
    float norm = 1.0f / sqrt((float)D);
    for (uint j = 0; j < D; j++) {
        out[offset + j] *= norm;
    }
"""

# Optimized fused forward path: one threadgroup (32 threads) per vector.
# Templates:
#   D   = head_dim
#   NPT = D / 32   (number of contiguous elements handled per lane)
#   BPT = D / 128  (number of 4x4 blocks handled per lane)
_OPT_FUSED_FORWARD_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    uint vec_idx = thread_position_in_grid.z;
    uint head_idx = vec_idx / seq_len[0];
    uint offset = vec_idx * D;
    uint block_base = head_idx * (D / 4);

    threadgroup float shared_a[D];
    threadgroup float shared_b[D];

    for (uint i = 0; i < NPT; ++i) {
        uint idx = lane * NPT + i;
        shared_a[idx] = x[offset + idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float* src = shared_a;
    threadgroup float* dst = shared_b;

    for (uint stride = 1; stride < D; stride <<= 1) {
        for (uint i = 0; i < NPT; ++i) {
            uint idx = lane * NPT + i;
            uint partner = idx ^ stride;
            float self = src[idx];
            float other = src[partner];
            dst[idx] = ((idx & stride) == 0) ? (self + other) : (other - self);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* tmp = src;
        src = dst;
        dst = tmp;
    }

    float norm = 1.0f / sqrt((float)D);
    for (uint bi = 0; bi < BPT; ++bi) {
        uint block_idx = lane * BPT + bi;
        uint base = block_idx * 4;
        uint blk_offset = (block_base + block_idx) * 16;

        float v0 = src[base + 0] * norm;
        float v1 = src[base + 1] * norm;
        float v2 = src[base + 2] * norm;
        float v3 = src[base + 3] * norm;

        out[offset + base + 0] =
            blocks[blk_offset + 0] * v0 + blocks[blk_offset + 1] * v1 +
            blocks[blk_offset + 2] * v2 + blocks[blk_offset + 3] * v3;
        out[offset + base + 1] =
            blocks[blk_offset + 4] * v0 + blocks[blk_offset + 5] * v1 +
            blocks[blk_offset + 6] * v2 + blocks[blk_offset + 7] * v3;
        out[offset + base + 2] =
            blocks[blk_offset + 8] * v0 + blocks[blk_offset + 9] * v1 +
            blocks[blk_offset + 10] * v2 + blocks[blk_offset + 11] * v3;
        out[offset + base + 3] =
            blocks[blk_offset + 12] * v0 + blocks[blk_offset + 13] * v1 +
            blocks[blk_offset + 14] * v2 + blocks[blk_offset + 15] * v3;
    }
"""

# Optimized fused inverse path: block multiply into threadgroup memory,
# then FWHT with one threadgroup per vector.
_OPT_FUSED_INVERSE_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    uint vec_idx = thread_position_in_grid.z;
    uint head_idx = vec_idx / seq_len[0];
    uint offset = vec_idx * D;
    uint block_base = head_idx * (D / 4);

    threadgroup float shared_a[D];
    threadgroup float shared_b[D];

    for (uint bi = 0; bi < BPT; ++bi) {
        uint block_idx = lane * BPT + bi;
        uint base = block_idx * 4;
        uint blk_offset = (block_base + block_idx) * 16;

        float v0 = x[offset + base + 0];
        float v1 = x[offset + base + 1];
        float v2 = x[offset + base + 2];
        float v3 = x[offset + base + 3];

        shared_a[base + 0] =
            blocks[blk_offset + 0] * v0 + blocks[blk_offset + 1] * v1 +
            blocks[blk_offset + 2] * v2 + blocks[blk_offset + 3] * v3;
        shared_a[base + 1] =
            blocks[blk_offset + 4] * v0 + blocks[blk_offset + 5] * v1 +
            blocks[blk_offset + 6] * v2 + blocks[blk_offset + 7] * v3;
        shared_a[base + 2] =
            blocks[blk_offset + 8] * v0 + blocks[blk_offset + 9] * v1 +
            blocks[blk_offset + 10] * v2 + blocks[blk_offset + 11] * v3;
        shared_a[base + 3] =
            blocks[blk_offset + 12] * v0 + blocks[blk_offset + 13] * v1 +
            blocks[blk_offset + 14] * v2 + blocks[blk_offset + 15] * v3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float* src = shared_a;
    threadgroup float* dst = shared_b;

    for (uint stride = 1; stride < D; stride <<= 1) {
        for (uint i = 0; i < NPT; ++i) {
            uint idx = lane * NPT + i;
            uint partner = idx ^ stride;
            float self = src[idx];
            float other = src[partner];
            dst[idx] = ((idx & stride) == 0) ? (self + other) : (other - self);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* tmp = src;
        src = dst;
        dst = tmp;
    }

    float norm = 1.0f / sqrt((float)D);
    for (uint i = 0; i < NPT; ++i) {
        uint idx = lane * NPT + i;
        out[offset + idx] = src[idx] * norm;
    }
"""

# ---------------------------------------------------------------------------
# Compiled kernel objects (lazy init, cached)
# ---------------------------------------------------------------------------

_kernel_cache: dict[str, any] = {}


def _get_fused_forward_kernel():
    if "fused_fwd" not in _kernel_cache:
        _kernel_cache["fused_fwd"] = mx.fast.metal_kernel(
            name="isoquant_fused_forward",
            input_names=["x", "blocks", "head_dim", "seq_len"],
            output_names=["out"],
            source=_FUSED_FORWARD_SOURCE,
        )
    return _kernel_cache["fused_fwd"]


def _get_fused_inverse_kernel():
    if "fused_inv" not in _kernel_cache:
        _kernel_cache["fused_inv"] = mx.fast.metal_kernel(
            name="isoquant_fused_inverse",
            input_names=["x", "blocks", "head_dim", "seq_len"],
            output_names=["out"],
            source=_FUSED_INVERSE_SOURCE,
        )
    return _kernel_cache["fused_inv"]


def _get_opt_fused_forward_kernel():
    if "opt_fused_fwd" not in _kernel_cache:
        _kernel_cache["opt_fused_fwd"] = mx.fast.metal_kernel(
            name="isoquant_opt_fused_forward",
            input_names=["x", "blocks", "seq_len"],
            output_names=["out"],
            source=_OPT_FUSED_FORWARD_SOURCE,
        )
    return _kernel_cache["opt_fused_fwd"]


def _get_opt_fused_inverse_kernel():
    if "opt_fused_inv" not in _kernel_cache:
        _kernel_cache["opt_fused_inv"] = mx.fast.metal_kernel(
            name="isoquant_opt_fused_inverse",
            input_names=["x", "blocks", "seq_len"],
            output_names=["out"],
            source=_OPT_FUSED_INVERSE_SOURCE,
        )
    return _kernel_cache["opt_fused_inv"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def metal_rotate_forward(
    x: mx.array,
    block_matrices: mx.array,
    use_hadamard: bool = True,
) -> mx.array:
    """Forward row-vector rotation via fused Metal kernel.

    Args:
        x: (num_heads, seq_len, head_dim) float32
        block_matrices: (num_heads, num_blocks, 4, 4) — forward block matrices
        use_hadamard: if False, skip FWHT (block-only rotation for non-power-of-2 dims)

    Returns:
        x_rot: (num_heads, seq_len, head_dim) float32
    """
    if not use_hadamard:
        # Block-only path — use the standalone block kernel
        return _metal_block_multiply_rowvec(x, block_matrices)

    num_heads, seq_len, head_dim = x.shape
    n_threads = num_heads * seq_len

    if _can_use_optimized_fwht(head_dim):
        return _metal_rotate_forward_optimized(x, block_matrices)

    # Flatten blocks for the kernel: (H, B, 4, 4) → (H * B * 16,)
    blocks_flat = block_matrices.reshape(-1)
    hd = mx.array([head_dim], dtype=mx.uint32)
    sl = mx.array([seq_len], dtype=mx.uint32)

    kernel = _get_fused_forward_kernel()
    (out,) = kernel(
        inputs=[x.reshape(-1).astype(mx.float32), blocks_flat, hd, sl],
        output_shapes=[(n_threads * head_dim,)],
        output_dtypes=[mx.float32],
        grid=(n_threads, 1, 1),
        threadgroup=(min(n_threads, 256), 1, 1),
    )
    return out.reshape(num_heads, seq_len, head_dim)


def metal_rotate_inverse(
    x_rot: mx.array,
    block_matrices_t: mx.array,
    use_hadamard: bool = True,
) -> mx.array:
    """Inverse row-vector rotation via fused Metal kernel.

    Args:
        x_rot: (num_heads, seq_len, head_dim) float32
        block_matrices_t: (num_heads, num_blocks, 4, 4) — transposed block matrices
        use_hadamard: if False, skip inverse FWHT

    Returns:
        x_hat: (num_heads, seq_len, head_dim) float32
    """
    if not use_hadamard:
        return _metal_block_multiply_rowvec(x_rot, block_matrices_t)

    num_heads, seq_len, head_dim = x_rot.shape
    n_threads = num_heads * seq_len

    if _can_use_optimized_fwht(head_dim):
        return _metal_rotate_inverse_optimized(x_rot, block_matrices_t)

    blocks_flat = block_matrices_t.reshape(-1)
    hd = mx.array([head_dim], dtype=mx.uint32)
    sl = mx.array([seq_len], dtype=mx.uint32)

    kernel = _get_fused_inverse_kernel()
    (out,) = kernel(
        inputs=[x_rot.reshape(-1).astype(mx.float32), blocks_flat, hd, sl],
        output_shapes=[(n_threads * head_dim,)],
        output_dtypes=[mx.float32],
        grid=(n_threads, 1, 1),
        threadgroup=(min(n_threads, 256), 1, 1),
    )
    return out.reshape(num_heads, seq_len, head_dim)


def _metal_block_multiply_rowvec(x: mx.array, blocks: mx.array) -> mx.array:
    """Block-diagonal row-vector 4×4 multiply only (no FWHT).

    The shader computes column-vector semantics (`M @ v`), so callers pass the
    matrix whose transpose matches the desired row-vector multiply.
    """
    num_heads, seq_len, head_dim = x.shape
    n_threads = num_heads * seq_len

    kernel_src = _BLOCK4x4_KERNEL_SOURCE
    if "block_only" not in _kernel_cache:
        _kernel_cache["block_only"] = mx.fast.metal_kernel(
            name="isoquant_block4x4",
            input_names=["x", "blocks", "head_dim", "seq_len"],
            output_names=["out"],
            source=kernel_src,
        )
    kernel = _kernel_cache["block_only"]

    blocks_flat = blocks.reshape(-1)
    hd = mx.array([head_dim], dtype=mx.uint32)
    sl = mx.array([seq_len], dtype=mx.uint32)

    (out,) = kernel(
        inputs=[x.reshape(-1).astype(mx.float32), blocks_flat, hd, sl],
        output_shapes=[(n_threads * head_dim,)],
        output_dtypes=[mx.float32],
        grid=(n_threads, 1, 1),
        threadgroup=(min(n_threads, 256), 1, 1),
    )
    return out.reshape(num_heads, seq_len, head_dim)


def _can_use_optimized_fwht(head_dim: int) -> bool:
    return head_dim in (128, 256, 512)


def _metal_rotate_forward_optimized(x: mx.array, blocks: mx.array) -> mx.array:
    num_heads, seq_len, head_dim = x.shape
    n_vectors = num_heads * seq_len
    blocks_flat = blocks.reshape(-1)
    sl = mx.array([seq_len], dtype=mx.uint32)
    kernel = _get_opt_fused_forward_kernel()
    (out,) = kernel(
        inputs=[x.reshape(-1).astype(mx.float32), blocks_flat, sl],
        template=[
            ("D", head_dim),
            ("NPT", head_dim // 32),
            ("BPT", head_dim // 128),
        ],
        output_shapes=[(n_vectors * head_dim,)],
        output_dtypes=[mx.float32],
        grid=(32, 1, n_vectors),
        threadgroup=(32, 1, 1),
    )
    return out.reshape(num_heads, seq_len, head_dim)


def _metal_rotate_inverse_optimized(
    x_rot: mx.array, blocks_t: mx.array
) -> mx.array:
    num_heads, seq_len, head_dim = x_rot.shape
    n_vectors = num_heads * seq_len
    blocks_flat = blocks_t.reshape(-1)
    sl = mx.array([seq_len], dtype=mx.uint32)
    kernel = _get_opt_fused_inverse_kernel()
    (out,) = kernel(
        inputs=[x_rot.reshape(-1).astype(mx.float32), blocks_flat, sl],
        template=[
            ("D", head_dim),
            ("NPT", head_dim // 32),
            ("BPT", head_dim // 128),
        ],
        output_shapes=[(n_vectors * head_dim,)],
        output_dtypes=[mx.float32],
        grid=(32, 1, n_vectors),
        threadgroup=(32, 1, 1),
    )
    return out.reshape(num_heads, seq_len, head_dim)
