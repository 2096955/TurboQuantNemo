"""Fused Metal kernels for the IsoQuant decode pipeline.

Implements two execution paths:

**Single-kernel fully-fused (preferred):**
  One `mx.fast.metal_kernel` dispatch that collapses QK dot + online softmax +
  V accumulation + inverse rotation (SO(4) + WHT) into a single GPU pass.
  32 threads/TG (1 SIMD group), each thread owns 4 dimensions.  Zero barriers
  in the main token loop — `simd_sum` broadcasts the dot product natively.

**Legacy 3-kernel pipeline (fallback):**
  Kernel A: fused_qk_dot_3bit  — Q·Kᵀ from packed 3-bit KV (no K reconstruction)
  Kernel C: fused_value_accum  — weighted V sum from packed storage (no V reconstruction)
  Kernel D: wht_so4_inverse    — structured inverse rotation on aggregated output (once)

The key insight: inverse rotation moves from O(T × d²) to O(d²) by accumulating
in rotated space and rotating the output once.

Usage:
    from .fused_kv_decode_kernels import (
        fully_fused_attention,      # single-kernel path (preferred)
        fused_qk_dot,               # legacy 3-kernel path
        fused_value_accum,
        fused_inverse_rotate,
        fused_isoquant_attention,
    )
"""

from __future__ import annotations

import mlx.core as mx

_MAX_KERNEL_HEAD_DIM = 512

# ---------------------------------------------------------------------------
# Metal shader sources
# ---------------------------------------------------------------------------

# Kernel A: Fused Q·Kᵀ — one threadgroup per token, 32 threads cooperate.
# Decodes 3-bit packed K on the fly, never materialises full K.
# Centroids in constant buffer for fast cache access.
_FUSED_QK_DOT_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    uint token_idx = threadgroup_position_in_grid.x;
    uint q_head_idx = threadgroup_position_in_grid.y;
    uint kv_head_idx = kv_head_map[q_head_idx];

    uint D = head_dim[0];
    uint T = seq_len[0];
    uint VALS_PER_WORD = 8;
    uint PACKED_WORDS = D / VALS_PER_WORD;
    uint TG_SIZE = threads_per_threadgroup.x;

    // Cache query in threadgroup memory
    threadgroup float q_local[512];  // max head_dim
    for (uint i = lane; i < D; i += TG_SIZE) {
        q_local[i] = q[q_head_idx * D + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pointer to this token's packed K for this head
    const device uint8_t* packed_base = K_packed + (kv_head_idx * T + token_idx) * PACKED_WORDS * 3;

    float acc = 0.0f;

    // Striped loop over dimensions (8 values per 3 bytes)
    for (uint w = lane; w < PACKED_WORDS; w += TG_SIZE) {
        // Read 3 bytes = 24 bits = 8 × 3-bit indices
        uint byte0 = packed_base[w * 3 + 0];
        uint byte1 = packed_base[w * 3 + 1];
        uint byte2 = packed_base[w * 3 + 2];
        uint word = byte0 | (byte1 << 8) | (byte2 << 16);

        uint base_dim = w * VALS_PER_WORD;

        // Unpack 8 indices and lookup centroids, accumulate dot product
        for (uint i = 0; i < VALS_PER_WORD; i++) {
            uint idx = (word >> (i * 3)) & 0x7;
            float k_val = centroids[idx];
            float q_val = q_local[base_dim + i];
            acc += k_val * q_val;
        }
    }

    // SIMD reduction
    acc = simd_sum(acc);

    if (lane == 0) {
        float norm_val = norms[(kv_head_idx * T + token_idx)];
        scores[q_head_idx * T + token_idx] = acc * norm_val;
    }
"""

# Kernel C: Fused weighted value accumulation in ROTATED space.
# One threadgroup per head, all threads cooperate over d_k.
# Accumulates sum_j(a_j * dequant(V_packed[j])) without inverse rotation.
_FUSED_VALUE_ACCUM_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    uint q_head_idx = threadgroup_position_in_grid.x;
    uint kv_head_idx = kv_head_map[q_head_idx];

    uint D = head_dim[0];
    uint T = seq_len[0];
    uint VALS_PER_WORD = 8;
    uint PACKED_WORDS = D / VALS_PER_WORD;
    uint TG_SIZE = threads_per_threadgroup.x;

    // Each thread accumulates partial sums for its assigned dimensions
    // We process all tokens, accumulating into thread-local registers
    for (uint d = lane; d < D; d += TG_SIZE) {
        float sum = 0.0f;
        uint w_idx = d / VALS_PER_WORD;
        uint bit_pos = d % VALS_PER_WORD;

        for (uint t = 0; t < T; t++) {
            float attn_w = attn_weights[q_head_idx * T + t];
            if (attn_w == 0.0f) continue;

            float norm_val = norms[kv_head_idx * T + t];
            const device uint8_t* packed_base = V_packed + (kv_head_idx * T + t) * PACKED_WORDS * 3;

            // Read the 3-byte word containing this dimension
            uint byte0 = packed_base[w_idx * 3 + 0];
            uint byte1 = packed_base[w_idx * 3 + 1];
            uint byte2 = packed_base[w_idx * 3 + 2];
            uint word = byte0 | (byte1 << 8) | (byte2 << 16);

            uint idx = (word >> (bit_pos * 3)) & 0x7;
            float v_val = centroids[idx] * norm_val;
            sum += attn_w * v_val;
        }

        output[q_head_idx * D + d] = sum;
    }
"""

# ---------------------------------------------------------------------------
# Single fully-fused kernel: QK dot + online softmax + V accum + inverse rot
# ---------------------------------------------------------------------------
#
# Design:
#   32 threads per threadgroup, 1 SIMD group.  Each thread owns NPT=D/32
#   consecutive dimensions (4 for D=128).  This gives:
#     - Zero barriers in the main token loop (simd_sum broadcasts natively)
#     - SO(4) block rotation fully thread-local (NPT=4 = block size)
#     - Each thread reads exactly 1 packed 3-byte word per K/V per token
#     - Shared memory only for WHT butterfly (7 stages, 1 barrier each)
#
# Online softmax (FlashAttention-style single-pass):
#   For each token t:
#     score = Q·K[t] (via simd_sum)
#     m_new = max(m, score)
#     correction = exp(m - m_new)
#     exp_score = exp(score - m_new)
#     O_d = O_d * correction + exp_score * V[t,d]
#     l = l * correction + exp_score
#   Final: O_d /= l, then inverse rotate once.
#
# Templates: D, NPT, PACKED_WORDS, N_BLOCKS, USE_HADAMARD

_FULLY_FUSED_ATTENTION_SOURCE = """
    // Grid: (32, H_q, 1) — 32 threads per TG, one TG per query head
    // Threadgroup: (32, 1, 1)

    uint lane = thread_position_in_threadgroup.x;
    uint q_head = threadgroup_position_in_grid.y;
    uint kv_head = kv_head_map[q_head];
    uint T = seq_len[0];
    uint use_mask = has_mask[0];

    // Each thread owns NPT consecutive dimensions
    uint dim_base = lane * NPT;

    // Packing: dims [dim_base..dim_base+NPT-1] always fall in one 3-byte word
    // because NPT=4 and words hold 8 values (4+3=7 < 8)
    uint w = dim_base / 8;
    uint bp_base = dim_base % 8;
    uint w_byte = w * 3;

    // Load rotated query values into registers
    float q_r[NPT];
    for (uint i = 0; i < NPT; i++) {
        q_r[i] = q[q_head * D + dim_base + i];
    }

    // Online softmax state
    float m_run = -1e38f;
    float l_run = 0.0f;
    float O_r[NPT];
    for (uint i = 0; i < NPT; i++) O_r[i] = 0.0f;

    // Pre-compute base offsets for K and V packed data
    uint kv_k_base = kv_head * T * PACKED_WORDS * 3;
    uint kv_v_base = kv_head * T * PACKED_WORDS * 3;
    uint stride_bytes = PACKED_WORDS * 3;

    // === Main loop: single pass over all KV tokens ===
    for (uint t = 0; t < T; t++) {
        // --- Decode K[t] for this thread's dims ---
        uint k_off = kv_k_base + t * stride_bytes + w_byte;
        uint kw = uint(K_packed[k_off]) | (uint(K_packed[k_off+1]) << 8) | (uint(K_packed[k_off+2]) << 16);
        float k_norm = k_norms[kv_head * T + t];

        // Partial dot product over this thread's NPT dimensions
        float partial = 0.0f;
        for (uint i = 0; i < NPT; i++) {
            float k_val = centroids[(kw >> ((bp_base + i) * 3)) & 0x7] * k_norm;
            partial += q_r[i] * k_val;
        }

        // Full dot product via SIMD reduction — broadcasts to all 32 lanes
        float score = simd_sum(partial) * scale_val[0];

        // Apply mask (skip read when no mask)
        if (use_mask) score += mask_data[q_head * T + t];

        // --- Online softmax update ---
        float m_new = max(m_run, score);
        float corr = exp(m_run - m_new);
        float es = exp(score - m_new);

        // --- Decode V[t] for this thread's dims and accumulate ---
        uint v_off = kv_v_base + t * stride_bytes + w_byte;
        uint vw = uint(V_packed[v_off]) | (uint(V_packed[v_off+1]) << 8) | (uint(V_packed[v_off+2]) << 16);
        float v_norm = v_norms[kv_head * T + t];

        for (uint i = 0; i < NPT; i++) {
            float v_val = centroids[(vw >> ((bp_base + i) * 3)) & 0x7] * v_norm;
            O_r[i] = O_r[i] * corr + es * v_val;
        }
        l_run = l_run * corr + es;
        m_run = m_new;
    }

    // === Normalize ===
    float inv_l = (l_run > 0.0f) ? (1.0f / l_run) : 0.0f;
    for (uint i = 0; i < NPT; i++) O_r[i] *= inv_l;

    // === Inverse rotation ===
    // Step 1: SO(4) block multiply (fully thread-local: NPT=4 = block size)
    // Each thread owns exactly one 4x4 block (lane = block index)
    uint bo = (kv_head * N_BLOCKS + lane) * 16;
    float r0 = blocks_t[bo+ 0]*O_r[0] + blocks_t[bo+ 1]*O_r[1] + blocks_t[bo+ 2]*O_r[2] + blocks_t[bo+ 3]*O_r[3];
    float r1 = blocks_t[bo+ 4]*O_r[0] + blocks_t[bo+ 5]*O_r[1] + blocks_t[bo+ 6]*O_r[2] + blocks_t[bo+ 7]*O_r[3];
    float r2 = blocks_t[bo+ 8]*O_r[0] + blocks_t[bo+ 9]*O_r[1] + blocks_t[bo+10]*O_r[2] + blocks_t[bo+11]*O_r[3];
    float r3 = blocks_t[bo+12]*O_r[0] + blocks_t[bo+13]*O_r[1] + blocks_t[bo+14]*O_r[2] + blocks_t[bo+15]*O_r[3];
    O_r[0] = r0; O_r[1] = r1; O_r[2] = r2; O_r[3] = r3;

    if (USE_HADAMARD) {
        // Step 2: Walsh-Hadamard butterfly via shared memory (ping-pong)
        threadgroup float sa[D];
        threadgroup float sb[D];

        for (uint i = 0; i < NPT; i++) sa[dim_base + i] = O_r[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float* src = sa;
        threadgroup float* dst = sb;

        for (uint stride = 1; stride < D; stride <<= 1) {
            for (uint i = 0; i < NPT; i++) {
                uint idx = dim_base + i;
                uint partner = idx ^ stride;
                float sv = src[idx];
                float pv = src[partner];
                dst[idx] = ((idx & stride) == 0) ? (sv + pv) : (pv - sv);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            threadgroup float* tmp = src; src = dst; dst = tmp;
        }

        float wht_norm = 1.0f / sqrt((float)D);
        for (uint i = 0; i < NPT; i++) {
            output[q_head * D + dim_base + i] = src[dim_base + i] * wht_norm;
        }
    } else {
        for (uint i = 0; i < NPT; i++) {
            output[q_head * D + dim_base + i] = O_r[i];
        }
    }
"""

# ---------------------------------------------------------------------------
# Kernel cache (lazy init)
# ---------------------------------------------------------------------------

_fused_kernel_cache: dict[str, any] = {}


def _get_fused_qk_kernel():
    if "fused_qk" not in _fused_kernel_cache:
        _fused_kernel_cache["fused_qk"] = mx.fast.metal_kernel(
            name="fused_qk_dot_3bit",
            input_names=[
                "K_packed",
                "centroids",
                "norms",
                "q",
                "kv_head_map",
                "head_dim",
                "seq_len",
            ],
            output_names=["scores"],
            source=_FUSED_QK_DOT_SOURCE,
        )
    return _fused_kernel_cache["fused_qk"]


def _get_fused_value_kernel():
    if "fused_val" not in _fused_kernel_cache:
        _fused_kernel_cache["fused_val"] = mx.fast.metal_kernel(
            name="fused_value_accum_3bit",
            input_names=[
                "V_packed",
                "centroids",
                "norms",
                "attn_weights",
                "kv_head_map",
                "head_dim",
                "seq_len",
            ],
            output_names=["output"],
            source=_FUSED_VALUE_ACCUM_SOURCE,
        )
    return _fused_kernel_cache["fused_val"]


def _get_fully_fused_kernel(use_hadamard: bool):
    key = f"fully_fused_h{int(use_hadamard)}"
    if key not in _fused_kernel_cache:
        _fused_kernel_cache[key] = mx.fast.metal_kernel(
            name=f"fully_fused_attn_3bit_h{int(use_hadamard)}",
            input_names=[
                "K_packed",
                "V_packed",
                "centroids",
                "k_norms",
                "v_norms",
                "q",
                "kv_head_map",
                "blocks_t",
                "scale_val",
                "seq_len",
                "mask_data",
                "has_mask",
            ],
            output_names=["output"],
            source=_FULLY_FUSED_ATTENTION_SOURCE,
        )
    return _fused_kernel_cache[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pack_indices_3bit(indices: mx.array) -> mx.array:
    """Pack uint8 quantisation indices (0-7) into 3-bit packed format.

    Input: (num_heads, seq_len, head_dim) uint8, values in [0, 7]
    Output: (num_heads, seq_len, packed_bytes) uint8

    Every 8 indices → 3 bytes (24 bits).
    """
    H, S, D = indices.shape
    assert D % 8 == 0, f"head_dim {D} must be divisible by 8"
    packed_words = D // 8

    # Reshape to groups of 8
    idx = indices.reshape(H, S, packed_words, 8).astype(mx.uint32)

    # Pack 8 × 3-bit values into 24 bits
    word = mx.zeros((H, S, packed_words), dtype=mx.uint32)
    for i in range(8):
        word = word | (idx[..., i] << (i * 3))

    # Split 24-bit word into 3 bytes
    byte0 = (word & 0xFF).astype(mx.uint8)
    byte1 = ((word >> 8) & 0xFF).astype(mx.uint8)
    byte2 = ((word >> 16) & 0xFF).astype(mx.uint8)

    # Interleave bytes: [b0, b1, b2, b0, b1, b2, ...]
    packed = mx.stack([byte0, byte1, byte2], axis=-1).reshape(H, S, packed_words * 3)
    return packed


def fused_qk_dot(
    K_packed: mx.array,
    centroids: mx.array,
    norms: mx.array,
    q: mx.array,
    kv_head_map: mx.array,
    num_heads: int,
    seq_len: int,
    head_dim: int,
) -> mx.array:
    """Compute attention scores Q·Kᵀ from 3-bit packed K without reconstruction.

    Args:
        K_packed: (num_heads, seq_len, packed_bytes) uint8 — 3-bit packed indices
        centroids: (num_centroids,) float32 — Lloyd-Max centroids (typically 8 for 3-bit)
        norms: (num_heads, seq_len) float32 — stored ||k|| norms
        q: (num_heads, head_dim) float32 — query vectors (one per head)
        num_heads, seq_len, head_dim: dimensions

    Returns:
        scores: (num_heads, seq_len) float32 — attention scores
    """
    if head_dim > _MAX_KERNEL_HEAD_DIM:
        raise ValueError(
            f"head_dim {head_dim} exceeds kernel maximum ({_MAX_KERNEL_HEAD_DIM})"
        )
    kernel = _get_fused_qk_kernel()
    hd = mx.array([head_dim], dtype=mx.uint32)
    sl = mx.array([seq_len], dtype=mx.uint32)

    (scores,) = kernel(
        inputs=[
            K_packed.reshape(-1),
            centroids.reshape(-1),
            norms.reshape(-1),
            q.reshape(-1),
            kv_head_map.reshape(-1),
            hd,
            sl,
        ],
        output_shapes=[(num_heads * seq_len,)],
        output_dtypes=[mx.float32],
        grid=(32 * seq_len, num_heads, 1),
        threadgroup=(32, 1, 1),
    )
    return scores.reshape(num_heads, seq_len)


def fused_value_accum(
    V_packed: mx.array,
    centroids: mx.array,
    norms: mx.array,
    attn_weights: mx.array,
    kv_head_map: mx.array,
    num_heads: int,
    seq_len: int,
    head_dim: int,
) -> mx.array:
    """Compute attention-weighted value sum from 3-bit packed V in rotated space.

    Accumulates sum_j(a_j * dequant(V[j])) without applying inverse rotation.
    The caller must apply inverse rotation once on the output.

    Args:
        V_packed: (num_heads, seq_len, packed_bytes) uint8 — 3-bit packed indices
        centroids: (num_centroids,) float32 — Lloyd-Max centroids
        norms: (num_heads, seq_len) float32 — stored ||v|| norms
        attn_weights: (num_heads, seq_len) float32 — softmax attention weights
        num_heads, seq_len, head_dim: dimensions

    Returns:
        output_rotated: (num_heads, head_dim) float32 — attention output in rotated space
    """
    if head_dim > _MAX_KERNEL_HEAD_DIM:
        raise ValueError(
            f"head_dim {head_dim} exceeds kernel maximum ({_MAX_KERNEL_HEAD_DIM})"
        )
    kernel = _get_fused_value_kernel()
    hd = mx.array([head_dim], dtype=mx.uint32)
    sl = mx.array([seq_len], dtype=mx.uint32)

    (output,) = kernel(
        inputs=[
            V_packed.reshape(-1),
            centroids.reshape(-1),
            norms.reshape(-1),
            attn_weights.reshape(-1),
            kv_head_map.reshape(-1),
            hd,
            sl,
        ],
        output_shapes=[(num_heads * head_dim,)],
        output_dtypes=[mx.float32],
        grid=(min(head_dim, 128) * num_heads, 1, 1),
        threadgroup=(min(head_dim, 128), 1, 1),
    )
    return output.reshape(num_heads, head_dim)


def fully_fused_attention(
    K_packed: mx.array,
    V_packed: mx.array,
    centroids: mx.array,
    k_norms: mx.array,
    v_norms: mx.array,
    q_rot: mx.array,
    kv_head_map: mx.array,
    blocks_t: mx.array,
    scale: float,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    use_hadamard: bool = True,
    mask: mx.array | None = None,
) -> mx.array:
    """Single-kernel fully-fused attention: QK dot + online softmax + V accum + inverse rotation.

    One `mx.fast.metal_kernel` dispatch replaces the entire 3-kernel pipeline + mx.softmax.
    32 threads per threadgroup (1 SIMD group), each thread owns D/32 dimensions.

    Args:
        K_packed: (H_kv, T, packed_bytes) uint8 — 3-bit packed key indices
        V_packed: (H_kv, T, packed_bytes) uint8 — 3-bit packed value indices
        centroids: (num_centroids,) float32 — Lloyd-Max centroids (8 for 3-bit)
        k_norms: (H_kv, T) float32 — key norms
        v_norms: (H_kv, T) float32 — value norms
        q_rot: (H_q, D) float32 — query vectors already rotated into compressed space
        kv_head_map: (H_q,) uint32 — maps query head → KV head for GQA
        blocks_t: (H_kv, N_BLOCKS, 4, 4) float32 — transposed SO(4) block matrices
        scale: attention scale (1/sqrt(d_k))
        num_heads: H_q (number of query heads)
        seq_len: T (number of KV tokens)
        head_dim: D (head dimension, must be divisible by 32)
        use_hadamard: whether to apply WHT in inverse rotation
        mask: optional (H_q, T) or broadcastable attention mask

    Returns:
        output: (H_q, D) float32 — attention output in original space
    """
    assert head_dim % 32 == 0, f"head_dim {head_dim} must be divisible by 32"
    npt = head_dim // 32
    assert npt == 4, (
        f"Fully fused kernel requires head_dim=128 (NPT=4), got head_dim={head_dim}"
    )

    kernel = _get_fully_fused_kernel(use_hadamard)

    # Prepare mask
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

    (output,) = kernel(
        inputs=[
            K_packed.reshape(-1),
            V_packed.reshape(-1),
            centroids.reshape(-1).astype(mx.float32),
            k_norms.reshape(-1).astype(mx.float32),
            v_norms.reshape(-1).astype(mx.float32),
            q_rot.reshape(-1).astype(mx.float32),
            kv_head_map.reshape(-1),
            blocks_t.reshape(-1).astype(mx.float32),
            scale_arr,
            sl,
            mask_flat,
            has_mask_val,
        ],
        template=[
            ("D", head_dim),
            ("NPT", npt),
            ("PACKED_WORDS", head_dim // 8),
            ("N_BLOCKS", head_dim // 4),
            ("USE_HADAMARD", int(use_hadamard)),
        ],
        output_shapes=[(num_heads * head_dim,)],
        output_dtypes=[mx.float32],
        grid=(32, num_heads, 1),
        threadgroup=(32, 1, 1),
    )
    return output.reshape(num_heads, head_dim)


def fused_inverse_rotate(
    output_rotated: mx.array,
    block_matrices: mx.array,
    use_hadamard: bool = True,
) -> mx.array:
    """Apply inverse rotation (SO(4) blocks + WHT) on aggregated output.

    This is Kernel D — applied ONCE on the attention-weighted sum, not per-token.
    Uses the existing optimized Metal backend from isoquant_metal_kernels.

    Args:
        output_rotated: (num_heads, head_dim) float32 — output in rotated space
        block_matrices: (num_heads, num_blocks, 4, 4) — forward block matrices
            (inverse for SO(4) is the transpose; WHT is self-inverse)
        use_hadamard: whether WHT was applied in forward rotation

    Returns:
        output: (num_heads, head_dim) float32 — output in original space
    """
    from .isoquant_metal_kernels import metal_rotate_inverse

    # Add seq_len=1 dimension for the rotation kernel interface
    x = output_rotated[:, None, :]  # (H, 1, D)
    block_matrices_t = mx.swapaxes(block_matrices, -2, -1)  # transpose for inverse
    result = metal_rotate_inverse(x, block_matrices_t, use_hadamard=use_hadamard)
    return result[:, 0, :]  # (H, D)


def fused_isoquant_attention(
    q: mx.array,
    K_packed: mx.array,
    V_packed: mx.array,
    centroids: mx.array,
    k_norms: mx.array,
    v_norms: mx.array,
    rotation_matrices: mx.array,
    block_matrices: mx.array,
    use_hadamard: bool,
    scale: float,
    mask: mx.array | None = None,
    num_kv_heads: int | None = None,
) -> mx.array:
    """Full fused IsoQuant attention: Q·Kᵀ → softmax → V·a → inverse rotate.

    This is the complete 4-kernel pipeline replacing the current
    reconstruct_keys() + get_values() + SDPA path.

    The query is rotated forward into the same space as the stored keys,
    so attention scores are computed in rotated space (valid because
    isometric rotation preserves inner products).

    Args:
        q: (batch, num_q_heads, 1, head_dim) float32 — query (decode = single token)
        K_packed: (num_kv_heads, seq_len, packed_bytes) uint8
        V_packed: (num_kv_heads, seq_len, packed_bytes) uint8
        centroids: (num_centroids,) float32
        k_norms: (num_kv_heads, seq_len) float32
        v_norms: (num_kv_heads, seq_len) float32
        rotation_matrices: (num_kv_heads, head_dim, head_dim) float32
        block_matrices: (num_kv_heads, num_blocks, 4, 4) float32
        use_hadamard: bool
        scale: float — attention scale (1/sqrt(d_k))
        mask: optional attention mask
        num_kv_heads: KV head count (defaults to num_q_heads if None)

    Returns:
        output: (batch, num_q_heads, 1, head_dim) — attention output
    """
    B, H_q, _, D = q.shape
    H_kv = num_kv_heads if num_kv_heads is not None else H_q
    T = K_packed.shape[1]
    repeats = H_q // H_kv if H_q != H_kv else 1

    # Build KV head mapping: query head i reads from KV head i // repeats
    kv_head_map = mx.arange(H_q, dtype=mx.uint32) // repeats

    # Step 1: Rotate query forward into compressed space
    R_T = mx.swapaxes(rotation_matrices, -2, -1)  # (H_kv, D, D)
    q_flat = q[0, :, 0, :]  # (H_q, D) — single decode token
    if repeats > 1:
        R_T_exp = mx.repeat(R_T, repeats, axis=0)
    else:
        R_T_exp = R_T
    q_rot = mx.squeeze(mx.matmul(q_flat[:, None, :], R_T_exp), axis=1)  # (H_q, D)

    # Step 2: Kernel A — fused Q·Kᵀ from packed storage (no K materialisation)
    scores = fused_qk_dot(
        K_packed,
        centroids,
        k_norms,
        q_rot,
        kv_head_map,
        num_heads=H_q,
        seq_len=T,
        head_dim=D,
    )
    scores = scores * scale  # (H_q, T)

    # Apply mask if provided
    if mask is not None:
        m = mask
        while m.ndim > scores.ndim:
            m = m.squeeze(0)
        scores = scores + m

    # Step 3: Kernel B — softmax
    scores = mx.softmax(scores, axis=-1)  # (H_q, T)

    # Step 4: Kernel C — fused value accumulation in rotated space
    output_rot = fused_value_accum(
        V_packed,
        centroids,
        v_norms,
        scores,
        kv_head_map,
        num_heads=H_q,
        seq_len=T,
        head_dim=D,
    )  # (H_q, D) — still in rotated space

    # Step 5: Kernel D — single inverse rotation on aggregated output
    if block_matrices is not None:
        if repeats > 1:
            # Group query heads back to KV-head space for rotation
            output_groups = output_rot.reshape(H_kv, repeats, D)
            rotated = []
            for g in range(repeats):
                rotated.append(
                    fused_inverse_rotate(
                        output_groups[:, g, :], block_matrices, use_hadamard
                    )
                )
            output = mx.stack(rotated, axis=1).reshape(H_q, D)
        else:
            output = fused_inverse_rotate(output_rot, block_matrices, use_hadamard)
    else:
        # Dense fallback
        if repeats > 1:
            R_exp = mx.repeat(rotation_matrices, repeats, axis=0)
        else:
            R_exp = rotation_matrices
        output = mx.matmul(output_rot[:, None, :], R_exp)[:, 0, :]

    # Reshape to match SDPA output: (B, H_q, 1, D)
    return output[None, :, None, :]
