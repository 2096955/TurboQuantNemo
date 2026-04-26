# Fused NPT=8 T-tiled attention kernel — design

**Status:** draft target design. Current Phase 3 v1 implementation is the
single-pass kernel in `mlx_lm.models.fused_kv_decode_npt8`; this spec remains
the more ambitious T-tiled v2 direction.

**Context:** `fully_fused_attention` in `fused_kv_decode_kernels.py` requires `head_dim=128` (NPT=4) because the Metal source assumes each thread’s `NPT` dimensions fit one 3-byte packed word in the inner loop. For `head_dim=256` (NPT=8), Phase 3 adds a new pipeline: T-tile parallel partials, FlashAttention-style merge, then inverse rotation.

## Function signature (target)

```text
fused_attention_npt8_tiled(
    K_packed: (H_kv, T, packed_bytes) uint8,
    V_packed: (H_kv, T, packed_bytes) uint8,
    centroids: (8,) float32,
    k_norms: (H_kv, T) float32,
    v_norms: (H_kv, T) float32,
    q_rot: (H_q, D=256) float32,
    kv_head_map: (H_q,) uint32,
    blocks_t: (H_kv, N_BLOCKS, 4, 4) float32,  # D=256 → N_BLOCKS=64
    scale: float,
    use_hadamard: bool,
    mask: (H_q, T) float32 | None,
    tile_size: int,
) -> (H_q, D=256) float32
```

If built, the wrapper would also pass `num_heads`, `seq_len`, `head_dim`,
optional `storage_stride` for padded KV buffers.

## Threadgroup map (target)

- **Grid:** `(num_tiles * 32, H_q, 1)` or equivalent — one tile per head slice of T; exact layout TBD in implementation.
- **Threadgroup:** `(32, 1, 1)` — 32 threads, NPT=8 means each thread owns `D/32 = 8` consecutive dimensions.
- **Per TG:** one `(tile, q_head)` tile of scores/values; emits partial `m`, `l`, `O` for merge.

## Two-pass strategy

1. **Per-tile kernel:** for each T-tile, compute unnormalized attention contributions and value partials; output per-tile statistics for softmax merge.
2. **Merge kernel:** reduce over tiles with the FlashAttention-2 style stable merge:

   - `m = max(m_a, m_b)`
   - `l = exp(m_a - m) * l_a + exp(m_b - m) * l_b`
   - `O_d = exp(m_a - m) * O_a_d + exp(m_b - m) * O_b_d`

   Then normalize `O` by `l`, apply SO(4) inverse rotation + optional Hadamard (same contract as the existing 3-kernel path).

## Numerical contract

- FP32 in merge; match the **3-kernel** reference (`fused_qk_dot` → `mx.softmax` → `fused_value_accum`) to `rtol=1e-3`, `atol=1e-4` (same as existing fused tests where applicable).
- Cross-check optional: match `fused_value_accum_tiled` for the V leg once the new kernel is correct.

## Risks

- NPT=8 increases per-thread register pressure vs NPT=4; watch occupancy in Metal GPU capture.
- Packed-index addressing for D=256 must stay consistent with `pack_indices_3bit` layout (3 bytes per 8 indices).
- Padded buffers: use `storage_stride` in kernels (same as Phase 2) so T indexing does not assume dense length.

## Gating (from Phase 1)

Phase 1 found the V-accum path was **not** memory-bandwidth-saturated. Phase 3 is justified mainly by **dispatch / sync reduction**, not by bandwidth — keep expectations measured, not forecast-only.
