# Copyright © 2026 — IsoQuant KV rotation via isoclinic decomposition of SO(4).
# SAFETY: V-cache dequant uses R.T (orthogonal inverse). Isoclinic matrices are
# orthogonal by construction; inverse_rotate is implicit via matmul with R.
#
# Math: each 4D block is rotated by T(v) = q_L * v * conj(q_R) where q_L, q_R
# are independent unit quaternions. This gives the full SO(4) on each block
# (no fixed axis). Paper: arXiv:2603.28430 "IsoQuant" (March 2026).

from __future__ import annotations

import os

import numpy as np
import mlx.core as mx

from .mlx_turboquant import (
    TurboQuantCompressor,
    TurboQuantKVCache,
    get_default_codebook_dir,
    quantize_scalar,
)

_METAL_BACKEND = None


# ---------------------------------------------------------------------------
# Module-level instrumentation. Aggregated across all per-layer cache
# instances. Reset before each benchmark run via ``reset_stats()`` and read
# after via ``stats_summary()``. Modelled on ``mx.get_peak_memory()`` —
# global state is fine because there is one cache pool per generation run.
# ---------------------------------------------------------------------------


class IsoQuantStats:
    """Counters for IsoQuant cache activity (write, read, fused metal)."""

    __slots__ = (
        "prefill_calls",
        "prefill_tokens",
        "decode_calls",
        "decode_tokens",
        "compress_calls",
        "decompress_calls",
        "read_keys_calls",
        "read_values_calls",
        "finalize_calls",
        "fused_metal_attempts",
        "fused_metal_failures",
        "packed_cache_hits",
        "packed_cache_misses",
        "fallback_invocations",
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        for f in self.__slots__:
            setattr(self, f, 0)

    def to_dict(self) -> dict:
        d = {f: getattr(self, f) for f in self.__slots__}
        # Derived rates — guard divisions
        packed_total = self.packed_cache_hits + self.packed_cache_misses
        d["packed_cache_hit_rate"] = (
            round(self.packed_cache_hits / packed_total, 4) if packed_total else None
        )
        fused_total = self.fused_metal_attempts
        d["fused_metal_success_rate"] = (
            round(1 - self.fused_metal_failures / fused_total, 4)
            if fused_total
            else None
        )
        return d


_GLOBAL_STATS = IsoQuantStats()


def get_stats() -> IsoQuantStats:
    return _GLOBAL_STATS


def reset_stats() -> None:
    _GLOBAL_STATS.reset()


def stats_summary() -> dict:
    return _GLOBAL_STATS.to_dict()


def _get_metal_backend():
    """Lazy-load the optional Metal structured-rotation backend.

    Kept behind an explicit opt-in because correctness should match the dense
    path first; performance closure remains a separate measurement task.
    """
    global _METAL_BACKEND
    if _METAL_BACKEND is None:
        from .isoquant_metal_kernels import metal_rotate_forward, metal_rotate_inverse

        _METAL_BACKEND = (metal_rotate_forward, metal_rotate_inverse)
    return _METAL_BACKEND


def _quat_mul_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, last axis 4, (w,x,y,z)."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return np.stack([ow, ox, oy, oz], axis=-1)


def _quat_conj_np(q: np.ndarray) -> np.ndarray:
    out = np.array(q, copy=True)
    out[..., 1:] *= -1.0
    return out


def _isoclinic_vec_np(q_L: np.ndarray, q_R: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Isoclinic SO(4) rotation: v' = q_L * v * conj(q_R)."""
    return _quat_mul_np(_quat_mul_np(q_L, v), _quat_conj_np(q_R))


def _isoclinic_block_matrix(q_L: np.ndarray, q_R: np.ndarray) -> np.ndarray:
    """4x4 orthogonal matrix M where M @ v = q_L * v * conj(q_R).

    Two independent unit quaternions give the full SO(4) per block —
    no fixed axis, all 4 dimensions mixed. This is the isoclinic
    decomposition from arXiv:2603.28430.
    """
    M = np.zeros((4, 4), dtype=np.float32)
    for i in range(4):
        e = np.zeros(4, dtype=np.float32)
        e[i] = 1.0
        M[:, i] = _isoclinic_vec_np(q_L, q_R, e)
    return M


def _random_unit_quat(rng: np.random.Generator) -> np.ndarray:
    g = rng.normal(size=4).astype(np.float32)
    return g / (np.linalg.norm(g) + 1e-8)


def _build_isoclinic_block_rotation_matrix(
    head_dim: int, rng: np.random.Generator
) -> np.ndarray:
    """Block-diagonal SO(4) rotation without any global pre-mixing."""
    blocks = _build_isoclinic_block_matrices(head_dim, rng)
    return _block_matrices_to_dense(blocks)


def _build_isoclinic_block_matrices(
    head_dim: int, rng: np.random.Generator
) -> np.ndarray:
    """Independent 4x4 SO(4) blocks for a single head."""
    n_blocks = head_dim // 4
    blocks = np.zeros((n_blocks, 4, 4), dtype=np.float32)
    for b in range(n_blocks):
        q_L = _random_unit_quat(rng)
        q_R = _random_unit_quat(rng)
        blocks[b] = _isoclinic_block_matrix(q_L, q_R)
    return blocks


def _block_matrices_to_dense(blocks: np.ndarray) -> np.ndarray:
    """Dense block-diagonal matrix from 4x4 block matrices."""
    n_blocks = blocks.shape[0]
    head_dim = n_blocks * 4
    R_block = np.zeros((head_dim, head_dim), dtype=np.float32)
    for b, blk in enumerate(blocks):
        sl = slice(b * 4, (b + 1) * 4)
        R_block[sl, sl] = blk
    return R_block


def _hadamard_matrix(d: int) -> np.ndarray:
    """Normalized Walsh-Hadamard matrix of size d (must be power of 2).

    O(d log d) global mixing matrix for decorrelating all dimensions before
    block rotation.
    """
    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < d:
        H = np.block([[H, H], [H, -H]]) / np.sqrt(2.0)
    return H.astype(np.float32)


def _combine_block_rotation_with_hadamard(
    R_block: np.ndarray, H: np.ndarray
) -> np.ndarray:
    """Combine block SO(4) rotations with a global Hadamard mix.

    Since ``R_block`` is block diagonal, each 4-row block only depends on the
    matching 4 rows of ``H``. Building the product blockwise avoids the large
    dense matmul path that emitted suspicious runtime warnings on Apple Silicon.
    """
    head_dim = R_block.shape[0]
    combined = np.zeros((head_dim, head_dim), dtype=np.float32)
    for start in range(0, head_dim, 4):
        stop = start + 4
        blk = R_block[start:stop, start:stop]
        H_chunk = H[start:stop, :]
        block_rows = np.zeros((4, head_dim), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                block_rows[i, :] += blk[i, j] * H_chunk[j, :]
        combined[start:stop, :] = block_rows
    return combined


def build_isoquant_rotation_matrix(
    head_dim: int, rng: np.random.Generator, apply_global_mix: bool = True
) -> np.ndarray:
    """Single head rotation matrix for IsoQuant.

    ``apply_global_mix=True`` builds the current WHT+SO(4) path.
    ``False`` preserves the legacy block-only SO(4) path for prompt-cache
    compatibility with older serialized ``iso_v1`` artifacts.
    """
    if head_dim % 4 != 0:
        raise ValueError(f"IsoQuant expects head_dim divisible by 4, got {head_dim}")
    return _build_isoquant_rotation_components_for_head(
        head_dim, rng, apply_global_mix=apply_global_mix
    )[0]


def _build_isoquant_rotation_components_for_head(
    head_dim: int, rng: np.random.Generator, apply_global_mix: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return dense and structured rotation components for a single head.

    The cache uses row vectors. Forward rotation is `x @ R.T`, where
    `R = R_block @ H` when Hadamard mixing is enabled. The structured forward
    path is therefore `x @ H @ R_block.T`.
    """
    if head_dim % 4 != 0:
        raise ValueError(f"IsoQuant expects head_dim divisible by 4, got {head_dim}")
    blocks = _build_isoclinic_block_matrices(head_dim, rng)
    blocks_t = np.swapaxes(blocks, -2, -1)
    R_block = _block_matrices_to_dense(blocks)
    is_pow2 = head_dim > 0 and (head_dim & (head_dim - 1)) == 0
    use_hadamard = bool(apply_global_mix and is_pow2)
    if use_hadamard:
        H = _hadamard_matrix(head_dim)
        dense = _combine_block_rotation_with_hadamard(R_block, H)
    else:
        dense = R_block
    return dense, blocks, blocks_t, use_hadamard


def build_isoquant_rotation_matrices(
    num_heads: int,
    head_dim: int,
    seed: int,
    layer_idx: int = 0,
    apply_global_mix: bool = True,
) -> mx.array:
    rng = np.random.default_rng(seed + (layer_idx or 0) * 1000)
    mats = [
        build_isoquant_rotation_matrix(head_dim, rng, apply_global_mix=apply_global_mix)
        for _ in range(num_heads)
    ]
    return mx.array(np.stack(mats, dtype=np.float32))


def build_isoquant_rotation_components(
    num_heads: int,
    head_dim: int,
    seed: int,
    layer_idx: int = 0,
    apply_global_mix: bool = True,
) -> dict[str, mx.array | bool]:
    """Structured per-head IsoQuant rotation state plus dense fallback matrix."""
    rng = np.random.default_rng(seed + (layer_idx or 0) * 1000)
    dense_mats = []
    block_mats = []
    block_mats_t = []
    use_hadamard = None
    for _ in range(num_heads):
        dense, blocks, blocks_t, head_use_hadamard = (
            _build_isoquant_rotation_components_for_head(
                head_dim, rng, apply_global_mix=apply_global_mix
            )
        )
        dense_mats.append(dense)
        block_mats.append(blocks)
        block_mats_t.append(blocks_t)
        if use_hadamard is None:
            use_hadamard = head_use_hadamard
    return {
        "rotation_matrices": mx.array(np.stack(dense_mats, dtype=np.float32)),
        "block_matrices": mx.array(np.stack(block_mats, dtype=np.float32)),
        "block_matrices_t": mx.array(np.stack(block_mats_t, dtype=np.float32)),
        "use_hadamard": bool(use_hadamard),
    }


def _fwht_last_axis(x: mx.array) -> mx.array:
    """Exact normalized Walsh-Hadamard transform on the last axis."""
    dim = x.shape[-1]
    if dim <= 0 or (dim & (dim - 1)) != 0:
        raise ValueError(f"FWHT expects power-of-2 last dimension, got {dim}")
    y = x.astype(mx.float32)
    scale = mx.array(np.sqrt(2.0), dtype=mx.float32)
    stride = 1
    while stride < dim:
        shape = y.shape[:-1] + (dim // (2 * stride), 2, stride)
        paired = y.reshape(shape)
        even = paired[..., 0, :]
        odd = paired[..., 1, :]
        y = mx.stack([(even + odd) / scale, (even - odd) / scale], axis=-2)
        y = y.reshape(x.shape)
        stride *= 2
    return y


def _apply_so4_blocks_last_axis(x: mx.array, blocks: mx.array) -> mx.array:
    """Apply per-head 4x4 block matvecs to the last axis.

    Args:
        x: shape (heads, ..., head_dim)
        blocks: shape (heads, n_blocks, 4, 4)
    """
    head_dim = x.shape[-1]
    n_blocks = head_dim // 4
    heads = x.shape[0]
    trailing = x.shape[1:-1]
    flat = 1
    for dim in trailing:
        flat *= dim
    x_blocks = x.reshape((heads, flat, n_blocks, 1, 4))
    block_view = blocks[:, None, :, :, :]
    rotated = mx.matmul(x_blocks, block_view).squeeze(-2)
    return rotated.reshape(x.shape)


def structured_rotate_forward(
    x: mx.array, block_matrices_t: mx.array, use_hadamard: bool
) -> mx.array:
    """Row-vector forward rotate matching `x @ R.T`."""
    y = _fwht_last_axis(x) if use_hadamard else x.astype(mx.float32)
    return _apply_so4_blocks_last_axis(y, block_matrices_t)


def structured_rotate_inverse(
    x_rot: mx.array, block_matrices: mx.array, use_hadamard: bool
) -> mx.array:
    """Row-vector inverse rotate matching `x_rot @ R`."""
    y = _apply_so4_blocks_last_axis(x_rot, block_matrices)
    return _fwht_last_axis(y) if use_hadamard else y


class IsoQuantKVCache(TurboQuantKVCache):
    """
    KV cache using IsoQuant-style rotations: a global Walsh-Hadamard pre-mix
    followed by independent isoclinic SO(4) rotations on each 4-D block.

    Reuses TurboQuant Lloyd–Max quantisation and asymmetric attention; only the
    orthogonal R differs (Hadamard + isoclinic blocks vs dense QR).
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        bit_width: int = 4,
        max_seq_len: int = 32768,
        layer_idx: int | None = None,
        codebook_dir: str = "codebooks",
        seed: int = 42,
    ):
        super().__init__(
            num_heads=num_heads,
            head_dim=head_dim,
            bit_width=bit_width,
            max_seq_len=max_seq_len,
            layer_idx=layer_idx,
            codebook_dir=codebook_dir,
            seed=seed,
        )
        # Deferred prefill: accumulate FP16 KV during prefill, bulk-compress
        # at the prefill→decode transition.  Zero error compounding during prefill.
        self._deferred = True
        self._fp16_keys: list[mx.array] = []
        self._fp16_values: list[mx.array] = []
        self._apply_global_mix = True
        self.block_matrices = None
        self.block_matrices_t = None
        self._use_hadamard = False
        self._use_structured_runtime = (
            os.environ.get("ISOQUANT_USE_STRUCTURED_MLX", "0") == "1"
        )
        self._use_metal_runtime = os.environ.get("ISOQUANT_USE_METAL", "0") == "1"
        self._metal_runtime_error: str | None = None
        self._fused_metal_ok: bool | None = (
            None  # None=untested, True=works, False=failed
        )
        self._packed_keys_cache: mx.array | None = None
        self._packed_values_cache: mx.array | None = None
        if self._fallback_cache is None:
            try:
                self._set_rotation_components(self.num_heads, self.head_dim)
            except Exception:
                self._activate_fallback_cache(self.num_heads, self.head_dim)

    @property
    def meta_state(self):
        return tuple(
            map(
                str,
                (
                    "iso_v2",
                    self.bit_width,
                    self.num_heads,
                    self.head_dim,
                    self.layer_idx if self.layer_idx is not None else -1,
                    self.codebook_dir,
                    self.seed,
                ),
            )
        )

    @meta_state.setter
    def meta_state(self, v):
        version = v[0]
        if version not in {"iso_v1", "iso_v2"}:
            raise ValueError(
                f"IsoQuantKVCache expects meta_state iso_v1/iso_v2, got {version!r}"
            )
        self.bit_width = int(v[1])
        self.num_heads = int(v[2])
        self.head_dim = int(v[3])
        layer_idx_val = int(v[4])
        self.layer_idx = layer_idx_val if layer_idx_val != -1 else None
        self.max_seq_len = 32768
        self.codebook_dir = (
            os.path.abspath(v[5]) if len(v) > 5 and v[5] else get_default_codebook_dir()
        )
        from .mlx_turboquant import _DEFAULT_TQ_SEED

        self.seed = int(v[6]) if len(v) >= 7 else _DEFAULT_TQ_SEED
        self._apply_global_mix = version == "iso_v2"
        self._deferred = False
        self._fp16_keys = []
        self._fp16_values = []
        self._fallback_cache = None
        self._warned_fallback = False
        self.block_matrices = None
        self.block_matrices_t = None
        self._use_hadamard = False
        self._use_structured_runtime = (
            os.environ.get("ISOQUANT_USE_STRUCTURED_MLX", "0") == "1"
        )
        self._use_metal_runtime = os.environ.get("ISOQUANT_USE_METAL", "0") == "1"
        self._metal_runtime_error = None
        self._fused_metal_ok = None
        self._packed_keys_cache = None
        self._packed_values_cache = None
        try:
            self.compressor = TurboQuantCompressor(
                bit_width=self.bit_width,
                head_dim=self.head_dim,
                codebook_dir=self.codebook_dir,
                seed=self.seed,
            )
            self._set_rotation_components(self.num_heads, self.head_dim)
        except Exception:
            self._activate_fallback_cache(self.num_heads, self.head_dim)

    def _set_rotation_components(self, num_heads: int, head_dim: int) -> None:
        components = build_isoquant_rotation_components(
            num_heads,
            head_dim,
            self.seed,
            self.layer_idx or 0,
            apply_global_mix=self._apply_global_mix,
        )
        self.rotation_matrices = components["rotation_matrices"]
        self.block_matrices = components["block_matrices"]
        self.block_matrices_t = components["block_matrices_t"]
        self._use_hadamard = bool(components["use_hadamard"])

    def _rotate_forward(self, x: mx.array) -> mx.array:
        q_t = mx.swapaxes(self.rotation_matrices, -2, -1)
        if self._use_metal_runtime and self.block_matrices_t is not None:
            try:
                metal_forward, _ = _get_metal_backend()
                return metal_forward(x, self.block_matrices, self._use_hadamard)
            except Exception as exc:
                self._use_metal_runtime = False
                self._metal_runtime_error = str(exc)
        if not self._use_structured_runtime or self.block_matrices_t is None:
            return mx.matmul(x, q_t)
        return structured_rotate_forward(x, self.block_matrices_t, self._use_hadamard)

    def _rotate_inverse(self, x_rot: mx.array) -> mx.array:
        if self._use_metal_runtime and self.block_matrices is not None:
            try:
                _, metal_inverse = _get_metal_backend()
                return metal_inverse(x_rot, self.block_matrices_t, self._use_hadamard)
            except Exception as exc:
                self._use_metal_runtime = False
                self._metal_runtime_error = str(exc)
        if not self._use_structured_runtime or self.block_matrices is None:
            return mx.matmul(x_rot, self.rotation_matrices)
        return structured_rotate_inverse(x_rot, self.block_matrices, self._use_hadamard)

    def _decompress_batch(self, compressed: dict[str, mx.array]) -> mx.array:
        _GLOBAL_STATS.decompress_calls += 1
        indices = compressed["indices"]
        x_rot_quant = self.compressor.centroids[indices]
        x_hat_unit = self._rotate_inverse(x_rot_quant)
        return x_hat_unit * compressed["x_norm"]

    def _invalidate_fused_caches(self) -> None:
        self._packed_keys_cache = None
        self._packed_values_cache = None

    def finalize_deferred_prefill(self) -> None:
        """Bulk-compress accumulated FP16 KV into IsoQuant at the prefill→decode boundary.

        During prefill, KV is stored in FP16 (zero error compounding).
        This method compresses the entire buffer in one pass, then switches
        to incremental compression for the decode phase.
        """
        if not self._deferred:
            return
        _GLOBAL_STATS.finalize_calls += 1
        self._deferred = False

        if self._fallback_cache is not None:
            # Fallback path — nothing to compress
            self._fp16_keys.clear()
            self._fp16_values.clear()
            return

        if not self._fp16_keys:
            return

        # Concatenate all prefill chunks: each is (heads, seq_chunk, dim)
        all_keys = mx.concatenate(self._fp16_keys, axis=1)
        all_values = mx.concatenate(self._fp16_values, axis=1)

        # Bulk compress
        self.compressed_keys = self._compress_batch(all_keys)
        self.compressed_values = self._compress_batch(all_values)
        self._invalidate_fused_caches()
        mx.eval(
            *[v for v in self.compressed_keys.values() if isinstance(v, mx.array)],
            *[v for v in self.compressed_values.values() if isinstance(v, mx.array)],
        )

        # Free the FP16 buffer
        self._fp16_keys.clear()
        self._fp16_values.clear()

    def _compress_batch(self, x: mx.array) -> dict[str, mx.array]:
        """Batched MLX quantization for (heads, seq, dim), avoiding Python per-head loops."""
        _GLOBAL_STATS.compress_calls += 1
        x_f32 = x.astype(mx.float32)
        x_norm = mx.linalg.norm(x_f32, axis=-1, keepdims=True)
        x_unit = x_f32 / mx.maximum(x_norm, mx.array(1e-8, dtype=mx.float32))
        x_rot = self._rotate_forward(x_unit)
        indices, _ = quantize_scalar(
            x_rot, self.compressor.centroids, self.compressor.boundaries
        )
        return {
            "indices": indices.astype(mx.uint8),
            "x_norm": x_norm.astype(mx.float16),
        }

    def _ensure_runtime_shape(self, n_kv_heads: int, head_dim: int) -> None:
        if self._fallback_cache is not None:
            return
        if n_kv_heads == self.num_heads and head_dim == self.head_dim:
            return

        try:
            self.num_heads = int(n_kv_heads)
            self.head_dim = int(head_dim)
            self.compressor = TurboQuantCompressor(
                bit_width=self.bit_width,
                head_dim=self.head_dim,
                codebook_dir=self.codebook_dir,
                seed=self.seed,
            )
            self._set_rotation_components(self.num_heads, self.head_dim)
            self._invalidate_fused_caches()
            self._fused_metal_ok = None
        except Exception:
            # Fallback for heterogeneous-cache layers without matching codebooks.
            self._activate_fallback_cache(n_kv_heads, head_dim)

    def update_and_fetch(
        self, keys: mx.array, values: mx.array, offset: int | None = None
    ) -> tuple[mx.array, mx.array]:
        """
        Update cache using batched MLX kernels only.

        This keeps Python out of the per-head hot path for IsoQuant.
        """
        batch_size, n_kv_heads, seq_len, _ = keys.shape
        assert batch_size == 1, "IsoQuantKVCache currently assumes batch=1"
        if offset is not None:
            self._seq_len = offset
            self.offset = offset
        self._ensure_runtime_shape(n_kv_heads, int(keys.shape[-1]))
        if self._fallback_cache is not None:
            _GLOBAL_STATS.fallback_invocations += 1
            out = self._fallback_cache.update_and_fetch(keys, values)
            self._sync_fallback_offset()
            return out

        k_batch = keys[0]  # (heads, seq, dim)
        v_batch = values[0]

        if self._deferred:
            # Prefill phase: accumulate FP16, no compression (zero error compounding).
            _GLOBAL_STATS.prefill_calls += 1
            _GLOBAL_STATS.prefill_tokens += seq_len
            self._fp16_keys.append(k_batch)
            self._fp16_values.append(v_batch)
            self._seq_len += seq_len
            self.offset += seq_len
            return keys, values

        # Decode phase (post-finalize): compress incrementally.
        _GLOBAL_STATS.decode_calls += 1
        _GLOBAL_STATS.decode_tokens += seq_len
        new_compressed_keys = self._compress_batch(k_batch)
        new_compressed_vals = self._compress_batch(v_batch)

        def _concat_compressed(existing, new):
            if not existing:
                return new
            for key in new:
                existing[key] = mx.concatenate([existing[key], new[key]], axis=1)
            return existing

        self.compressed_keys = _concat_compressed(
            self.compressed_keys, new_compressed_keys
        )
        self.compressed_values = _concat_compressed(
            self.compressed_values, new_compressed_vals
        )
        self._invalidate_fused_caches()
        mx.eval(
            *[v for v in self.compressed_keys.values() if isinstance(v, mx.array)],
            *[v for v in self.compressed_values.values() if isinstance(v, mx.array)],
        )
        self._seq_len += seq_len
        self.offset += seq_len
        return keys, values

    def reconstruct_keys(self) -> mx.array:
        """Full key sequence for SDPA; supports deferred FP16 prefill before bulk compress."""
        _GLOBAL_STATS.read_keys_calls += 1
        if self._fallback_cache is not None:
            return self._get_fallback_keys()
        if self._deferred and self._fp16_keys:
            full = mx.concatenate(self._fp16_keys, axis=1)
            return full[None, ...].astype(self._dtype)
        x_hat = self._decompress_batch(self.compressed_keys)
        return x_hat[None, ...].astype(self._dtype)

    def get_values(self) -> mx.array:
        """Full value sequence for SDPA (pair with ``reconstruct_keys``)."""
        _GLOBAL_STATS.read_values_calls += 1
        if self._fallback_cache is not None:
            return self._get_fallback_values()
        if self._deferred and self._fp16_values:
            full = mx.concatenate(self._fp16_values, axis=1)
            return full[None, ...].astype(self._dtype)
        x_hat = self._decompress_batch(self.compressed_values)
        return x_hat[None, ...].astype(self._dtype)

    # ------------------------------------------------------------------
    # Fused decode pipeline (Section 6.6)
    # ------------------------------------------------------------------
    #
    # The fused path eliminates the per-value inverse rotation by computing
    # attention entirely in rotated space and applying a single inverse
    # rotation on the aggregated output.
    #
    # Cost reduction: O(T × d²) → O(d²) for rotation.
    # ------------------------------------------------------------------

    @property
    def supports_fused_attention(self) -> bool:
        """True if the fused decode pipeline can be used."""
        return (
            self._fallback_cache is None
            and not self._deferred
            and self.compressed_keys is not None
            and self.compressed_values is not None
            and self.bit_width == 3
        )

    def fused_attention(
        self,
        queries: mx.array,
        scale: float,
        mask: mx.array | None = None,
    ) -> mx.array:
        """Fused IsoQuant attention: compute in rotated space, inverse rotate once.

        Two execution paths:
          Metal-fused: packed 3-bit decode in-register, no K/V materialisation.
          MLX-ops:     centroid gather + dense matmul (correct but no bandwidth win).

        The Metal path is tried first; on failure it falls back to MLX-ops
        and latches ``_fused_metal_ok = False`` so subsequent calls skip the
        attempt.

        Args:
            queries: (B, num_heads, 1, head_dim) — decode query (single token)
            scale: attention scale factor (1/sqrt(d_k))
            mask: optional attention mask

        Returns:
            output: (B, num_heads, 1, head_dim) — attention output
        """
        if not self.supports_fused_attention:
            keys = self.reconstruct_keys()
            values = self.get_values()
            from .base import scaled_dot_product_attention

            return scaled_dot_product_attention(
                queries, keys, values, cache=None, scale=scale, mask=mask
            )

        B, H_q, _, D = queries.shape
        H_kv = self.num_heads
        T = self.compressed_keys["indices"].shape[1]
        repeats = H_q // H_kv if H_q != H_kv else 1

        # --- Rotate query forward (always dense — one vector, O(d²)) ---
        R_T = mx.swapaxes(self.rotation_matrices, -2, -1)
        q_flat = queries[0, :, 0, :]
        R_T_exp = mx.repeat(R_T, repeats, axis=0) if repeats > 1 else R_T
        q_rot = mx.squeeze(mx.matmul(q_flat[:, None, :], R_T_exp), axis=1)  # (H_q, D)

        # --- Try Metal-fused kernels (packed 3-bit, no materialisation) ---
        if self._fused_metal_ok is not False:
            _GLOBAL_STATS.fused_metal_attempts += 1
            try:
                out = self._fused_attention_metal(
                    q_rot, scale, mask, H_q, H_kv, T, D, repeats
                )
                self._fused_metal_ok = True
                return out[None, :, None, :].astype(queries.dtype)
            except Exception:
                _GLOBAL_STATS.fused_metal_failures += 1
                self._fused_metal_ok = False

        # --- MLX-ops fallback (centroid gather + dense matmul) ---
        out = self._fused_attention_mlx(q_rot, scale, mask, H_q, H_kv, T, D, repeats)
        return out[None, :, None, :].astype(queries.dtype)

    # ----- Metal-fused path (Kernels A/C/D) -----

    # Data-driven threshold (sweep_threshold.py, 100 iters, M4 Max, H=8 D=128):
    # The 3-kernel path wins at all practical T because it parallelizes QK dot
    # across T tokens with grid=(32*T, H).  The single kernel processes T serially
    # in one 32-thread group, losing ~30-50% even at T=16.
    # Disabled (threshold=0) until the single kernel gains T-parallel tiling.
    _SINGLE_KERNEL_T_THRESHOLD = 0

    def _fused_attention_metal(
        self,
        q_rot: mx.array,
        scale: float,
        mask: mx.array | None,
        H_q: int,
        H_kv: int,
        T: int,
        D: int,
        repeats: int,
    ) -> mx.array:
        """Execute attention via Metal kernels on packed 3-bit storage.

        Adaptively selects between:
          - Single fully-fused kernel (T <= threshold): one dispatch, online softmax,
            eliminates 3 kernel launch overheads (~150μs saved).
          - Legacy 3-kernel pipeline (T > threshold): parallelizes QK dot across T
            tokens with grid=(32*T, H), better GPU occupancy at long context.
        """
        from .fused_kv_decode_kernels import pack_indices_3bit

        if self._packed_keys_cache is None:
            _GLOBAL_STATS.packed_cache_misses += 1
            self._packed_keys_cache = pack_indices_3bit(self.compressed_keys["indices"])
        else:
            _GLOBAL_STATS.packed_cache_hits += 1
        if self._packed_values_cache is None:
            _GLOBAL_STATS.packed_cache_misses += 1
            self._packed_values_cache = pack_indices_3bit(
                self.compressed_values["indices"]
            )
        else:
            _GLOBAL_STATS.packed_cache_hits += 1

        k_packed = self._packed_keys_cache
        v_packed = self._packed_values_cache

        # Norms: squeeze trailing dim, cast to f32
        k_norms = self.compressed_keys["x_norm"][:, :, 0].astype(mx.float32)
        v_norms = self.compressed_values["x_norm"][:, :, 0].astype(mx.float32)
        centroids = self.compressor.centroids.reshape(-1).astype(mx.float32)
        kv_head_map = mx.arange(H_q, dtype=mx.uint32) // repeats

        if T <= self._SINGLE_KERNEL_T_THRESHOLD:
            return self._fused_attention_single_kernel(
                k_packed,
                v_packed,
                centroids,
                k_norms,
                v_norms,
                q_rot,
                kv_head_map,
                scale,
                mask,
                H_q,
                T,
                D,
            )
        else:
            return self._fused_attention_3kernel(
                k_packed,
                v_packed,
                centroids,
                k_norms,
                v_norms,
                q_rot,
                kv_head_map,
                scale,
                mask,
                H_q,
                H_kv,
                T,
                D,
                repeats,
            )

    def _fused_attention_single_kernel(
        self,
        k_packed,
        v_packed,
        centroids,
        k_norms,
        v_norms,
        q_rot,
        kv_head_map,
        scale,
        mask,
        H_q,
        T,
        D,
    ) -> mx.array:
        """Single fully-fused kernel: QK + online softmax + V + inverse rotation."""
        from .fused_kv_decode_kernels import fully_fused_attention

        return fully_fused_attention(
            K_packed=k_packed,
            V_packed=v_packed,
            centroids=centroids,
            k_norms=k_norms,
            v_norms=v_norms,
            q_rot=q_rot,
            kv_head_map=kv_head_map,
            blocks_t=self.block_matrices_t,
            scale=scale,
            num_heads=H_q,
            seq_len=T,
            head_dim=D,
            use_hadamard=self._use_hadamard,
            mask=mask,
        )

    def _fused_attention_3kernel(
        self,
        k_packed,
        v_packed,
        centroids,
        k_norms,
        v_norms,
        q_rot,
        kv_head_map,
        scale,
        mask,
        H_q,
        H_kv,
        T,
        D,
        repeats,
    ) -> mx.array:
        """Legacy 3-kernel pipeline: QK dot + softmax + V accum + inverse rotation."""
        from .fused_kv_decode_kernels import fused_qk_dot, fused_value_accum

        scores = fused_qk_dot(
            k_packed, centroids, k_norms, q_rot, kv_head_map, H_q, T, D
        )
        scores = scores * scale

        if mask is not None:
            m = mask
            while m.ndim > scores.ndim:
                m = m.squeeze(0)
            scores = scores + m

        attn_weights = mx.softmax(scores, axis=-1)

        output_rot = fused_value_accum(
            v_packed, centroids, v_norms, attn_weights, kv_head_map, H_q, T, D
        )

        return self._apply_inverse_rotation(
            output_rot, H_q, H_kv, D, repeats, use_metal_kernel=True
        )

    # ----- MLX-ops fallback (centroid gather + dense matmul) -----

    def _fused_attention_mlx(
        self,
        q_rot: mx.array,
        scale: float,
        mask: mx.array | None,
        H_q: int,
        H_kv: int,
        T: int,
        D: int,
        repeats: int,
    ) -> mx.array:
        """Rotated-space attention using standard MLX ops.

        Algorithmically identical to the Metal path but uses centroid gather
        + dense matmul instead of fused kernels.  Correct, but no bandwidth
        reduction — K and V are fully materialised in FP32 before the dot
        product.
        """
        k_rot_quant = self.compressor.centroids[
            self.compressed_keys["indices"]
        ]  # (H_kv, T, D)
        k_rot_scaled = k_rot_quant * self.compressed_keys["x_norm"].astype(mx.float32)

        v_rot_quant = self.compressor.centroids[
            self.compressed_values["indices"]
        ]  # (H_kv, T, D)
        v_rot_scaled = v_rot_quant * self.compressed_values["x_norm"].astype(mx.float32)

        if repeats > 1:
            k_rot_scaled = mx.repeat(k_rot_scaled, repeats, axis=0)
            v_rot_scaled = mx.repeat(v_rot_scaled, repeats, axis=0)

        # Q·Kᵀ in rotated space
        scores = mx.matmul(
            q_rot[:, None, :], mx.swapaxes(k_rot_scaled, -2, -1)
        )  # (H_q, 1, T)
        scores = scores * scale

        if mask is not None:
            m = mask
            while m.ndim > scores.ndim:
                m = m.squeeze(0)
            scores = scores + m

        attn_weights = mx.softmax(scores, axis=-1)  # (H_q, 1, T)

        # Weighted value sum in rotated space
        output_rot = mx.matmul(attn_weights, v_rot_scaled)[:, 0, :]  # (H_q, D)

        return self._apply_inverse_rotation(
            output_rot, H_q, H_kv, D, repeats, use_metal_kernel=False
        )

    # ----- Shared inverse rotation -----

    def _apply_inverse_rotation(
        self,
        output_rot: mx.array,
        H_q: int,
        H_kv: int,
        D: int,
        repeats: int,
        use_metal_kernel: bool = False,
    ) -> mx.array:
        """Apply single inverse rotation on aggregated attention output.

        For GQA, groups query heads back to KV-head space, rotates per group,
        then re-interleaves.
        """
        if repeats > 1:
            output_groups = output_rot.reshape(H_kv, repeats, D)
            rotated_groups = []
            for g in range(repeats):
                group_out = output_groups[:, g, :]  # (H_kv, D)
                rotated_groups.append(
                    self._inverse_rotate_one(group_out, use_metal_kernel)
                )
            return mx.stack(rotated_groups, axis=1).reshape(H_q, D)
        else:
            return self._inverse_rotate_one(output_rot, use_metal_kernel)

    def _inverse_rotate_one(self, x: mx.array, use_metal_kernel: bool) -> mx.array:
        """Inverse-rotate (H, D) tensor.  Metal kernel or dense fallback."""
        if use_metal_kernel and self.block_matrices is not None:
            from .fused_kv_decode_kernels import fused_inverse_rotate

            return fused_inverse_rotate(x, self.block_matrices, self._use_hadamard)
        return self._rotate_inverse(x[:, None, :])[:, 0, :]
