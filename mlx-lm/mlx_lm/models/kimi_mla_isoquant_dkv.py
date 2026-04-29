# Copyright © 2026 Apple Inc.
"""Kimi MLA + IsoQuant DKV cache.

Compresses kv_latent (512-D) via IsoQuant rotation+quantization.
Stores k_pe (64-D RoPE) raw — never rotated or quantized.
"""

from __future__ import annotations

import os

import mlx.core as mx

from .mlx_isoquant import IsoQuantKVCache
from .mlx_turboquant import get_default_codebook_dir


class KimiMLAIsoQuantCache:
    """IsoQuant-compressed MLA cache for DeepseekV3/Kimi models.

    Uses composition with IsoQuantKVCache for compression internals.
    The kv_latent is the full kv_lora_rank dimension (no internal split).
    k_pe is a separate tensor that must round-trip exactly.
    """

    def __init__(
        self,
        kv_lora_rank: int = 512,
        qk_rope_head_dim: int = 64,
        bit_width: int = 3,
        layer_idx: int | None = None,
        codebook_dir: str | None = None,
        seed: int = 42,
    ):
        self._kv_lora_rank = kv_lora_rank
        self._rope_dim = qk_rope_head_dim

        if codebook_dir is None:
            codebook_dir = get_default_codebook_dir()

        self._iso = IsoQuantKVCache(
            num_heads=1,
            head_dim=kv_lora_rank,
            bit_width=bit_width,
            layer_idx=layer_idx,
            codebook_dir=codebook_dir,
            seed=seed,
        )

        self._deferred = True
        self._fp16_latent: list[mx.array] = []
        self._fp16_pe: list[mx.array] = []
        self._compressed_latent: dict[str, mx.array] | None = None
        self._pe_buffer: mx.array | None = None
        self._packed_latent_cache: mx.array | None = None
        self.offset = 0

    @classmethod
    def from_state(cls, state: dict, meta_state=None, **kwargs):
        if meta_state is None:
            raise ValueError(
                "meta_state is required for from_state reconstruction. "
                "Cannot restore KimiMLAIsoQuantCache without dimensions and codebook metadata."
            )
        obj = cls.__new__(cls)
        obj.meta_state = meta_state
        obj.state = state
        return obj

    def update_and_fetch(
        self, kv_latent: mx.array, k_pe: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Cache kv_latent and k_pe, return full accumulated history.

        Args:
            kv_latent: (B, 1, L, kv_lora_rank) — MLA latent, compressible
            k_pe: (B, 1, L, qk_rope_head_dim) — RoPE keys, stored raw

        Returns:
            (all_kv_latent, all_k_pe) — full history, same shapes with
            accumulated seq dimension.
        """
        seq_len = kv_latent.shape[2]
        lat_3d = kv_latent[0]  # (1, L, kv_lora_rank)
        pe_3d = k_pe[0]  # (1, L, qk_rope_head_dim)

        if self._deferred:
            self._fp16_latent.append(lat_3d)
            self._fp16_pe.append(pe_3d)
            self.offset += seq_len
            all_lat = mx.concatenate(self._fp16_latent, axis=1)
            all_pe = mx.concatenate(self._fp16_pe, axis=1)
            return all_lat[None], all_pe[None]

        # Decode phase: compress new latent, append k_pe raw
        new_compressed = self._iso._compress_batch(lat_3d)

        if self._compressed_latent is None:
            self._compressed_latent = new_compressed
        else:
            for key in new_compressed:
                self._compressed_latent[key] = mx.concatenate(
                    [self._compressed_latent[key], new_compressed[key]], axis=1
                )

        if self._pe_buffer is None:
            self._pe_buffer = pe_3d
        else:
            self._pe_buffer = mx.concatenate([self._pe_buffer, pe_3d], axis=1)

        # Incrementally extend the packed cache (avoids O(T) repack per step)
        if self._packed_latent_cache is not None:
            from .fused_kv_decode_kernels import pack_indices_3bit

            new_packed = pack_indices_3bit(new_compressed["indices"])
            self._packed_latent_cache = mx.concatenate(
                [self._packed_latent_cache, new_packed], axis=1
            )

        self.offset += seq_len

        all_lat = self._iso._decompress_batch(self._compressed_latent)
        return all_lat[None], self._pe_buffer[None]

    @property
    def supports_fused_latent_attention(self) -> bool:
        """True when fused Metal attention can run on compressed latent.

        Does NOT delegate to self._iso.supports_fused_attention because the
        inner IsoQuantKVCache is used as a compression engine, not a
        standalone cache — its _deferred flag and compressed_keys/values
        are never set through the normal MLA composition flow.
        """
        return (
            not self._deferred
            and self._compressed_latent is not None
            and self._iso._fallback_cache is None
            and self._iso.bit_width == 3
        )

    def fused_latent_attention(
        self,
        q_absorbed: mx.array,
        pe_scores: mx.array,
        scale: float,
    ) -> mx.array:
        """Fused attention on compressed kv_latent without materialising FP16.

        Args:
            q_absorbed: (B, H_q, 1, kv_lora_rank) — query after embed_q absorption
            pe_scores: (B, H_q, 1, T) — RoPE position scores (used as additive mask)
            scale: attention scale factor

        Returns:
            output: (B, H_q, 1, kv_lora_rank) — attention output in latent space
        """
        # Build packed cache once (after finalize), then extend incrementally
        if self._packed_latent_cache is None and self._compressed_latent is not None:
            from .fused_kv_decode_kernels import pack_indices_3bit

            self._packed_latent_cache = pack_indices_3bit(
                self._compressed_latent["indices"]
            )

        self._iso.compressed_keys = self._compressed_latent
        self._iso.compressed_values = self._compressed_latent
        self._iso._seq_len = self.offset
        # Share the single packed buffer for both K and V (MLA: K=V)
        self._iso._packed_keys_cache = self._packed_latent_cache
        self._iso._packed_values_cache = self._packed_latent_cache
        return self._iso.fused_attention(q_absorbed, scale, mask=pe_scores)

    def finalize_deferred_prefill(self) -> None:
        """Bulk-compress accumulated FP16 latent at prefill→decode boundary."""
        if not self._deferred:
            return
        self._deferred = False

        if not self._fp16_latent:
            return

        all_latent = mx.concatenate(self._fp16_latent, axis=1)
        self._compressed_latent = self._iso._compress_batch(all_latent)
        self._packed_latent_cache = None

        self._pe_buffer = mx.concatenate(self._fp16_pe, axis=1)

        self._fp16_latent.clear()
        self._fp16_pe.clear()

    @property
    def state(self):
        return {
            "compressed_latent": self._compressed_latent,
            "pe_buffer": self._pe_buffer,
            "offset": mx.array(self.offset, dtype=mx.int32),
        }

    @state.setter
    def state(self, v):
        self._compressed_latent = v.get("compressed_latent")
        self._pe_buffer = v.get("pe_buffer")
        offset = v.get("offset", 0)
        if isinstance(offset, mx.array):
            offset = int(offset.item())
        self.offset = int(offset)
        self._fp16_latent = []
        self._fp16_pe = []
        self._packed_latent_cache = None
        self._deferred = False

    @property
    def meta_state(self):
        return (
            "kimi_mla_iso_v1",
            str(self._iso.bit_width),
            str(self._kv_lora_rank),
            str(self._rope_dim),
            str(self._iso.layer_idx if self._iso.layer_idx is not None else -1),
            str(self._iso.codebook_dir),
            str(self._iso.seed),
        )

    @meta_state.setter
    def meta_state(self, v):
        version = v[0]
        if version != "kimi_mla_iso_v1":
            raise ValueError(
                f"KimiMLAIsoQuantCache expects meta_state kimi_mla_iso_v1, got {version!r}"
            )
        bit_width = int(v[1])
        self._kv_lora_rank = int(v[2])
        self._rope_dim = int(v[3])
        layer_idx_val = int(v[4]) if len(v) > 4 else -1
        layer_idx = layer_idx_val if layer_idx_val != -1 else None
        codebook_dir = (
            os.path.abspath(v[5]) if len(v) > 5 and v[5] else get_default_codebook_dir()
        )
        seed = int(v[6]) if len(v) > 6 else 42
        self._iso = IsoQuantKVCache(
            num_heads=1,
            head_dim=self._kv_lora_rank,
            bit_width=bit_width,
            layer_idx=layer_idx,
            codebook_dir=codebook_dir,
            seed=seed,
        )
