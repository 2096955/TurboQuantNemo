# Copyright © 2026 Apple Inc.
"""Tests for KimiMLAIsoQuantCache — synthetic tensors, no checkpoint needed."""

import tempfile
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache
from mlx_lm.models.mlx_turboquant import get_default_codebook_dir


def _make_cache(**kwargs):
    from mlx_lm.models.kimi_mla_isoquant_dkv import KimiMLAIsoQuantCache

    defaults = dict(
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        bit_width=3,
        layer_idx=0,
        codebook_dir=get_default_codebook_dir(),
        seed=42,
    )
    defaults.update(kwargs)
    return KimiMLAIsoQuantCache(**defaults)


def _random_latent(seq_len, kv_lora_rank=512):
    return mx.random.normal((1, 1, seq_len, kv_lora_rank))


def _random_pe(seq_len, qk_rope_head_dim=64):
    return mx.random.normal((1, 1, seq_len, qk_rope_head_dim))


class TestInit:
    def test_init(self):
        cache = _make_cache()
        assert cache._kv_lora_rank == 512
        assert cache._rope_dim == 64
        assert cache.offset == 0
        assert cache._deferred is True


class TestPrefillPassthrough:
    def test_prefill_passthrough(self):
        """During deferred prefill, returned kv_latent matches input exactly."""
        cache = _make_cache()
        lat = _random_latent(16)
        pe = _random_pe(16)
        mx.eval(lat, pe)

        out_lat, out_pe = cache.update_and_fetch(lat, pe)
        mx.eval(out_lat, out_pe)

        np.testing.assert_allclose(
            np.array(out_lat), np.array(lat), atol=1e-6, rtol=1e-5
        )
        assert cache.offset == 16


class TestPeExactRoundtrip:
    def test_pe_exact_roundtrip(self):
        """k_pe is NEVER compressed — exact bit-for-bit roundtrip."""
        cache = _make_cache()
        pe_prefill = _random_pe(32)
        lat_prefill = _random_latent(32)
        mx.eval(pe_prefill, lat_prefill)

        cache.update_and_fetch(lat_prefill, pe_prefill)
        cache.finalize_deferred_prefill()

        pe_decode = _random_pe(1)
        lat_decode = _random_latent(1)
        mx.eval(pe_decode, lat_decode)

        _, out_pe = cache.update_and_fetch(lat_decode, pe_decode)
        mx.eval(out_pe)

        expected_pe = mx.concatenate([pe_prefill, pe_decode], axis=2)
        np.testing.assert_array_equal(np.array(out_pe), np.array(expected_pe))


class TestLatentCompressedWithinTolerance:
    def test_latent_compressed_within_tolerance(self):
        """After finalize, decompressed latent is close to original (cosine sim > 0.95)."""
        cache = _make_cache()
        lat = _random_latent(64)
        pe = _random_pe(64)
        mx.eval(lat, pe)

        cache.update_and_fetch(lat, pe)
        cache.finalize_deferred_prefill()

        lat_decode = _random_latent(1)
        pe_decode = _random_pe(1)
        mx.eval(lat_decode, pe_decode)

        out_lat, _ = cache.update_and_fetch(lat_decode, pe_decode)
        mx.eval(out_lat)

        original = np.array(lat[0, 0])  # (64, 512)
        decompressed = np.array(out_lat[0, 0, :64, :])  # (64, 512)

        for t in range(original.shape[0]):
            cos_sim = np.dot(original[t], decompressed[t]) / (
                np.linalg.norm(original[t]) * np.linalg.norm(decompressed[t]) + 1e-8
            )
            assert cos_sim > 0.95, f"Token {t}: cosine sim {cos_sim:.4f} < 0.95"


class TestDeferredPrefillFlow:
    def test_deferred_prefill_flow(self):
        """prefill → finalize → decode sequence works end-to-end."""
        cache = _make_cache()

        lat1 = _random_latent(8)
        pe1 = _random_pe(8)
        lat2 = _random_latent(4)
        pe2 = _random_pe(4)
        mx.eval(lat1, pe1, lat2, pe2)

        out_lat1, out_pe1 = cache.update_and_fetch(lat1, pe1)
        out_lat2, out_pe2 = cache.update_and_fetch(lat2, pe2)
        mx.eval(out_lat1, out_pe1, out_lat2, out_pe2)

        assert cache.offset == 12
        assert cache._deferred is True

        expected_lat = mx.concatenate([lat1, lat2], axis=2)
        np.testing.assert_allclose(
            np.array(out_lat2), np.array(expected_lat), atol=1e-6
        )

        cache.finalize_deferred_prefill()
        assert cache._deferred is False
        assert cache._compressed_latent is not None
        assert cache._pe_buffer is not None
        assert len(cache._fp16_latent) == 0

        lat_dec = _random_latent(1)
        pe_dec = _random_pe(1)
        mx.eval(lat_dec, pe_dec)

        out_lat, out_pe = cache.update_and_fetch(lat_dec, pe_dec)
        mx.eval(out_lat, out_pe)

        assert cache.offset == 13
        assert out_lat.shape == (1, 1, 13, 512)
        assert out_pe.shape == (1, 1, 13, 64)


class TestOffsetTracking:
    def test_offset_tracking(self):
        """offset increments correctly through prefill and decode."""
        cache = _make_cache()
        assert cache.offset == 0

        cache.update_and_fetch(_random_latent(10), _random_pe(10))
        assert cache.offset == 10

        cache.update_and_fetch(_random_latent(5), _random_pe(5))
        assert cache.offset == 15

        cache.finalize_deferred_prefill()
        assert cache.offset == 15

        cache.update_and_fetch(_random_latent(1), _random_pe(1))
        assert cache.offset == 16

        cache.update_and_fetch(_random_latent(1), _random_pe(1))
        assert cache.offset == 17


class TestMultiDecodeConcat:
    def test_multi_decode_concat(self):
        """Multiple decode steps accumulate correctly."""
        cache = _make_cache()

        lat_pf = _random_latent(4)
        pe_pf = _random_pe(4)
        mx.eval(lat_pf, pe_pf)

        cache.update_and_fetch(lat_pf, pe_pf)
        cache.finalize_deferred_prefill()

        pe_tokens = []
        for i in range(5):
            pe_tok = _random_pe(1)
            lat_tok = _random_latent(1)
            mx.eval(pe_tok, lat_tok)
            pe_tokens.append(pe_tok)
            out_lat, out_pe = cache.update_and_fetch(lat_tok, pe_tok)
            mx.eval(out_lat, out_pe)

        assert cache.offset == 9
        assert out_lat.shape == (1, 1, 9, 512)
        assert out_pe.shape == (1, 1, 9, 64)

        expected_pe = mx.concatenate([pe_pf] + pe_tokens, axis=2)
        np.testing.assert_array_equal(np.array(out_pe), np.array(expected_pe))


class TestMakePromptCacheDispatch:
    def test_kimi_model_uses_mla_isoquant_cache(self, monkeypatch):
        """make_prompt_cache dispatches Kimi/MLA isoquant caches, not generic KV caches."""
        from mlx_lm.models.kimi_mla_isoquant_dkv import KimiMLAIsoQuantCache

        monkeypatch.setenv("TURBOQUANT_SKIP_LAYERS", "0")
        model = SimpleNamespace(
            args=SimpleNamespace(
                model_type="kimi_k25",
                text_config=SimpleNamespace(
                    model_type="kimi_k2",
                    kv_lora_rank=512,
                    qk_rope_head_dim=64,
                    num_attention_heads=64,
                    num_key_value_heads=64,
                    hidden_size=7168,
                ),
            ),
            layers=[object(), object()],
        )

        caches = make_prompt_cache(model, kv_cache_type="isoquant")

        assert len(caches) == 2
        assert all(isinstance(c, KimiMLAIsoQuantCache) for c in caches)

    def test_deepseek_mla_config_uses_mla_isoquant_cache(self, monkeypatch):
        """The dispatch is MLA-based, so text-only DeepSeek/Kimi configs also route correctly."""
        from mlx_lm.models.kimi_mla_isoquant_dkv import KimiMLAIsoQuantCache

        monkeypatch.setenv("TURBOQUANT_SKIP_LAYERS", "0")
        model = SimpleNamespace(
            args=SimpleNamespace(
                model_type="deepseek_v3",
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                num_attention_heads=64,
                num_key_value_heads=64,
                hidden_size=7168,
            ),
            layers=[object()],
        )

        caches = make_prompt_cache(model, kv_cache_type="isoquant")

        assert len(caches) == 1
        assert isinstance(caches[0], KimiMLAIsoQuantCache)


class TestStateRoundTrip:
    def test_save_load_roundtrip_preserves_state(self):
        """Prompt-cache serialization can reconstruct KimiMLAIsoQuantCache."""
        cache = _make_cache()
        lat = _random_latent(8)
        pe = _random_pe(8)
        mx.eval(lat, pe)

        cache.update_and_fetch(lat, pe)
        cache.finalize_deferred_prefill()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/kimi_cache.safetensors"
            save_prompt_cache(path, [cache])
            loaded = load_prompt_cache(path)[0]

        assert type(loaded).__name__ == "KimiMLAIsoQuantCache"
        assert loaded.offset == cache.offset

        lat_dec = _random_latent(1)
        pe_dec = _random_pe(1)
        _, out_pe = loaded.update_and_fetch(lat_dec, pe_dec)
        mx.eval(out_pe)

        expected_pe = mx.concatenate([pe, pe_dec], axis=2)
        np.testing.assert_array_equal(np.array(out_pe), np.array(expected_pe))
