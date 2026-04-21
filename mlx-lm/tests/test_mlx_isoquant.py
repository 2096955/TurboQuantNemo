import os
import unittest
from unittest.mock import patch

import numpy as np
import mlx.core as mx

from mlx_lm.models.cache import ArraysCache, CacheList, KVCache, make_prompt_cache
from mlx_lm.models.mlx_isoquant import (
    IsoQuantKVCache,
    build_isoquant_rotation_components,
    build_isoquant_rotation_matrix,
    build_isoquant_rotation_matrices,
    structured_rotate_forward,
    structured_rotate_inverse,
)
from mlx_lm.models.mlx_turboquant import _DEFAULT_TQ_SEED
from mlx_lm.models.mlx_turboquant import TurboQuantKVCache
from mlx_lm.models.mlx_turboquant import get_default_codebook_dir

try:
    from mlx_lm.models.isoquant_metal_kernels import (
        metal_rotate_forward,
        metal_rotate_inverse,
    )
except Exception:  # pragma: no cover - exercised only in MLX/Metal envs
    metal_rotate_forward = None
    metal_rotate_inverse = None


class IsoQuantRotationTest(unittest.TestCase):
    def test_block_matrices_orthogonal(self):
        rng = np.random.default_rng(0)
        M = build_isoquant_rotation_matrix(128, rng).astype(np.float64)
        I = M @ M.T
        err = np.linalg.norm(I - np.eye(128, dtype=np.float64))
        self.assertLess(err, 1e-4)

    def test_stacked_heads_same_count(self):
        R = build_isoquant_rotation_matrices(4, 128, seed=7, layer_idx=0)
        self.assertEqual(tuple(R.shape), (4, 128, 128))

    def test_structured_forward_matches_dense_rotation(self):
        rng = np.random.default_rng(0)
        components = build_isoquant_rotation_components(2, 128, seed=7, layer_idx=0)
        x = mx.array(rng.normal(size=(2, 3, 128)).astype(np.float32))
        dense = mx.matmul(x, mx.swapaxes(components["rotation_matrices"], -2, -1))
        structured = structured_rotate_forward(
            x, components["block_matrices_t"], components["use_hadamard"]
        )
        np.testing.assert_allclose(
            np.array(structured), np.array(dense), atol=1e-5, rtol=1e-5
        )

    def test_structured_inverse_matches_dense_rotation(self):
        rng = np.random.default_rng(1)
        components = build_isoquant_rotation_components(2, 128, seed=11, layer_idx=1)
        x_rot = mx.array(rng.normal(size=(2, 5, 128)).astype(np.float32))
        dense = mx.matmul(x_rot, components["rotation_matrices"])
        structured = structured_rotate_inverse(
            x_rot, components["block_matrices"], components["use_hadamard"]
        )
        np.testing.assert_allclose(
            np.array(structured), np.array(dense), atol=1e-5, rtol=1e-5
        )

    def test_structured_round_trip_matches_dense_reference(self):
        rng = np.random.default_rng(2)
        # head_dim=128: same orthogonality coverage as 256 at lower cost (smaller matmuls).
        components = build_isoquant_rotation_components(1, 128, seed=13, layer_idx=0)
        x = mx.array(rng.normal(size=(1, 4, 128)).astype(np.float32))
        dense_rot = mx.matmul(x, mx.swapaxes(components["rotation_matrices"], -2, -1))
        dense_back = mx.matmul(dense_rot, components["rotation_matrices"])
        structured_rot = structured_rotate_forward(
            x, components["block_matrices_t"], components["use_hadamard"]
        )
        structured_back = structured_rotate_inverse(
            structured_rot, components["block_matrices"], components["use_hadamard"]
        )
        np.testing.assert_allclose(
            np.array(structured_rot), np.array(dense_rot), atol=1e-5, rtol=1e-5
        )
        np.testing.assert_allclose(
            np.array(structured_back), np.array(dense_back), atol=1e-5, rtol=1e-5
        )

    def test_structured_block_only_path_matches_dense_reference(self):
        rng = np.random.default_rng(3)
        components = build_isoquant_rotation_components(
            1, 12, seed=17, layer_idx=0, apply_global_mix=True
        )
        self.assertFalse(components["use_hadamard"])
        x = mx.array(rng.normal(size=(1, 2, 12)).astype(np.float32))
        dense = mx.matmul(x, mx.swapaxes(components["rotation_matrices"], -2, -1))
        structured = structured_rotate_forward(
            x, components["block_matrices_t"], components["use_hadamard"]
        )
        np.testing.assert_allclose(
            np.array(structured), np.array(dense), atol=1e-5, rtol=1e-5
        )

    @unittest.skipUnless(
        os.environ.get("ISOQUANT_RUN_METAL_TESTS") == "1"
        and metal_rotate_forward is not None
        and metal_rotate_inverse is not None,
        "set ISOQUANT_RUN_METAL_TESTS=1 in a Metal-capable MLX environment",
    )
    def test_metal_forward_matches_dense_rotation(self):
        rng = np.random.default_rng(4)
        components = build_isoquant_rotation_components(2, 128, seed=19, layer_idx=0)
        x = mx.array(rng.normal(size=(2, 3, 128)).astype(np.float32))
        dense = mx.matmul(x, mx.swapaxes(components["rotation_matrices"], -2, -1))
        metal = metal_rotate_forward(
            x, components["block_matrices"], components["use_hadamard"]
        )
        mx.eval(dense, metal)
        np.testing.assert_allclose(
            np.array(metal), np.array(dense), atol=1e-5, rtol=1e-5
        )

    @unittest.skipUnless(
        os.environ.get("ISOQUANT_RUN_METAL_TESTS") == "1"
        and metal_rotate_forward is not None
        and metal_rotate_inverse is not None,
        "set ISOQUANT_RUN_METAL_TESTS=1 in a Metal-capable MLX environment",
    )
    def test_metal_round_trip_matches_dense_reference(self):
        rng = np.random.default_rng(5)
        components = build_isoquant_rotation_components(1, 256, seed=23, layer_idx=0)
        x = mx.array(rng.normal(size=(1, 4, 256)).astype(np.float32))
        dense_rot = mx.matmul(x, mx.swapaxes(components["rotation_matrices"], -2, -1))
        dense_back = mx.matmul(dense_rot, components["rotation_matrices"])
        metal_rot = metal_rotate_forward(
            x, components["block_matrices"], components["use_hadamard"]
        )
        metal_back = metal_rotate_inverse(
            metal_rot, components["block_matrices_t"], components["use_hadamard"]
        )
        mx.eval(dense_rot, dense_back, metal_rot, metal_back)
        np.testing.assert_allclose(
            np.array(metal_rot), np.array(dense_rot), atol=1e-5, rtol=1e-5
        )
        np.testing.assert_allclose(
            np.array(metal_back), np.array(dense_back), atol=1e-5, rtol=1e-5
        )

    def test_wht_global_mix_breaks_block_diagonal_structure(self):
        rng = np.random.default_rng(0)
        M = build_isoquant_rotation_matrix(128, rng)
        off_block_max = 0.0
        for i in range(0, 128, 4):
            for j in range(0, 128, 4):
                if i == j:
                    continue
                off_block_max = max(
                    off_block_max, float(np.max(np.abs(M[i : i + 4, j : j + 4])))
                )
        self.assertGreater(off_block_max, 1e-5)

    def test_non_power_of_two_head_dim_uses_block_only_rotation(self):
        rng = np.random.default_rng(0)
        M = build_isoquant_rotation_matrix(12, rng)
        off_block_max = 0.0
        for i in range(0, 12, 4):
            for j in range(0, 12, 4):
                if i == j:
                    continue
                off_block_max = max(
                    off_block_max, float(np.max(np.abs(M[i : i + 4, j : j + 4])))
                )
        self.assertLess(off_block_max, 1e-6)

    def test_make_cache_wraps_native_kv_for_isoquant(self):
        class _Args:
            head_dim = 128
            num_attention_heads = 8
            num_key_value_heads = 8

        class _Model:
            args = _Args()

            def make_cache(self):
                return [ArraysCache(size=2), KVCache()]

        with patch.dict("os.environ", {"TURBOQUANT_SKIP_LAYERS": "0"}):
            caches = make_prompt_cache(_Model(), kv_cache_type="isoquant")
        self.assertIsInstance(caches[0], ArraysCache)
        self.assertIsInstance(caches[1], IsoQuantKVCache)

    def test_make_cache_wraps_native_kv_for_turboquant(self):
        class _Args:
            head_dim = 128
            num_attention_heads = 8
            num_key_value_heads = 8

        class _Model:
            args = _Args()

            def make_cache(self):
                return [ArraysCache(size=2), KVCache()]

        with patch.dict("os.environ", {"TURBOQUANT_SKIP_LAYERS": "0"}):
            caches = make_prompt_cache(_Model(), kv_cache_type="turboquant")
        self.assertIsInstance(caches[0], ArraysCache)
        self.assertIsInstance(caches[1], TurboQuantKVCache)

    def test_make_cache_preserves_skip_layers_per_top_level_layer(self):
        class _Args:
            head_dim = 128
            num_attention_heads = 8
            num_key_value_heads = 8

        class _Layer:
            pass

        class _Model:
            args = _Args()
            layers = [_Layer(), _Layer(), _Layer(), _Layer()]

            def make_cache(self):
                return [CacheList(KVCache(), KVCache()) for _ in self.layers]

        with patch.dict("os.environ", {"TURBOQUANT_SKIP_LAYERS": "2"}):
            caches = make_prompt_cache(_Model(), kv_cache_type="isoquant")

        for idx, cache in enumerate(caches):
            self.assertIsInstance(cache, CacheList)
            first, second = cache.caches
            if idx < 2:
                self.assertIsInstance(first, KVCache)
                self.assertIsInstance(second, KVCache)
            else:
                self.assertIsInstance(first, IsoQuantKVCache)
                self.assertIsInstance(second, IsoQuantKVCache)

    def test_isoquant_constructor_falls_back_without_codebook(self):
        cache = IsoQuantKVCache(
            num_heads=4,
            head_dim=16,
            bit_width=3,
            codebook_dir=get_default_codebook_dir(),
        )
        self.assertIsNotNone(cache._fallback_cache)

    def test_isoquant_meta_state_uses_v2_for_wht_variant(self):
        cache = IsoQuantKVCache(
            num_heads=4,
            head_dim=128,
            bit_width=3,
            codebook_dir=get_default_codebook_dir(),
        )
        self.assertEqual(cache.meta_state[0], "iso_v2")
        self.assertFalse(cache._use_structured_runtime)

    def test_isoquant_structured_runtime_opt_in_via_env(self):
        with patch.dict("os.environ", {"ISOQUANT_USE_STRUCTURED_MLX": "1"}):
            cache = IsoQuantKVCache(
                num_heads=4,
                head_dim=128,
                bit_width=3,
                codebook_dir=get_default_codebook_dir(),
            )
        self.assertTrue(cache._use_structured_runtime)

    def test_isoquant_metal_runtime_opt_in_via_env(self):
        with patch.dict("os.environ", {"ISOQUANT_USE_METAL": "1"}):
            cache = IsoQuantKVCache(
                num_heads=4,
                head_dim=128,
                bit_width=3,
                codebook_dir=get_default_codebook_dir(),
            )
        self.assertTrue(cache._use_metal_runtime)

    def test_isoquant_meta_state_loads_legacy_v1_without_global_mix(self):
        cache = IsoQuantKVCache.__new__(IsoQuantKVCache)
        cache.meta_state = (
            "iso_v1",
            "3",
            "4",
            "128",
            "0",
            get_default_codebook_dir(),
            str(_DEFAULT_TQ_SEED),
        )
        self.assertFalse(cache._apply_global_mix)

    def test_isoquant_fallback_delegates_offset_and_reconstruction(self):
        cache = IsoQuantKVCache(
            num_heads=4,
            head_dim=16,
            bit_width=3,
            codebook_dir=get_default_codebook_dir(),
        )
        keys = mx.zeros((1, 4, 2, 16), dtype=mx.float16)
        values = mx.ones((1, 4, 2, 16), dtype=mx.float16)
        fetched_keys, fetched_values = cache.update_and_fetch(keys, values)
        self.assertEqual(cache.offset, 2)
        self.assertEqual(cache.size(), 2)
        np.testing.assert_allclose(
            np.array(cache.reconstruct_keys()), np.array(fetched_keys)
        )
        np.testing.assert_allclose(
            np.array(cache.get_values()), np.array(fetched_values)
        )

    def test_isoquant_runtime_shape_reconfigures_for_gemma_heads(self):
        cache = IsoQuantKVCache(
            num_heads=8,
            head_dim=128,
            bit_width=3,
            codebook_dir=get_default_codebook_dir(),
        )
        keys = mx.zeros((1, 8, 4, 256), dtype=mx.float16)
        values = mx.zeros((1, 8, 4, 256), dtype=mx.float16)
        cache.update_and_fetch(keys, values)
        self.assertEqual(cache.num_heads, 8)
        self.assertEqual(cache.head_dim, 256)

    def test_turboquant_runtime_shape_reconfigures_for_gemma_heads(self):
        cache = TurboQuantKVCache(
            num_heads=8,
            head_dim=128,
            bit_width=3,
            codebook_dir=get_default_codebook_dir(),
        )
        keys = mx.zeros((1, 2, 4, 512), dtype=mx.float16)
        values = mx.zeros((1, 2, 4, 512), dtype=mx.float16)
        cache.update_and_fetch(keys, values)
        self.assertEqual(cache.num_heads, 2)
        self.assertEqual(cache.head_dim, 512)

    def test_rotate_forward_uses_metal_backend_when_enabled(self):
        cache = IsoQuantKVCache.__new__(IsoQuantKVCache)
        cache.rotation_matrices = mx.eye(4, dtype=mx.float32)[None, :, :]
        cache.block_matrices_t = mx.eye(4, dtype=mx.float32).reshape(1, 1, 4, 4)
        cache.block_matrices = mx.eye(4, dtype=mx.float32).reshape(1, 1, 4, 4)
        cache._use_hadamard = False
        cache._use_structured_runtime = False
        cache._use_metal_runtime = True
        cache._metal_runtime_error = None
        x = mx.arange(4, dtype=mx.float32).reshape(1, 1, 4)

        def _metal_forward(x_arg, blocks_arg, hadamard_arg):
            self.assertIs(x_arg, x)
            self.assertFalse(hadamard_arg)
            return mx.ones_like(x_arg) * 7.0

        with patch(
            "mlx_lm.models.mlx_isoquant._get_metal_backend",
            return_value=(_metal_forward, lambda *_: None),
        ):
            out = cache._rotate_forward(x)

        np.testing.assert_allclose(np.array(out), np.full((1, 1, 4), 7.0))
        self.assertTrue(cache._use_metal_runtime)
        self.assertIsNone(cache._metal_runtime_error)

    def test_rotate_forward_falls_back_to_dense_after_metal_error(self):
        cache = IsoQuantKVCache.__new__(IsoQuantKVCache)
        cache.rotation_matrices = mx.eye(4, dtype=mx.float32)[None, :, :]
        cache.block_matrices_t = mx.eye(4, dtype=mx.float32).reshape(1, 1, 4, 4)
        cache.block_matrices = mx.eye(4, dtype=mx.float32).reshape(1, 1, 4, 4)
        cache._use_hadamard = False
        cache._use_structured_runtime = False
        cache._use_metal_runtime = True
        cache._metal_runtime_error = None
        x = mx.arange(4, dtype=mx.float32).reshape(1, 1, 4)

        with patch(
            "mlx_lm.models.mlx_isoquant._get_metal_backend",
            return_value=(
                lambda *_: (_ for _ in ()).throw(RuntimeError("metal failed")),
                lambda *_: None,
            ),
        ):
            out = cache._rotate_forward(x)

        np.testing.assert_allclose(np.array(out), np.array(x))
        self.assertFalse(cache._use_metal_runtime)
        self.assertEqual(cache._metal_runtime_error, "metal failed")

    def test_rotate_inverse_uses_metal_backend_when_enabled(self):
        cache = IsoQuantKVCache.__new__(IsoQuantKVCache)
        cache.rotation_matrices = mx.eye(4, dtype=mx.float32)[None, :, :]
        cache.block_matrices_t = mx.eye(4, dtype=mx.float32).reshape(1, 1, 4, 4)
        cache.block_matrices = mx.eye(4, dtype=mx.float32).reshape(1, 1, 4, 4)
        cache._use_hadamard = False
        cache._use_structured_runtime = False
        cache._use_metal_runtime = True
        cache._metal_runtime_error = None
        x_rot = mx.arange(4, dtype=mx.float32).reshape(1, 1, 4)

        def _metal_inverse(x_arg, blocks_arg, hadamard_arg):
            self.assertIs(x_arg, x_rot)
            self.assertFalse(hadamard_arg)
            return mx.ones_like(x_arg) * 3.0

        with patch(
            "mlx_lm.models.mlx_isoquant._get_metal_backend",
            return_value=(lambda *_: None, _metal_inverse),
        ):
            out = cache._rotate_inverse(x_rot)

        np.testing.assert_allclose(np.array(out), np.full((1, 1, 4), 3.0))


class TestUnfusedPathWarning(unittest.TestCase):
    def test_bit_width_4_warns_unfused(self):
        import warnings

        cache = IsoQuantKVCache(
            num_heads=2,
            head_dim=128,
            bit_width=4,
            codebook_dir=None,
        )
        self.assertFalse(cache.supports_fused_attention)

        keys = mx.random.normal((1, 2, 4, 128))
        values = mx.random.normal((1, 2, 4, 128))
        cache.update_and_fetch(keys, values)
        cache.finalize_deferred_prefill()

        q = mx.random.normal((1, 2, 1, 128))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cache.fused_attention(q, scale=1.0 / (128**0.5))
            unfused_warnings = [
                x for x in w if "fused path unavailable" in str(x.message).lower()
            ]
            self.assertGreater(
                len(unfused_warnings),
                0,
                "Expected a warning about fused path being unavailable for bit_width=4",
            )

    def test_bit_width_3_no_warning(self):
        import warnings

        cache = IsoQuantKVCache(
            num_heads=2,
            head_dim=128,
            bit_width=3,
            codebook_dir=None,
        )
        keys = mx.random.normal((1, 2, 4, 128))
        values = mx.random.normal((1, 2, 4, 128))
        cache.update_and_fetch(keys, values)
        cache.finalize_deferred_prefill()

        q = mx.random.normal((1, 2, 1, 128))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cache.fused_attention(q, scale=1.0 / (128**0.5))
            unfused_warnings = [
                x for x in w if "fused path unavailable" in str(x.message).lower()
            ]
            self.assertEqual(
                len(unfused_warnings),
                0,
                "bit_width=3 should not emit unfused warning",
            )

    def test_unfused_counter_increments(self):
        from mlx_lm.models.mlx_isoquant import reset_stats, get_stats

        reset_stats()
        cache = IsoQuantKVCache(
            num_heads=2,
            head_dim=128,
            bit_width=4,
            codebook_dir=None,
        )
        keys = mx.random.normal((1, 2, 4, 128))
        values = mx.random.normal((1, 2, 4, 128))
        cache.update_and_fetch(keys, values)
        cache.finalize_deferred_prefill()

        q = mx.random.normal((1, 2, 1, 128))
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cache.fused_attention(q, scale=1.0 / (128**0.5))

        self.assertGreater(get_stats().unfused_fallback_calls, 0)


class TestIsoquantBitsEnvVar(unittest.TestCase):
    def test_isoquant_bits_overrides_turboquant_bits(self):
        with patch.dict(os.environ, {"TURBOQUANT_BITS": "4", "ISOQUANT_BITS": "3"}):
            from mlx_lm.models.cache import _get_isoquant_bits

            self.assertEqual(_get_isoquant_bits(), 3)

    def test_turboquant_bits_fallback_when_isoquant_bits_unset(self):
        env = {"TURBOQUANT_BITS": "2"}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("ISOQUANT_BITS", None)
            from mlx_lm.models.cache import _get_isoquant_bits

            self.assertEqual(_get_isoquant_bits(), 2)

    def test_default_is_3_when_neither_set(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ISOQUANT_BITS", None)
            os.environ.pop("TURBOQUANT_BITS", None)
            from mlx_lm.models.cache import _get_isoquant_bits, _get_turboquant_bits

            self.assertEqual(_get_turboquant_bits(), 3)
            self.assertEqual(_get_isoquant_bits(), 3)


if __name__ == "__main__":
    unittest.main()
