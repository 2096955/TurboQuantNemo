"""Failing tests for Kimi K2.6 expert offload — Phase 1.

Covers five tasks:
  1.1  Expert-key parsing (regex + table builder for kimi_k25/kimi_k2)
  1.2  Non-expert load exclusion (Kimi expert keys filtered from non-expert set)
  1.5  Compressed-tensors aliasing (weight_packed→uint32, weight_scale→scales, biases=-8*s)
  1.3  Module swap (SwitchGLU → OffloadQuantizedSwitchGLU for quantized Kimi)
  1.4  Manager attach (60 MoE layers wired to ExpertOffloadManager)

Design decisions:
  - No repack needed: Kimi/DeepSeek-V3 checkpoints store experts as individual
    per-expert tensors (the sanitizer stacks them into SwitchGLU format on full load,
    but for offload we skip that step). This is unlike Gemma 4 which needs
    repack_experts.py to split stacked SwitchGLU shards.
  - Compressed-tensors aliasing: The raw checkpoint uses weight_packed (int32) /
    weight_scale (bfloat16) / weight_shape (int32) suffixes (INT4 group-32).
    _load_expert_pair_tensors must be extended to remap these:
    weight_packed→.view(uint32), weight_scale→scales, biases=-8*scale.
  - Module swap: Kimi's model.model property (kimi_k25.py) returns
    language_model.model (DeepseekV3Model), so _swap_qwen3_offload_modules works
    via getattr(model, "model", model). The implementation may either add a
    dedicated _swap_kimi_offload_modules or route kimi through the Qwen3 swap
    in the loader dispatch. Both are valid.
  - Quantization config: Kimi stores quantization under text_config.quantization_config
    (not top-level). The loader _is_quantized detection must resolve this nested path.
  - set_expert_manager stores manager on child linears (gate_proj.manager, etc.),
    not on switch_mlp itself (switch_layers.py:696-704).

Checkpoint facts (verified against /Volumes/Samsung9904tb/Kimi-K2.6):
  - Shard names: model-00001-of-000064.safetensors (6-digit zero-padded, 64 shards)
  - weight_packed: int32, shape e.g. (2048, 896)
  - weight_scale: bfloat16, shape e.g. (2048, 224)
  - weight_shape: int32, shape (2,)
  - quantization_config is under text_config, not top-level
"""

import inspect
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx

from mlx_lm.expert_offload import (
    EXPERT_OFFLOAD_MODEL_TYPES,
    MOE_MODEL_TYPES,
    ExpertOffloadManager,
    attach_expert_offload_manager,
    is_expert_weight_key,
    parse_expert_key,
)
from mlx_lm.expert_weight_loader import load_non_expert_weights
from mlx_lm.models.switch_layers import (
    OffloadQuantizedSwitchGLU,
    OffloadSwitchGLU,
    SwitchGLU,
)

try:
    from mlx_lm.expert_offload import build_kimi_expert_key_table
except ImportError:
    build_kimi_expert_key_table = None


def _resolve_kimi_swap_fn():
    """Resolve the swap function for Kimi model types.

    A valid implementation may either:
    (a) Export a dedicated _swap_kimi_offload_modules, or
    (b) Route kimi_k25/kimi_k2 through the existing _swap_qwen3_offload_modules
        via the loader dispatch in utils.py.

    Returns the callable swap function, or None if kimi is not yet dispatched.
    """
    # Path (a): dedicated function
    try:
        from mlx_lm.utils import _swap_kimi_offload_modules

        return _swap_kimi_offload_modules
    except ImportError:
        pass

    # Path (b): check if load_model dispatches kimi to an existing swap
    from mlx_lm import utils

    try:
        src = inspect.getsource(utils.load_model)
    except Exception:
        return None

    if "kimi_k25" not in src and "kimi_k2" not in src:
        return None

    # Dispatch exists — resolve the function it routes to
    try:
        from mlx_lm.utils import _swap_qwen3_offload_modules

        return _swap_qwen3_offload_modules
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SHARD = "model-00001-of-000064.safetensors"


def _kimi_expert_key(layer, expert, proj, suffix):
    return f"language_model.model.layers.{layer}.mlp.experts.{expert}.{proj}.{suffix}"


def _kimi_expert_weight_map(layers, experts, shard=_SHARD):
    """Build a synthetic Kimi weight map with compressed-tensors suffixes."""
    wm = {}
    for layer in layers:
        for expert in experts:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                for suffix in ("weight_packed", "weight_scale", "weight_shape"):
                    wm[_kimi_expert_key(layer, expert, proj, suffix)] = shard
    return wm


def _make_kimi_model(num_layers=4, n_routed_experts=8, first_k_dense_replace=1):
    """Instantiate a tiny Kimi K2.5 model for swap/attach tests."""
    from mlx_lm.models.deepseek_v3 import ModelArgs as TextConfig
    from mlx_lm.models.kimi_k25 import Model, ModelArgs

    text_config = TextConfig(
        model_type="kimi_k2",
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=2,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=16,
        v_head_dim=16,
        qk_nope_head_dim=16,
        first_k_dense_replace=first_k_dense_replace,
        moe_layer_freq=1,
        rms_norm_eps=1e-6,
        vocab_size=1000,
        rope_theta=10000.0,
        max_position_embeddings=512,
    )
    config = ModelArgs(text_config=text_config, model_type="kimi_k25")
    return Model(config)


def _make_kimi_expert_table_manual(layers, experts):
    """Build expert key table by hand (for tests that run before build_kimi_expert_key_table exists)."""
    tbl = {}
    for layer in layers:
        for expert in experts:
            spec = {}
            for proj_short, proj_full in [
                ("gate", "gate_proj"),
                ("up", "up_proj"),
                ("down", "down_proj"),
            ]:
                spec[f"{proj_short}_weight"] = _kimi_expert_key(
                    layer, expert, proj_full, "weight_packed"
                )
                spec[f"{proj_short}_scales"] = _kimi_expert_key(
                    layer, expert, proj_full, "weight_scale"
                )
            tbl[(layer, expert)] = spec
    return tbl


# ===================================================================
# Task 1.1: Expert-key parsing
# ===================================================================


class TestKimiExpertKeyParsing(unittest.TestCase):
    """parse_expert_key and is_expert_weight_key must handle kimi_k25/kimi_k2."""

    def test_parse_kimi_k25_weight_packed(self):
        result = parse_expert_key(
            _kimi_expert_key(1, 0, "gate_proj", "weight_packed"),
            model_type="kimi_k25",
        )
        self.assertIsNotNone(result, "parse_expert_key returned None for kimi_k25")
        self.assertEqual(result, (1, 0, "gate_proj", "weight_packed"))

    def test_parse_kimi_k25_weight_scale(self):
        result = parse_expert_key(
            _kimi_expert_key(30, 383, "down_proj", "weight_scale"),
            model_type="kimi_k25",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result, (30, 383, "down_proj", "weight_scale"))

    def test_parse_kimi_k25_weight_shape(self):
        result = parse_expert_key(
            _kimi_expert_key(60, 100, "up_proj", "weight_shape"),
            model_type="kimi_k25",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result, (60, 100, "up_proj", "weight_shape"))

    def test_parse_kimi_k2_model_type(self):
        result = parse_expert_key(
            _kimi_expert_key(5, 10, "up_proj", "weight_packed"),
            model_type="kimi_k2",
        )
        self.assertIsNotNone(result, "kimi_k2 model_type must be handled")
        self.assertEqual(result, (5, 10, "up_proj", "weight_packed"))

    def test_parse_kimi_without_language_model_prefix(self):
        """Text-only model keys omit the language_model. prefix."""
        result = parse_expert_key(
            "model.layers.5.mlp.experts.10.gate_proj.weight_packed",
            model_type="kimi_k25",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result, (5, 10, "gate_proj", "weight_packed"))

    def test_is_expert_weight_key_kimi_true(self):
        self.assertTrue(
            is_expert_weight_key(
                _kimi_expert_key(1, 0, "gate_proj", "weight_packed"),
                model_type="kimi_k25",
            )
        )

    def test_is_expert_weight_key_kimi_false_for_attention(self):
        self.assertFalse(
            is_expert_weight_key(
                "language_model.model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
                model_type="kimi_k25",
            )
        )

    def test_is_expert_weight_key_kimi_false_for_shared_expert(self):
        self.assertFalse(
            is_expert_weight_key(
                "language_model.model.layers.1.mlp.shared_experts.gate_proj.weight",
                model_type="kimi_k25",
            )
        )

    def test_kimi_k25_in_expert_offload_model_types(self):
        self.assertIn("kimi_k25", EXPERT_OFFLOAD_MODEL_TYPES)

    def test_kimi_k2_in_expert_offload_model_types(self):
        self.assertIn("kimi_k2", EXPERT_OFFLOAD_MODEL_TYPES)

    def test_kimi_k25_in_moe_model_types(self):
        self.assertIn("kimi_k25", MOE_MODEL_TYPES)

    def test_kimi_k2_in_moe_model_types(self):
        self.assertIn("kimi_k2", MOE_MODEL_TYPES)


class TestKimiExpertKeyTable(unittest.TestCase):
    """build_kimi_expert_key_table must map (layer, expert) → projection specs."""

    def test_build_kimi_expert_key_table_exists(self):
        if build_kimi_expert_key_table is None:
            self.fail(
                "build_kimi_expert_key_table not yet implemented in expert_offload"
            )

    def test_basic_single_expert(self):
        if build_kimi_expert_key_table is None:
            self.fail("build_kimi_expert_key_table not yet implemented")
        wm = _kimi_expert_weight_map(layers=[1], experts=[0])
        tbl = build_kimi_expert_key_table(wm)
        self.assertIn((1, 0), tbl)
        spec = tbl[(1, 0)]
        self.assertIn("gate_weight", spec)
        self.assertIn("up_weight", spec)
        self.assertIn("down_weight", spec)

    def test_scales_mapped(self):
        if build_kimi_expert_key_table is None:
            self.fail("build_kimi_expert_key_table not yet implemented")
        wm = _kimi_expert_weight_map(layers=[1], experts=[0])
        tbl = build_kimi_expert_key_table(wm)
        spec = tbl[(1, 0)]
        self.assertIn("gate_scales", spec)
        self.assertIn("up_scales", spec)
        self.assertIn("down_scales", spec)

    def test_weight_keys_point_to_weight_packed_safetensors_key(self):
        """gate_weight must map to the *.weight_packed safetensors key (not *.weight)."""
        if build_kimi_expert_key_table is None:
            self.fail("build_kimi_expert_key_table not yet implemented")
        wm = _kimi_expert_weight_map(layers=[1], experts=[0])
        tbl = build_kimi_expert_key_table(wm)
        spec = tbl[(1, 0)]
        self.assertIn("weight_packed", spec["gate_weight"])

    def test_entry_count_two_layers_two_experts(self):
        if build_kimi_expert_key_table is None:
            self.fail("build_kimi_expert_key_table not yet implemented")
        wm = _kimi_expert_weight_map(layers=[1, 2], experts=[0, 1])
        tbl = build_kimi_expert_key_table(wm)
        self.assertEqual(len(tbl), 4)
        for layer in [1, 2]:
            for expert in [0, 1]:
                self.assertIn((layer, expert), tbl)

    def test_full_kimi_scale_60_layers(self):
        """60 MoE layers × 2 experts = 120 entries. Uses 2 (not 384) for test speed."""
        if build_kimi_expert_key_table is None:
            self.fail("build_kimi_expert_key_table not yet implemented")
        wm = _kimi_expert_weight_map(layers=range(1, 61), experts=[0, 1])
        tbl = build_kimi_expert_key_table(wm)
        self.assertEqual(len(tbl), 120)


# ===================================================================
# Task 1.2: Non-expert load exclusion
# ===================================================================


class TestKimiNonExpertExclusion(unittest.TestCase):
    """load_non_expert_weights must filter Kimi expert keys."""

    def test_kimi_expert_keys_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "model.safetensors"

            expert_key = _kimi_expert_key(1, 0, "gate_proj", "weight_packed")
            non_expert_key = "language_model.model.embed_tokens.weight"

            mx.save_safetensors(
                str(shard),
                {
                    expert_key: mx.zeros((4, 4), dtype=mx.int32),
                    non_expert_key: mx.ones((8, 4)),
                },
            )
            loaded = load_non_expert_weights(
                root,
                {
                    expert_key: "model.safetensors",
                    non_expert_key: "model.safetensors",
                },
            )
            self.assertIn(non_expert_key, loaded)
            self.assertNotIn(expert_key, loaded, "Kimi expert key was not filtered out")

    def test_kimi_shared_expert_not_excluded(self):
        """Shared expert weights must NOT be filtered — only routed experts."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "model.safetensors"

            shared_key = (
                "language_model.model.layers.1.mlp.shared_experts.gate_proj.weight"
            )
            routed_key = _kimi_expert_key(1, 0, "gate_proj", "weight_packed")

            mx.save_safetensors(
                str(shard),
                {
                    shared_key: mx.ones((4, 4)),
                    routed_key: mx.zeros((4, 4), dtype=mx.int32),
                },
            )
            loaded = load_non_expert_weights(
                root,
                {
                    shared_key: "model.safetensors",
                    routed_key: "model.safetensors",
                },
            )
            self.assertIn(shared_key, loaded, "Shared expert was incorrectly filtered")
            self.assertNotIn(routed_key, loaded)

    def test_kimi_weight_scale_and_shape_also_excluded(self):
        """All three compressed-tensors suffixes must be excluded."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "model.safetensors"
            non_expert_key = "language_model.lm_head.weight"
            expert_keys = [
                _kimi_expert_key(1, 0, "gate_proj", "weight_packed"),
                _kimi_expert_key(1, 0, "gate_proj", "weight_scale"),
                _kimi_expert_key(1, 0, "gate_proj", "weight_shape"),
            ]
            tensors = {non_expert_key: mx.ones((4, 4))}
            for ek in expert_keys:
                tensors[ek] = mx.zeros((4, 4))
            mx.save_safetensors(str(shard), tensors)

            wm = {k: "model.safetensors" for k in [non_expert_key] + expert_keys}
            loaded = load_non_expert_weights(root, wm)
            self.assertIn(non_expert_key, loaded)
            for ek in expert_keys:
                self.assertNotIn(ek, loaded, f"Expert key not filtered: {ek}")


# ===================================================================
# Task 1.5: Compressed-tensors aliasing
# ===================================================================

# Shapes match real Kimi K2.6 checkpoint structure at reduced scale.
# Real: weight_packed int32 (2048, 896), weight_scale bfloat16 (2048, 224).
# Test: weight_packed int32 (8, 4), weight_scale bfloat16 (8, 1), group_size=32.
_CT_PACKED_SHAPE = (8, 4)
_CT_SCALE_SHAPE = (8, 1)


class TestKimiCompressedTensorsAliasing(unittest.TestCase):
    """_load_expert_pair_tensors must remap weight_packed/weight_scale/weight_shape."""

    def _create_compressed_expert_safetensors(self, tmp_dir):
        """Create a safetensors file with Kimi compressed-tensors format.

        Uses int32 for weight_packed and bfloat16 for weight_scale,
        matching the real Kimi K2.6 checkpoint dtypes.
        """
        shard_name = "model.safetensors"
        shard_path = Path(tmp_dir) / shard_name

        tensors = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            prefix = f"language_model.model.layers.1.mlp.experts.0.{proj}"
            tensors[f"{prefix}.weight_packed"] = mx.zeros(
                _CT_PACKED_SHAPE, dtype=mx.int32
            )
            tensors[f"{prefix}.weight_scale"] = mx.ones(
                _CT_SCALE_SHAPE, dtype=mx.bfloat16
            )
            tensors[f"{prefix}.weight_shape"] = mx.array(
                [_CT_PACKED_SHAPE[0], _CT_PACKED_SHAPE[1] * 8], dtype=mx.int32
            )

        mx.save_safetensors(str(shard_path), tensors)
        return shard_name

    def _build_compressed_spec(self):
        """Build the spec dict that a Kimi table builder should produce."""
        spec = {}
        for proj_short, proj_full in [
            ("gate", "gate_proj"),
            ("up", "up_proj"),
            ("down", "down_proj"),
        ]:
            prefix = f"language_model.model.layers.1.mlp.experts.0.{proj_full}"
            spec[f"{proj_short}_weight"] = f"{prefix}.weight_packed"
            spec[f"{proj_short}_scales"] = f"{prefix}.weight_scale"
        return spec

    def _build_weight_map_and_mgr(self, tmp_dir, shard_name, spec):
        """Helper: build weight map and ExpertOffloadManager for compressed-tensors tests."""
        wm = {}
        for v in spec.values():
            wm[v] = shard_name
        for proj in ("gate_proj", "up_proj", "down_proj"):
            shape_key = (
                f"language_model.model.layers.1.mlp.experts.0.{proj}.weight_shape"
            )
            wm[shape_key] = shard_name
        return ExpertOffloadManager(
            base_path=Path(tmp_dir),
            weight_map=wm,
            expert_key_table={(1, 0): spec},
            max_resident_experts=4,
            projections=("gate", "up", "down"),
        )

    def test_loaded_weight_is_uint32(self):
        """weight_packed (int32) must be .view(mx.uint32) on load."""
        with tempfile.TemporaryDirectory() as tmp:
            shard_name = self._create_compressed_expert_safetensors(tmp)
            spec = self._build_compressed_spec()
            mgr = self._build_weight_map_and_mgr(tmp, shard_name, spec)
            result = mgr._load_expert_pair_tensors(spec)
            self.assertEqual(
                result["gate_weight"].dtype,
                mx.uint32,
                "weight_packed must be viewed as uint32",
            )

    def test_biases_synthesized_as_negative_8x_scale(self):
        """biases = -8 * scale must be synthesized (not stored in checkpoint)."""
        with tempfile.TemporaryDirectory() as tmp:
            shard_name = self._create_compressed_expert_safetensors(tmp)
            spec = self._build_compressed_spec()
            mgr = self._build_weight_map_and_mgr(tmp, shard_name, spec)
            result = mgr._load_expert_pair_tensors(spec)
            self.assertIn("gate_biases", result, "biases must be synthesized")
            expected_bias = -8.0 * result["gate_scales"]
            mx.eval(expected_bias, result["gate_biases"])
            self.assertTrue(
                mx.allclose(result["gate_biases"], expected_bias).item(),
                "biases must equal -8 * scales",
            )

    def test_all_three_projections_aliased(self):
        """gate, up, and down projections must all be aliased."""
        with tempfile.TemporaryDirectory() as tmp:
            shard_name = self._create_compressed_expert_safetensors(tmp)
            spec = self._build_compressed_spec()
            mgr = self._build_weight_map_and_mgr(tmp, shard_name, spec)
            result = mgr._load_expert_pair_tensors(spec)
            for proj in ("gate", "up", "down"):
                self.assertIn(f"{proj}_weight", result)
                self.assertIn(f"{proj}_scales", result)
                self.assertIn(f"{proj}_biases", result)
                self.assertEqual(
                    result[f"{proj}_weight"].dtype,
                    mx.uint32,
                    f"{proj}_weight must be uint32",
                )

    def test_compressed_tensors_through_prepare_gather(self):
        """Compressed-tensors loaded via manager produce valid gather output.

        This is the Phase 1.5 gate: prepare_gather_triple_quantized must
        successfully load, alias, and stack Kimi compressed-tensors.
        Does NOT test OffloadQuantizedSwitchGLU forward (that is Phase 2).
        """
        with tempfile.TemporaryDirectory() as tmp:
            shard_name = self._create_compressed_expert_safetensors(tmp)
            spec = self._build_compressed_spec()
            mgr = self._build_weight_map_and_mgr(tmp, shard_name, spec)
            idx = mx.array([[0]], dtype=mx.int32)
            try:
                gate, up, down, remapped = mgr.prepare_gather_triple_quantized(1, idx)
                gate_w, gate_s, gate_b = gate
                mx.eval(gate_w, gate_s, remapped)
                self.assertEqual(gate_w.dtype, mx.uint32)
                self.assertIsNotNone(gate_b, "biases must be present after aliasing")
            except (KeyError, ExpertLoadError) as e:
                self.fail(
                    f"prepare_gather_triple_quantized failed with aliased tensors: {e}"
                )


# ===================================================================
# Task 1.3: Module swap + dispatch
# ===================================================================


class TestKimiModuleSwap(unittest.TestCase):
    """SwitchGLU → OffloadQuantizedSwitchGLU for quantized Kimi checkpoint."""

    def test_model_starts_with_plain_switchglu(self):
        """Before swap, MoE layers should have plain SwitchGLU."""
        model = _make_kimi_model(num_layers=4, first_k_dense_replace=1)
        moe_count = 0
        for layer in model.language_model.model.layers:
            mlp = layer.mlp
            if hasattr(mlp, "switch_mlp"):
                self.assertIsInstance(mlp.switch_mlp, SwitchGLU)
                self.assertNotIsInstance(mlp.switch_mlp, OffloadSwitchGLU)
                moe_count += 1
        self.assertEqual(moe_count, 3, "Layers 1-3 should be MoE (layer 0 dense)")

    def test_kimi_model_type_dispatched_in_loader(self):
        """utils.load_model must dispatch swap for kimi_k25/kimi_k2 model types.

        Always runs — never short-circuits. Verifies the loader source contains
        kimi model type dispatch, regardless of whether a dedicated swap function exists.
        """
        from mlx_lm import utils

        try:
            src = inspect.getsource(utils.load_model)
        except Exception:
            self.fail("Could not inspect utils.load_model source")

        has_kimi_dispatch = "kimi_k25" in src or "kimi_k2" in src
        self.assertTrue(
            has_kimi_dispatch,
            "utils.load_model must dispatch swap for kimi_k25 or kimi_k2 model types "
            "(currently only qwen3_moe, qwen3_5_moe, gemma4_text, gemma4)",
        )

    def test_nested_text_config_quantization_detected(self):
        """The loader must propagate text_config.quantization_config before the swap dispatch.

        Kimi's config.json has quantization_config nested under text_config, not at top level.
        The loader already propagates this at lines 133-136 (before the swap dispatch at 487).
        This test verifies that after propagation, _is_quantized resolves correctly for a
        config that has ONLY text_config.quantization_config (no top-level quantization).
        """
        kimi_config = {
            "model_type": "kimi_k25",
            "text_config": {
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "group_size": 32,
                }
            },
        }
        # Simulate the loader's propagation logic (utils.py lines 133-136)
        if "quantization_config" not in kimi_config:
            tc = kimi_config.get("text_config", {})
            if "quantization_config" in tc:
                kimi_config["quantization_config"] = tc["quantization_config"]

        # Now the swap dispatch's _is_quantized check (utils.py line 489-490) must work
        is_quantized = bool(
            kimi_config.get("quantization") or kimi_config.get("quantization_config")
        )
        self.assertTrue(
            is_quantized,
            "After text_config propagation, _is_quantized must be True for Kimi "
            "(quantization_config should have been hoisted to top-level)",
        )

        # Verify the propagation code actually exists in load_model
        from mlx_lm import utils

        src = inspect.getsource(utils.load_model)
        has_propagation = (
            "text_config" in src
            and "quantization_config" in src
            and 'config["quantization_config"]' in src
        )
        self.assertTrue(
            has_propagation,
            "load_model must propagate text_config.quantization_config to top-level",
        )

    def test_quantized_swap_produces_correct_offload_type(self):
        """After swap with is_quantized=True, MoE layers get OffloadQuantizedSwitchGLU."""
        swap_fn = _resolve_kimi_swap_fn()
        if swap_fn is None:
            self.fail("No kimi swap function available (no dispatch or dedicated fn)")
        model = _make_kimi_model(num_layers=4, first_k_dense_replace=1)
        config = {"quantization": {"group_size": 32, "bits": 4, "mode": "affine"}}
        swap_fn(model, is_quantized=True, config=config)
        moe_count = 0
        for layer in model.language_model.model.layers:
            mlp = layer.mlp
            if hasattr(mlp, "switch_mlp"):
                self.assertIsInstance(
                    mlp.switch_mlp,
                    OffloadQuantizedSwitchGLU,
                    "MoE layers must have OffloadQuantizedSwitchGLU after quantized swap",
                )
                self.assertTrue(hasattr(mlp.switch_mlp, "set_expert_manager"))
                moe_count += 1
        self.assertGreater(moe_count, 0, "No MoE layers found to verify")

    def test_dense_swap_gives_offload_switchglu(self):
        """After swap with is_quantized=False, MoE layers get OffloadSwitchGLU."""
        swap_fn = _resolve_kimi_swap_fn()
        if swap_fn is None:
            self.fail("No kimi swap function available (no dispatch or dedicated fn)")
        model = _make_kimi_model(num_layers=4, first_k_dense_replace=1)
        swap_fn(model, is_quantized=False, config={})
        for layer in model.language_model.model.layers:
            mlp = layer.mlp
            if hasattr(mlp, "switch_mlp"):
                self.assertIsInstance(mlp.switch_mlp, OffloadSwitchGLU)
                self.assertNotIsInstance(mlp.switch_mlp, OffloadQuantizedSwitchGLU)

    def test_dense_layer_0_untouched(self):
        """Layer 0 (dense MLP, no switch_mlp) must not be modified."""
        swap_fn = _resolve_kimi_swap_fn()
        if swap_fn is None:
            self.fail("No kimi swap function available (no dispatch or dedicated fn)")
        model = _make_kimi_model(num_layers=4, first_k_dense_replace=1)
        swap_fn(model, is_quantized=True, config={"quantization": {}})
        layer0_mlp = model.language_model.model.layers[0].mlp
        self.assertFalse(
            hasattr(layer0_mlp, "switch_mlp"),
            "Dense layer 0 should not have switch_mlp",
        )
        self.assertNotIsInstance(layer0_mlp, OffloadSwitchGLU)
        self.assertNotIsInstance(layer0_mlp, OffloadQuantizedSwitchGLU)


# ===================================================================
# Task 1.4: Manager attach
# ===================================================================


class TestKimiManagerAttach(unittest.TestCase):
    """attach_expert_offload_manager must wire manager into Kimi MoE layers."""

    def test_attach_kimi_k25_accepted(self):
        """model_type='kimi_k25' must not raise NotImplementedError."""
        model = _make_kimi_model(num_layers=4, first_k_dense_replace=1)
        wm = _kimi_expert_weight_map(layers=[1, 2, 3], experts=[0, 1])
        tbl = _make_kimi_expert_table_manual([1, 2, 3], [0, 1])
        mgr = ExpertOffloadManager(
            base_path=Path("/tmp"),
            weight_map=wm,
            expert_key_table=tbl,
            max_resident_experts=4,
            projections=("gate", "up", "down"),
        )
        try:
            attach_expert_offload_manager(model, mgr, model_type="kimi_k25")
        except NotImplementedError as e:
            self.fail(f"kimi_k25 should be supported: {e}")

    def test_attach_kimi_k2_accepted(self):
        """model_type='kimi_k2' must also be accepted."""
        model = _make_kimi_model()
        wm = _kimi_expert_weight_map(layers=[1, 2, 3], experts=[0])
        tbl = _make_kimi_expert_table_manual([1, 2, 3], [0])
        mgr = ExpertOffloadManager(
            base_path=Path("/tmp"),
            weight_map=wm,
            expert_key_table=tbl,
            max_resident_experts=4,
            projections=("gate", "up", "down"),
        )
        try:
            attach_expert_offload_manager(model, mgr, model_type="kimi_k2")
        except NotImplementedError as e:
            self.fail(f"kimi_k2 should be supported: {e}")

    def test_attach_wires_child_proj_managers(self):
        """After attach + swap, each child linear (gate_proj, up_proj, down_proj) must
        have manager and layer_idx set. set_expert_manager stores on child linears,
        not on switch_mlp itself (switch_layers.py:696-704).
        """
        swap_fn = _resolve_kimi_swap_fn()
        if swap_fn is None:
            self.fail("No kimi swap function available for attach wiring test")

        model = _make_kimi_model(num_layers=4, first_k_dense_replace=1)
        config = {"quantization": {"group_size": 32, "bits": 4, "mode": "affine"}}
        swap_fn(model, is_quantized=True, config=config)

        wm = _kimi_expert_weight_map(layers=[1, 2, 3], experts=range(8))
        if build_kimi_expert_key_table is not None:
            tbl = build_kimi_expert_key_table(wm)
        else:
            tbl = _make_kimi_expert_table_manual([1, 2, 3], range(8))

        mgr = ExpertOffloadManager(
            base_path=Path("/tmp"),
            weight_map=wm,
            expert_key_table=tbl,
            max_resident_experts=16,
            projections=("gate", "up", "down"),
        )
        attach_expert_offload_manager(model, mgr, model_type="kimi_k25")

        attached = 0
        for i, layer in enumerate(model.language_model.model.layers):
            mlp = layer.mlp
            if not hasattr(mlp, "switch_mlp"):
                continue
            sw = mlp.switch_mlp
            if not isinstance(sw, (OffloadSwitchGLU, OffloadQuantizedSwitchGLU)):
                continue
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                proj = getattr(sw, proj_name)
                self.assertIs(
                    proj.manager,
                    mgr,
                    f"layer {i} {proj_name}.manager must be the attached manager",
                )
                self.assertEqual(
                    proj.layer_idx,
                    i,
                    f"layer {i} {proj_name}.layer_idx must match layer index",
                )
            attached += 1
        self.assertEqual(attached, 3, "All 3 MoE layers (1-3) should be wired")

    def test_attach_60_moe_layers(self):
        """Full Kimi topology: 61 layers, layer 0 dense, layers 1-60 MoE.

        Verifies that attach wires exactly 60 MoE layers, matching the real
        Kimi K2.6 architecture (61 layers total, first_k_dense_replace=1).
        """
        swap_fn = _resolve_kimi_swap_fn()
        if swap_fn is None:
            self.fail("No kimi swap function available for 60-layer attach test")

        model = _make_kimi_model(
            num_layers=61, n_routed_experts=8, first_k_dense_replace=1
        )
        config = {"quantization": {"group_size": 32, "bits": 4, "mode": "affine"}}
        swap_fn(model, is_quantized=True, config=config)

        moe_layers = list(range(1, 61))
        wm = _kimi_expert_weight_map(layers=moe_layers, experts=range(8))
        if build_kimi_expert_key_table is not None:
            tbl = build_kimi_expert_key_table(wm)
        else:
            tbl = _make_kimi_expert_table_manual(moe_layers, range(8))

        mgr = ExpertOffloadManager(
            base_path=Path("/tmp"),
            weight_map=wm,
            expert_key_table=tbl,
            max_resident_experts=32,
            projections=("gate", "up", "down"),
        )
        attach_expert_offload_manager(model, mgr, model_type="kimi_k25")

        attached = 0
        for i, layer in enumerate(model.language_model.model.layers):
            mlp = layer.mlp
            if not hasattr(mlp, "switch_mlp"):
                continue
            sw = mlp.switch_mlp
            if isinstance(sw, (OffloadSwitchGLU, OffloadQuantizedSwitchGLU)):
                attached += 1
        self.assertEqual(
            attached, 60, "All 60 MoE layers (1-60) must be wired by attach"
        )

    def test_attach_sets_model_attribute(self):
        """model.expert_offload_manager should be set after attach."""
        model = _make_kimi_model()
        wm = _kimi_expert_weight_map(layers=[1, 2, 3], experts=[0])
        tbl = _make_kimi_expert_table_manual([1, 2, 3], [0])
        mgr = ExpertOffloadManager(
            base_path=Path("/tmp"),
            weight_map=wm,
            expert_key_table=tbl,
            max_resident_experts=4,
            projections=("gate", "up", "down"),
        )
        try:
            attach_expert_offload_manager(model, mgr, model_type="kimi_k25")
        except NotImplementedError:
            self.fail("kimi_k25 not yet supported")
        self.assertIs(model.expert_offload_manager, mgr)


# Import at bottom to avoid circular issues in test compressed-tensors
try:
    from mlx_lm.expert_offload import ExpertLoadError
except ImportError:
    ExpertLoadError = RuntimeError


if __name__ == "__main__":
    unittest.main()
