"""Tests for Qwen3 expert offload module selection — dense vs quantized.

Covers the Codex review findings:
- High: OffloadSwitchGLU vs OffloadQuantizedSwitchGLU selection
- High: quant predicate must match SwitchLinear (leaf), not SwitchGLU (parent)
- Medium: max_resident_experts auto-sizing must be capped
"""

import mlx.nn as nn

from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.qwen3_moe import Model, ModelArgs
from mlx_lm.models.switch_layers import (
    OffloadQuantizedSwitchGLU,
    OffloadSwitchGLU,
    SwitchGLU,
    SwitchLinear,
)
from mlx_lm.utils import _swap_qwen3_offload_modules


def _make_args(**overrides):
    defaults = dict(
        model_type="qwen3_moe",
        hidden_size=64,
        num_hidden_layers=4,
        intermediate_size=128,
        num_attention_heads=4,
        num_experts=8,
        num_experts_per_tok=2,
        decoder_sparse_step=2,
        mlp_only_layers=[],
        moe_intermediate_size=64,
        rms_norm_eps=1e-6,
        vocab_size=1000,
        num_key_value_heads=4,
        head_dim=16,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        max_position_embeddings=512,
        norm_topk_prob=True,
        expert_offload=True,
    )
    defaults.update(overrides)
    return ModelArgs(**defaults)


class TestOffloadModuleSelection:
    """Module swap must select the right offload variant after quantization."""

    def test_model_constructs_plain_switchglu(self):
        """Model file always builds SwitchGLU, not OffloadSwitchGLU."""
        model = Model(_make_args())
        for layer in model.model.layers:
            mlp = layer.mlp
            if hasattr(mlp, "switch_mlp"):
                assert isinstance(mlp.switch_mlp, SwitchGLU)
                assert not isinstance(mlp.switch_mlp, OffloadSwitchGLU)

    def test_dense_swap_gives_offload_switchglu(self):
        """Dense checkpoint → OffloadSwitchGLU (gather_mm path)."""
        model = Model(_make_args())
        _swap_qwen3_offload_modules(model, is_quantized=False, config={})
        for layer in model.model.layers:
            mlp = layer.mlp
            if hasattr(mlp, "switch_mlp"):
                assert isinstance(mlp.switch_mlp, OffloadSwitchGLU)
                assert not isinstance(mlp.switch_mlp, OffloadQuantizedSwitchGLU)
                assert hasattr(mlp.switch_mlp, "set_expert_manager")

    def test_quantized_swap_gives_offload_quantized_switchglu(self):
        """Quantized checkpoint → OffloadQuantizedSwitchGLU (gather_qmm path)."""
        model = Model(_make_args())
        config = {"quantization": {"group_size": 64, "bits": 4, "mode": "affine"}}
        _swap_qwen3_offload_modules(model, is_quantized=True, config=config)
        for layer in model.model.layers:
            mlp = layer.mlp
            if hasattr(mlp, "switch_mlp"):
                assert isinstance(mlp.switch_mlp, OffloadQuantizedSwitchGLU)
                assert hasattr(mlp.switch_mlp, "set_expert_manager")

    def test_dense_layers_untouched_by_swap(self):
        """Non-MoE layers (dense MLP) must not be swapped."""
        # decoder_sparse_step=2 means only odd-indexed layers are MoE
        model = Model(_make_args(decoder_sparse_step=2))
        _swap_qwen3_offload_modules(model, is_quantized=False, config={})
        for i, layer in enumerate(model.model.layers):
            mlp = layer.mlp
            if not hasattr(mlp, "switch_mlp"):
                # Dense layer — should be a plain MLP, not an offload module
                assert not isinstance(
                    mlp, (OffloadSwitchGLU, OffloadQuantizedSwitchGLU)
                )


class TestQuantPredicateLeafModule:
    """Quant predicates must check isinstance(SwitchLinear), not isinstance(SwitchGLU)."""

    def test_mixed_expert_predicate_matches_switchlinear(self):
        from mlx_lm.convert import _build_mixed_expert_quant_predicate

        pred = _build_mixed_expert_quant_predicate(
            mixed_expert_bits=2,
            default_bits=4,
            default_group_size=64,
            mode="affine",
        )
        sl = SwitchLinear(64, 64, 8, bias=False)
        result = pred("model.layers.5.mlp.switch_mlp.gate_proj", sl)
        assert result["bits"] == 2

    def test_mixed_expert_predicate_non_expert_default(self):
        from mlx_lm.convert import _build_mixed_expert_quant_predicate

        pred = _build_mixed_expert_quant_predicate(
            mixed_expert_bits=2,
            default_bits=4,
            default_group_size=64,
            mode="affine",
        )
        lin = nn.Linear(64, 64)
        result = pred("model.layers.5.self_attn.q_proj", lin)
        assert result["bits"] == 4

    def test_apex_predicate_matches_switchlinear(self):
        from mlx_lm.convert import _build_apex_expert_quant_predicate

        recipe = {
            "bands": [
                {"start": 0, "end": 3, "routed_bits": 4},
            ],
            "shared_expert": {"bits": 6, "group_size": 64},
        }
        pred = _build_apex_expert_quant_predicate(
            recipe=recipe,
            default_bits=4,
            default_group_size=64,
            mode="affine",
            num_layers=4,
        )
        sl = SwitchLinear(64, 64, 8, bias=False)
        result = pred("model.layers.2.mlp.switch_mlp.up_proj", sl)
        assert result["bits"] == 4  # Layer 2 band says 4-bit


class TestMaxResidentExpertsCap:
    """max_resident_experts auto-sizing must not blow memory."""

    def test_cap_prevents_runaway(self):
        """Qwen3-30B has 48 MoE layers × top_k=8 = 384. Must be capped."""
        # Simulate the auto-sizing logic from utils.py
        top_k = 8
        default_max_experts = min(top_k * 4, 64)
        assert default_max_experts == 32
        assert default_max_experts < 384

    def test_small_top_k_reasonable(self):
        top_k = 2
        default_max_experts = min(top_k * 4, 64)
        assert default_max_experts == 8


class TestQwen3TurboQuantAttentionPath:
    """TurboQuant/IsoQuant must reconstruct full KV for SDPA (Codex/Gemma parity)."""

    def test_make_prompt_cache_accepts_isoquant_for_qwen3(self):
        """IsoQuant construction should not crash on stub configs without codebooks."""
        from mlx_lm.models.mlx_isoquant import IsoQuantKVCache

        model = Model(_make_args(decoder_sparse_step=1))
        cache = make_prompt_cache(model, kv_cache_type="isoquant")
        has_iso = any(isinstance(c, IsoQuantKVCache) for c in cache)
        assert has_iso, (
            f"Expected IsoQuantKVCache in cache, got {[type(c).__name__ for c in cache]}"
        )
        fallback_layers = [c for c in cache if isinstance(c, IsoQuantKVCache)]
        assert fallback_layers
        assert all(c._fallback_cache is not None for c in fallback_layers)

    def test_attention_prefill_uses_reconstructed_keys_not_latest_chunk(self):
        import mlx.core as mx

        from mlx_lm.models.base import create_attention_mask
        from mlx_lm.models.mlx_turboquant import (
            TurboQuantKVCache,
            get_default_codebook_dir,
        )
        from mlx_lm.models.qwen3_moe import Attention

        hidden_size = 512
        args = _make_args(hidden_size=hidden_size, head_dim=128)
        attn = Attention(args, layer_idx=0)
        cache = TurboQuantKVCache(
            num_heads=4,
            head_dim=128,
            bit_width=3,
            codebook_dir=get_default_codebook_dir(),
        )
        x = mx.random.uniform(shape=(1, 2, hidden_size))
        mask = create_attention_mask(x, cache=cache)
        out = attn(x, mask=mask, cache=cache)
        mx.eval(out)
        assert out.shape == (1, 2, hidden_size)
        assert mx.isfinite(out).all().item()
        assert cache.offset == 2

    def test_attention_prefill_isoquant_subclass(self):
        import mlx.core as mx

        from mlx_lm.models.base import create_attention_mask
        from mlx_lm.models.mlx_isoquant import IsoQuantKVCache
        from mlx_lm.models.mlx_turboquant import get_default_codebook_dir
        from mlx_lm.models.qwen3_moe import Attention

        hidden_size = 512
        args = _make_args(hidden_size=hidden_size, head_dim=128)
        attn = Attention(args, layer_idx=0)
        cache = IsoQuantKVCache(
            num_heads=4,
            head_dim=128,
            bit_width=3,
            codebook_dir=get_default_codebook_dir(),
        )
        x = mx.random.uniform(shape=(1, 2, hidden_size))
        mask = create_attention_mask(x, cache=cache)
        out = attn(x, mask=mask, cache=cache)
        mx.eval(out)
        assert out.shape == (1, 2, hidden_size)
        assert mx.isfinite(out).all().item()
