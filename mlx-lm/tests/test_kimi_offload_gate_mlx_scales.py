"""Regression tests for the expert_offload load gate.

The gate at mlx_lm.utils._has_per_expert_scales_for_offload accepts per-expert
scale tensors for the ExpertOffloadManager. For Kimi (kimi_k25 / kimi_k2) it
must accept BOTH:

  - Compressed-tensors-style: .weight_packed / .weight_scale / .weight_shape
    (the upstream Kimi K2.x checkpoint as distributed by Moonshot)
  - MLX-native style: .weight / .scales / .biases
    (output of `mlx_lm.convert --quantize --mixed-expert-bits N` followed by
    the unstack-to-per-expert step in scripts/unstack_kimi_2bit.py)

The downstream code (build_kimi_expert_key_table, _load_expert_pair_tensors,
prepare_gather_triple_quantized) was already format-agnostic. The gate was
narrower than the downstream code, blocking MLX-quantized Kimi checkpoints
from loading even though the runtime would have handled them fine.
"""

import unittest

from mlx_lm.utils import _has_per_expert_scales_for_offload


class TestKimiOffloadGate(unittest.TestCase):
    """The gate must accept either compressed-tensors or MLX-native per-expert
    scale keys for Kimi, and reject weight_maps with no per-expert scales."""

    @staticmethod
    def _kimi_compressed_tensors_wm() -> dict:
        """A weight_map fragment in the compressed-tensors convention."""
        prefix = "language_model.model.layers.1.mlp.experts.0"
        return {
            f"{prefix}.gate_proj.weight_packed": "model-00001-of-00064.safetensors",
            f"{prefix}.gate_proj.weight_scale": "model-00001-of-00064.safetensors",
            f"{prefix}.gate_proj.weight_shape": "model-00001-of-00064.safetensors",
            f"{prefix}.up_proj.weight_packed": "model-00001-of-00064.safetensors",
            f"{prefix}.up_proj.weight_scale": "model-00001-of-00064.safetensors",
            f"{prefix}.up_proj.weight_shape": "model-00001-of-00064.safetensors",
            f"{prefix}.down_proj.weight_packed": "model-00001-of-00064.safetensors",
            f"{prefix}.down_proj.weight_scale": "model-00001-of-00064.safetensors",
            f"{prefix}.down_proj.weight_shape": "model-00001-of-00064.safetensors",
        }

    @staticmethod
    def _kimi_mlx_native_wm() -> dict:
        """A weight_map fragment in the MLX-native convention."""
        prefix = "language_model.model.layers.1.mlp.experts.0"
        return {
            f"{prefix}.gate_proj.weight": "model-00001-of-00061.safetensors",
            f"{prefix}.gate_proj.scales": "model-00001-of-00061.safetensors",
            f"{prefix}.gate_proj.biases": "model-00001-of-00061.safetensors",
            f"{prefix}.up_proj.weight": "model-00001-of-00061.safetensors",
            f"{prefix}.up_proj.scales": "model-00001-of-00061.safetensors",
            f"{prefix}.up_proj.biases": "model-00001-of-00061.safetensors",
            f"{prefix}.down_proj.weight": "model-00001-of-00061.safetensors",
            f"{prefix}.down_proj.scales": "model-00001-of-00061.safetensors",
            f"{prefix}.down_proj.biases": "model-00001-of-00061.safetensors",
        }

    def test_compressed_tensors_passes_for_kimi(self):
        wm = self._kimi_compressed_tensors_wm()
        self.assertTrue(
            _has_per_expert_scales_for_offload(wm, "kimi_k25"),
            "compressed-tensors per-expert scales must pass the gate for Kimi",
        )

    def test_mlx_native_passes_for_kimi(self):
        """The new behavior: MLX-native .scales keys must also pass for Kimi.

        This is the regression test for the gate fix that unblocks
        `mlx_lm.convert --quantize --mixed-expert-bits` outputs for Kimi.
        Without the fix, this would raise ValueError at load time.
        """
        wm = self._kimi_mlx_native_wm()
        self.assertTrue(
            _has_per_expert_scales_for_offload(wm, "kimi_k25"),
            "MLX-native .scales per-expert keys must pass the gate for Kimi",
        )

    def test_mlx_native_passes_for_kimi_k2(self):
        wm = self._kimi_mlx_native_wm()
        self.assertTrue(
            _has_per_expert_scales_for_offload(wm, "kimi_k2"),
            "kimi_k2 model_type must accept the same MLX-native scales as kimi_k25",
        )

    def test_no_per_expert_scales_fails(self):
        """Stacked SwitchLinear layout (switch_mlp.<proj>.<kind>) must NOT
        pass the gate — those keys aren't per-expert and the offload manager
        can't load them."""
        wm = {
            "language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight": "shard.safetensors",
            "language_model.model.layers.1.mlp.switch_mlp.gate_proj.scales": "shard.safetensors",
            "language_model.model.layers.1.mlp.switch_mlp.gate_proj.biases": "shard.safetensors",
        }
        self.assertFalse(
            _has_per_expert_scales_for_offload(wm, "kimi_k25"),
            "Stacked switch_mlp keys (not per-expert) must NOT pass the gate",
        )

    def test_dense_only_fails(self):
        """Pure non-expert weight_map must not falsely pass."""
        wm = {
            "language_model.model.embed_tokens.weight": "shard.safetensors",
            "language_model.model.layers.0.self_attn.q_proj.weight": "shard.safetensors",
            "language_model.model.layers.0.self_attn.q_proj.scales": "shard.safetensors",
        }
        self.assertFalse(
            _has_per_expert_scales_for_offload(wm, "kimi_k25"),
            "Non-expert .scales keys must NOT satisfy the per-expert gate",
        )

    def test_other_model_type_unchanged(self):
        """Non-Kimi model types still only accept .scales (the existing behavior).
        This pins the scope of the fix to Kimi to avoid regressing other models."""
        # Compressed-tensors-style keys must NOT pass for a non-Kimi model
        wm = {
            "model.layers.1.mlp.experts.0.gate_proj.weight_scale": "shard.safetensors",
        }
        self.assertFalse(
            _has_per_expert_scales_for_offload(wm, "qwen3_moe"),
            "Compressed-tensors weight_scale keys must NOT pass for non-Kimi models "
            "(this would be a breaking change to the existing accepted convention)",
        )


if __name__ == "__main__":
    unittest.main()
