import unittest

import mlx.nn as nn

from mlx_lm.convert import _build_mixed_expert_quant_predicate
from mlx_lm.models.switch_layers import SwitchLinear


class ConvertMixedExpertBitsTest(unittest.TestCase):
    def test_predicate_targets_only_routed_switch_mlp_layers(self):
        pred = _build_mixed_expert_quant_predicate(
            mixed_expert_bits=2,
            default_bits=4,
            default_group_size=64,
            mode="affine",
        )
        expert_module = SwitchLinear(16, 32, 4, bias=False)
        dense_module = nn.Linear(16, 32, bias=False)

        expert_cfg = pred("backbone.layers.1.mixer.switch_mlp.fc1", expert_module)
        dense_cfg = pred("backbone.layers.1.mixer.shared_experts.up_proj", dense_module)

        self.assertEqual(expert_cfg["bits"], 2)
        self.assertEqual(expert_cfg["group_size"], 64)
        self.assertEqual(dense_cfg["bits"], 4)
        self.assertEqual(dense_cfg["group_size"], 64)

    def test_predicate_does_not_match_non_switch_modules(self):
        pred = _build_mixed_expert_quant_predicate(
            mixed_expert_bits=2,
            default_bits=4,
            default_group_size=32,
            mode="affine",
        )
        dense_module = nn.Linear(16, 16, bias=False)
        cfg = pred("backbone.layers.2.mixer.switch_mlp.fc1", dense_module)
        self.assertEqual(cfg["bits"], 4)
        self.assertEqual(cfg["group_size"], 32)

    def test_invalid_mixed_bits_raises(self):
        with self.assertRaises(ValueError):
            _build_mixed_expert_quant_predicate(
                mixed_expert_bits=0,
                default_bits=4,
                default_group_size=64,
                mode="affine",
            )

    def test_shared_expert_bits_applies_to_singular_path(self):
        """Llama 4 / Qwen3-Next / Qwen2-MoE use singular `.shared_expert.`."""
        pred = _build_mixed_expert_quant_predicate(
            mixed_expert_bits=2,
            shared_expert_bits=6,
            default_bits=4,
            default_group_size=64,
            mode="affine",
        )
        dense_module = nn.Linear(16, 32, bias=False)
        cfg = pred("model.layers.1.mlp.shared_expert.gate_proj", dense_module)
        self.assertEqual(cfg["bits"], 6)

    def test_shared_expert_bits_applies_to_plural_path(self):
        """Kimi K2.x / DeepSeek V3 use plural `.shared_experts.`. Without
        explicit support, --shared-expert-bits silently no-ops on those
        models (Codex review 2026-05-07, MEDIUM #3)."""
        pred = _build_mixed_expert_quant_predicate(
            mixed_expert_bits=2,
            shared_expert_bits=6,
            default_bits=4,
            default_group_size=64,
            mode="affine",
        )
        dense_module = nn.Linear(16, 32, bias=False)
        cfg = pred(
            "language_model.model.layers.1.mlp.shared_experts.gate_proj",
            dense_module,
        )
        self.assertEqual(
            cfg["bits"],
            6,
            "Kimi/DeepSeek shared_experts (plural) must honor --shared-expert-bits",
        )


if __name__ == "__main__":
    unittest.main()
