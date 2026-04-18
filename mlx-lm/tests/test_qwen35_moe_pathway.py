import unittest
from pathlib import Path


class TestQwen35MoeRepacking(unittest.TestCase):
    def test_repack_accepts_qwen3_5_moe_model_type(self):
        from mlx_lm.repack_experts import _repack_supported_types

        self.assertIn("qwen3_5_moe", _repack_supported_types)


class TestQwen35MoeOffload(unittest.TestCase):
    def test_offload_type_sets_include_qwen3_5_moe(self):
        from mlx_lm.expert_offload import EXPERT_OFFLOAD_MODEL_TYPES, MOE_MODEL_TYPES

        self.assertIn("qwen3_5_moe", EXPERT_OFFLOAD_MODEL_TYPES)
        self.assertIn("qwen3_5_moe", MOE_MODEL_TYPES)

    def test_parse_expert_key_qwen3_5_moe(self):
        from mlx_lm.expert_offload import parse_expert_key

        result = parse_expert_key(
            "language_model.model.layers.5.mlp.experts.3.gate_proj.weight",
            model_type="qwen3_5_moe",
        )
        self.assertIsNotNone(result)
        layer_idx, expert_id, proj, suffix = result
        self.assertEqual(layer_idx, 5)
        self.assertEqual(expert_id, 3)
        self.assertEqual(proj, "gate_proj")
        self.assertEqual(suffix, "weight")

    def test_is_expert_weight_key_qwen3_5_moe(self):
        from mlx_lm.expert_offload import is_expert_weight_key

        self.assertTrue(
            is_expert_weight_key(
                "model.layers.1.mlp.experts.0.gate_proj.weight",
                model_type="qwen3_5_moe",
            )
        )
        self.assertFalse(
            is_expert_weight_key(
                "model.layers.1.mlp.shared_expert.gate_proj.weight",
                model_type="qwen3_5_moe",
            )
        )


class TestQwen35MoeUtils(unittest.TestCase):
    def test_offload_supported_types_in_utils(self):
        src = (Path(__file__).resolve().parents[1] / "mlx_lm" / "utils.py").read_text()
        self.assertIn('"qwen3_5_moe"', src)


if __name__ == "__main__":
    unittest.main()
