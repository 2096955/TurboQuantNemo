import unittest
from mlx_lm.models.nemotron_h import ModelArgs, NemotronHMoE

class TestNemotronQuantizedOffloadConfig(unittest.TestCase):
    def setUp(self):
        self.base_config = {
            "model_type": "nemotron_h",
            "vocab_size": 256,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "max_position_embeddings": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "attention_bias": False,
            "mamba_num_heads": 4,
            "hybrid_override_pattern": "M-E",
            "n_routed_experts": 4,
            "n_shared_experts": 2,
            "moe_intermediate_size": 128,
            "topk_group": 2,
            "n_group": 1,
            "num_experts_per_tok": 2,
            "routed_scaling_factor": 1.0,
            "mamba_head_dim": 16,
            "mamba_proj_bias": False,
            "ssm_state_size": 16,
            "conv_kernel": 4,
            "n_groups": 1,
            "mlp_bias": False,
            "layer_norm_epsilon": 1e-5,
            "use_bias": False,
            "use_conv_bias": True,
            "time_step_limit": [0.001, 0.1],
            "expert_offload": True,
        }

    def test_matching_fc1_fc2_quant_config(self):
        config = self.base_config.copy()
        config["quantization"] = {
            "backbone.layers.1.mixer.switch_mlp.fc1": {"bits": 2, "group_size": 32, "mode": "affine"},
            "backbone.layers.1.mixer.switch_mlp.fc2": {"bits": 2, "group_size": 32, "mode": "affine"},
            "bits": 4,
            "group_size": 64
        }
        args = ModelArgs.from_dict(config)
        # Should not raise
        moe = NemotronHMoE(args, layer_idx=1)
        self.assertEqual(moe.switch_mlp.fc1.bits, 2)
        self.assertEqual(moe.switch_mlp.fc2.bits, 2)
        self.assertEqual(moe.switch_mlp.fc1.group_size, 32)
        self.assertEqual(moe.switch_mlp.fc2.group_size, 32)

    def test_mismatching_bits_raises_value_error(self):
        config = self.base_config.copy()
        config["quantization"] = {
            "backbone.layers.1.mixer.switch_mlp.fc1": {"bits": 2, "group_size": 32, "mode": "affine"},
            "backbone.layers.1.mixer.switch_mlp.fc2": {"bits": 3, "group_size": 32, "mode": "affine"},
            "bits": 4,
            "group_size": 64
        }
        args = ModelArgs.from_dict(config)
        with self.assertRaises(ValueError) as ctx:
            NemotronHMoE(args, layer_idx=1)
        self.assertIn("matching fc1/fc2 quantization config", str(ctx.exception))

    def test_mismatching_group_size_raises_value_error(self):
        config = self.base_config.copy()
        config["quantization"] = {
            "backbone.layers.1.mixer.switch_mlp.fc1": {"bits": 2, "group_size": 32, "mode": "affine"},
            "backbone.layers.1.mixer.switch_mlp.fc2": {"bits": 2, "group_size": 64, "mode": "affine"},
            "bits": 4,
            "group_size": 64
        }
        args = ModelArgs.from_dict(config)
        with self.assertRaises(ValueError) as ctx:
            NemotronHMoE(args, layer_idx=1)
        self.assertIn("matching fc1/fc2 quantization config", str(ctx.exception))

    def test_fallback_to_global_config_if_missing(self):
        config = self.base_config.copy()
        config["quantization"] = {
            "bits": 4,
            "group_size": 64
        }
        args = ModelArgs.from_dict(config)
        # Should not raise, falls back to global
        moe = NemotronHMoE(args, layer_idx=1)
        self.assertEqual(moe.switch_mlp.fc1.bits, 4)
        self.assertEqual(moe.switch_mlp.fc2.bits, 4)
        self.assertEqual(moe.switch_mlp.fc1.group_size, 64)
        self.assertEqual(moe.switch_mlp.fc2.group_size, 64)

if __name__ == "__main__":
    unittest.main()
