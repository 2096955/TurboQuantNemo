import json
import importlib
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.expert_offload import ExpertOffloadManager, build_nemotron_expert_key_table
from mlx_lm.expert_weight_loader import load_non_expert_weights
from mlx_lm.models.nemotron_h import ModelArgs, Model
from mlx_lm.repack_experts import repack_checkpoint
from mlx_lm.utils import load_model


class TestQuantizedOffloadEndToEnd(unittest.TestCase):
    def test_load_non_expert_weights_preserves_uint32_dtype(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            shard_name = "model.safetensors"
            weights = {
                "backbone.layers.0.mixer.in_proj.weight": mx.array(
                    [1, 2, 3], dtype=mx.uint32
                ),
                "backbone.layers.0.mixer.experts.0.up_proj.weight": mx.array(
                    [4, 5, 6], dtype=mx.uint32
                ),
            }
            mx.save_safetensors(str(model_path / shard_name), weights)
            weight_map = {k: shard_name for k in weights}

            loaded = load_non_expert_weights(model_path, weight_map)

            self.assertEqual(
                loaded["backbone.layers.0.mixer.in_proj.weight"].dtype, mx.uint32
            )
            self.assertNotIn(
                "backbone.layers.0.mixer.experts.0.up_proj.weight", loaded
            )

    def test_expert_offload_manager_preserves_quantized_uint32_dtype(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            shard_name = "model.safetensors"
            weights = {
                "backbone.layers.0.mixer.experts.0.up_proj.weight": mx.array(
                    [[1, 2], [3, 4]], dtype=mx.uint32
                ),
                "backbone.layers.0.mixer.experts.0.up_proj.scales": mx.ones(
                    (2, 1), dtype=mx.float16
                ),
                "backbone.layers.0.mixer.experts.0.down_proj.weight": mx.array(
                    [[5, 6], [7, 8]], dtype=mx.uint32
                ),
                "backbone.layers.0.mixer.experts.0.down_proj.scales": mx.ones(
                    (2, 1), dtype=mx.float16
                ),
            }
            mx.save_safetensors(str(model_path / shard_name), weights)
            weight_map = {k: shard_name for k in weights}
            key_table = build_nemotron_expert_key_table(weight_map)
            mgr = ExpertOffloadManager(
                model_path,
                weight_map,
                key_table,
                max_resident_experts=1,
            )

            loaded = mgr._load_expert_pair_tensors(key_table[(0, 0)])

            self.assertEqual(loaded["fc1_weight"].dtype, mx.uint32)
            self.assertEqual(loaded["fc2_weight"].dtype, mx.uint32)

    def test_nemotron_time_step_limit_defaults_to_hf_behavior(self):
        args = ModelArgs.from_dict(
            {
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
                "time_step_min": 0.001,
                "time_step_max": 0.1,
            }
        )

        self.assertEqual(args.time_step_limit[0], 0.0)
        self.assertTrue(args.time_step_limit[1] == float("inf"))

    def test_mock_quantized_model_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)

            # Create a tiny mock config
            config = {
                "model_type": "nemotron_h",
                "vocab_size": 256,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,  # Layer 0 is M, Layer 1 is E (has switch_mlp)
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
                "quantization": {"group_size": 32, "bits": 4, "mode": "affine"},
            }

            # Write config.json
            with open(model_path / "config.json", "w") as f:
                json.dump(config, f)

            # Create dummy model
            model_args = ModelArgs.from_dict(config)
            model = Model(model_args)

            # Quantize it (similar to what mlx_lm.utils._quantize does)
            nn.quantize(model, group_size=32, bits=4)

            # We want to save this model's weights into a safetensor, but repack the switch_mlp experts
            from mlx.utils import tree_flatten

            weights = dict(tree_flatten(model.parameters()))

            # Repack layer 1 switch_mlp manually for the test
            new_weights = {}
            for k, v in weights.items():
                if "switch_mlp" in k:
                    # k is e.g. "layers.1.mixer.switch_mlp.fc1.weight"
                    parts = k.split(".")
                    l_idx = parts[2]
                    proj = parts[5]
                    suffix = parts[6]
                    orig_proj = "up_proj" if proj == "fc1" else "down_proj"

                    for e in range(model_args.n_routed_experts):
                        expert_key = f"backbone.layers.{l_idx}.mixer.experts.{e}.{orig_proj}.{suffix}"
                        new_weights[expert_key] = v[e]
                else:
                    new_weights[k] = v

            # Save the safetensors
            shard_name = "model.safetensors"
            mx.save_safetensors(str(model_path / shard_name), new_weights)

            # Create index
            index = {"weight_map": {k: shard_name for k in new_weights.keys()}}
            with open(model_path / "model.safetensors.index.json", "w") as f:
                json.dump(index, f)

            # Now, attempt to load the model with expert offload enabled
            model_config = {"expert_offload": True, "max_resident_experts": 2}

            offload_model, _ = load_model(
                model_path, lazy=True, model_config=model_config
            )

            # Compare offload vs non-offload execution from the same checkpoint.
            x = mx.array([[3, 11, 7, 5]], dtype=mx.int32)
            offload_out = offload_model(x)
            eager_out = model(x)
            if isinstance(offload_out, tuple):
                offload_out = offload_out[0]
            if isinstance(eager_out, tuple):
                eager_out = eager_out[0]

            mx.eval(offload_out, eager_out)
            self.assertTrue(mx.allclose(offload_out, eager_out, rtol=1e-4, atol=1e-4))

    def test_convert_then_repack_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "mlx_model"
            convert_module = importlib.import_module("mlx_lm.convert")
            cfg = {
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
            }
            model = Model(ModelArgs.from_dict(cfg))

            def fake_load(*args, **kwargs):
                return model, object(), cfg

            def fake_save(mlx_path, hf_path, model_obj, tokenizer, config):
                mlx_path = Path(mlx_path)
                mlx_path.mkdir(parents=True, exist_ok=False)
                weights = dict(tree_flatten(model_obj.parameters()))
                shard = "model.safetensors"
                mx.save_safetensors(str(mlx_path / shard), weights)
                (mlx_path / "model.safetensors.index.json").write_text(
                    json.dumps({"weight_map": {k: shard for k in weights}}),
                    encoding="utf-8",
                )
                (mlx_path / "config.json").write_text(
                    json.dumps(config), encoding="utf-8"
                )

            with (
                patch.object(convert_module, "load", side_effect=fake_load),
                patch.object(convert_module, "save", side_effect=fake_save),
            ):
                convert_module.convert(
                    hf_path="fake-hf-repo",
                    mlx_path=str(out_path),
                    quantize=True,
                    q_bits=4,
                    q_group_size=32,
                    mixed_expert_bits=2,
                )

            repack_checkpoint(out_path)
            index = json.loads(
                (out_path / "model.safetensors.index.json").read_text(encoding="utf-8")
            )["weight_map"]
            self.assertTrue(
                any(".mixer.experts.0.up_proj.weight" in k for k in index.keys())
            )
            self.assertFalse(any(".switch_mlp." in k for k in index.keys()))
            self.assertTrue(any(v.startswith("repacked-") for v in index.values()))


if __name__ == "__main__":
    unittest.main()
