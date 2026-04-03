import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
from mlx_lm.repack_experts import main as repack_main


class TestRepackExperts(unittest.TestCase):
    def test_repack_experts_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)

            # Create dummy config
            config = {
                "model_type": "nemotron_h",
                "n_routed_experts": 2,
            }
            with open(model_path / "config.json", "w") as f:
                json.dump(config, f)

            # Create stacked switch_mlp tensors mimicking mlx_lm.convert output
            stacked_fc1_weight = mx.zeros((2, 4, 8))
            stacked_fc1_weight[0] = mx.ones((4, 8))
            stacked_fc1_weight[1] = mx.ones((4, 8)) * 2

            stacked_fc1_scales = mx.zeros((2, 4, 1))
            stacked_fc1_scales[0] = mx.ones((4, 1)) * 0.5
            stacked_fc1_scales[1] = mx.ones((4, 1)) * 1.5

            stacked_fc1_biases = mx.zeros((2, 4))
            stacked_fc1_biases[0] = mx.ones((4,)) * 0.1
            stacked_fc1_biases[1] = mx.ones((4,)) * 0.2

            stacked_fc2_weight = mx.zeros((2, 8, 4))
            stacked_fc2_weight[0] = mx.ones((8, 4)) * 3
            stacked_fc2_weight[1] = mx.ones((8, 4)) * 4

            stacked_fc2_scales = mx.zeros((2, 8, 1))
            stacked_fc2_scales[0] = mx.ones((8, 1)) * 0.3
            stacked_fc2_scales[1] = mx.ones((8, 1)) * 0.4

            stacked_fc2_biases = mx.zeros((2, 8))
            stacked_fc2_biases[0] = mx.ones((8,)) * 0.05
            stacked_fc2_biases[1] = mx.ones((8,)) * 0.06

            # Shard 1: fc1 weight + scales + biases
            weights_1 = {
                "backbone.layers.0.mixer.switch_mlp.fc1.weight": stacked_fc1_weight,
                "backbone.layers.0.mixer.switch_mlp.fc1.scales": stacked_fc1_scales,
                "backbone.layers.0.mixer.switch_mlp.fc1.biases": stacked_fc1_biases,
                "backbone.embeddings.weight": mx.zeros((10, 8)),
            }

            # Shard 2: fc2 weight + scales + biases
            weights_2 = {
                "backbone.layers.0.mixer.switch_mlp.fc2.weight": stacked_fc2_weight,
                "backbone.layers.0.mixer.switch_mlp.fc2.scales": stacked_fc2_scales,
                "backbone.layers.0.mixer.switch_mlp.fc2.biases": stacked_fc2_biases,
                "lm_head.weight": mx.zeros((10, 4)),
            }

            shard_name_1 = "model-00001-of-00002.safetensors"
            shard_name_2 = "model-00002-of-00002.safetensors"
            mx.save_safetensors(str(model_path / shard_name_1), weights_1)
            mx.save_safetensors(str(model_path / shard_name_2), weights_2)

            # Create index pointing to the stacked tensor
            index = {
                "weight_map": {
                    "backbone.layers.0.mixer.switch_mlp.fc1.weight": shard_name_1,
                    "backbone.layers.0.mixer.switch_mlp.fc1.scales": shard_name_1,
                    "backbone.layers.0.mixer.switch_mlp.fc1.biases": shard_name_1,
                    "backbone.embeddings.weight": shard_name_1,
                    "backbone.layers.0.mixer.switch_mlp.fc2.weight": shard_name_2,
                    "backbone.layers.0.mixer.switch_mlp.fc2.scales": shard_name_2,
                    "backbone.layers.0.mixer.switch_mlp.fc2.biases": shard_name_2,
                    "lm_head.weight": shard_name_2,
                }
            }
            with open(model_path / "model.safetensors.index.json", "w") as f:
                json.dump(index, f)

            # Run the repacker script by mocking sys.argv
            import sys
            from unittest.mock import patch

            test_args = ["repack_experts.py", "--model", str(model_path)]
            with patch.object(sys, "argv", test_args):
                repack_main()

            # Verify the resulting index
            with open(model_path / "model.safetensors.index.json", "r") as f:
                new_index = json.load(f)

            wm = new_index.get("weight_map", {})
            self.assertIn("backbone.embeddings.weight", wm)
            self.assertIn("lm_head.weight", wm)
            self.assertNotIn("backbone.layers.0.mixer.switch_mlp.fc1.weight", wm)
            self.assertNotIn("backbone.layers.0.mixer.switch_mlp.fc1.scales", wm)
            self.assertNotIn("backbone.layers.0.mixer.switch_mlp.fc1.biases", wm)
            self.assertNotIn("backbone.layers.0.mixer.switch_mlp.fc2.weight", wm)
            self.assertNotIn("backbone.layers.0.mixer.switch_mlp.fc2.scales", wm)
            self.assertNotIn("backbone.layers.0.mixer.switch_mlp.fc2.biases", wm)
            self.assertIn("backbone.layers.0.mixer.experts.0.up_proj.weight", wm)
            self.assertIn("backbone.layers.0.mixer.experts.1.up_proj.scales", wm)
            self.assertIn("backbone.layers.0.mixer.experts.1.up_proj.biases", wm)
            self.assertIn("backbone.layers.0.mixer.experts.0.down_proj.weight", wm)
            self.assertIn("backbone.layers.0.mixer.experts.0.down_proj.scales", wm)
            self.assertIn("backbone.layers.0.mixer.experts.1.down_proj.biases", wm)

            # Verify resulting shard 1
            new_weights_1 = dict(mx.load(str(model_path / f"repacked-{shard_name_1}")))
            self.assertIn("backbone.embeddings.weight", new_weights_1)
            self.assertNotIn(
                "backbone.layers.0.mixer.switch_mlp.fc1.weight", new_weights_1
            )

            exp0_w = new_weights_1["backbone.layers.0.mixer.experts.0.up_proj.weight"]
            exp1_s = new_weights_1["backbone.layers.0.mixer.experts.1.up_proj.scales"]
            exp1_b = new_weights_1["backbone.layers.0.mixer.experts.1.up_proj.biases"]

            self.assertTrue(mx.array_equal(exp0_w, mx.ones((4, 8))))
            self.assertTrue(mx.array_equal(exp1_s, mx.ones((4, 1)) * 1.5))
            self.assertTrue(mx.array_equal(exp1_b, mx.ones((4,)) * 0.2))

            # Verify resulting shard 2
            new_weights_2 = dict(mx.load(str(model_path / f"repacked-{shard_name_2}")))
            self.assertIn("lm_head.weight", new_weights_2)
            self.assertNotIn(
                "backbone.layers.0.mixer.switch_mlp.fc2.weight", new_weights_2
            )

            self.assertNotIn(
                "backbone.layers.0.mixer.switch_mlp.fc2.scales", new_weights_2
            )
            self.assertNotIn(
                "backbone.layers.0.mixer.switch_mlp.fc2.biases", new_weights_2
            )

            exp0_fc2 = new_weights_2[
                "backbone.layers.0.mixer.experts.0.down_proj.weight"
            ]
            exp1_fc2_s = new_weights_2[
                "backbone.layers.0.mixer.experts.1.down_proj.scales"
            ]
            exp0_fc2_b = new_weights_2[
                "backbone.layers.0.mixer.experts.0.down_proj.biases"
            ]

            self.assertTrue(mx.array_equal(exp0_fc2, mx.ones((8, 4)) * 3))
            self.assertTrue(mx.array_equal(exp1_fc2_s, mx.ones((8, 1)) * 0.4))
            self.assertTrue(mx.array_equal(exp0_fc2_b, mx.ones((8,)) * 0.05))


if __name__ == "__main__":
    unittest.main()
