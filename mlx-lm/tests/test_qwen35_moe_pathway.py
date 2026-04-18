import json
import subprocess
import sys
import unittest
from pathlib import Path


def _require_mlx_lm_runtime(test_case):
    proc = subprocess.run(
        [sys.executable, "-c", "import mlx_lm"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        output = (proc.stdout or "") + (proc.stderr or "")
        test_case.skipTest(
            f"MLX runtime unavailable in this environment: {output[:240]}"
        )


class TestQwen35MoeRepacking(unittest.TestCase):
    def test_repack_accepts_qwen3_5_moe_model_type(self):
        _require_mlx_lm_runtime(self)
        from mlx_lm.repack_experts import _repack_supported_types

        self.assertIn("qwen3_5_moe", _repack_supported_types)


class TestQwen35MoeOffload(unittest.TestCase):
    def test_offload_type_sets_include_qwen3_5_moe(self):
        _require_mlx_lm_runtime(self)
        from mlx_lm.expert_offload import EXPERT_OFFLOAD_MODEL_TYPES, MOE_MODEL_TYPES

        self.assertIn("qwen3_5_moe", EXPERT_OFFLOAD_MODEL_TYPES)
        self.assertIn("qwen3_5_moe", MOE_MODEL_TYPES)

    def test_parse_expert_key_qwen3_5_moe(self):
        _require_mlx_lm_runtime(self)
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
        _require_mlx_lm_runtime(self)
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


class TestSharedExpertQuantPredicate(unittest.TestCase):
    def _run_shared_predicate_case(self, shared_expert_bits):
        script = f"""
import json
import mlx.nn as nn
from mlx_lm.convert import _build_mixed_expert_quant_predicate

predicate = _build_mixed_expert_quant_predicate(
    mixed_expert_bits=2,
    shared_expert_bits={shared_expert_bits!r},
    default_bits=4,
    default_group_size=64,
    mode="affine",
)

result = {{
    "shared_proj": predicate("model.layers.0.mlp.shared_expert.gate_proj", nn.Linear(16, 16)),
    "mlp_gate": predicate("model.layers.0.mlp.gate", nn.Linear(16, 8)),
    "shared_gate": predicate("model.layers.0.mlp.shared_expert_gate", nn.Linear(16, 8)),
    "dense_proj": predicate("model.layers.0.mlp.down_proj", nn.Linear(16, 16)),
}}
print(json.dumps(result))
"""
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            output = (proc.stdout or "") + (proc.stderr or "")
            self.skipTest(
                f"MLX runtime unavailable in this environment: {output[:240]}"
            )
        return json.loads(proc.stdout)

    def test_shared_expert_gets_higher_bits(self):
        result = self._run_shared_predicate_case(8)

        self.assertEqual(
            result["shared_proj"],
            {"bits": 8, "group_size": 64, "mode": "affine"},
        )
        self.assertEqual(
            result["mlp_gate"],
            {"bits": 8, "group_size": 64, "mode": "affine"},
        )
        self.assertEqual(
            result["shared_gate"],
            {"bits": 8, "group_size": 64, "mode": "affine"},
        )
        self.assertEqual(
            result["dense_proj"],
            {"bits": 4, "group_size": 64, "mode": "affine"},
        )

    def test_shared_expert_bits_defaults_to_default_bits(self):
        result = self._run_shared_predicate_case(None)

        self.assertEqual(
            result["shared_proj"],
            {"bits": 4, "group_size": 64, "mode": "affine"},
        )


class TestIsoQuantDeltaNetSkip(unittest.TestCase):
    """Gate G3: IsoQuant must NOT replace ArraysCache (DeltaNet layers)."""

    def test_deltanet_layers_keep_arrays_cache(self):
        _require_mlx_lm_runtime(self)
        from mlx_lm.models.cache import KVCache, ArraysCache, _replace_attention_caches

        # Simulate Qwen3.5 cache pattern: 3 DeltaNet + 1 Attention, repeated
        caches = []
        for i in range(8):
            if (i + 1) % 4 == 0:  # full attention layer
                caches.append(KVCache())
            else:  # DeltaNet layer
                caches.append(ArraysCache(size=2))

        def make_quant_cache(layer_idx):
            return f"quant_cache_{layer_idx}"

        layer_idx_ref = [0]
        replaced = []
        for c in caches:
            result = _replace_attention_caches(
                c,
                make_quant_cache=make_quant_cache,
                skip_layers=0,
                layer_idx=None,
                layer_idx_ref=layer_idx_ref,
            )
            replaced.append(result)

        # ArraysCache (DeltaNet) must be untouched; KVCache (attention) must be replaced
        for i, (original, result) in enumerate(zip(caches, replaced)):
            if isinstance(original, ArraysCache):
                self.assertIs(
                    result, original, f"Layer {i} (DeltaNet) must keep ArraysCache"
                )
            else:
                self.assertIsInstance(
                    result, str, f"Layer {i} (attention) must be replaced"
                )
                self.assertTrue(
                    result.startswith("quant_cache_"),
                    f"Layer {i} (attention) must be replaced with quant cache",
                )


if __name__ == "__main__":
    unittest.main()
