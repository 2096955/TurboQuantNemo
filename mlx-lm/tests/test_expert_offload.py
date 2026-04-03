# Copyright © 2026 Apple Inc.

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx

from mlx_lm.expert_offload import (
    ExpertLoadError,
    ExpertOffloadManager,
    build_nemotron_expert_key_table,
    http_status_for_generation_failure,
    is_nemotron_routed_expert_weight_key,
    parse_nemotron_expert_key,
)
from mlx_lm.expert_weight_loader import (
    load_non_expert_weights,
    resolve_weight_map,
)


class ExpertOffloadHttpStatusTest(unittest.TestCase):
    def test_generation_failure_maps_expert_load_and_memory_to_503(self):
        self.assertEqual(http_status_for_generation_failure(ExpertLoadError("x")), 503)
        self.assertEqual(http_status_for_generation_failure(MemoryError()), 503)
        self.assertEqual(http_status_for_generation_failure(ValueError("missing")), 500)


class ExpertOffloadParseTest(unittest.TestCase):
    def test_parse_expert_key(self):
        self.assertEqual(
            parse_nemotron_expert_key(
                "backbone.layers.2.mixer.experts.5.down_proj.weight"
            ),
            (2, 5, "down_proj", "weight"),
        )
        self.assertIsNone(
            parse_nemotron_expert_key("backbone.layers.2.mixer.gate.weight")
        )

    def test_is_routed_key(self):
        self.assertTrue(
            is_nemotron_routed_expert_weight_key(
                "backbone.layers.0.mixer.experts.0.up_proj.weight"
            )
        )
        self.assertFalse(is_nemotron_routed_expert_weight_key("lm_head.weight"))


def _fake_expert_tensors():
    return {"fc1_weight": mx.zeros((4, 8)), "fc2_weight": mx.zeros((4, 8))}


class ExpertOffloadManagerTest(unittest.TestCase):
    def test_key_table(self):
        wm = {
            "backbone.layers.0.mixer.experts.0.up_proj.weight": "m.safetensors",
            "backbone.layers.0.mixer.experts.0.down_proj.weight": "m.safetensors",
            "backbone.layers.0.mixer.experts.1.up_proj.weight": "m.safetensors",
            "backbone.layers.0.mixer.experts.1.down_proj.weight": "m.safetensors",
        }
        tbl = build_nemotron_expert_key_table(wm)
        self.assertIn((0, 0), tbl)
        self.assertEqual(set(tbl[(0, 0)].keys()), {"fc1_weight", "fc2_weight"})

    @patch.object(ExpertOffloadManager, "_load_expert_pair_tensors")
    def test_prepare_gather_stacks(self, mock_load):
        mock_load.return_value = _fake_expert_tensors()

        wm = {
            "backbone.layers.0.mixer.experts.0.up_proj.weight": "m.safetensors",
            "backbone.layers.0.mixer.experts.0.down_proj.weight": "m.safetensors",
            "backbone.layers.0.mixer.experts.1.up_proj.weight": "m.safetensors",
            "backbone.layers.0.mixer.experts.1.down_proj.weight": "m.safetensors",
        }
        tbl = build_nemotron_expert_key_table(wm)
        mgr = ExpertOffloadManager(
            base_path=Path("/tmp"),
            weight_map=wm,
            expert_key_table=tbl,
            max_resident_experts=4,
        )
        idx = mx.array([[0, 1]], dtype=mx.int32)
        c, r = mgr.prepare_gather(0, "fc1", idx)
        mx.eval(c, r)
        self.assertEqual(tuple(c.shape), (2, 4, 8))
        self.assertEqual(mgr.misses, 2)

    @patch.object(ExpertOffloadManager, "_load_expert_pair_tensors")
    def test_prepare_gather_loop_mode_alias(self, mock_load):
        mock_load.return_value = _fake_expert_tensors()
        wm = {
            "backbone.layers.0.mixer.experts.0.up_proj.weight": "m.safetensors",
            "backbone.layers.0.mixer.experts.0.down_proj.weight": "m.safetensors",
        }
        tbl = build_nemotron_expert_key_table(wm)
        mgr = ExpertOffloadManager(
            base_path=Path("/tmp"),
            weight_map=wm,
            expert_key_table=tbl,
            max_resident_experts=4,
        )
        idx = mx.array([[0]], dtype=mx.int32)
        c, r = mgr.prepare_gather(0, "fc1", idx, mode="loop")
        mx.eval(c, r)
        self.assertEqual(tuple(c.shape), (1, 4, 8))

    @patch.object(ExpertOffloadManager, "_load_expert_pair_tensors")
    def test_prepare_gather_pair_returns_fc1_fc2_remapped(self, mock_load):
        mock_load.return_value = _fake_expert_tensors()
        wm = {
            "backbone.layers.0.mixer.experts.0.up_proj.weight": "m.safetensors",
            "backbone.layers.0.mixer.experts.0.down_proj.weight": "m.safetensors",
            "backbone.layers.0.mixer.experts.1.up_proj.weight": "m.safetensors",
            "backbone.layers.0.mixer.experts.1.down_proj.weight": "m.safetensors",
        }
        tbl = build_nemotron_expert_key_table(wm)
        mgr = ExpertOffloadManager(
            base_path=Path("/tmp"),
            weight_map=wm,
            expert_key_table=tbl,
            max_resident_experts=4,
        )
        idx = mx.array([[0, 1]], dtype=mx.int32)
        c1, c2, r = mgr.prepare_gather_pair(0, idx)
        mx.eval(c1, c2, r)
        self.assertEqual(tuple(c1.shape), (2, 4, 8))
        self.assertEqual(tuple(c2.shape), (2, 4, 8))
        self.assertEqual(tuple(r.shape), (1, 2))

    @patch.object(ExpertOffloadManager, "_load_expert_pair_tensors")
    def test_eviction_when_cache_full(self, mock_load):
        mock_load.return_value = _fake_expert_tensors()
        wm = {
            "backbone.layers.0.mixer.experts.0.up_proj.weight": "m.safetensors",
            "backbone.layers.0.mixer.experts.0.down_proj.weight": "m.safetensors",
            "backbone.layers.0.mixer.experts.1.up_proj.weight": "m.safetensors",
            "backbone.layers.0.mixer.experts.1.down_proj.weight": "m.safetensors",
            "backbone.layers.0.mixer.experts.2.up_proj.weight": "m.safetensors",
            "backbone.layers.0.mixer.experts.2.down_proj.weight": "m.safetensors",
        }
        tbl = build_nemotron_expert_key_table(wm)
        mgr = ExpertOffloadManager(
            base_path=Path("/tmp"),
            weight_map=wm,
            expert_key_table=tbl,
            max_resident_experts=1,
        )
        mgr.prepare_gather(0, "fc1", mx.array([[0]], dtype=mx.int32))
        self.assertEqual(mgr.evictions, 0)
        mgr.prepare_gather(0, "fc1", mx.array([[1]], dtype=mx.int32))
        self.assertEqual(mgr.evictions, 1)
        self.assertEqual(mgr.misses, 2)

    @patch.object(ExpertOffloadManager, "_load_expert_pair_tensors")
    def test_stats_summary_concurrent_with_prepare_gather(self, mock_load):
        mock_load.return_value = _fake_expert_tensors()
        wm = {
            "backbone.layers.0.mixer.experts.0.up_proj.weight": "m.safetensors",
            "backbone.layers.0.mixer.experts.0.down_proj.weight": "m.safetensors",
            "backbone.layers.0.mixer.experts.1.up_proj.weight": "m.safetensors",
            "backbone.layers.0.mixer.experts.1.down_proj.weight": "m.safetensors",
        }
        tbl = build_nemotron_expert_key_table(wm)
        mgr = ExpertOffloadManager(
            base_path=Path("/tmp"),
            weight_map=wm,
            expert_key_table=tbl,
            max_resident_experts=4,
        )
        errors: list = []

        def spin_stats():
            try:
                for _ in range(200):
                    mgr.stats_summary()
            except Exception as e:
                errors.append(e)

        th = threading.Thread(target=spin_stats)
        th.start()
        for _ in range(5):
            mgr.prepare_gather(0, "fc1", mx.array([[0, 1]], dtype=mx.int32))
        th.join(timeout=10.0)
        self.assertFalse(errors)
        self.assertFalse(th.is_alive())

    def test_set_phase_increments_prefill_decode_counters(self):
        wm = {
            "backbone.layers.0.mixer.experts.0.up_proj.weight": "m.safetensors",
            "backbone.layers.0.mixer.experts.0.down_proj.weight": "m.safetensors",
        }
        tbl = build_nemotron_expert_key_table(wm)
        mgr = ExpertOffloadManager(
            base_path=Path("/tmp"),
            weight_map=wm,
            expert_key_table=tbl,
            max_resident_experts=4,
        )
        with patch.object(
            ExpertOffloadManager, "_load_expert_pair_tensors"
        ) as mock_load:
            mock_load.return_value = _fake_expert_tensors()
            mgr.set_phase("prefill")
            mgr.prepare_gather(0, "fc1", mx.array([[0]], dtype=mx.int32))
            mgr.prepare_gather(0, "fc1", mx.array([[0]], dtype=mx.int32))
            mgr.set_phase("decode")
            mgr.prepare_gather(0, "fc1", mx.array([[0]], dtype=mx.int32))
        self.assertEqual(mgr.prefill_evals, 2)
        self.assertEqual(mgr.decode_evals, 1)
        s = mgr.stats_summary()
        self.assertEqual(s["prefill_gather_calls"], 2)
        self.assertEqual(s["decode_gather_calls"], 1)
        self.assertEqual(s["prefill_hits"], 1)
        self.assertEqual(s["prefill_misses"], 1)
        self.assertEqual(s["decode_hits"], 1)
        self.assertEqual(s["decode_misses"], 0)
        self.assertEqual(s["prefill_hit_rate"], 0.5)
        self.assertEqual(s["decode_hit_rate"], 1.0)


class ExpertWeightLoaderTest(unittest.TestCase):
    def test_load_non_expert_weights_empty_map_raises(self):
        with self.assertRaises(ValueError) as ctx:
            load_non_expert_weights(Path("/tmp/nonexistent"), {})
        self.assertIn("empty weight_map", str(ctx.exception).lower())

    def test_resolve_weight_map_missing_shard_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = {
                "weight_map": {
                    "lm_head.weight": "missing_shard.safetensors",
                }
            }
            (root / "model.safetensors.index.json").write_text(
                json.dumps(index), encoding="utf-8"
            )
            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_weight_map(root)
            self.assertIn("missing shard", str(ctx.exception).lower())

    def test_is_nemotron_expert_key_matches_scales_biases(self):
        self.assertTrue(
            is_nemotron_routed_expert_weight_key(
                "backbone.layers.5.mixer.experts.3.up_proj.scales"
            )
        )
        self.assertTrue(
            is_nemotron_routed_expert_weight_key(
                "backbone.layers.5.mixer.experts.3.down_proj.biases"
            )
        )

    def test_manager_is_quantized_property(self):
        wm = {
            "backbone.layers.0.mixer.experts.0.up_proj.weight": "m.safetensors",
            "backbone.layers.0.mixer.experts.0.up_proj.scales": "m.safetensors",
        }
        tbl = build_nemotron_expert_key_table(wm)
        mgr = ExpertOffloadManager(
            base_path=Path("/tmp"),
            weight_map=wm,
            expert_key_table=tbl,
            max_resident_experts=4,
        )
        self.assertTrue(mgr.is_quantized)

    def test_gather_qmm_compact_subset(self):
        from mlx_lm.models.switch_layers import QuantizedSwitchLinear

        sl = QuantizedSwitchLinear(
            input_dims=32, output_dims=64, num_experts=4, bits=4, group_size=32
        )
        x = mx.random.normal((1, 2, 32))

        compact_w = sl.weight[mx.array([0, 2])]
        compact_s = sl.scales[mx.array([0, 2])]
        compact_b = sl.biases[mx.array([0, 2])] if sl.biases is not None else None
        remapped = mx.array([[0, 1]])  # 0->0, 2->1

        result = mx.gather_qmm(
            x,
            compact_w,
            compact_s,
            compact_b,
            rhs_indices=remapped,
            transpose=True,
            group_size=32,
            bits=4,
        )
        mx.eval(result)
        self.assertEqual(result.shape, (1, 2, 2, 64))

    @patch.object(ExpertOffloadManager, "_load_expert_pair_tensors")
    def test_prepare_gather_pair_quantized(self, mock_load):
        def fake_load(spec):
            res = {}
            for k in spec:
                if (
                    k.endswith("_weight")
                    or k.endswith("_scales")
                    or k.endswith("_biases")
                ):
                    res[k] = mx.zeros((4, 8))
            return res

        mock_load.side_effect = fake_load

        wm = {
            "backbone.layers.0.mixer.experts.0.up_proj.weight": "m.safetensors",
            "backbone.layers.0.mixer.experts.0.up_proj.scales": "m.safetensors",
            "backbone.layers.0.mixer.experts.0.down_proj.weight": "m.safetensors",
            "backbone.layers.0.mixer.experts.0.down_proj.scales": "m.safetensors",
        }
        tbl = build_nemotron_expert_key_table(wm)
        mgr = ExpertOffloadManager(
            base_path=Path("/tmp"),
            weight_map=wm,
            expert_key_table=tbl,
            max_resident_experts=4,
        )
        idx = mx.array([[0]], dtype=mx.int32)
        c1, c2, r = mgr.prepare_gather_pair_quantized(0, idx)

        c1_w, c1_s, c1_b = c1
        mx.eval(c1_w, c1_s, r)
        self.assertEqual(tuple(c1_w.shape), (1, 4, 8))
        self.assertEqual(tuple(c1_s.shape), (1, 4, 8))
        self.assertIsNone(c1_b)
        self.assertEqual(tuple(r.shape), (1, 1))


if __name__ == "__main__":
    unittest.main()
