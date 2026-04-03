import unittest

import mlx.core as mx

from mlx_lm.models.switch_layers import QuantizedSwitchLinear


class GatherQmmCompactSubsetTest(unittest.TestCase):
    def test_compact_subset_matches_full_expert_tensor(self):
        layer = QuantizedSwitchLinear(
            input_dims=32,
            output_dims=24,
            num_experts=4,
            bits=4,
            group_size=32,
            bias=False,
        )
        x = mx.random.normal((1, 3, 32))
        full_indices = mx.array([[0, 2, 2]], dtype=mx.int32)
        compact_indices = mx.array([0, 2], dtype=mx.int32)
        remapped = mx.array([[0, 1, 1]], dtype=mx.int32)

        y_full = mx.gather_qmm(
            x,
            layer.weight,
            layer.scales,
            layer.biases,
            rhs_indices=full_indices,
            transpose=True,
            group_size=layer.group_size,
            bits=layer.bits,
            mode=layer.mode,
        )
        y_compact = mx.gather_qmm(
            x,
            layer.weight[compact_indices],
            layer.scales[compact_indices],
            layer.biases[compact_indices] if layer.biases is not None else None,
            rhs_indices=remapped,
            transpose=True,
            group_size=layer.group_size,
            bits=layer.bits,
            mode=layer.mode,
        )
        mx.eval(y_full, y_compact)
        self.assertTrue(mx.allclose(y_full, y_compact, rtol=1e-5, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
