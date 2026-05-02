import mlx.core as mx
import unittest
from mlx_lm.models.mlx_turboquant import RotorQuantCompressor


class TestRotorQuantCompressor(unittest.TestCase):
    def test_rotorquant_compressor(self):
        head_dim = 128
        # Using 4 bit as it's typically present in codebooks directory
        compressor = RotorQuantCompressor(bit_width=4, head_dim=head_dim)

        # 1. Test compression forward pass
        x = mx.random.normal((2, 10, head_dim))

        # Initialize some valid rotors (norm = 1)
        raw_rotors = mx.random.normal((compressor.num_chunks, 4))
        rotors = raw_rotors / mx.linalg.norm(raw_rotors, axis=-1, keepdims=True)

        compressed = compressor.compress(x, rotors)
        self.assertIn("indices", compressed)
        self.assertIn("x_norm", compressed)
        self.assertIn("x_rot_quant", compressed)

        # 2. Test shapes
        # Indices are stored at padded_dim width (ceil(head_dim/3)*3)
        self.assertEqual(compressed["indices"].shape, (2, 10, compressor.padded_dim))
        self.assertEqual(compressed["x_norm"].shape, (2, 10, 1))

        # 3. Test decompression
        decompressed = compressor.decompress(
            compressed["indices"],
            compressed["indices"].shape,
            compressed["x_norm"],
            rotors,
        )
        self.assertEqual(decompressed.shape, (2, 10, head_dim))


if __name__ == "__main__":
    unittest.main()
