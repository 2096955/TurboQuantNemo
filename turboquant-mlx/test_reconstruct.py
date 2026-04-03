import mlx.core as mx
from mlx_turboquant import TurboQuantCompressor
import numpy as np

compressor = TurboQuantCompressor(bit_width=4, head_dim=128, seed=42)
x = mx.random.normal((10, 128))
rotation, _ = np.linalg.qr(np.random.normal(size=(128, 128)))
rotation = mx.array(rotation, dtype=mx.float32)

compressed = compressor.compress(x, rotation)

reconstructed = compressor.centroids[compressed["indices"]]

diff = mx.max(mx.abs(compressed["x_rot_quant"] - reconstructed)).item()
print("Max diff:", diff)

