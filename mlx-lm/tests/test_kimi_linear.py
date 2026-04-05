import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.kimi_linear import BlockAttnRes

def test_block_attn_res():
    dim = 64
    module = BlockAttnRes(dim)
    blocks = [mx.random.normal((2, 10, dim)) for _ in range(3)]
    partial = mx.random.normal((2, 10, dim))
    out, alpha = module(blocks, partial)
    assert out.shape == (2, 10, dim)
    assert alpha.shape == (4, 2, 10)
