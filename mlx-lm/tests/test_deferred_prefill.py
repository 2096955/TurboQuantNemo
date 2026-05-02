import pytest
import mlx.core as mx
from mlx_lm.models.mlx_turboquant import RotorQuantKVCache

def test_deferred_prefill_compression():
    cache = RotorQuantKVCache(head_dim=128, bit_width=3)
    keys = mx.random.normal((1, 4, 2048, 128))
    values = mx.random.normal((1, 4, 2048, 128))
    
    # Simulate prefill without compression (seq_len > 1 infers prefill)
    cache.update_and_fetch(keys, values)
    assert cache.is_deferred == True
    
    # Simulate transition to decode (seq_len == 1 infers decode)
    k_hat, v_hat = cache.update_and_fetch(mx.random.normal((1, 4, 1, 128)), mx.random.normal((1, 4, 1, 128)))
    assert cache.is_deferred == False
    assert cache.keys is not None # Compression happened
