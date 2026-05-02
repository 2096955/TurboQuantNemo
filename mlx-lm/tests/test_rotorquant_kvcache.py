import mlx.core as mx
import unittest
from mlx_lm.models.mlx_turboquant import RotorQuantKVCache

class TestRotorQuantKVCache(unittest.TestCase):
    def test_rotorquant_kvcache(self):
        head_dim = 128
        cache = RotorQuantKVCache(bit_width=4, head_dim=head_dim)
        
        B, n_kv_heads, seq_len = 1, 2, 10
        keys = mx.random.normal((B, n_kv_heads, seq_len, head_dim))
        values = mx.random.normal((B, n_kv_heads, seq_len, head_dim))
        
        k_hat, v_hat = cache.update_and_fetch(keys, values)
        
        self.assertEqual(k_hat.shape, (B, n_kv_heads, seq_len, head_dim))
        self.assertEqual(v_hat.shape, (B, n_kv_heads, seq_len, head_dim))
        self.assertEqual(cache.offset, seq_len)
        
        # Test appending more
        keys2 = mx.random.normal((B, n_kv_heads, 5, head_dim))
        values2 = mx.random.normal((B, n_kv_heads, 5, head_dim))
        
        k_hat2, v_hat2 = cache.update_and_fetch(keys2, values2)
        self.assertEqual(k_hat2.shape, (B, n_kv_heads, 15, head_dim))
        self.assertEqual(v_hat2.shape, (B, n_kv_heads, 15, head_dim))
        self.assertEqual(cache.offset, 15)

if __name__ == '__main__':
    unittest.main()
