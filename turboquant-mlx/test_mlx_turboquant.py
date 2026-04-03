import math
import numpy as np
import mlx.core as mx

from mlx_turboquant import (
    TurboQuantCompressor,
    TurboQuantKVCache,
    asymmetric_attention_scores,
)


def random_unit_vectors(n, d, seed=42):
    mx.random.seed(seed)
    x = mx.random.normal((n, d))
    norm = mx.linalg.norm(x, axis=-1, keepdims=True)
    return x / norm


def test_compressor_shapes_and_keys():
    d = 128
    n = 10
    bits = 4

    x = mx.random.normal((n, d)).astype(mx.float32)
    x_norm = mx.linalg.norm(x, axis=-1, keepdims=True)
    x_unit = x / mx.maximum(x_norm, mx.array(1e-8))

    mx.random.seed(42)
    Q, _ = np.linalg.qr(np.random.normal(size=(d, d)).astype(np.float32))
    Q = mx.array(Q)

    compressor = TurboQuantCompressor(bits, d, seed=42)
    compressed = compressor.compress(x, Q)

    assert "indices" in compressed
    assert "x_rot_quant" in compressed
    assert "x_norm" in compressed
    assert "residual_signs" in compressed
    assert "residual_norm" in compressed

    assert compressed["indices"].shape == (n, d)
    assert compressed["residual_signs"].shape == (n, d)
    assert compressed["x_norm"].shape == (n, 1)

    print("test_compressor_shapes_and_keys: OK")


def test_compressor_value():
    d = 128
    n = 10
    bits = 4

    x = mx.random.normal((n, d)).astype(mx.float32)

    mx.random.seed(42)
    Q, _ = np.linalg.qr(np.random.normal(size=(d, d)).astype(np.float32))
    Q = mx.array(Q)

    compressor = TurboQuantCompressor(bits, d, seed=42)
    comp_v = compressor.compress_value(x, Q)
    assert "indices" in comp_v
    assert "x_norm" in comp_v

    x_hat = compressor.decompress_value(comp_v, Q)
    assert x_hat.shape == x.shape

    print("test_compressor_value: OK")


def test_asymmetric_scores_match_reference():
    d = 128
    n_k = 10
    n_q = 5
    bits = 3

    mx.random.seed(42)
    q = mx.random.normal((n_q, d))
    k = mx.random.normal((n_k, d))

    Q, _ = np.linalg.qr(np.random.normal(size=(d, d)).astype(np.float32))
    Q = mx.array(Q)

    compressor = TurboQuantCompressor(bits, d, seed=42)
    compressed = compressor.compress(k, Q)

    # Reference (non-associative) score calculation
    x_hat_unit = mx.matmul(compressed["x_rot_quant"], Q)
    k_hat = x_hat_unit * compressed["x_norm"]

    term1_ref = mx.matmul(q, mx.swapaxes(k_hat, -2, -1))

    Sq = mx.matmul(q, mx.transpose(compressor.S))
    signs_T = mx.swapaxes(compressed["residual_signs"], -2, -1)
    correction_ref = mx.matmul(Sq, signs_T)

    r_norm = mx.swapaxes(compressed["residual_norm"], -2, -1)
    term2_ref = compressor.qjl_scale * r_norm * correction_ref

    scale = 1.0 / math.sqrt(d)
    scores_ref = (term1_ref + term2_ref) * scale

    # Optimized associative score calculation
    scores_opt = asymmetric_attention_scores(
        q, compressed, Q, compressor.S, compressor.qjl_scale, scale
    )

    max_diff = mx.max(mx.abs(scores_ref - scores_opt)).item()
    assert max_diff < 1e-5, f"Optimized scores diverged from reference by {max_diff}"

    print(f"test_asymmetric_scores_match_reference: OK (max_abs={max_diff:.2e})")


def test_kv_cache_gqa_scores_finite():
    b = 1
    seq_len = 10
    num_kv_heads = 2
    num_q_heads = 8
    head_dim = 128

    cache = TurboQuantKVCache(num_kv_heads, head_dim, bit_width=3, max_seq_len=2048)

    mx.random.seed(123)
    k = mx.random.normal((b, num_kv_heads, seq_len, head_dim))
    v = mx.random.normal((b, num_kv_heads, seq_len, head_dim))
    q = mx.random.normal((b, num_q_heads, 1, head_dim))

    keys, values = cache.update_and_fetch(k, v)

    assert cache.size() == seq_len

    # Values should be reconstructed from compressed storage
    vals = cache.get_values()
    assert vals.shape == (b, num_kv_heads, seq_len, head_dim)
    assert not mx.any(mx.isnan(vals)).item()

    # Keys should be reconstructable
    keys_recon = cache.reconstruct_keys()
    assert keys_recon.shape == (b, num_kv_heads, seq_len, head_dim)
    assert not mx.any(mx.isnan(keys_recon)).item()

    # Standard attention with reconstructed keys/values should produce finite scores
    scale = 1.0 / math.sqrt(head_dim)
    output = mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32),
        keys_recon.astype(mx.float32),
        vals.astype(mx.float32),
        scale=scale,
        mask=None,
    )
    assert output.shape == (b, num_q_heads, 1, head_dim)
    assert not mx.any(mx.isnan(output)).item()
    assert not mx.any(mx.isinf(output)).item()

    print("test_kv_cache_gqa_scores_finite: OK")


def test_trim_reduces_sequence():
    num_kv_heads = 1
    head_dim = 128
    seq_len = 10

    cache = TurboQuantKVCache(num_kv_heads, head_dim, bit_width=4)

    k = mx.random.normal((1, num_kv_heads, seq_len, head_dim))
    v = mx.random.normal((1, num_kv_heads, seq_len, head_dim))

    cache.update_and_fetch(k, v)
    assert cache.size() == seq_len

    cache.trim(3)
    assert cache.size() == seq_len - 3

    vals = cache.get_values()
    assert vals.shape[2] == seq_len - 3

    print("test_trim_reduces_sequence: OK")


def test_from_state_roundtrip():
    cache = TurboQuantKVCache(num_heads=2, head_dim=128, bit_width=3, layer_idx=0)
    keys = mx.random.normal((1, 2, 10, 128))
    values = mx.random.normal((1, 2, 10, 128))
    cache.update_and_fetch(keys, values)

    # Save state
    state = cache.state
    meta = cache.meta_state

    # Restore state
    cache_restored = TurboQuantKVCache.from_state(state, meta)

    # Check properties
    assert cache_restored.num_heads == cache.num_heads
    assert cache_restored.layer_idx == cache.layer_idx
    assert cache_restored.bit_width == cache.bit_width
    assert cache_restored.head_dim == cache.head_dim
    assert cache_restored.codebook_dir == cache.codebook_dir
    assert cache_restored.seed == cache.seed
    assert cache_restored._seq_len == cache._seq_len

    # Reconstruct keys from both and compare
    keys_orig = cache.reconstruct_keys()
    keys_restored = cache_restored.reconstruct_keys()

    diff = mx.max(mx.abs(keys_orig - keys_restored)).item()
    assert diff < 1e-4, f"Restored keys differ by {diff}"

    # Values should also match
    vals_orig = cache.get_values()
    vals_restored = cache_restored.get_values()
    vdiff = mx.max(mx.abs(vals_orig - vals_restored)).item()
    assert vdiff < 1e-4, f"Restored values differ by {vdiff}"

    print("test_from_state_roundtrip: OK")


def test_cosine_vs_dense():
    # Reconstructed keys vs original, cosine > 0.98 at 3-bit
    cache = TurboQuantKVCache(num_heads=1, head_dim=128, bit_width=3, layer_idx=1)
    keys = mx.random.normal((1, 1, 50, 128))
    values = mx.random.normal((1, 1, 50, 128))
    cache.update_and_fetch(keys, values)

    keys_recon = cache.reconstruct_keys()
    query = mx.random.normal((1, 1, 1, 128))

    # Compare attention outputs
    dense_scores = mx.matmul(query, mx.transpose(keys, (0, 1, 3, 2)))
    recon_scores = mx.matmul(
        query, mx.transpose(keys_recon.astype(keys.dtype), (0, 1, 3, 2))
    )

    tq_flat = recon_scores.flatten()
    dense_flat = dense_scores.flatten()

    cosine = mx.sum(tq_flat * dense_flat) / (
        mx.linalg.norm(tq_flat) * mx.linalg.norm(dense_flat)
    )
    assert cosine.item() > 0.98, f"Cosine similarity too low: {cosine.item()}"

    print("test_cosine_vs_dense: OK")


def test_nbytes_positive():
    cache = TurboQuantKVCache(num_heads=4, head_dim=128, bit_width=3, layer_idx=2)

    assert cache.nbytes == 0

    keys = mx.random.normal((1, 4, 12, 128))
    values = mx.random.normal((1, 4, 12, 128))
    cache.update_and_fetch(keys, values)

    assert cache.nbytes > 0

    print("test_nbytes_positive: OK")


if __name__ == "__main__":
    test_compressor_shapes_and_keys()
    test_compressor_value()
    test_asymmetric_scores_match_reference()
    test_kv_cache_gqa_scores_finite()
    test_trim_reduces_sequence()
    test_from_state_roundtrip()
    test_cosine_vs_dense()
    test_nbytes_positive()

    print("\nAll Phase 2 tests passed.")
