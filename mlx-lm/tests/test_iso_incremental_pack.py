"""Phase 2: incremental packed-cache append correctness invariant.

The only correctness requirement: after N decode steps, the incrementally-built
packed cache (`_packed_keys_cache` / `_packed_values_cache`) must be **bit-exact
equal** to a fresh `pack_indices_3bit(compressed_indices)` rebuild on the same
stored compressed indices.

Two scales:
- Short (10 decode steps): catches off-by-one and finalize-boundary bugs
- Long (8000 decode steps): catches drift, slot-aliasing, or accumulated bugs

If both pass, the implementation is correct. End-to-end decode-time deltas are
*evidence* about whether the work moved, not the correctness criterion.
"""

import os

os.environ["ISOQUANT_BITS"] = "3"

import mlx.core as mx
import numpy as np

from mlx_lm.models.fused_kv_decode_kernels import pack_indices_3bit
from mlx_lm.models.mlx_isoquant import IsoQuantKVCache
from mlx_lm.models.mlx_turboquant import get_default_codebook_dir

NUM_HEADS, HEAD_DIM = 2, 256
CODEBOOK_DIR = get_default_codebook_dir()


def _make_cache():
    """Build a real IsoQuant cache wired to the production codebook dir."""
    return IsoQuantKVCache(
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        bit_width=3,
        layer_idx=0,
        codebook_dir=CODEBOOK_DIR,
    )


def _gen_kv(seq_len: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    keys = mx.array(
        rng.standard_normal((1, NUM_HEADS, seq_len, HEAD_DIM)).astype(np.float32)
    )
    values = mx.array(
        rng.standard_normal((1, NUM_HEADS, seq_len, HEAD_DIM)).astype(np.float32)
    )
    mx.eval(keys, values)
    return keys, values


def test_incremental_append_matches_rebuild_short():
    """After 10 decode steps, incrementally-built packed cache equals fresh rebuild."""
    cache = _make_cache()
    assert cache._fallback_cache is None, (
        "test must exercise IsoQuant path, not fallback KVCache"
    )
    keys, values = _gen_kv(256)
    cache.update_and_fetch(keys, values)
    cache.finalize_deferred_prefill()

    for step in range(10):
        k_new, v_new = _gen_kv(1, seed=100 + step)
        cache.update_and_fetch(k_new, v_new)

    inc_keys = cache._packed_keys_cache
    inc_vals = cache._packed_values_cache
    assert inc_keys is not None and inc_vals is not None, (
        "incremental cache must be populated after Phase 2 implementation"
    )

    ref_keys = pack_indices_3bit(cache.compressed_keys["indices"])
    ref_vals = pack_indices_3bit(cache.compressed_values["indices"])
    mx.eval(inc_keys, inc_vals, ref_keys, ref_vals)

    np.testing.assert_array_equal(np.asarray(inc_keys), np.asarray(ref_keys))
    np.testing.assert_array_equal(np.asarray(inc_vals), np.asarray(ref_vals))


def test_incremental_append_matches_rebuild_long():
    """8K total tokens: catches drift / slot-aliasing across many appends."""
    cache = _make_cache()
    assert cache._fallback_cache is None, (
        "test must exercise IsoQuant path, not fallback KVCache"
    )
    keys, values = _gen_kv(2048)
    cache.update_and_fetch(keys, values)
    cache.finalize_deferred_prefill()

    # 8000 appends — meaningful long-context regression check
    for step in range(8000):
        k_new, v_new = _gen_kv(1, seed=10000 + step)
        cache.update_and_fetch(k_new, v_new)

    inc_keys = cache._packed_keys_cache
    inc_vals = cache._packed_values_cache
    assert inc_keys is not None and inc_vals is not None

    ref_keys = pack_indices_3bit(cache.compressed_keys["indices"])
    ref_vals = pack_indices_3bit(cache.compressed_values["indices"])
    mx.eval(inc_keys, inc_vals, ref_keys, ref_vals)

    np.testing.assert_array_equal(np.asarray(inc_keys), np.asarray(ref_keys))
    np.testing.assert_array_equal(np.asarray(inc_vals), np.asarray(ref_vals))
