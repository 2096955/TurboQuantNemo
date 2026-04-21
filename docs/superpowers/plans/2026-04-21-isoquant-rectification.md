# IsoQuant/TurboQuant Rectification Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 3 critical bugs, 5 high-severity gaps, and key medium/low issues found in the code audit of the IsoQuant fused attention system against plan `staged-discovering-micali`.

**Architecture:** The IsoQuant KV cache compression system has a fused Metal decode pipeline that correctly solves the O(T*d^2) inverse rotation problem, but the surrounding infrastructure has critical gaps: broken prompt cache persistence, missing safety guards on Metal kernels, silent fallback to the buggy path, zero integration test coverage, and a shared env var that can silently degrade performance. This plan fixes each issue with TDD, smallest-change-first.

**Tech Stack:** Python 3.11+, MLX, Metal shaders (via mx.fast.metal_kernel), pytest

---

## File Map

| File | Role | Tasks |
|------|------|-------|
| `mlx-lm/mlx_lm/models/cache.py` | Cache factory + prompt cache persistence | 1, 4 |
| `mlx-lm/mlx_lm/models/fused_kv_decode_kernels.py` | Metal kernel dispatch wrappers | 2 |
| `mlx-lm/mlx_lm/models/mlx_isoquant.py` | IsoQuantKVCache class | 3, 7 |
| `mlx-lm/mlx_lm/models/mlx_turboquant.py` | TurboQuantKVCache class | 6 |
| `mlx-lm/mlx_lm/models/qwen3_next.py` | Model attention dispatch (reference) | 3 (read-only) |
| `mlx-lm/tests/test_prompt_cache.py` | Prompt cache test suite | 1, 9 |
| `mlx-lm/tests/test_fused_isoquant_attention.py` | Fused attention tests | 2, 7, 8 |
| `mlx-lm/tests/test_mlx_isoquant.py` | IsoQuant unit tests | 5 |
| `mlx-lm/tests/test_models.py` | Main model test suite | 5 |
| `docs/FROM_ATTENTION_TO_CONSUMER_HARDWARE.md` | Paper | 10 |

---

## Task 1: Fix IsoQuant prompt cache load (CRITICAL)

**Bug:** `load_prompt_cache` at `cache.py:212` calls `globals()["IsoQuantKVCache"]` but `IsoQuantKVCache` is only imported inside `make_prompt_cache()` (function-local scope at line 31). Saving an IsoQuant prompt cache and reloading crashes with `KeyError`.

**Root cause:** `TurboQuantKVCache` works because it has a module-level import. `IsoQuantKVCache` does not.

**Files:**
- Modify: `mlx-lm/mlx_lm/models/cache.py:30-36` (add module-level import)
- Test: `mlx-lm/tests/test_prompt_cache.py`

- [ ] **Step 1: Write the failing test**

Add to `mlx-lm/tests/test_prompt_cache.py`:

```python
class TestIsoQuantPromptCachePersistence(unittest.TestCase):
    def test_isoquant_cache_save_load_roundtrip(self):
        """Verify IsoQuantKVCache survives save/load via globals() lookup."""
        from mlx_lm.models.mlx_isoquant import IsoQuantKVCache
        from mlx_lm.models.cache import save_prompt_cache, load_prompt_cache
        import tempfile, os

        cache = IsoQuantKVCache(
            num_heads=2,
            head_dim=128,
            bit_width=3,
            codebook_dir=None,
        )
        keys = mx.random.normal((1, 2, 4, 128))
        values = mx.random.normal((1, 2, 4, 128))
        cache.update_and_fetch(keys, values)
        cache.finalize_deferred_prefill()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_iso_cache.safetensors")
            save_prompt_cache(path, [cache])
            loaded = load_prompt_cache(path)

        self.assertEqual(len(loaded), 1)
        self.assertIsInstance(loaded[0], IsoQuantKVCache)
        self.assertTrue(loaded[0].supports_fused_attention)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mlx-lm && python -m pytest tests/test_prompt_cache.py::TestIsoQuantPromptCachePersistence::test_isoquant_cache_save_load_roundtrip -v`
Expected: FAIL with `KeyError: 'IsoQuantKVCache'`

- [ ] **Step 3: Add module-level import of IsoQuantKVCache in cache.py**

At the bottom of `cache.py`, near the existing module-level TurboQuantKVCache import (around line 1479), add:

```python
from .mlx_isoquant import IsoQuantKVCache  # noqa: E402
```

This mirrors the existing pattern for `TurboQuantKVCache` and ensures `globals()["IsoQuantKVCache"]` resolves during `load_prompt_cache`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mlx-lm && python -m pytest tests/test_prompt_cache.py::TestIsoQuantPromptCachePersistence -v`
Expected: PASS

- [ ] **Step 5: Run full prompt cache test suite for regressions**

Run: `cd mlx-lm && python -m pytest tests/test_prompt_cache.py -v --timeout=60`
Expected: All existing tests PASS

- [ ] **Step 6: Commit**

```bash
git add mlx-lm/mlx_lm/models/cache.py mlx-lm/tests/test_prompt_cache.py
git commit -m "fix: add module-level IsoQuantKVCache import for prompt cache persistence"
```

---

## Task 2: Guard Kernel A against head_dim > 512 buffer overflow (CRITICAL)

**Bug:** `fused_kv_decode_kernels.py:46` allocates `threadgroup float q_local[512]` statically. The dispatch wrapper at `fused_qk_dot` (line 386) passes `head_dim` from the query tensor with no upper-bound check. Any model with `head_dim > 512` causes out-of-bounds Metal shared memory writes.

**Files:**
- Modify: `mlx-lm/mlx_lm/models/fused_kv_decode_kernels.py:386-427`
- Test: `mlx-lm/tests/test_fused_isoquant_attention.py`

- [ ] **Step 1: Write the failing test**

Add to `mlx-lm/tests/test_fused_isoquant_attention.py`:

```python
class TestKernelSafetyGuards(unittest.TestCase):
    def test_fused_qk_dot_rejects_head_dim_above_512(self):
        """Kernel A has q_local[512]; head_dim > 512 must raise, not overflow."""
        from mlx_lm.models.fused_kv_decode_kernels import fused_qk_dot

        num_heads, seq_len, head_dim = 2, 8, 768
        q = mx.random.normal((1, num_heads, 1, head_dim))
        packed = mx.zeros((num_heads, seq_len * head_dim // 8 * 3), dtype=mx.uint8)
        centroids = mx.random.normal((8, 1))
        norms = mx.ones((num_heads, seq_len, 1))

        with self.assertRaises(ValueError, msg="head_dim 768 exceeds kernel maximum"):
            fused_qk_dot(q, packed, centroids, norms, seq_len, head_dim)

    def test_fused_qk_dot_accepts_head_dim_128(self):
        """head_dim=128 (common case) must not raise."""
        cache = _make_populated_cache(num_heads=2, head_dim=128, seq_len=8)
        q = mx.random.normal((1, 2, 1, 128))
        try:
            cache.fused_attention(q, scale=1.0 / (128 ** 0.5))
        except ValueError:
            self.fail("fused_attention raised ValueError for head_dim=128")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mlx-lm && python -m pytest tests/test_fused_isoquant_attention.py::TestKernelSafetyGuards::test_fused_qk_dot_rejects_head_dim_above_512 -v`
Expected: FAIL (no ValueError raised, or Metal crash)

- [ ] **Step 3: Add guard to fused_qk_dot dispatch**

In `fused_kv_decode_kernels.py`, at the start of `fused_qk_dot` (around line 395, after parameter extraction):

```python
_MAX_KERNEL_HEAD_DIM = 512

def fused_qk_dot(q, packed_keys, centroids, norms, seq_len, head_dim, ...):
    if head_dim > _MAX_KERNEL_HEAD_DIM:
        raise ValueError(
            f"head_dim {head_dim} exceeds kernel maximum ({_MAX_KERNEL_HEAD_DIM}). "
            f"Fused attention will use the MLX-ops fallback path."
        )
    # ... rest of function
```

Add the same guard to `fused_value_accum` and `fully_fused_attention` dispatch functions.

- [ ] **Step 4: Run tests to verify both pass**

Run: `cd mlx-lm && python -m pytest tests/test_fused_isoquant_attention.py::TestKernelSafetyGuards -v`
Expected: Both PASS

- [ ] **Step 5: Run full fused attention test suite for regressions**

Run: `cd mlx-lm && python -m pytest tests/test_fused_isoquant_attention.py -v --timeout=120`
Expected: All existing tests PASS

- [ ] **Step 6: Commit**

```bash
git add mlx-lm/mlx_lm/models/fused_kv_decode_kernels.py mlx-lm/tests/test_fused_isoquant_attention.py
git commit -m "fix: guard Metal kernels against head_dim > 512 buffer overflow"
```

---

## Task 3: Emit warning when unfused path is taken (HIGH)

**Bug:** When `supports_fused_attention` returns `False` due to `bit_width != 3`, the code silently falls through to the O(T*d^2) `reconstruct_keys()` + SDPA path. No log, no metric, no warning. Users with `TURBOQUANT_BITS=4` get the buggy path with no indication.

**Files:**
- Modify: `mlx-lm/mlx_lm/models/mlx_isoquant.py:692-701` (supports_fused_attention property)
- Modify: `mlx-lm/mlx_lm/models/mlx_isoquant.py:34-89` (IsoQuantStats — add unfused counter)
- Test: `mlx-lm/tests/test_mlx_isoquant.py`

- [ ] **Step 1: Write the failing test**

Add to `mlx-lm/tests/test_mlx_isoquant.py`:

```python
class TestUnfusedPathWarning(unittest.TestCase):
    def test_bit_width_4_warns_unfused(self):
        """IsoQuant with bit_width=4 must warn that fused path is unavailable."""
        import warnings
        from mlx_lm.models.mlx_isoquant import IsoQuantKVCache

        cache = IsoQuantKVCache(
            num_heads=2, head_dim=128, bit_width=4, codebook_dir=None,
        )
        self.assertFalse(cache.supports_fused_attention)

        keys = mx.random.normal((1, 2, 4, 128))
        values = mx.random.normal((1, 2, 4, 128))
        cache.update_and_fetch(keys, values)
        cache.finalize_deferred_prefill()

        q = mx.random.normal((1, 2, 1, 128))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cache.fused_attention(q, scale=1.0 / (128 ** 0.5))
            unfused_warnings = [x for x in w if "unfused" in str(x.message).lower()
                                or "fused path unavailable" in str(x.message).lower()]
            self.assertGreater(len(unfused_warnings), 0,
                "Expected a warning about fused path being unavailable for bit_width=4")

    def test_bit_width_3_no_warning(self):
        """IsoQuant with bit_width=3 must NOT warn."""
        import warnings
        from mlx_lm.models.mlx_isoquant import IsoQuantKVCache

        cache = IsoQuantKVCache(
            num_heads=2, head_dim=128, bit_width=3, codebook_dir=None,
        )
        keys = mx.random.normal((1, 2, 4, 128))
        values = mx.random.normal((1, 2, 4, 128))
        cache.update_and_fetch(keys, values)
        cache.finalize_deferred_prefill()

        q = mx.random.normal((1, 2, 1, 128))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cache.fused_attention(q, scale=1.0 / (128 ** 0.5))
            unfused_warnings = [x for x in w if "unfused" in str(x.message).lower()
                                or "fused path unavailable" in str(x.message).lower()]
            self.assertEqual(len(unfused_warnings), 0,
                "bit_width=3 should not emit unfused warning")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mlx-lm && python -m pytest tests/test_mlx_isoquant.py::TestUnfusedPathWarning::test_bit_width_4_warns_unfused -v`
Expected: FAIL (no warning emitted)

- [ ] **Step 3: Add warning to fused_attention fallback path**

In `mlx_isoquant.py`, inside the `fused_attention` method, at the point where it falls back because `supports_fused_attention` is False (the reconstruct_keys path around lines 727-734):

```python
import warnings

def fused_attention(self, queries, scale, mask=None):
    if not self.supports_fused_attention:
        if self.bit_width != 3:
            warnings.warn(
                f"IsoQuant fused path unavailable: bit_width={self.bit_width} "
                f"(only 3-bit supported). Using unfused O(T*d^2) path.",
                stacklevel=2,
            )
        _GLOBAL_STATS.unfused_fallback_calls += 1
        # ... existing fallback code
```

Also add `"unfused_fallback_calls"` to `IsoQuantStats.__slots__` (around line 34) and initialise it to 0 in `__init__`.

- [ ] **Step 4: Run tests to verify both pass**

Run: `cd mlx-lm && python -m pytest tests/test_mlx_isoquant.py::TestUnfusedPathWarning -v`
Expected: Both PASS

- [ ] **Step 5: Commit**

```bash
git add mlx-lm/mlx_lm/models/mlx_isoquant.py mlx-lm/tests/test_mlx_isoquant.py
git commit -m "fix: emit warning when IsoQuant fused path is unavailable"
```

---

## Task 4: Add ISOQUANT_BITS env var and fix docs mismatch (HIGH)

**Bug:** Both TurboQuant and IsoQuant read bit_width from `TURBOQUANT_BITS`. Setting it to 4 for TurboQuant silently pushes IsoQuant off its fused path. Additionally, CLAUDE.md documents the default as `2` but the code defaults to `3`.

**Files:**
- Modify: `mlx-lm/mlx_lm/models/cache.py:55-65`
- Modify: `CLAUDE.md` (fix default from `2` to `3`)
- Test: `mlx-lm/tests/test_mlx_isoquant.py`

- [ ] **Step 1: Write the failing test**

Add to `mlx-lm/tests/test_mlx_isoquant.py`:

```python
class TestIsoquantBitsEnvVar(unittest.TestCase):
    def test_isoquant_bits_overrides_turboquant_bits(self):
        """ISOQUANT_BITS should take precedence over TURBOQUANT_BITS for IsoQuant."""
        import os
        from unittest.mock import patch
        from mlx_lm.models.cache import make_prompt_cache

        class FakeModel:
            class layers:
                pass
            layers = [type("L", (), {"self_attn": type("A", (), {"n_kv_heads": 2, "head_dim": 128})()})()]

        with patch.dict(os.environ, {"TURBOQUANT_BITS": "4", "ISOQUANT_BITS": "3"}):
            # Reload to pick up env change — or test the parsing function directly
            from mlx_lm.models.cache import _get_isoquant_bits
            bits = _get_isoquant_bits()
            self.assertEqual(bits, 3, "ISOQUANT_BITS=3 should override TURBOQUANT_BITS=4")

    def test_turboquant_bits_fallback_when_isoquant_bits_unset(self):
        """When ISOQUANT_BITS is not set, fall back to TURBOQUANT_BITS."""
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"TURBOQUANT_BITS": "2"}, clear=False):
            os.environ.pop("ISOQUANT_BITS", None)
            from mlx_lm.models.cache import _get_isoquant_bits
            bits = _get_isoquant_bits()
            self.assertEqual(bits, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mlx-lm && python -m pytest tests/test_mlx_isoquant.py::TestIsoquantBitsEnvVar -v`
Expected: FAIL (`_get_isoquant_bits` does not exist)

- [ ] **Step 3: Extract bit-width helpers and add ISOQUANT_BITS**

In `cache.py`, replace the inline env var read (around line 55-58) with:

```python
def _get_turboquant_bits() -> int:
    try:
        return int(os.environ.get("TURBOQUANT_BITS", 3))
    except ValueError:
        return 3

def _get_isoquant_bits() -> int:
    iso_bits = os.environ.get("ISOQUANT_BITS")
    if iso_bits is not None:
        try:
            return int(iso_bits)
        except ValueError:
            pass
    return _get_turboquant_bits()
```

Then in `make_quant_cache`, use `_get_isoquant_bits()` for the isoquant branch and `_get_turboquant_bits()` for the turboquant branch.

- [ ] **Step 4: Fix CLAUDE.md default**

In `CLAUDE.md`, the TurboQuant-specific table says `TURBOQUANT_BITS | 2`. Change it to `3` to match the code at `cache.py:56`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd mlx-lm && python -m pytest tests/test_mlx_isoquant.py::TestIsoquantBitsEnvVar -v`
Expected: Both PASS

- [ ] **Step 6: Commit**

```bash
git add mlx-lm/mlx_lm/models/cache.py mlx-lm/tests/test_mlx_isoquant.py CLAUDE.md
git commit -m "fix: add ISOQUANT_BITS env var, fix TURBOQUANT_BITS default in docs"
```

---

## Task 5: Add IsoQuant integration test to test_models.py (HIGH)

**Bug:** `test_models.py` has zero references to IsoQuant or TurboQuant. If someone removed the `isinstance(cache, IsoQuantKVCache)` check from any model file, no test would fail.

**Files:**
- Modify: `mlx-lm/tests/test_models.py`

- [ ] **Step 1: Write the integration test**

Add to `mlx-lm/tests/test_models.py`:

```python
class TestIsoQuantIntegration(unittest.TestCase):
    """Verify IsoQuant fused attention is wired through the model forward pass."""

    def _check_model_isoquant_wiring(self, model_name, model_cls, config_dict):
        """Helper: create model, run generate_step with isoquant cache, verify output."""
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.models.mlx_isoquant import IsoQuantKVCache

        config = model_cls.ModelArgs(**config_dict)
        model = model_cls.Model(config)
        model.eval()

        cache = make_prompt_cache(model, kv_cache_type="isoquant")
        # Verify at least one layer got an IsoQuantKVCache
        has_isoquant = any(isinstance(c, IsoQuantKVCache) for c in cache)
        self.assertTrue(has_isoquant, f"{model_name}: no IsoQuantKVCache in prompt cache")

        # Run a single-token forward pass
        x = mx.array([[0]])
        logits = model(x, cache=cache)
        mx.eval(logits)

        self.assertTrue(mx.isfinite(logits).all().item(),
            f"{model_name}: non-finite logits with IsoQuant cache")
        self.assertEqual(logits.shape[0], 1)

    def test_qwen2_isoquant_wiring(self):
        from mlx_lm.models import qwen2
        self._check_model_isoquant_wiring("qwen2", qwen2, {
            "model_type": "qwen2",
            "hidden_size": 256,
            "num_hidden_layers": 2,
            "intermediate_size": 512,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "vocab_size": 1000,
            "tie_word_embeddings": False,
        })
```

- [ ] **Step 2: Run test to verify it passes (baseline)**

Run: `cd mlx-lm && python -m pytest tests/test_models.py::TestIsoQuantIntegration::test_qwen2_isoquant_wiring -v --timeout=120`
Expected: PASS (this confirms the wiring currently works)

- [ ] **Step 3: Verify the test would catch a wiring removal**

Temporarily comment out the `isinstance(cache, IsoQuantKVCache)` check in `qwen2.py:82` and re-run. Verify the test fails (logits should still be finite but the fused path is not taken). If the test passes even without the check, add an assertion that verifies `cache.stats.fused_metal_attempts > 0` or similar.

Revert the temporary change after verification.

- [ ] **Step 4: Commit**

```bash
git add mlx-lm/tests/test_models.py
git commit -m "test: add IsoQuant integration test to test_models.py"
```

---

## Task 6: Fix TurboQuant from_state crash on meta_state=None (MEDIUM)

**Bug:** `TurboQuantKVCache.from_state` at `mlx_turboquant.py:674` calls `obj.meta_state = meta_state`. The setter at line 761 accesses `v[0]`, which raises `TypeError` when `meta_state=None`.

**Files:**
- Modify: `mlx-lm/mlx_lm/models/mlx_turboquant.py:761-783`
- Test: `mlx-lm/tests/test_prompt_cache.py`

- [ ] **Step 1: Write the failing test**

Add to `mlx-lm/tests/test_prompt_cache.py`:

```python
class TestTurboQuantFromStateEdgeCases(unittest.TestCase):
    def test_from_state_with_none_meta_state_raises_clearly(self):
        """from_state(state, meta_state=None) should raise ValueError, not TypeError."""
        from mlx_lm.models.mlx_turboquant import TurboQuantKVCache

        with self.assertRaises(ValueError, msg="meta_state is required"):
            TurboQuantKVCache.from_state({}, meta_state=None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mlx-lm && python -m pytest tests/test_prompt_cache.py::TestTurboQuantFromStateEdgeCases -v`
Expected: FAIL with `TypeError` (not `ValueError`)

- [ ] **Step 3: Guard the meta_state setter**

In `mlx_turboquant.py`, at the start of the `meta_state` setter (line 761):

```python
@meta_state.setter
def meta_state(self, v):
    if v is None:
        raise ValueError(
            "meta_state is required for from_state reconstruction. "
            "Cannot restore cache without version, bit_width, num_heads, head_dim."
        )
    version = v[0]
    # ... rest of setter
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mlx-lm && python -m pytest tests/test_prompt_cache.py::TestTurboQuantFromStateEdgeCases -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mlx-lm/mlx_lm/models/mlx_turboquant.py mlx-lm/tests/test_prompt_cache.py
git commit -m "fix: TurboQuant from_state raises ValueError instead of TypeError on None meta_state"
```

---

## Task 7: Guard fused_attention against empty cache (LOW)

**Bug:** `fused_attention()` with T=0 (no tokens stored yet) is untested. Metal kernel dispatch with `seq_len=0` produces a zero-size grid which may cause undefined behavior.

**Files:**
- Modify: `mlx-lm/mlx_lm/models/mlx_isoquant.py` (fused_attention method)
- Test: `mlx-lm/tests/test_fused_isoquant_attention.py`

- [ ] **Step 1: Write the failing test**

Add to `mlx-lm/tests/test_fused_isoquant_attention.py`:

```python
class TestFusedAttentionEdgeCases(unittest.TestCase):
    def test_fused_attention_empty_cache_returns_zeros(self):
        """fused_attention on an empty cache should return zeros, not crash."""
        from mlx_lm.models.mlx_isoquant import IsoQuantKVCache
        from mlx_lm.models.cache import get_default_codebook_dir

        cache = IsoQuantKVCache(
            num_heads=2, head_dim=128, bit_width=3,
            codebook_dir=get_default_codebook_dir(),
        )
        q = mx.random.normal((1, 2, 1, 128))
        result = cache.fused_attention(q, scale=1.0 / (128 ** 0.5))
        mx.eval(result)
        self.assertEqual(result.shape, (1, 2, 1, 128))
        # With no KV data, output should be zeros (no attention to attend to)
        self.assertTrue(mx.allclose(result, mx.zeros_like(result)).item())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mlx-lm && python -m pytest tests/test_fused_isoquant_attention.py::TestFusedAttentionEdgeCases::test_fused_attention_empty_cache_returns_zeros -v`
Expected: FAIL (crash or unexpected output)

- [ ] **Step 3: Add early return for empty cache**

In `mlx_isoquant.py`, at the start of the `fused_attention` method (after the `supports_fused_attention` check):

```python
def fused_attention(self, queries, scale, mask=None):
    if self.compressed_keys is None or self.compressed_keys["indices"].shape[1] == 0:
        return mx.zeros_like(queries)
    # ... rest of method
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mlx-lm && python -m pytest tests/test_fused_isoquant_attention.py::TestFusedAttentionEdgeCases -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mlx-lm/mlx_lm/models/mlx_isoquant.py mlx-lm/tests/test_fused_isoquant_attention.py
git commit -m "fix: guard fused_attention against empty cache dispatch"
```

---

## Task 8: Add head_dim != 128 fused attention test (LOW)

**Bug:** All fused attention tests hardcode `head_dim=128`. The 3-kernel path is dimension-agnostic but the fully-fused kernel asserts `npt == 4` (head_dim 128 only). There is no test that verifies the fallback behavior when head_dim differs.

**Files:**
- Test: `mlx-lm/tests/test_fused_isoquant_attention.py`

- [ ] **Step 1: Write the test**

Add to `mlx-lm/tests/test_fused_isoquant_attention.py`:

```python
class TestHeadDimVariations(unittest.TestCase):
    def test_head_dim_256_uses_3kernel_not_single(self):
        """head_dim=256 should work via 3-kernel path, not the npt=4 single kernel."""
        cache = _make_populated_cache(num_heads=2, head_dim=256, seq_len=16, bit_width=3)

        q = mx.random.normal((1, 2, 1, 256))
        result = cache.fused_attention(q, scale=1.0 / (256 ** 0.5))
        mx.eval(result)

        self.assertEqual(result.shape, (1, 2, 1, 256))
        self.assertTrue(mx.isfinite(result).all().item())

    def test_head_dim_64_works(self):
        """head_dim=64 (smaller than 128) should also work."""
        cache = _make_populated_cache(num_heads=4, head_dim=64, seq_len=16, bit_width=3)

        q = mx.random.normal((1, 4, 1, 64))
        result = cache.fused_attention(q, scale=1.0 / (64 ** 0.5))
        mx.eval(result)

        self.assertEqual(result.shape, (1, 4, 1, 64))
        self.assertTrue(mx.isfinite(result).all().item())
```

- [ ] **Step 2: Run tests**

Run: `cd mlx-lm && python -m pytest tests/test_fused_isoquant_attention.py::TestHeadDimVariations -v --timeout=120`
Expected: PASS (3-kernel path should handle these dimensions)

- [ ] **Step 3: Commit**

```bash
git add mlx-lm/tests/test_fused_isoquant_attention.py
git commit -m "test: add head_dim != 128 coverage for fused attention"
```

---

## Task 9: Add IsoQuant prompt cache save/load round-trip test (MEDIUM)

**Depends on:** Task 1 (module-level import fix)

**Bug:** No test verifies that an IsoQuant cache survives a full save -> load -> use cycle. The `meta_state` setter must correctly reconstruct rotation matrices and compressor from stored parameters.

**Files:**
- Test: `mlx-lm/tests/test_prompt_cache.py`

- [ ] **Step 1: Write the round-trip test**

Add to `mlx-lm/tests/test_prompt_cache.py`:

```python
class TestIsoQuantRoundTrip(unittest.TestCase):
    def test_isoquant_save_load_produces_same_attention_output(self):
        """Save IsoQuant cache, load it, verify fused_attention output matches."""
        from mlx_lm.models.mlx_isoquant import IsoQuantKVCache
        from mlx_lm.models.cache import save_prompt_cache, load_prompt_cache
        import tempfile, os

        cache = IsoQuantKVCache(
            num_heads=2, head_dim=128, bit_width=3, codebook_dir=None,
        )
        keys = mx.random.normal((1, 2, 8, 128))
        values = mx.random.normal((1, 2, 8, 128))
        cache.update_and_fetch(keys, values)
        cache.finalize_deferred_prefill()

        q = mx.random.normal((1, 2, 1, 128))
        original_output = cache.fused_attention(q, scale=1.0 / (128 ** 0.5))
        mx.eval(original_output)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "roundtrip.safetensors")
            save_prompt_cache(path, [cache])
            loaded = load_prompt_cache(path)

        loaded_cache = loaded[0]
        loaded_output = loaded_cache.fused_attention(q, scale=1.0 / (128 ** 0.5))
        mx.eval(loaded_output)

        self.assertTrue(
            mx.allclose(original_output, loaded_output, atol=1e-5).item(),
            "Loaded cache should produce identical attention output",
        )
```

- [ ] **Step 2: Run test**

Run: `cd mlx-lm && python -m pytest tests/test_prompt_cache.py::TestIsoQuantRoundTrip -v --timeout=60`
Expected: PASS (if Task 1 is done)

- [ ] **Step 3: Commit**

```bash
git add mlx-lm/tests/test_prompt_cache.py
git commit -m "test: add IsoQuant prompt cache save/load round-trip with output verification"
```

---

## Task 10: Fix paper cross-references (MEDIUM)

**Bug:** `docs/FROM_ATTENTION_TO_CONSUMER_HARDWARE.md` references "Section 6.6" (lines 335, 656) and "Section 13" which do not exist as headings. Content lives under sections 6.3 and 10.1.6.

**Files:**
- Modify: `docs/FROM_ATTENTION_TO_CONSUMER_HARDWARE.md`

- [ ] **Step 1: Find and audit all cross-references**

Run: `grep -n "Section 6\\.6\|Section 13\|section 6\\.6\|section 13" docs/FROM_ATTENTION_TO_CONSUMER_HARDWARE.md`

- [ ] **Step 2: Fix each dangling reference**

For each "Section 6.6" reference: replace with the correct section number (likely 6.3 or wherever the fused pipeline content actually lives).

For each "Section 13" reference: replace with the correct section number (likely 10.1.6 where the go/no-go table lives, or the relevant appendix).

- [ ] **Step 3: Verify no dangling references remain**

Run: `grep -n "Section [0-9]" docs/FROM_ATTENTION_TO_CONSUMER_HARDWARE.md | head -30`

Cross-check each reference against actual headings.

- [ ] **Step 4: Commit**

```bash
git add docs/FROM_ATTENTION_TO_CONSUMER_HARDWARE.md
git commit -m "docs: fix dangling section cross-references in paper"
```

---

## Task 11: Update parent plan accuracy (LOW)

**Bug:** The parent plan `staged-discovering-micali.md` references `scripts/profile_decode_pipeline.py` (Step 1) but the actual file is `scripts/decode_profiler.py`. Step 1 status says NOT STARTED but the tool is built — only the execution is missing.

**Files:**
- Modify: `/Users/anthonylui/.claude/plans/staged-discovering-micali.md`

- [ ] **Step 1: Fix filename reference**

In the plan, Step 1, change:
- `scripts/profile_decode_pipeline.py` -> `scripts/decode_profiler.py`
- Status: `NOT STARTED` -> `TOOL BUILT — execution pending`

- [ ] **Step 2: Commit**

```bash
git add .claude/plans/staged-discovering-micali.md 2>/dev/null || true
```

(Plan files may not be tracked in git — commit only if tracked.)

---

## Prioritised Execution Order

| Order | Task | Severity | Est. Time | Dependencies |
|-------|------|----------|-----------|--------------|
| 1 | Task 1: IsoQuant prompt cache load | CRITICAL | 5 min | None |
| 2 | Task 2: Kernel A head_dim guard | CRITICAL | 5 min | None |
| 3 | Task 3: Unfused path warning | HIGH | 5 min | None |
| 4 | Task 6: from_state None guard | MEDIUM | 3 min | None |
| 5 | Task 7: Empty cache guard | LOW | 3 min | None |
| 6 | Task 4: ISOQUANT_BITS env var | HIGH | 10 min | None |
| 7 | Task 5: Integration test | HIGH | 10 min | None |
| 8 | Task 8: head_dim variation tests | LOW | 5 min | Task 2 |
| 9 | Task 9: Save/load round-trip test | MEDIUM | 5 min | Task 1 |
| 10 | Task 10: Paper cross-refs | MEDIUM | 5 min | None |
| 11 | Task 11: Parent plan fix | LOW | 2 min | None |

## What This Plan Does NOT Cover

These were identified in the audit but are out of scope for a rectification plan (they are feature work, not bug fixes):

- **TurboQuant has no fused path** (H2) — This is a design limitation, not a bug. TurboQuant is the older baseline. Adding a fused path to TurboQuant is feature work.
- **Non-fused fallback retains all bugs** (H1) — The fallback paths (Metal failure, deferred prefill) are algorithmically correct but slow. The fused path is the fix. Making the fallback equally fast would mean reimplementing the fused path in pure MLX ops, which is a separate project.
- **GQA inverse rotation Python loop** (M1) — Performance optimization, not a correctness bug.
- **Eager mx.eval preventing graph fusion** (M4) — Performance optimization requiring MLX graph analysis.
- **Single kernel threshold = 0** (M3) — The single kernel path is intentionally disabled pending T-parallel tiling. Not a bug.
- **Ranking metrics** (Plan Step 5) — New evaluation infrastructure, not a rectification.
- **Long-context testing** (Plan Step 6) — New test infrastructure, not a rectification.
- **test_hybrid_cache env-locked** (L4) — Pre-existing issue, unrelated to IsoQuant.

---

## Discovered During Execution (2026-04-21)

Additions found while implementing the audit, not in the original plan.

### Task 1b: RotorQuant cache deserialization (HIGH — latent)

**Discovered by:** codex review of Task 1 commit `9afd247`.

**Bug:** `RotorQuantKVCache` (in `mlx_turboquant.py:823-983`) is constructible via the new `make_prompt_cache(..., kv_cache_type="rotorquant")` route, but it cannot be deserialized:
1. Not registered in `cache.py` module-level globals (only `TurboQuantKVCache` and `IsoQuantKVCache` are).
2. No `meta_state` property/setter or `from_state` classmethod implemented.
3. `state` setter restores tensors/rotors only, not `offset`, `compressor`, or codebook config.

**Trigger:** Any user who runs `save_prompt_cache` on a RotorQuant cache and reloads it. Currently latent because no benchmark exercises this path.

**Fix outline:**
1. Add module-level `from .mlx_turboquant import RotorQuantKVCache  # noqa: E402, F401` in `cache.py`.
2. Implement `meta_state` (carry `bit_width`, `head_dim`, `codebook_dir`, `seed`, version tag).
3. Implement `from_state` (reconstruct compressor + restore offset/seq_len from state).
4. Test: parallel of `TestIsoQuantPromptCachePersistence::test_isoquant_cache_save_load_roundtrip` for RotorQuant.

**Status:** Pending. Schedule before any production use of `kv_cache_type="rotorquant"`.

### Task 1c: Bit-for-bit content assertion in IsoQuant roundtrip test (MEDIUM)

**Discovered by:** codex review.

**Gap:** Original Task 1 test only asserted `len`, `isinstance`, and `supports_fused_attention`. Did not prove compressed state survived save/load.

**Status:** ✅ Resolved in commit follow-up. `test_isoquant_cache_save_load_roundtrip` now also asserts `_seq_len` equality and `mx.array_equal` on `reconstruct_keys()` and `get_values()` before/after roundtrip.

### Task 1d: `CacheList.__len__` and `__iter__` (MEDIUM — defensive)

**Discovered by:** codex review of `cache.py:105-107` `make_prompt_cache` walker.

**Bug:** `_replace_attention_caches` walker calls `len(native)` and `enumerate(native)` on whatever `model.make_cache()` returns. `CacheList` had `__getitem__` but no `__len__`/`__iter__`, so a top-level `CacheList` return would raise `TypeError`.

**Status:** ✅ Resolved in commit follow-up. Added both dunder methods to `CacheList`.

### Task 1e: TurboQuant/IsoQuant `state` save-side `int` bug (CRITICAL — concurrent with Task 1)

**Discovered by:** writing the Task 1 test.

**Bug:** `TurboQuantKVCache.state` (parent class) returned plain Python `int` for `_seq_len` and `bit_width`. `mx.save_safetensors` rejects non-array values with `RuntimeError: std::bad_cast`. Test could never reach the load-side fix until save-side was fixed.

**Status:** ✅ Resolved in Task 1 commit `9afd247`. Wrapped both as `mx.array(dtype=mx.int32)`. Setter handles both `mx.array` and `int` for forward compat (no prior caches exist on disk).
