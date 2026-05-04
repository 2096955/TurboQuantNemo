# Codex Second-Pass Review — 2026-05-04

**Reviewer:** Codex (OpenAI o3 family, via `codex exec`)
**Subject:** 4 substantive commits landed and pushed to `origin/main` during the
2026-05-03 → 2026-05-04 session.
**Why this audit:** First implementation AND first review pass were both done by
sonnet subagents (Anthropic family). I wanted a different lineage to spot what
we collectively missed.

## Commits reviewed

| Hash | Message | Codex verdict |
|---|---|---|
| `b4ebfb5` | fix(kimi-mla): validate_real_kv.py actually exercises IsoQuant MLA path | SHIP-WITH-FIXES |
| `55e2d64` | fix(dispatch): make IsoQuant fallback explicit in attention paths | RECONSIDER |
| `056701c` | fix(gemma3): parameterise RotorQuant bit_width via ROTORQUANT_BITS | SHIP |
| `ab4f936` | fix(native): close 4 BLOCKERs in IsoQuant C++/Metal pipeline (8117c76 audit) | SHIP-WITH-FIXES |
| `736a88c` | fix(native): close 3 HIGH + 2 MEDIUM in IsoQuant pipeline (8117c76 audit) | RECONSIDER |

## Verified BLOCKERs

### 1. `qwen3_next.py:77` — `_isoquant_warned` not initialized in `__init__`

The dispatch-fix commit `55e2d64` added explicit IsoQuant fallback elif blocks
to 5 model files. Each fallback reads `self._isoquant_warned`. The init for
that attribute was added to qwen2/qwen3_moe/nemotron_h/gemma4_text but missed
in qwen3_next. First IsoQuant fallback decode → `AttributeError`.

Verification:
```bash
$ for f in mlx-lm/mlx_lm/models/{qwen2,qwen3_moe,qwen3_next,nemotron_h,gemma4_text}.py; do
    grep -c "self._isoquant_warned = False" "$f"
  done
1   # qwen2.py
1   # qwen3_moe.py
0   # qwen3_next.py  ← BLOCKER
1   # nemotron_h.py
1   # gemma4_text.py
```

### 2. `gemma4_text.py:287` — missing TurboQuantKVCache reconstruct branch

The other 4 model files have an explicit elif:
```python
elif isinstance(cache, TurboQuantKVCache) and not isinstance(cache, IsoQuantKVCache):
    keys_reconstructed = cache.reconstruct_keys()
    values_reconstructed = cache.get_values()
    output = scaled_dot_product_attention(...)
```

Gemma4's commit removed the old TurboQuant branch and never restored it. The
ladder now falls through `IsoQuant fused → IsoQuant fallback → generic else`.
Pure TurboQuant ends up in the generic SDPA path with `cache=cache` and only
sees `update_and_fetch()` output, NOT the full reconstructed history. **Quality
regression on Gemma4 + TurboQuant.**

### 3. `isoquant_kv_runtime.cpp:154` — non-existent Metal-cpp API

The HIGH 6 fix added:
```cpp
size_t max_tg_mem = impl_->pso_inverse_rot->maxTotalThreadgroupMemory();
```

`MTL::ComputePipelineState::maxTotalThreadgroupMemory()` is not a real Metal-cpp
method. The actual Metal APIs are:
- `MTL::ComputePipelineState::staticThreadgroupMemoryLength()` — bytes statically
  declared by kernel
- `MTL::ComputePipelineState::maxTotalThreadsPerThreadgroup()` — threads (not
  bytes)
- `MTL::Device::maxThreadgroupMemoryLength()` — device limit, in bytes

The prior agent's syntax-only check (`clang++ -fsyntax-only`) does not catch
unknown method calls without Metal-cpp headers, and the CMake build was already
broken on environmental issues — so nothing actually compiled this path.

## Verified HIGHs (correctness/coverage gaps)

### 4. `validate_real_kv.py:210` — never asserts fused path actually runs

The harness assumes `finalize_deferred_kv_caches(cache)` makes the IsoQuant
cache fused-capable, but never asserts `supports_fused_latent_attention is
True` post-finalize. With the wrong bit_width (e.g., `--bits 4`) or unsupported
head_dim, the cache would silently stay in reconstruct mode and the gate would
report PASS while validating the wrong code path.

### 5. `validate_real_kv.py:269` — NaN/Inf silently passes the gate

`nan < threshold` and `nan > threshold` both evaluate False. If FP16 reference
or IsoQuant logits contain non-finite values, the verdict logic happily reports
PASS while metrics are nonsense.

### 6. `isoquant_kv_test.cpp:296` — stride regression test does not exercise the fix

The BLOCKER 4 fix added a `cache_stride` parameter and updated tests. But the
test still passes `SEQ_LEN` as the stride, while the runtime cache has
`capacity = SEQ_LEN * 2`. The test never hits the `cache_stride != T` path,
defeating the regression test's purpose.

## Other findings (defer or document)

### `ab4f936` runtime.cpp — exception leaks (HIGH, deferred)

`pack_indices()` status-check throws leak `src_buf`/`dst_buf`.
`append_compressed()` chunk-2 throw leaks chunk-1. Same pattern in `736a88c`
for `q_scaled_buf` / `q_buf` / `out_buf`. Real concern, but only on error
paths (where the process likely dies anyway). RAII refactor needed across all
Metal buffer ownership — bigger scope, deferred to separate work.

### `ab4f936` kernels.metal:348 — added barrier may be a no-op (LOW)

Codex argues the existing post-init barrier already ordered `max_logit`/`denom`
before reads. The new pre-init barrier may be redundant. Worth re-reading the
kernel critically; could keep as documented "q_local synchronization" rather
than removing. Not correctness-affecting either way.

### `55e2d64` — mask-shape handling diverges across 5 model files (MEDIUM)

Gemma4's IsoQuant branch does:
```python
if mask is not None and isinstance(mask, mx.array):
    if mask.shape[-1] != keys.shape[-2]:
        mask = mask[..., -keys.shape[-2]:]
```
The other 4 files (qwen2/qwen3_moe/qwen3_next/nemotron_h) don't. Copy-paste
diverged on a real shape guard. Worth normalizing.

## Overstated / misrepresented commit messages

These are fair criticisms of the close-out language used:

- `736a88c` claimed "D=512 budget validation closed" — only checks one kernel
  and via a non-existent API
- `ab4f936` claimed "max_logit init race fix" — added barrier is redundant; the
  existing post-init barrier already ordered the writes
- `ab4f936` claimed stride coverage — but tests still pass `SEQ_LEN`, never hit
  the cache.capacity ≠ T path
- `55e2d64` claimed "all five files guard TurboQuant" — Gemma4 does not
- `b4ebfb5` claimed "actually exercises fused MLA" — could silently validate
  fallback

## Disposition

| Finding | Action |
|---|---|
| BLOCKER 1 (qwen3_next init) | Fix in follow-up commit |
| BLOCKER 2 (gemma4 TurboQuant) | Fix in follow-up commit |
| BLOCKER 3 (Metal API) | Fix in follow-up commit |
| HIGH 4 (validate fused assertion) | Fix in follow-up commit |
| HIGH 5 (validate NaN guard) | Fix in follow-up commit |
| HIGH 6 (stride test) | Fix in follow-up commit |
| HIGH (RAII leak paths) | Defer — separate refactor |
| MEDIUM (mask handling divergence) | Defer — separate cleanup |
| LOW (max_logit barrier redundancy) | Defer — non-correctness |
| 3 SMELL findings (unused param, hardcoded venv, isfinite check) | Defer — non-correctness |

## Process lessons

- Same-family-of-models doing implementation AND review is a blind spot.
  Different lineage (Codex via OpenAI) caught what 2 sonnet agents missed.
- Syntax-only checks (`clang++ -fsyntax-only`) without target headers don't
  catch unknown method calls. Need a real CMake build to validate native
  pipeline changes.
- Spot-grep before trusting agent claims of "all files updated" — copy-paste
  drift is real (qwen3_next missing init, gemma4 missing TurboQuant elif).
- "Verdict" claims in commit messages should be more cautious — "addressed"
  ≠ "verified" ≠ "compiled" ≠ "tested under load".
