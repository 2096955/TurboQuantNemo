# Phase 3 Handover: Kimi MLA IsoQuant Cache

**Date:** 2026-04-29
**Status:** Design complete, implementation not started
**Predecessor:** Phase 2 complete (load + decode smoke passed)
**Plan:** `docs/superpowers/plans/2026-04-28-kimi-k26-rotaryquant-pathway.md`

---

## What to build

A `KimiMLAIsoQuantCache` class that compresses `kv_latent` (512-D) via IsoQuant rotation+quantization while storing `k_pe` (64-D) raw. This is the DKV constraint — RoPE dimensions must never be rotated or quantized.

## Key finding from Task 3.1 (already verified)

The placeholder in `mlx-lm/mlx_lm/models/kimi_mla_isoquant_dkv.py` assumes a **448+64 split inside the 512-D latent**. This is WRONG.

**Reality from K2.6 checkpoint (`modeling_deepseek.py:776-777`):**
```python
compressed_kv, k_pe = torch.split(
    compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
```

The split happens at `kv_a_proj_with_mqa` output (576-D = 512 + 64):
- `kv_latent` = full 512-D (ALL compressible, already separated from RoPE)
- `k_pe` = 64-D (separate tensor, stored raw)

There is NO internal split within `kv_latent`. The `DKV_CONTENT_DIM=448` constant is wrong and must be removed.

## MLX attention code path (`deepseek_v3.py:118-169`)

```python
# Line 134: split at kv_lora_rank
compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
kv_latent = self.kv_a_layernorm(compressed_kv)  # (B, L, 512)
kv_latent = mx.expand_dims(kv_latent, axis=1)   # (B, 1, L, 512) — 1 KV head

# Line 144-145: cache stores (kv_latent, k_pe)
if cache is not None:
    kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)

# Line 147: RoPE scores from raw k_pe
pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)

# Lines 155-166: decode vs prefill attention
if L == 1:  # decode
    q_nope = self.embed_q(q_nope)    # project q to latent space (512-D)
    k = v = kv_latent                 # latent IS both K and V
else:       # prefill
    k = self.embed_q(kv_latent, transpose=False)  # per-head K from latent
    v = self.unembed_out(kv_latent)                # per-head V from latent

output = scaled_dot_product_attention(q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores)
```

**Critical:** `update_and_fetch` must return FULL accumulated history (not just new tokens). The attention code uses returned `kv_latent` and `k_pe` directly — there is no `cache.reconstruct_keys()` call in MLA.

`scaled_dot_product_attention` (`base.py:108-137`) does NOT check for fused attention — it delegates directly to `mx.fast.scaled_dot_product_attention` with the passed K/V.

## Cache design (composition, not inheritance)

```python
class KimiMLAIsoQuantCache:
    """Compress kv_latent (512-D) via IsoQuant, store k_pe (64-D) raw."""

    def __init__(self, kv_lora_rank, qk_rope_head_dim, bit_width, layer_idx, codebook_dir, seed=42):
        # Use IsoQuantKVCache instance for compression machinery only
        # (rotation matrices, codebooks, _compress_batch/_decompress_batch)
        self._iso = IsoQuantKVCache(
            num_heads=1, head_dim=kv_lora_rank,
            bit_width=bit_width, layer_idx=layer_idx,
            codebook_dir=codebook_dir, seed=seed,
        )
        self._kv_lora_rank = kv_lora_rank
        self._rope_dim = qk_rope_head_dim

        # State
        self._deferred = True
        self._fp16_latent: list[mx.array] = []   # 3D: (1, seq, 512)
        self._fp16_pe: list[mx.array] = []        # 3D: (1, seq, 64)
        self._compressed_latent: dict | None = None
        self._pe_buffer: mx.array | None = None
        self.offset = 0
```

### update_and_fetch contract

- **Input:** `kv_latent (B=1, 1, L, 512)`, `k_pe (B=1, 1, L, 64)`
- **Output:** `(all_kv_latent, all_k_pe)` — full history, 4D
- **Prefill (deferred=True):** Accumulate FP16, return concatenated history (zero error)
- **Decode (deferred=False):** Compress new latent via `self._iso._compress_batch(lat[0])`, append to `_compressed_latent`. Append k_pe to `_pe_buffer`. Decompress full `_compressed_latent` via `self._iso._decompress_batch()` and return with full `_pe_buffer`.

### finalize_deferred_prefill contract

- Concatenate all `_fp16_latent` chunks along axis=1
- Bulk compress via `self._iso._compress_batch(all_latent)`
- Concatenate all `_fp16_pe` into `_pe_buffer`
- Set `_deferred = False`
- Clear FP16 buffers

### IsoQuant internal methods used

- `_compress_batch(x)` — input: 3D `(heads=1, seq, dim=512)`, output: `{"indices": uint8, "x_norm": float16}`
- `_decompress_batch(compressed)` — input: dict, output: 3D `(heads=1, seq, dim=512)`
- These use the instance's rotation matrices and codebook centroids/boundaries internally

### Codebooks

`dim_512_3bit.npz` exists in `mlx-lm/mlx_lm/models/turboquant_codebooks/`. No new codebook generation needed.

## Files to create/modify

### 1. Replace: `mlx-lm/mlx_lm/models/kimi_mla_isoquant_dkv.py`

Current file is a 29-line placeholder with wrong constants. Replace entirely with `KimiMLAIsoQuantCache` class.

### 2. Create: `mlx-lm/tests/test_kimi_mla_isoquant_dkv.py`

Tests (all synthetic, no real checkpoint needed):

| Test | What it proves |
|------|---------------|
| `test_init` | Cache creates with kv_lora_rank=512, qk_rope_head_dim=64 |
| `test_prefill_passthrough` | During deferred prefill, returned kv_latent matches input exactly |
| `test_pe_exact_roundtrip` | k_pe is NEVER compressed — exact bit-for-bit roundtrip |
| `test_latent_compressed_within_tolerance` | After finalize, decompressed latent is close to original (cosine sim > 0.95) |
| `test_deferred_prefill_flow` | prefill → finalize → decode sequence works end-to-end |
| `test_offset_tracking` | offset increments correctly through prefill and decode |
| `test_multi_decode_concat` | Multiple decode steps accumulate correctly |

### 3. Modify: `mlx-lm/mlx_lm/models/cache.py`

In `make_prompt_cache`, detect Kimi/DeepSeek-V3 MLA models and dispatch `KimiMLAIsoQuantCache`:

```python
# Inside make_prompt_cache, after config extraction:
model_type = getattr(config, "model_type", "")
is_mla = model_type in ("deepseek_v3", "kimi_k2", "kimi_k25") or hasattr(config, "kv_lora_rank")

if is_mla and kv_cache_type == "isoquant":
    from .kimi_mla_isoquant_dkv import KimiMLAIsoQuantCache
    kv_lora_rank = getattr(config, "kv_lora_rank", 512)
    qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 64)
    # ... dispatch KimiMLAIsoQuantCache instead of IsoQuantKVCache
```

### 4. Update: plan checkboxes in `2026-04-28-kimi-k26-rotaryquant-pathway.md`

Check Task 3.1 (steps 1-2) and Task 3.2 (steps 1-2) when tests pass.

## What NOT to do

- Don't add fused attention — Phase 3 is correctness-first (reconstruct path)
- Don't optimize decode performance — O(T×D) decompression per step is fine
- Don't modify `deepseek_v3.py` — the attention code already uses `cache.update_and_fetch` correctly
- Don't split kv_latent internally — the full 512-D is compressible

## Gate criteria

From the plan:
> Default MLA attention and Kimi compressed-MLA attention match within tolerance on synthetic tensors for prefill + decode.

Concretely:
1. Tests pass
2. k_pe round-trips exactly (bit-for-bit)
3. kv_latent decompression error within IsoQuant tolerance (cosine sim > 0.95 for 3-bit)
4. Offset tracking correct through prefill→finalize→decode
