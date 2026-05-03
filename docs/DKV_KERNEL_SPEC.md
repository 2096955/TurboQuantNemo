# DKV-Aware Kernel Spec for MLA-Based Models (Kimi K2.5 / K2.6)

**Status:** draft v1, intended for `docs/DKV_KERNEL_SPEC.md` in the RotaryQuant repo.
**Owners:** kernel and model-port tracks.
**Blocks:** Kimi K2.5/K2.6 model port; long-context (≥ 16 K) IsoQuant validation on MLA.
**Companion:** `scripts/bandwidth_budget.py` (predict-vs-measure for the same models).

---

## 1. Background

The four fused Metal kernels in `mlx-lm/mlx_lm/fused_kv_decode_kernels.py` (Kernels A/B/C/D) currently treat every KV cache entry as a monolithic `head_dim` vector. They apply WHT + SO(4) rotation, scalar-quantise to 3 bits, attend in rotated space, and inverse-rotate the aggregated output once per attention call.

This works for standard GQA models — Gemma 4, Nemotron-H, Qwen 3, Qwen 3.6-35B-A3B — where every dimension of the cache entry carries content. It is **not safe** for Multi-Head Latent Attention (MLA) as used by Kimi K2.5 and (presumed) K2.6.

The failure mode is silent at short context. Rotating the RoPE-bearing dimensions of an MLA latent smears positional phase. The model still attends to plausible tokens in a 500-token quality gate and the gate passes. PPL drift only becomes obvious past ~8 K tokens, by which point the cause is hard to attribute. This spec exists to make the split explicit at the kernel boundary so the failure mode cannot occur.

---

## 2. The DKV constraint

MLA caches a low-rank latent per token. Two layout conventions are in circulation; the kernel spec must support both.

**Standard DeepSeek MLA layout** stores `(c_t, K^R_t)` as separate tensors:
- `c_t ∈ ℝ^{d_c}` — content latent (typically `d_c = 512` for Kimi-class), no positional information.
- `K^R_t ∈ ℝ^{d_R}` — decoupled RoPE'd key (typically `d_R = 64`), shared across heads in the absorbed form.
- Total per layer per token: `d_c + d_R = 576` floats.

**Project-documented Kimi K2.5 layout** describes the 512-dim latent as splitting *internally* into 448 content + 64 RoPE positional sub-spaces, with no separate `K^R_t`:
- Total per layer per token: 512 floats.
- Compressible portion: dims `[0:448]`.
- RoPE portion: dims `[448:512]`.

**Confirm the actual layout from the K2.5 / K2.6 checkpoint config before committing kernels.** The spec below is parameterised on the split offset so it survives either interpretation; the model-port code (`mlx-lm/mlx_lm/models/kimi_k2.py`) is responsible for setting the correct values.

The DKV invariant is:

> **WHT and SO(4) rotation are valid on content dimensions only. Lloyd-Max scalar quantisation is valid on rotated content dimensions only. Both are forbidden on RoPE-bearing dimensions, which must remain in FP16 throughout the cache lifetime.**

Violating this invariant produces context-length-dependent quality loss with no clean signal at short contexts. It is the single most important correctness rule for MLA support.

---

## 3. Cache entry format

Add a per-cache descriptor:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class MLAEntryLayout:
    total_dim:      int          # e.g. 512 (Kimi K2.5 user split) or 576 (standard)
    content_offset: int          # e.g. 0
    content_dim:    int          # e.g. 448 or 512
    rope_offset:    int          # e.g. 448 or 512
    rope_dim:       int          # e.g. 64
    quantise_content: bool = True
    quantise_rope:    bool = False     # invariant — never True

    def __post_init__(self):
        assert self.content_offset + self.content_dim == self.rope_offset
        assert self.rope_offset + self.rope_dim == self.total_dim
        assert not self.quantise_rope, "DKV invariant: RoPE dims must stay FP16"
```

The cache stores two arrays per layer rather than one:

```python
class IsoQuantMLACache:
    content_packed: mx.array   # uint8, shape (T, content_dim * 3 / 8) after pack
    content_meta:   mx.array   # per-block SO(4) quaternions + Lloyd-Max scales
    rope_fp16:      mx.array   # float16, shape (T, rope_dim) — stored verbatim
    layout:         MLAEntryLayout
```

This separation is the single most consequential structural change. The current `IsoQuantKVCache` exposes one packed plane with one set of metadata; the MLA path needs two storage planes with different access patterns and different invalidation rules. The packed plane is invalidated on append and rebuilt lazily (existing behaviour). The FP16 RoPE plane is append-only and never rebuilt.

---

## 4. Kernel A — `fused_qk_dot_mla`

**Current behaviour (monolithic):**
```
score[t] = q · dequantise(K[t])   for t in [0, T)
```
where `q` and the dequantised `K[t]` are both `head_dim`-wide.

**New behaviour (DKV-aware):**
```
score_content[t] = q_content · dequantise(K.content_packed[t])     # 448 dims
score_rope[t]    = q_rope · K.rope_fp16[t]                          # 64 dims, plain FP16
score[t]         = (score_content[t] + score_rope[t]) / sqrt(head_dim_for_softmax)
```

**Implementation notes:**
- `q_content` and `q_rope` are pre-split by the model layer before the kernel is called. The kernel does not know how to split `q`; it expects two inputs.
- The content path is identical to the current Kernel A on a vector of width `content_dim` (e.g. 448), with one change: the threadgroup memory budget rises from `head_dim * 4 bytes` to `content_dim * 4 bytes`. For 448 this is 1.75 KB, well within Apple Silicon's per-threadgroup memory.
- The RoPE path is a straight FP16 dot product of width `rope_dim`. No decode, no rotation, no quantisation. It can run on the same threadgroup or be delegated to a small companion kernel — measurement should decide.
- The `/ sqrt(head_dim_for_softmax)` divisor is the **original per-head dimension** (e.g. 128 for the Q head), not the latent dimension. This matters because the absorbed-form Q projection has already collapsed the per-head structure.
- Threadgroup layout unchanged: `grid=(32 * seq_len, num_heads, 1)`, `threadgroup=(32, 1, 1)`. The split happens inside the threadgroup.

**Error budget:**
- Content score error: same bound as monolithic Kernel A on a 448-dim vector — `O(σ_q · sqrt(content_dim))` per the Hanson-Wright analysis in the paper.
- RoPE score error: bit-exact in FP16 (no compression).
- Combined: dominated by content path. No new error source.

---

## 5. Kernel B — softmax (unchanged)

Standard `mx.softmax`. No change. The split happens upstream and is invisible by the time scores reach softmax.

---

## 6. Kernel C — `fused_value_accum_mla`

In MLA absorbed form, V is reconstructed from `c_t` only via `W_UV`. **There is no value RoPE.** The current Kernel C semantics carry over, with one change of dimension.

**New behaviour:**
```
v_acc_content[d] = sum_t weights[t] * dequantise(V.content_packed[t, d])
                       for d in [0, content_dim)
```

The result `v_acc_content` has shape `(content_dim,)` and is in **rotated space**. The rope plane is unused for V and must not be read.

If a future implementation caches K and V independently rather than recomputing both from `c_t` (which would defeat the purpose of MLA but is technically possible), the rope plane is still unused for V. The kernel signature must enforce this: `value_rope` is not an input.

**Threadgroup layout:** `grid=(min(content_dim, 128) * num_heads, 1, 1)`. Per-head striping over content dimensions. The dual-strategy short-vs-long sequence selection from the existing Kernel C carries over unchanged (word-parallel for `T < 512`, dim-parallel for `T ≥ 512`).

---

## 7. Kernel D — `metal_rotate_inverse_mla`

The current Kernel D applies WHT + SO(4) inverse over the full `head_dim`. The new kernel applies the inverse only to the content sub-vector, then projects to model dimension via the absorbed `W_UV`.

**New behaviour:**
```
out_content_unrotated = WHT_inverse(SO4_inverse(v_acc_content))   # 448 dims, FP16
out_model_dim         = W_UV @ out_content_unrotated              # hidden dims, FP16
```

Two subtleties:

1. **`W_UV` projection placement.** In the current pipeline this happens outside the kernel, in the model file. Kernel D should still end at returning the unrotated FP16 content. The model file calls `W_UV` afterwards. This keeps Kernel D as a pure rotation-inverse and avoids tying the kernel to absorbed-form specifics.

2. **FMA accounting.** In monolithic GQA the inverse runs once per head per token: `1408 FMAs × n_heads`. For Kimi K2.6 with 128 query heads that is ~180 K FMAs per token. In MLA absorbed form, the inverse runs once per token at the latent dimension — the projection back to per-head model space happens via `W_UV` afterwards, which is a normal matmul. The kernel saves roughly `n_heads × (1408 - 1408 / scale_factor)` FMAs per token, where the scale factor depends on whether `content_dim` is per-head or per-latent. **This is the empirical question the bandwidth_budget.py meter resolves.**

---

## 8. Compressor (write path) changes

`IsoQuantCompressor.compress()` currently takes a single `(T, head_dim)` array and produces a packed 3-bit cache with metadata. For MLA, split first then compress separately:

```python
def compress_mla(self, c_t: mx.array, layout: MLAEntryLayout) -> tuple[mx.array, mx.array, mx.array]:
    """Returns (content_packed, content_meta, rope_fp16)."""
    content = c_t[..., layout.content_offset : layout.content_offset + layout.content_dim]
    rope    = c_t[..., layout.rope_offset    : layout.rope_offset    + layout.rope_dim]
    content_packed, content_meta = self._compress_content(content)
    rope_fp16 = rope.astype(mx.float16)
    return content_packed, content_meta, rope_fp16
```

For deferred prefill: the FP16 buffer holds the full `total_dim` cache. At the prefill→decode boundary, the bulk-compress step splits each token, compresses content via WHT + SO(4) + Lloyd-Max + pack, and writes both planes into the `IsoQuantMLACache` structure. The RoPE plane is a straight `astype(mx.float16)` copy.

---

## 9. Correctness tests

These must all pass before the DKV path is enabled by default. Land them as `tests/test_mla_kernels.py` alongside the existing `tests/test_fused_attention_correctness.py`.

1. **Round-trip with split.** Compress a known `c_t` of shape `(T=128, total_dim)` with the documented layout. Reconstruct via dequantise + inverse rotation on content, concat with the rope plane. Compare to original.
   - Content tolerance: cosine ≥ 0.997, max element error < 0.01.
   - RoPE plane: bit-exact in FP16 (it must not have been touched).

2. **Attention score equivalence.** Run attention over `(T=128, total_dim)` cached entries with (a) full FP16 reference and (b) DKV-aware compressed path. Compare the top-10 score ordering across ≥ 100 random query vectors.
   - Required: top-10 overlap ≥ 9/10, top-1 match rate ≥ 0.99.

3. **Long-context drift sentinel.** Generate 4096, then 8192, then 16384 tokens with both the FP16 reference path and the DKV-aware compressed path. Measure PPL divergence per 1 K context window.
   - Required: monotone-flat or decreasing.
   - **A linearly-increasing divergence is the signature of RoPE smearing.** If you see this, either the layout offsets are wrong or a kernel is rotating into the rope range. This test is the canary; it must run on every PR that touches MLA kernels.

4. **Per-token RoPE phase preservation.** For each cached position, recover the rotation angle from the rope plane and confirm it matches the expected RoPE phase for that position.
   - Required: bit-exact in FP16. Any drift means a kernel wrote into the rope plane by mistake.

5. **DKV-aware kernel parity vs MLX-ops fallback.** Run both paths on identical inputs of shape `(T=512, total_dim)` and compare element-wise.
   - Required: max element error < 4e-6 (matches the existing Kernel A/C/D parity bound).

6. **Threadgroup-memory bound check.** Statically assert at kernel-launch time that `content_dim * 4 ≤ threadgroup_memory_limit`. For 448 dims this is 1.75 KB and is fine on M-series. For larger content dims (a hypothetical K3 with 1024-dim latent) the kernel must refuse to launch and fall back to MLX-ops rather than silently corrupt.

---

## 10. Fallback and validation gates

Extend the current `_fused_metal_ok` flag to a tri-state for the MLA path:

```python
from enum import Enum

class FusedMetalState(Enum):
    UNTESTED        = 0
    OK_MONOLITHIC   = 1   # works for GQA models
    OK_DKV          = 2   # works for MLA models
    FAILED_DKV      = 3   # MLA path failed parity, fall back to MLX-ops
```

If `OK_DKV` cannot be reached after 3 attempts on a fresh process, refuse to enable IsoQuant on the MLA path. Log loud, fall back to the MLX-ops reference. This must not be silent: a visible warning at first generation token, plus a structured log entry with the parity error breakdown.

**Pre-deployment validation gate** (matches the existing `kimi-mla-rotorquant-validation` plan item from `8 April Plan` Appendix H.3):

- Run `validate_real_kv.py` on Kimi K2.5/K2.6 MLA latent representations with the DKV-aware kernels.
- Measure cosine similarity and top-k retrieval accuracy with vs without IsoQuant on content.
- **Decision rule:** if the marginal compression benefit over MLA alone is < 10%, OR if PPL @ 8 K rises by > 0.5 versus FP16 reference: skip IsoQuant on this path entirely. MLA's architectural compression is enough on its own and the kernel work is not worth the correctness surface area. Document the negative result either way — it is publishable.

---

## 11. Performance budget (predicted)

Per-token kernel-side cost comparison for Kimi K2.6 at `content_dim=448`, `rope_dim=64`, `n_heads=128`, `T=4096`. These are FMA counts only; real throughput depends on memory bandwidth, dispatch overhead, and threadgroup occupancy.

| Operation | Monolithic GQA reference (per head, hypothetical) | DKV-aware MLA (per latent) |
|---|---|---|
| Kernel A (QK score) | `head_dim × T = 524 K` FMAs | `(content_dim + rope_dim) × T = 2.10 M` FMAs |
| Kernel C (V accum) | `head_dim × T = 524 K` FMAs | `content_dim × T = 1.83 M` FMAs |
| Kernel D (inverse) | `1408 × n_heads = 180 K` FMAs | `1408 × 1 = 1.4 K` FMAs |

Kernel A is *larger* in absolute FMAs because it now operates on the full latent for scoring rather than on each per-head `head_dim` projection separately. Whether this is faster end-to-end depends on whether the ~130× saving in Kernel D and the elimination of per-head reconstruction work offset the score-path increase.

**Prediction:** at long context (T ≥ 16 K) MLA's KV size advantage swamps the score-path FMA increase, so the DKV-aware path wins net. At short context (T < 1 K) the monolithic GQA path is faster because there is no compression overhead at all. Crossover point is the empirical question.

This is exactly what `scripts/bandwidth_budget.py` and the `BandwidthMeter` instrumentation hooks are designed to resolve. Run the predict mode first to set expectations; then wire the meter into the decode loop and compare measured-vs-predicted before committing.

---

## 12. What this spec does not cover

- **Vision encoder integration (MoonViT-3D)** — out of scope for the text-decode KV path.
- **Streaming / MTP speculative decoding** — Kimi K2.5 does not document MTP. Even if K2.6 adds it, speculative decoding remains incompatible with offloaded MoE per the existing project decision (cache thrashing).
- **CUDA / Vulkan ports** — Metal-only spec. The Mojo prototypes in `mojo-bench/` can mirror the structure but are not part of the production path.
- **The `W_UV` absorbed projection itself** — assumed to be implemented in `mlx-lm/mlx_lm/models/kimi_k2.py`, not in the Metal kernels.
- **Q-LoRA and absorbed-form Q projection** — same: model-file responsibility, not kernel-spec scope.

---

## 13. Sequencing for implementation

In order:

1. Confirm the K2.6 MLA layout (`d_c`, `d_R`, whether RoPE is folded into the latent or decoupled). **Do not write kernel code before this.**
2. Land `MLAEntryLayout` and `IsoQuantMLACache` (compressor + storage), with the round-trip test (test 1 above) passing first. No kernels yet.
3. Land MLX-ops MLA reference path (slow but correct) and the long-context drift sentinel (test 3). This becomes the baseline against which fused kernels are validated.
4. Land `fused_qk_dot_mla` and `metal_rotate_inverse_mla`. Land `fused_value_accum_mla` last (it's the dominant runtime cost and most worth tuning when the surrounding scaffolding is stable).
5. Run the validation gate. If it fails the < 10% / > 0.5 PPL test, **stop and document the negative result.** Do not optimise a path the gate has rejected.
6. Wire `BandwidthMeter` into the decode loop and confirm measured tok/s sits within ±20% of predicted. Divergence beyond that means either the kernels are not behaving as the FMA accounting suggests or the meter is missing a category.

This sequencing is conservative on purpose. The DKV invariant is the kind of bug that does not show up until production-scale long-context use, and by then the cause is hard to attribute. Every step above is designed to catch a violation early and loudly.
