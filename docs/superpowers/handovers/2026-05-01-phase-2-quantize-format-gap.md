# Phase 2 Quantize — Lessons + Format Gap Handover

**Date:** 2026-05-01
**Plan:** `docs/superpowers/plans/2026-04-29-self-speculative-kimi.md` (Phase 2)
**Status:** Phase 2 quantization succeeded structurally; loader gate-blocked due to format mismatch. User chose to step back and reconsider.

## What we built (committed scripts)

| Script | Purpose | Reusable for |
|---|---|---|
| `scripts/quantize_kimi_2bit.sh` | Two-stage convert wrapper (dequantize → re-quantize) with disk-space gate | Any large MLX requantize where source is already pack-quantized |
| `scripts/convert_kimi_2bit_chunked.py` | Per-shard, per-tensor materialize-and-save loop | **Any large MLX quantize that hits `kIOGPUCommandBufferCallbackErrorTimeout`** |
| `scripts/unstack_kimi_2bit.py` | Stacked SwitchLinear → per-expert layout for ExpertOffloadManager | Any future Kimi/DeepseekV3 conversion that needs offload-compatible output |
| `scripts/recover_unstack_kimi_2bit.py` | Recover from a partial unstack (rebuild index + config + passthrough) | Any failed unstack run |

## Three failures we fixed

### 1. `mlx_lm.convert --mixed-expert-bits 2` silently no-op'd
Root cause: source was `compressed-tensors pack-quantized` 4-bit. `mlx_lm.convert` doesn't dequantize source pack-quantized weights before applying the `--mixed-expert-bits` predicate. The already-`QuantizedSwitchLinear` modules are silently skipped by `nn.quantize`. Output came out at 4.996 bpw (i.e. the source 4-bit + per-layer config bloat), not the requested 2-bit.

Fix: explicit two-stage convert. Stage 1 `--dequantize` source → bf16 intermediate (~1.9 TB for Kimi K2.6). Stage 2 `--quantize --mixed-expert-bits 2` on the bf16 intermediate. See `scripts/quantize_kimi_2bit.sh` (commit 6d75ce3).

### 2. `kIOGPUCommandBufferCallbackErrorTimeout` during `save_safetensors`
Root cause: MLX's lazy-eval graph stays unmaterialized until `mx.save_safetensors` triggers `mx.eval` per shard. On Kimi K2.6 each shard's eval pulls in massive routed-expert quantization work, generating Metal command buffers that exceed the macOS GPU watchdog (~5 second limit). Crash signature: `libc++abi: terminating due to uncaught exception of type std::runtime_error: [METAL] Command buffer execution failed`.

Fix: per-shard, per-tensor materialize loop with `mx.eval(arr); mx.synchronize()` between every leaf. Each Metal command buffer stays small enough to clear the watchdog. See `scripts/convert_kimi_2bit_chunked.py`.

### 3. OOM (SIGKILL) during chunked save
Root cause: even with per-tensor eval, the model object held references to every parameter through its module hierarchy. `del shard` in the save loop didn't actually free anything because `model.parameters()` still pointed to the tensors. Memory grew unboundedly, kernel killed the process.

Fix: after building shards, call `model.update(tree_map(lambda _: mx.array([]), model.parameters()))` to break the model's references. Then `del shard; gc.collect(); mx.clear_cache()` per iteration actually releases memory. See `convert_kimi_2bit_chunked.py:110-120`.

## The remaining gate (NOT fixed in this session)

After all three fixes above, the chunked converter produced a structurally correct 2-bit checkpoint:
- 180 routed-expert layers @ 2-bit ✓
- 551 attn/shared/dense layers @ 4-bit ✓
- 302 GB total, 2.525 bits/weight average ✓
- 64 stacked shards, clean exit code 0

But the resulting checkpoint **cannot be loaded with `expert_offload=True`** for two cumulative reasons:

### Gate A: layout (FIXED via unstack)
`ExpertOffloadManager` expects per-expert keys:
```
language_model.model.layers.<L>.mlp.experts.<E>.{gate,up,down}_proj.<kind>
```
But MLX's `SwitchLinear` saves stacked tensors:
```
language_model.model.layers.<L>.mlp.switch_mlp.{gate,up,down}_proj.<kind>  shape (384, ...)
```
Fix: `scripts/unstack_kimi_2bit.py` slices each `(384, ...)` tensor into 384 per-expert tensors with the right names. Output: 209,378 keys, 302 GB, 61 shards.

### Gate B: quantization storage convention (NOT FIXED)
After unstacking, the loader (utils.py:374-388) still rejects with:
> `expert_offload for kimi_k25 is not compatible with this quantized checkpoint layout. Routed expert scale tensors must be available as per-expert keys for offloading.`

The check looks for the suffix `.weight_scale` (compressed-tensors naming). MLX's standard quantize emits `.scales`. The `_KIMI_EXPERT_KEY_RE` regex accepts both, but the loader's gate is narrower than the regex.

```python
# utils.py:374-388
scale_suffixes = (".weight_scale",) if _mt in ("kimi_k25", "kimi_k2") else (".scales",)
has_repacked_scales = any(
    k.endswith(scale_suffixes) and is_expert_weight_key(k, model_type=_mt)
    for k in wm
)
if not has_repacked_scales:
    raise ValueError(...)
```

Beyond the gate, deeper format differences exist:
| Convention | weight | scale | bias / zero-point |
|---|---|---|---|
| `compressed-tensors pack-quantized` (Kimi source) | `weight_packed` (uint32) | `weight_scale` | none (symmetric); zero-point reconstructed at runtime as `-8 * scale` for sym int4 |
| MLX affine (our convert output) | `weight` (uint32) | `scales` | `biases` (per-group affine offset) |
| Match expected? | rename only | rename only | **Real semantic gap** |

The bias gap is the hard part. Compressed-tensors source assumes symmetric int4 and reconstructs the zero-point algebraically. MLX affine 2-bit has explicit per-group biases that aren't `-8 * scale`. Dropping our biases would change the math and likely break inference.

## Three options for next time

When the user comes back to this:

### Option B1 (recommended if pursued): patch mlx_lm
1. Modify the gate at `mlx-lm/mlx_lm/utils.py:380-388` to also accept `(.scales,)` for Kimi.
2. Verify `_load_expert_pair_tensors` and `prepare_gather_triple_quantized` (in `mlx-lm/mlx_lm/expert_offload.py`) actually load and use `.scales` + `.biases` for inference, not just `.weight_scale`.
3. Verify the MoE forward path's `mx.gather_qmm` call gets the right `mode='affine'` and uses the biases.

If steps 2-3 require more changes, they're likely small (the underlying quantized linear primitive is the same). If they're large, this option blows up.

### Option B2: transcode our checkpoint to compressed-tensors
1. Rename `.weight` → `.weight_packed`, `.scales` → `.weight_scale`.
2. Synthesize `.weight_shape` tensors (compute pre-pack shape from `.weight` shape and `bits`).
3. Drop `.biases`. **Risk**: MLX 2-bit affine biases are not `-8 * scale`. Dropping them changes inference output.
4. Set `quantization_config.quant_method = "compressed-tensors"`, `format = "pack-quantized"`.

This is fast to implement but inference correctness is in question.

### Option B3: skip this entirely
The 1-3 tok/s ceiling on this hardware doesn't change regardless of which sub-bit draft works. The whole self-speculative-on-Kimi pursuit is fighting a ~150× gap from the M4 Max bandwidth-and-compute floor. A different model that fits in RAM (e.g. Qwen3-30B-A3B) would deliver speedups that actually matter to a user.

## Disk state at handover

After cleanup (per user direction):
- `/Volumes/Samsung9904tb/Kimi-K2.6/` — 554 GB, source 4-bit checkpoint (kept)
- bf16 intermediate, 2-bit stacked, 2-bit per-expert all DELETED (~2.5 TB freed)

To re-derive the 2-bit per-expert checkpoint from scratch with current scripts:
- Stage 1: `bash scripts/quantize_kimi_2bit.sh` (~30 min wall, ~1.9 TB peak intermediate)
  - But this uses the OLD `mlx_lm convert` path which crashes; would need to call `convert_kimi_2bit_chunked.py` for Stage 2 instead
- Cleaner: skip Stage 1 dequantize when patched script supports loading source directly, OR keep the two-stage approach with chunked Stage 2

## Time spent

This session burned ~6 hours of orchestrator time on Phase 2 alone. The infrastructure scripts (chunked convert + unstack + recover) are reusable assets. The 2-bit Kimi checkpoint itself is gone. Path B investigation deferred.
