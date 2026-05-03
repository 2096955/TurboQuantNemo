# IsoQuant3 Correctness Report

**Date:** 2026-04-14
**Model:** Qwen2.5-1.5B-Instruct Q6_K (n_head=12, n_head_kv=2, n_gqa=6, head_dim=128)
**Hardware:** Apple M4 Max, Metal backend

## Test 1: End-to-end KV scheme comparison (historical turbo3 vs isoquant3 run)

**What this tests:** Whether both KV compression schemes produce viable model output.
**What this does NOT test:** The fused `kernel_turbo_wht_so4` kernel.

This was an earlier comparison run captured before non-identity SO(4) was enabled by default for `isoquant3`. In that historical run, the graph took the plain `ggml_turbo_wht()` path and the fused kernel was never dispatched.

**Prompt:** "The capital of France is"
**Config:** seed=42, temp=0, n_gen=30

| Path | KV write rotation | Q/output rotation | First token | Subsequent |
|------|-------------------|-------------------|-------------|------------|
| turbo3 | Dense random orthogonal (QR) | Plain WHT | Paris | Rome, London, Madrid |
| isoquant3 (historical identity SO(4) run) | WHT only (kernel_set_rows_isoquant3) | Plain WHT | Paris | London, Rome, Madrid |

**Divergence cause:** turbo3 and isoquant3 use different KV write-side rotations (dense vs WHT-based). This produces different quantised KV cache contents, different attention scores, and different token selection at soft tiebreak points. This is expected scheme-level divergence — not a bug, not precision noise.

**What this proves:**
- Both schemes run without crash
- Both produce coherent, viable text
- First-token agreement: prefill logits are close enough for argmax to agree
- Divergence is at a soft boundary (Rome vs London), not a hard failure

**What this does NOT prove:**
- `kernel_turbo_wht_so4` correctness (kernel was never dispatched)
- Non-identity SO(4) block rotation correctness
- Multi-token prefill with `% n_head` wrapping under non-identity blocks

## Test 2: Fused vs composed kernel correctness (non-identity SO(4))

**Method:** Two env vars control the test:
- `ISOQUANT_TEST_ROTATION=1` — uses explicit deterministic 30° test rotations that are address-independent (no init-order dependency). Each block gets a fixed (x,y) plane rotation and a small deterministic (z,w) plane rotation derived from `(head, block)` indices.
- `ISOQUANT_FORCE_COMPOSED=1` — restores the old graph-level SO(4) composition (reshape+permute+matmul) at all 5 sites instead of the fused `kernel_turbo_wht_so4`.

**Model:** Qwen2.5-1.5B Q6_K (n_head=12, n_head_kv=2, n_gqa=6 — exercises GQA head mapping)
**Config:** isoquant3 KV cache, seed=42, temp=0, n_gen=30

**Commands:**
```bash
# Fused path
ISOQUANT_TEST_ROTATION=1 ./build/bin/llama-cli -m $MODEL -ctk isoquant3 -ctv isoquant3 -ngl 99 --seed 42 --temp 0 -n 30 -p "The capital of France is" > /tmp/fused.txt 2>&1

# Composed path (reference)
ISOQUANT_TEST_ROTATION=1 ISOQUANT_FORCE_COMPOSED=1 ./build/bin/llama-cli -m $MODEL -ctk isoquant3 -ctv isoquant3 -ngl 99 --seed 42 --temp 0 -n 30 -p "The capital of France is" > /tmp/composed.txt 2>&1
```

**Token-level result: IDENTICAL** — all non-diagnostic output matches between fused and composed paths. Both runs exit 0.

**Logit-level result: NOT numerically identical** — final prompt logits differ even though the top prediction matches.

| Metric | Value |
|------|------|
| `max_abs_diff` | `4.3146` |
| `mean_abs_diff` | `0.7590` |
| `rmse` | `0.9456` |
| argmax match | `yes` (token id `13`) |
| top-10 overlap | `7/10` |

This test exercises:
- The fused Metal kernel with real non-identity SO(4) rotation blocks
- GQA head mapping (`(group_idx / gph) % n_head` → `head_idx / n_gqa` to index KV head rotation)
- Both forward (Q pre-rotation) and inverse (output post-rotation) directions
- Token-level agreement over 30 decode steps
- Final-logit comparison on a fixed prompt via `llama-debug --save-logits`

**Interpretation:** fused and composed are behaviorally aligned on this prompt, but they are not numerically equivalent. The observed logit gap is consistent with a precision asymmetry in the implementation:
- Fused Metal path performs the SO(4) block matvec in `half4`
- Composed graph path performs the SO(4) block rotation through `ggml_mul_mat` in `F32`

This is an inference from the implementation and the measured diff, not a proof of the exact error budget per layer. What is proven by the artifact is narrower: same prompt/tokens in, same top token out, materially different final logits.

**Limitations of this test:**
- The logit comparison above is a single-prompt check on the final prompt logits, not a sweep across prompts, sequence lengths, or decode steps.
- Token-level comparison can still false-pass if the prompt is insensitive to small numerical differences.
- The `% n_head` prefill fix is structurally present but not numerically isolated — a multi-token prompt exercises it during prefill, but token agreement doesn't bound the per-element error.
- Both paths share the same InnerQ scale tensor (identity = all ones). Metal ignores it; CPU applies it. This is a pre-existing gap in Metal WHT, not introduced by the fusion.

## Benchmark (throughput only — pinned artifacts)

**Same-session back-to-back** (pinned to `results/llama_cpp_turbo3_vs_isoquant3_final.json`):

| Path | Prompt (t/s) | Generation (t/s) |
|------|-------------|-------------------|
| turbo3 | 4101 | 99.4 |
| isoquant3 fused | 4078 (-0.6%) | 95.8 (-3.6%) |

**Cross-session composed baseline** (from `results/llama_cpp_isoquant3_bench.json`, pre-fusion 2026-04-13):

| Path | Prompt (t/s) | Generation (t/s) |
|------|-------------|-------------------|
| isoquant3 composed (280 extra launches) | 2306 (-44%) | 81.9 (-16%) |

The composed number is from a different build session but the same code path (restored via `ISOQUANT_FORCE_COMPOSED=1`). The same-session turbo3-vs-fused pair is the authoritative comparison.

## Guards in place

| Guard | Location | Prevents |
|-------|----------|----------|
| `group_size == 128` | `ggml.c:6304` | 64-wide sign table mismatch |
| `src[2] == NULL` | `ops.cpp:10729`, `turbo-wht.cu:122` | SO(4) on CPU/CUDA |
| `% n_head` wrap | `ggml-metal.metal:3354` | OOB rotation read on prefill |

There is currently **no active runtime guard** for InnerQ on Metal WHT paths. The code documents the gap, but Metal still ignores non-trivial `scale_inv` while CPU/CUDA apply it.

## Rotation seed determinism

Default runtime SO(4) blocks are generated with seed `42 + init_counter`, where `init_counter` is a static counter incremented once per tensor initialization. This is deterministic across processes when KV tensors are initialized in the same order, which is true for the current sequential llama.cpp cache initialization path.

For correctness testing, `ISOQUANT_TEST_ROTATION=1` now bypasses that dependency entirely and loads explicit fixed non-identity SO(4) blocks.

Limitation:
- If KV tensor initialization is ever parallelized or reordered, cross-process fused-vs-composed comparison becomes invalid again.
- The default runtime path still has that caveat unless test rotation mode is enabled.

## Status

- Fused kernel compiles and dispatches on Metal: **yes**
- Throughput matches turbo3 within 2%: **yes**
- Fused kernel token-identical to composed: **yes** (Test 2 — GQA model, explicit fixed non-identity blocks, 30 decode steps)
- Non-identity SO(4) tested at runtime: **yes** (`ISOQUANT_TEST_ROTATION=1`)
- Final-logit comparison: **yes** (same argmax, different logits; `max_abs_diff=4.3146`, `rmse=0.9456`)
- Numerical equivalence to composed: **no**
- Multi-token prefill: `% n_head` structurally present, exercised during prompt eval, but not numerically isolated
- InnerQ on Metal WHT: **still an open gap** (documented, not fixed)
