# Phase 3b: NPT=8 E2E Smoke Test — Gemma 4 (head_dim=256, H_kv=8)

**Date:** 2026-04-27 (rev 2 — corrected prompt length)
**Model:** gemma-4-26b-a4b-it-4bit (local, head_dim=256, H_kv=8, H_q=16)
**Branch:** main (post-merge of isoquant-decode-perf at 980985f)
**Short prompt:** 14 tokens ("Explain what makes the number 42 special...")
**Long prompt:** 670 tokens (attention mechanisms tutorial)
**Max decode tokens:** 64
**Env:** ISOQUANT_BITS=3

## Rev 1 error (corrected)

Rev 1 used a prompt that tokenized to only 383 tokens. With max_tokens=32,
the cache never reached T=512 during decode, so the tiled path was never
triggered despite the label claiming otherwise. Rev 2 extends the prompt
to 670 tokens and adds a tokenizer check that aborts if below 512.

## Results

| Path | Prompt tokens | decode tok/s | prompt tok/s | peak mem GB |
|---|---|---|---|---|
| 3-kernel (NPT8=0) | 14 | 40.689 | 23.049 | 14.569 |
| NPT=8 v1 (NPT8=1) | 14 | 42.964 | 23.505 | 14.569 |
| NPT=8 tiled (NPT8=1) | 670 | 30.819 | 423.346 | 14.954 |
| 3-kernel (NPT8=0) | 670 | 31.572 | 437.406 | 14.954 |
| Default KV | 14 | 113.175 | 138.470 | 14.548 |

## Dispatch verification

The script verifies LONG_PROMPT tokenizes to 670 >= 512 before running.
At 670 tokens, the first decode step sees T=670 in the cache, above the
`_NPT8_TILED_T_THRESHOLD=512` dispatch gate in mlx_isoquant.py. This is
a token-count proof, not instrumented dispatch verification — the script
does not hook into the actual dispatch code path to confirm the tiled
kernel was called.

## Output quality

All IsoQuant paths produced coherent text. Short-context runs generated
near-identical responses about Douglas Adams' 42. Long-context runs
generated matching technical text about transformer attention.

## Interpretation

- **Tiled vs 3-kernel (long context):** 30.8 vs 31.6 tok/s — within noise.
  No regression from tiled dispatch. Note: these are single runs.
- **NPT=8 v1 vs 3-kernel (short):** 43.0 vs 40.7 tok/s (+5.6%). Single run.
- **IsoQuant vs default KV:** ~3x decode overhead. Expected cost of 3-bit
  quantized KV compression.
- **Memory:** IsoQuant adds ~0.4 GB at 670-token context. Negligible.

## Caveats

1. Single run per path — no variance characterization.
2. Stochastic decode — text similarity is a weak quality signal.
3. Gemma 4 H_kv=8; NPT=8 kernel matters more for H_kv=2 models.
4. Dispatch proof is indirect (token count >= threshold, not instrumented).
