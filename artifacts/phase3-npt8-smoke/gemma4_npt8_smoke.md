# Phase 3 v1: NPT=8 E2E Smoke Test — Gemma 4 (head_dim=256, H_kv=8)

**Date:** 2026-04-26
**Model:** gemma-4-26b-a4b-it-4bit (local, head_dim=256, H_kv=8, H_q=16)
**Prompt:** "Explain what makes the number 42 special in exactly three sentences."
**Max tokens:** 64
**Cache mode:** concat_append (default)
**Env:** ISOQUANT_BITS=3

## Results

| Path | decode tok/s | prompt tok/s | peak memory GB |
|---|---|---|---|
| 3-kernel (NPT8=0) | 40.624 | 7.469 | 14.569 |
| NPT=8 (NPT8=1) | 43.640 | 7.513 | 14.569 |
| Default KV (no iso) | 113.194 | 293.404 | 14.569 |

## Output quality

All three paths produced coherent, near-identical responses about Douglas Adams'
Hitchhiker's Guide. Minor wording variations (expected from IsoQuant lossy compression):

- **3-kernel:** "...a pronic number and the sum of the..."
- **NPT=8:** "...a pronic number and a primary factor in..."
- **Default KV:** "...a pronic number and the sum of the first..."

## Interpretation

- **Quality:** No regression. NPT=8 matches 3-kernel quality; both are close
  to default KV. Differences are within expected IsoQuant quantization noise.
- **Throughput:** NPT=8 at 43.6 tok/s vs 3-kernel at 40.6 tok/s (+7.4%).
  This is a **single pair of runs** with no variance control. Treat as
  encouraging smoke, not a proven speedup.
- **Memory:** Identical across all three paths at this context length.

## Caveats

1. Single run, no repeats — no variance characterization.
2. Short context (27-token prompt + 64 decode) — does not stress long-T behavior.
3. Gemma 4 is H_kv=8; the NPT=8 kernel matters more for H_kv=2 models where
   decode kernel overhead is a larger fraction of total time.
4. Stochastic decode — text similarity is a weak correctness signal. The unit
   and cache-level tests (8/8 pass) are the real safety net.
