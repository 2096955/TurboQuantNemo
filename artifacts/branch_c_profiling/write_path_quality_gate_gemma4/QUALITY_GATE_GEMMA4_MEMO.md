# §3.4 Quality Gate (Gemma4 default suite — closes gap 2)

**Roadmap:** `docs/superpowers/plans/2026-05-02-wrap-loose-ends-and-bandwidth-roadmap.md`
Step 3.4 — written 2026-05-05 to close gap 2 from the self-review of the
2026-05-05 §3.4 closure (the original Qwen3.6 micro-suite quality check was
2 prompts × 48 tokens — too narrow to detect numerical drift).

**Date:** 2026-05-05 (clean boot)
**Model:** `gemma-4-26b-a4b-it-4bit` (head_dim=256 → IsoQuant fused
attention dispatches; non-fallback read path)
**Cache type:** `isoquant`
**Settings:** `--suite default` (5 prompts), `max_tokens=200`, `--strict`,
`--seed 42`, `--temp 0.0` (greedy/deterministic)

## Result: harness PASS for all 4 conditions; byte-divergence found for FUSED_ENCODE

| Config | n_pass | peak MB | Total latency | Response identity vs baseline_iso |
|---|---|---|---|---|
| `baseline_iso` | 5/5 | 13989.0 | 49.96s | (reference) |
| `fused_encode` | 5/5 | 13989.0 | 46.73s | **4/5 differ** |
| `prealloc` | 5/5 | 14018.9 | 51.52s | **5/5 byte-identical** |
| `combined` | 5/5 | 14018.9 | **38.63s** (-23%) | **4/5 differ** |

The 4 prompts that differ for FUSED_ENCODE / combined: Instruction
Following, Basic Reasoning, List Generation, Math. Code Generation
matches baseline byte-for-byte across all 4 configs.

### Sample diff (Math prompt at character 261)

| Config | Slice |
|---|---|
| baseline | `**Multiply 17 by 20:**\n    $17 \times 20` |
| fused_encode | `Multiply 17 by 20:\n    $17 \times 20` |

Same content, different markdown formatting. Under greedy + seed=42, byte
divergence means the underlying float values diverge, producing different
sampled tokens. The drift accumulates over the response.

## Interpretation

- **`prealloc` is numerically equivalent to baseline.** The pre-allocated
  buffer + pointer-bump append doesn't change the compress/pack math.
  Outputs are byte-identical across all 5 prompts.
- **`FUSED_ENCODE=1` introduces measurable numerical drift.** The Metal
  fused compress/pack kernel does normalize/FWHT/SO(4)/quantize/pack in a
  different float-op order than the python path. The drift is real.
- **The drift does NOT cause harness failure.** All 5 prompts still meet
  per-task pass criteria (formatting, length, content — whichever the
  prompt specifies). Per-task quality holds.
- **The drift does change what the model says.** Outputs differ from
  baseline at the byte level. For applications that depend on
  bit-reproducibility, this is a regression. For applications that just
  need correct output, it isn't.

## Correction to the prior §3.4 quality claim

The earlier quality memo (`write_path_quality_gate/QUALITY_GATE_MEMO.md`)
claimed "no quality regression — byte-identical responses across all 4
configs" based on Qwen3.6 micro suite (2 prompts × 48 tokens, responses
"9" and "def f(): pass"). That claim was a coincidence of trivially-short
responses, not evidence of numerical equivalence. The Gemma4 default suite
(5 prompts × 200 tokens) reveals what the micro suite missed.

Honest restatement: under §3.4 candidate envs, **`prealloc` is numerically
equivalent; `FUSED_ENCODE` is not, but produces output that still passes
per-task harness criteria.**

## Disposition (per user direction 2026-05-05)

User chose: "Promote combined anyway, document drift as 'within harness
gate'." All 4 conditions PASS harness; combined is 23% faster end-to-end.
The byte divergence is documented but not blocking promotion.
