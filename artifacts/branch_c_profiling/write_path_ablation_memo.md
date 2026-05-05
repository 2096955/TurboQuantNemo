# Write-path §3.4 ablation memo

**Roadmap:** `docs/superpowers/plans/2026-05-02-wrap-loose-ends-and-bandwidth-roadmap.md` Step 3.4
("Decide whether fused encode and prealloc graduate")

**Spec artifact:** `artifacts/branch_c_profiling/write_path_ablation_paired.json`

## Sign convention

`delta_ms = ref_ms(iso baseline) − variant_ms(ablated cfg)` so **positive delta means
the variant is faster than the iso baseline** (env defaults: `FUSED_ENCODE=0`,
`NPT8=1`, `CACHE_MODE=concat_append`, `USE_METAL=0`, `BITS=3`).

## Provenance — CAVEATED

- Source runs: `ablation_paired.json` (2026-04-28T21:03:08Z) +
  `ablation_combined.json` (2026-04-28T21:35:16Z), merged 2026-05-05.
- Method: `paired_alternating_with_priming`, 7 repeats per cell + 1 priming pair,
  80 decode steps, model `Qwen3.6-35B-A3B-nvfp4`.
- Both source runs predate the 2026-05-02 MLX/Metal hang.
- **Code-identity gap (Codex audit 2026-05-05):** the FUSED_ENCODE Metal path
  was committed AFTER the 2026-04-28 runs:
  - `mlx-lm/mlx_lm/models/fused_kv_compress.py` first committed in `57cb5f5`
    on 2026-05-02 (file did not exist in any prior committed tree).
  - `from .fused_kv_compress import fused_compress_and_pack` added to
    `mlx_isoquant.py` in `40501b2` on 2026-04-29.
  - Both occurred after 2026-04-28 21:03 when the source ablation was written.
  - The 2026-04-28 activation receipts show `fused_metal_ok_after: True`, so
    FUSED_ENCODE was active at run time — but only because that work existed
    as uncommitted local files at the time. We cannot now prove the local
    version is byte-identical to what is currently committed.
- Implication: the perf delta numbers are **provisional** evidence for the
  currently-committed code. The quality gate (`write_path_quality_gate/`) was
  run on the current committed code on 2026-05-05 and remains directly valid.
- Earlier wording in this memo claimed "kernel/cache code unchanged since
  `980985f`, 2026-04-27" — that was wrong and has been corrected.

## Results

Per decode step paired delta vs iso baseline (positive = variant faster):

| Config | T=4096 | signal | T=8192 | signal |
|---|---|---|---|---|
| `fused_enc` (FUSED_ENCODE=1) | **+1.23** | stable | **+1.92** | outlier×1 |
| `prealloc` (CACHE_MODE=prealloc) | **+0.61** | stable | **+1.27** | outlier×2 |
| `combined` (FUSED_ENCODE=1 + prealloc) | **+1.33** | outlier×1 | **+2.14** | outlier×1 |
| `no_npt8` (NPT8=0, sanity check) | -1.08 | outlier×1 | -2.55 | stable |
| `metal_fwd` (USE_METAL=1, sanity) | +0.06 | noisy | -0.22 | stable |

All three §3.4 candidates show **positive median delta** (variant faster than
iso baseline) across both context lengths. Signal mix per the harness's MAD
outlier check: `fused_enc` is stable at 4K and one-outlier at 8K; `prealloc`
is stable at 4K and two-outlier at 8K; `combined` is one-outlier at both.
Earlier wording "stable improvement" was imprecise — the medians and p25
quartiles are positive in every cell, but only one cell (fused_enc 4K) is
strictly outlier-free. The `no_npt8` row confirms NPT8 is helping (turning
it off is consistently slower). `metal_fwd` is noisy at 4K and stable-near-zero
at 8K — kept off in env defaults.

## Activation receipts

`ablation_paired.json` `activation_after` blocks confirm:

- `fused_enc`: `fused_encode_after=True, fused_metal_ok_after=True, fallback_after=False`
- `prealloc`: `fused_encode_after=False, fused_metal_ok_after=True, fallback_after=False`
- `combined`: `fused_encode_after=True, fused_metal_ok_after=True, fallback_after=False`

## Quality gate

§3.4 graduation also requires "no quality/PPL regression appears." That gate is
tracked separately under `artifacts/branch_c_profiling/write_path_quality_gate/`
(see Step 3.4 in the roadmap; populated 2026-05-05 — link the JSONs there
before final §3.4 sign-off).

## Decision

**GRADUATE** for `combined` (FUSED_ENCODE=1 + CACHE_MODE=prealloc).

Quality gate — Qwen3.6 micro-suite ONLY (2 prompts × 48 tokens, current code,
2026-05-05): byte-identical responses + identical peak memory across all 4
configs (`write_path_quality_gate/QUALITY_GATE_MEMO.md`).
**This finding alone is not sufficient evidence of numerical equivalence —
see the Quality caveat below for the Gemma4 default-suite finding which
shows FUSED_ENCODE introduces drift.**

Perf gate v2 (current code, 2026-05-05T15:51Z, clean boot): all three candidates
reproduce as wins. **Sign reproduces in 6/6 cells (all positive).** Magnitude
reproduction is **5/6 cells equal or larger** vs v1; one cell (`prealloc` at
T=8192) is smaller in v2 (+0.75 vs v1 +1.27, ratio 0.58×). The other five
cells are larger: v2/v1 ratios at T=4096 are fused_enc 2.35×, prealloc 1.11×,
combined 2.20×; at T=8192 fused_enc 1.41× and combined 1.27×. v2 deliberately
ran only the three §3.4 candidates (`--configs fused_enc prealloc combined`)
and did not re-measure the v1 sanity configs (`no_npt8`, `metal_fwd`); the
v1 sanity rows above remain pinned for context. Artifact:
`write_path_ablation_paired_v2.json`.

| Config | T=4096 v1 → v2 (Δ ms, signal) | T=8192 v1 → v2 (Δ ms, signal) |
|---|---|---|
| `fused_enc` | +1.23 stable → **+2.89** outlier×2 | +1.92 outlier → **+2.70** outlier×1 |
| `prealloc` | +0.61 stable → +0.68 noisy | +1.27 outlier → +0.75 outlier×2 |
| `combined` | +1.33 outlier → **+2.93 stable** | +2.14 outlier → **+2.73 stable** |

Key observations:

- **`combined` is now stable signal at BOTH T values** (was outlier×1 at both
  in v1). p25/p75 are tight (T=4096: +2.66/+3.34; T=8192: +2.68/+3.17) and
  positive.
- T=8192 paired_gap (default vs iso) is harness-flagged INVALID due to
  2 outliers in the gap measurement itself, so the % gap denominator is
  suppressed for T=8192. Absolute Δ remains valid.
- `fused_enc` median at T=4096 doubled (+1.23 → +2.89) but with 2 outliers
  inflating std. The reproducible win is in `combined`, where
  outlier-free runs at both T values support promotion as the new default.
- `prealloc` alone gives small benefit (+0.68/+0.75); the value is mostly
  through stacking with FUSED_ENCODE in `combined`.
- All activation receipts confirm kernels fired (fused_encode_after=True for
  fused_enc/combined; fallback_after=False everywhere).

§3.4 closes as GRADUATE on the strength of v2 perf reproduction + harness
quality PASS. Promote `combined` (FUSED_ENCODE=1 + CACHE_MODE=prealloc) as
the recommended default for IsoQuant write path. Single-flag variants remain
opt-in.

**Quality caveat (2026-05-05 self-review correction):** the original Qwen3.6
micro-suite byte-identity check was a coincidence of trivially-short
responses. The Gemma4 default suite
(`write_path_quality_gate_gemma4/QUALITY_GATE_GEMMA4_MEMO.md`) shows
`prealloc` is byte-identical to baseline but `FUSED_ENCODE=1` introduces
measurable numerical drift — outputs differ from baseline at the byte level
on 4 of 5 default prompts. The drift does NOT break per-task harness
criteria (5/5 PASS) but does change what the model says. Per user direction
2026-05-05 ("promote combined anyway, document drift as 'within harness
gate'"), this is documented but not blocking promotion.

**Runtime default flipped (2026-05-05):** `mlx_isoquant.py` env-default
reads at lines ~441 and ~523 now resolve to `prealloc` and `1` instead of
`concat_append` and `0`. Every IsoQuant cache instance constructed without
explicit env vars now runs in `combined` mode. 50-test IsoQuant pytest suite
still passes. Opt-out: `ISOQUANT_FUSED_ENCODE=0` and/or
`ISOQUANT_CACHE_MODE=concat_append`.
