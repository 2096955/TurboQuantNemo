# DW Fix — Grounded Diff (Pre-fix vs Post-fix Provenanced Run)

**Date:** 2026-04-21
**Author:** Phase 0 cleanup (subagent-driven, ARB review)
**Run UUID (post-fix):** `b14dfeeb6eca41eeb540acca51249554`
**Old data:** `results-pre-dw-fix/` (preserved for audit)
**New data:** `results/` (publishable baseline)

---

## What's Different

**Observed:** every JSON in `results-pre-dw-fix/` reports `dw_statistic ≈ 0.0157`,
a near-constant value across kernels of wildly different shapes. Every JSON in
the new `results/` run reports a plausible per-kernel DW value (range observed:
0.07 – 2.47), responding to the underlying serial dependence of the
measurements as a healthy DW should.

**Code state:** the current `harness/stats.mojo` already computes
`var dw = durbin_watson(timings)` on the unsorted, time-ordered timing list at
`stats.mojo:179`, and the `durbin_watson` function applies the standard formula
without re-sorting. **No uncommitted change in this worktree explains the DW
value transition** — the only unstaged stats.mojo diff is import/syntax
cleanup (`from collections` → `from std.collections`, `inout` → `mut`).

**What we cannot prove:** which earlier commit, environment change, or harness
state produced the buggy constant DW in the pre-fix JSONs. The pre-fix JSONs
have no provenance, so we can't pin them to a specific code SHA, mojo version,
or run timestamp. They predate any change made in this session.

**What we can claim with confidence:** the new run's DW values are plausible;
the old run's DW values were not. Treat the new run as the source of truth for
DW going forward. Treat the old run's DW column as irrelevant data, but note
the medians/p95s in the old data are independent of DW and still tell us
something about the pre-fix system state (see Findings below).

---

## Provenance — What's New This Run

The post-fix run captures full reproducibility metadata for every JSON. Old
JSONs have **no provenance object**, so direct same-machine claims about the
old run cannot be verified.

```json
"provenance": {
  "mojo_version": "Mojo 0.26.3.0.dev2026040805 (c200c78a)",
  "hostname": "Anthonys-MacBook-Pro-128.local",
  "chip": "Apple M4 Max",
  "macos_version": "26.3.1",
  "pixi_lock_sha256": "f4c72f6aaa325a499426f77e53ff3d1488b42f58793f95cba2e51541e104b496",
  "run_timestamp_utc": "2026-04-21T08:54:54Z",
  "run_uuid": "b14dfeeb6eca41eeb540acca51249554"
}
```

Captured by `scripts/bench_with_provenance.py` (stdlib only). All 21 JSONs in
this run share the same `run_uuid`. The wrapper is the canonical entry point
for any benchmark output going forward.

**Caveat (`framework_version`):** the field still reports the source-literal
`"0.26.3"` as written in `bench_*.mojo`. The `provenance.mojo_version` field
captures the actual runtime version. Treat `provenance.mojo_version` as
authoritative; the top-level `framework_version` is left as-is to keep the
schema stable.

---

## Per-Kernel Diff (Sorted by Δ%)

`old` = `results-pre-dw-fix/`, `new` = post-fix run with provenance.
`std_r` = new_std / old_std. `dw` columns: pre-fix dw was a buggy near-constant.

| Kernel | old_med µs | new_med µs | Δ% | old_std | new_std | std_r | old_dw | new_dw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| matmul_decode_gemv_6144                  |    23178.00 |    5682.50 |  -75.5 |   5493.89 |    213.65 | 0.04 | 0.0157 | 1.5701 |
| matmul_prefill_qkv_2048x6144             |  1398614.00 |  420064.00 |  -70.0 |  97050.31 |  16177.98 | 0.17 | 0.0217 | 0.5381 |
| matmul_ffn_up_decode_6144x16384          |    43775.00 |   15068.00 |  -65.6 |   9283.28 |    235.24 | 0.03 | 0.0269 | 2.4656 |
| softmax_seq_8192                         |   293011.50 |  227642.50 |  -22.3 |  23917.02 |   1331.34 | 0.06 | 0.0494 | 1.6599 |
| matmul_ffn_up_prefill_2048x6144x16384    |  1711661.50 | 1397970.00 |  -18.3 | 908886.99 | 224442.75 | 0.25 | 0.0630 | 1.1839 |
| matmul_square_2048                       |    62423.00 |   54804.00 |  -12.2 |   2079.67 |   1950.24 | 0.94 | 0.0296 | 0.0693 |
| matmul_square_1024                       |     6200.50 |    5854.50 |   -5.6 |    217.23 |    116.42 | 0.54 | 0.0152 | 0.8642 |
| matmul_square_4096                       |   480099.50 |  455123.50 |   -5.2 |  29484.13 |   9858.14 | 0.33 | 0.0343 | 0.2336 |
| matmul_ffn_down_prefill_2048x16384x6144  |  1525117.00 | 1459508.50 |   -4.3 |  77151.42 |  57818.84 | 0.75 | 0.0196 | 0.5980 |
| matmul_square_8192                       |  4184755.00 | 4042893.50 |   -3.4 | 212817.93 | 327050.51 | 1.54 | 0.0161 | 0.2902 |
| softmax_seq_2048                         |    10397.00 |   10224.00 |   -1.7 |    442.75 |    106.22 | 0.24 | 0.1281 | 1.1978 |
| matmul_moe_expert_5120x11008             |     8354.00 |    8229.00 |   -1.5 |    254.80 |    196.83 | 0.77 | 0.0388 | 1.2530 |
| rope_seq_32768                           |    30993.50 |   30789.50 |   -0.7 |   1041.18 |     84.51 | 0.08 | 0.6449 | 1.6870 |
| matmul_ffn_down_decode_16384x6144        |    12886.50 |   12880.50 |   -0.0 |    668.18 |    460.20 | 0.69 | 0.0429 | 2.3380 |
| rope_seq_8192                            |     7242.50 |    7247.50 |   +0.1 |    842.18 |     81.86 | 0.10 | 0.0391 | 1.2961 |
| matmul_square_512                        |      948.00 |     969.00 |   +2.2 |     82.98 |     38.18 | 0.46 | 0.0236 | 1.1888 |
| softmax_seq_512                          |      735.00 |     757.00 |   +3.0 |     16.31 |     44.59 | 2.73 | 0.1008 | 1.6459 |
| rope_seq_2048                            |     1773.50 |    1828.50 |   +3.1 |    456.90 |     57.41 | 0.13 | 0.0718 | 1.0685 |
| rope_seq_512                             |      544.00 |     602.00 |  +10.7 |     27.98 |    338.79 | 12.11 | 0.0533 | 0.1566 |
| softmax_seq_128                          |      328.00 |     446.00 |  +36.0 |    164.61 |     98.49 | 0.60 | 0.0304 | 0.9289 |
| rope_seq_128                             |      315.00 |     591.50 |  +87.8 |    140.19 |    115.67 | 0.83 | 0.0774 | 0.3787 |

---

## Findings

### 1. DW values are now plausible
Post-fix DW values vary 0.07 – 2.47 across kernels, consistent with what you'd
expect from honestly-computed serial-correlation statistics. The pre-fix
column shows the constant ~0.0157 artifact across the entire suite — that is
a bug signature, regardless of which historical state produced it.

### 2. Big speedups reproduce — they are real, not run-to-run noise
A separate post-fix run earlier in this session showed the same large
speedups within ≤1pp on the top movers
(matmul_decode_gemv_6144, matmul_prefill_qkv, matmul_ffn_up_decode,
softmax_seq_8192). Reproducibility across two independent provenanced runs
strongly indicates these are real changes vs the pre-fix data, not measurement
jitter on a single run.

### 3. Old run was on a noisier system
Standard deviations dropped sharply on the kernels where speedups are largest
(std_r 0.03 – 0.25 on the top 5 movers). The old run's std on
matmul_decode_gemv_6144 was 5494 µs on a 23ms kernel (CV ≈ 24%); the new run
shows std 214 µs (CV ≈ 4%). This is the dominant source of the apparent
speedup — the old absolute numbers were inflated by noise the old machine
state was carrying. **We cannot attribute the delta to any specific cause**
(thermal state, background load, etc.) because the old run has no provenance.

### 4. softmax_seq_128 +36% slowdown reproduces — flag for follow-up
The +36% slowdown on softmax_seq_128 was present in the prior post-fix run as
well. With std actually *lower* in the new run (60% of old), this is not
explainable as noise. Same pattern on rope_seq_128 (+87.8%). At small T
(seq=128), kernel cost is dominated by launch + boundary work; a real
regression at this size is plausible. **Do not publish softmax/rope numbers at
T ≤ 512 without re-investigating the small-tensor path first.**

### 5. Two kernels still noisy on the new run
- `rope_seq_512`: new_std 339 µs vs old 28 µs (std_r 12.11). The new run
  itself was noisier here.
- `matmul_square_8192`: new_std 327k µs vs old 213k µs (std_r 1.54). Largest
  kernel in the suite, system probably approached thermal limit during this
  one.

These don't invalidate the DW fix, but they're worth re-running back-to-back
before quoting absolute small-T or 8192² square numbers.

---

## Status of Old Data

`results-pre-dw-fix/` is preserved for audit only.

**Do not cite old absolute timings** in any publishable artifact. They are:
- Computed on a system with no captured provenance (cannot prove same machine)
- Affected by the buggy DW statistic (irrelevant to medians/p95s but reflects
  on harness trustworthiness)
- Noisier than necessary, biasing means/medians upward where std was 5–25× the
  new value

The new `results/` directory is the publishable baseline.

---

## What's Pending

- **Same-machine reproducibility evidence:** ideally run the suite once more,
  back-to-back, to show the inter-run delta on the post-fix provenanced
  pipeline is small. Rough check (two post-fix runs in this session) was
  ≤1pp on big movers but should be formalized.
- **Investigate softmax_seq_128 / rope_seq_128 small-T regression** before any
  publication that touches the small-T regime.
- **Stop hardcoding `framework_version` in `bench_*.mojo`** — promote the
  provenance field to the canonical version source in a follow-up.
