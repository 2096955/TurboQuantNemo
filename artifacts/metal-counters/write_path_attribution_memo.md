# Write-path attribution memo

**Evidence index**

| Artifact | Role |
|---------|------|
| `artifacts/metal-counters/profile_with_write.json` | Phase **3.2** synthetic isolation only (`--skip-e2e`; fast CI / smoke). |
| `artifacts/metal-counters/profile_with_write_e2e.json` | Phase **3.2+3.3**: synthetic **plus** default vs IsoQuant E2E gap and `comparison` block. |
| `artifacts/metal-counters/system_state_20260502.json` | Phase **0.x** logged smoke + pytest (predates some reruns — see JSON timestamps inside profiles). |

**Automation:** `scripts/run_write_path_attribution_gate.sh` (MLX smoke → NPT8 tests → synthetic profile → optional `RUN_E2E=1`).

**Profiler:** canonical tool is **`scripts/profile_metal_counters.py`** (`read` + write components, `2×` K/V write). **`scripts/profile_npt8_metal.py`** is a separate **high-overhead NPT8 read-path** instrumented splitter — do not confuse with §3.2.

---

## E2E reconciliation (closes roadmap §3.3 for this checkpoint)

Run: `profile_metal_counters.py` on **`Qwen3.6-35B-A3B-nvfp4`**, **`--prefill 4096 8192`**, **`--skip-traces`**, decode **35** steps / reduced warmup iters (~34 s wall-clock on reference run).

Measured **default vs IsoQuant** gap per decode step (**isoquant − default**):

| T | `e2e_gap_ms` | `predicted_10layer_ms` (10-layer proxy) | `prediction_to_gap_ratio` |
|---|--------------|----------------------------------------|---------------------------|
| 4096 | **8.48** | 8.05 | **94.9%** |
| 8192 | **9.01** | 9.20 | **102.2%** |

Residuals are ~**±0.2 ms/step** — the **read + 2× active-write** synthetic sum explains essentially the entire E2E gap on this model/config. Largest named buckets in the attribution table:** `compress_python`**, **`pack_3bit`**, **`inverse_rotation`**, **`tiled_kernel`** — consistent with doubling down on **write path (`_compress_batch` / packing / concat)** next, not speculative read tiling.

## Remaining roadmap items (not closed here)

- **§3.4** — fused encode (`--fused-encode`) + prealloc (`--prealloc`) paired ablations vs quality.
- **Kimi residency sweep** — `scripts/sweep_kimi_default_cache_residency.py` (needs Kimi checkpoint + time).
- **Nemotron RC** — `docs/RELEASE_CANDIDATE_CHECKLIST.md` (strict matrices, soak duration, serving).
- **`run_kimi_npt16_parity_gate.py`** without `--synthetic-only` still **exits 2** until real-weight MLA parity exists; **`artifacts/kimi_k26_profiling/npt16_synthetic_gate.json`** records **synthetic-only** PASS for CI/smoke.
