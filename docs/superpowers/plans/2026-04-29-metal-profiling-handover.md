# Handover: IsoQuant Metal Counter Profiling — Write-Path Attribution

**Date:** 2026-04-29
**Branch:** `main` (all work committed)
**Model:** Qwen3.6-35B-A3B-nvfp4, head_dim=256, H_kv=2, H_q=16, 10 IsoQuant layers, 30 Mamba layers
**Machine:** M4 Max 128 GB, macOS 26.3.1

---

## What Was Done

### Phase A — Paired Ablation (complete, committed)
Validated the iso-vs-default gap with paired interleaving and MAD outlier rejection:
- **T=4096:** +7.75 ms/step gap
- **T=8192:** +9.79 ms/step gap
- `fused_enc` + `prealloc` combined recovers ~1.3–2.1 ms (16–25%)
- Artifact: `artifacts/branch_c_profiling/ablation_paired.json`

### Phase B — Read-Path Synthetic Isolation (complete, committed)
`scripts/profile_metal_counters.py` benchmarks 4 read-path components in isolation (no model, no per-component fence overhead). Results at `artifacts/metal-counters/profile.json`:

| Component | T=4K corrected | T=4K serial10 | T=8K corrected | T=8K serial10 |
|-----------|---------------|---------------|----------------|---------------|
| query_rotation | 0.038 ms | 0.094 ms | 0.034 ms | 0.180 ms |
| tiled_kernel | 0.161 ms | 0.496 ms | 0.175 ms | 0.938 ms |
| fa2_merge | 0.050 ms | 0.440 ms | 0.076 ms | 0.445 ms |
| inverse_rotation | 0.198 ms | 1.372 ms | 0.202 ms | 1.368 ms |
| **Sum predicted 10-layer** | | **2.40 ms** | | **2.93 ms** |

E2E gap measured: T=4K: 9.41 ms, T=8K: 9.26 ms.

**Read-path explains only 25–32% of the gap.** ~6–7 ms remains.

### Phase C — 6-Component Instrumented Decomposition (complete, committed)
Separate script (`scripts/profile_npt8_metal.py`) with per-component fences (high overhead, ranking reliable, absolutes directional):

| Component | T=4K ms | T=8K ms | Role |
|-----------|---------|---------|------|
| compress_batch | 12.5 | 13.2 | **Write** |
| pack_indices_3bit | 5.2 | 5.3 | **Write** |
| metal_kernel | 3.5 | 3.9 | Read |
| inverse_rotation | 3.7 | 3.8 | Read |
| fa2_merge | 2.4 | 2.4 | Read |
| query_rotation | 2.3 | 2.4 | Read |

**Finding:** Write path (`compress_batch` + `pack_indices_3bit`) dominates. Instrumentation overhead was ~83–92% so absolute numbers are inflated, but the ranking is reliable.
- Artifact: `artifacts/branch_c_profiling/npt8_profile_fused_d128.json`
- Memo: `artifacts/branch_c_profiling/profiling_memo.md`

---

## What Is In Progress (uncommitted)

### Write-Path Synthetic Benchmarks — added to `scripts/profile_metal_counters.py`

The script (`scripts/profile_metal_counters.py`, currently **untracked** on main) has been extended with write-path synthetic benchmarks to get low-overhead, fence-corrected timings for each write-path component. This closes the attribution gap between Phase B (read-only, 25–32%) and Phase C (high-overhead instrumented).

**Completed edits:**
1. Module-level globals for write-path functions (`_structured_rotate_forward`, `_quantize_scalar`, `_fused_compress_and_pack`, `_load_codebook`)
2. `import_mlx_modules()` extended to import these from `mlx_isoquant`, `mlx_turboquant`, `fused_kv_compress`
3. `build_synthetic_fixture()` extended with write-path data:
   - `kv_new`: (H_KV, 1, HEAD_DIM) — single decode token
   - `centroids_full`, `boundaries` — loaded from codebook
   - `existing_indices/norms/packed` — pre-existing cache buffers at size T (for concat simulation)
   - `prealloc_indices/norms/packed` — oversized buffers (T+256, for slice-assign simulation)
   - `new_indices_1`, `new_norms_1`, `new_packed_1` — single-token write data
4. `write_component_fns(fixture)` function (line 463) returning 5 callables:
   - `compress_python` — full Python compress path (norm → rotate_forward → quantize_scalar)
   - `compress_fused_metal` — fused Metal kernel (if available)
   - `pack_3bit` — 3-bit packing only
   - `cache_concat` — `mx.concatenate` of 3 arrays (indices, norms, packed)
   - `cache_prealloc` — slice-assign into preallocated buffers

**NOT completed:**
- `run_synthetic_phase()` (line 528) still only benchmarks read-path via `component_fns()`. It needs to also call `write_component_fns()`, benchmark each, and include results in the output JSON.
- `print_attribution_table()` needs write-path rows.
- `compare_prediction_to_e2e()` needs write-path in the predicted total.
- The script has not been run with write-path components.

---

## What Needs To Happen Next

### Step 1: Wire write-path into `run_synthetic_phase()`

In `run_synthetic_phase()` (line 528), after the existing read-path loop:

```python
# After read-path components loop...
write_fns = write_component_fns(fixture)
write_serial_fns = {
    name: [write_component_fns(lf)[name] for lf in fixtures]
    for name in write_fns
}
for name, fn in write_fns.items():
    # Same bench_component + bench_serial_10x pattern
    # Add to components dict, accumulate into sum_single/sum_serial
```

Important: write-path components run 2× per decode step (K + V), so `predicted_10layer_ms` should account for this. Read-path runs 1× per step. The multiplier should be: `predicted = read_serial_sum + (write_serial_sum × 2)`.

### Step 2: Verify and run

```bash
python -m py_compile scripts/profile_metal_counters.py
python scripts/profile_metal_counters.py \
  --model /Users/anthonylui/Models/Qwen3.6-35B-A3B-nvfp4 \
  --output artifacts/metal-counters/profile_with_write.json \
  --prefill 4096 8192 --skip-e2e --skip-traces
```

### Step 3: Analyze

Compare `predicted_10layer_ms` (read + 2× write) against the known E2E gap (9.4–9.3 ms). If read + write synthetic sum is still < gap, the residual is Python/MLX dispatch overhead (graph construction, memory allocation, `mx.eval` scheduling across 10 layers × 2 caches × 6 ops per layer).

### Step 4: Decision point

Based on the write-path decomposition:

| If... | Then... |
|-------|---------|
| `compress_python` dominates and `compress_fused_metal` is much faster | Fused encode is the right optimization lever (already exists, just enable `ISOQUANT_FUSED_ENCODE=1`) |
| `cache_concat` dominates and `cache_prealloc` is much faster | Prealloc is the right lever (already exists, `ISOQUANT_CACHE_MODE=prealloc`) |
| Both fused encode + prealloc combined still leave a large residual | Python dispatch overhead is the bottleneck — need to reduce the number of `mx.eval` calls per step |
| Sum of all components × layers ≈ E2E gap | Attribution is complete, optimize the dominant component |

---

## Key Files

| File | Status | Purpose |
|------|--------|---------|
| `scripts/profile_metal_counters.py` | Untracked, partially complete | Main profiling script (read + write synthetic) |
| `scripts/profile_npt8_metal.py` | Committed | High-overhead 6-component instrumented profiler |
| `artifacts/metal-counters/profile.json` | Committed | Read-path-only synthetic results |
| `artifacts/branch_c_profiling/ablation_paired.json` | Committed | Paired ablation gap measurements |
| `artifacts/branch_c_profiling/profiling_memo.md` | Committed | Analysis memo (compress_batch dominates) |
| `mlx-lm/mlx_lm/models/mlx_isoquant.py` | Source | IsoQuantKVCache, _compress_batch, rotation functions |
| `mlx-lm/mlx_lm/models/fused_kv_compress.py` | Source | Fused Metal encode kernel |
| `mlx-lm/mlx_lm/models/mlx_turboquant.py` | Source | quantize_scalar, load_codebook |
| `mlx-lm/mlx_lm/models/fused_kv_decode_npt8_tiled.py` | Source | Tiled NPT=8 Metal decode kernel |

## Active Plan File

`/Users/anthonylui/.claude/plans/zazzy-hugging-castle.md` — the original plan for this profiling work. Still relevant for context on the three-phase approach.

## Prior Plan Documents

- `docs/superpowers/plans/2026-04-24-isoquant-decode-performance.md` — broader decode perf plan
- `docs/superpowers/plans/2026-04-24-isoquant-kernel-c-execution.md` — Branch C execution plan
- `docs/superpowers/plans/2026-04-21-isoquant-rectification.md` — IsoQuant correctness rectification

---

## TL;DR

Read-path synthetic profiling is done and committed (explains 25–32% of gap). Write-path component functions are written but not yet wired into the benchmark loop. The immediate task is: integrate `write_component_fns()` into `run_synthetic_phase()`, run it, and see whether read + 2× write closes the ~9 ms gap. If a residual remains, it's Python dispatch overhead and requires a different optimization strategy (fewer `mx.eval` calls, fused multi-component kernels, or graph-level batching).
