# IsoQuant Decode Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring NVFP4+IsoQuant decode at 8K context to parity-or-better with NVFP4+default by attacking the two remaining IsoQuant-specific costs (packed-cache rebuild and 3-kernel pipeline fragmentation), then validate the depth-reduction scaling model at 32K.

**Architecture:** Three measurement phases bracket two implementation phases. Phase 0 establishes a clean baseline and persists the per-kernel attribution that proves the V-accum tiled win. Phase 1 characterizes the tiled V-accum kernel as bandwidth-bound or execution-bound (DeepSeek's pre-fusion check) — the answer changes how aggressive Phase 3 needs to be. Phase 2 adds incremental packed-cache append to eliminate the O(T) repeat-pack cost. Phase 3 adds an NPT=8-generalised, T-tiled fully-fused single-dispatch kernel — but only if Phase 1 says fusion's upside is real. Phase 4 runs at T∈{4K,8K,16K,32K} with the Brent-bound prediction overlaid; Phase 5 picks one of three pre-defined decision branches.

**Tech Stack:** Python 3.12, MLX (mx.fast.metal_kernel), Metal Shading Language, pytest, M4 Max (this machine).

---

## Hard architectural constraint

The existing single-kernel fully-fused path
(`mlx-lm/mlx_lm/models/fused_kv_decode_kernels.py:527`) hard-asserts `NPT=4`, i.e.
`head_dim=128`. The qwen3_5_moe test model has `head_dim=256` → `NPT=8`. Implications:

1. The existing fully-fused kernel does **not** apply to this model; the 3-kernel pipeline
   was always the only viable path
2. Phase 3 is "generalise the kernel to NPT=8 AND add T-tiling on top", two distinct
   changes
3. The tiled V-accum work already merged (32.66 → 4.52 ms at 4K, 7.2× kernel-level) was
   the correct first move because it has no NPT constraint

## File Structure

**Created:**
- `scripts/instrument_bandwidth_iso.py` — bytes-moved counter wrapping the tiled V kernel
- `mlx-lm/tests/test_iso_incremental_pack.py` — bit-exact correctness test for Phase 2
- `mlx-lm/tests/test_fused_npt8.py` — NPT=8 + T-tiling equivalence test for Phase 3
- `mlx-lm/mlx_lm/models/fused_kv_decode_npt8_tiled.py` — Phase 3's new fused kernel + wrapper
- `scripts/run_phase0_baseline.sh` — wraps `benchmark_nvfp4_isoquant.py` for repeated runs
- `scripts/run_scaling_validation.py` — Phase 4 sweep + chart producer
- `docs/PHASE5_DECISION_2026-04-24.md` — Phase 5 output
- `artifacts/phase{0,1,2,3,4}_*/` — per-phase JSON/PNG outputs

**Modified:**
- `mlx-lm/mlx_lm/models/mlx_isoquant.py` — Phase 2 incremental append in `IsoQuantKVCache`;
  Phase 3 new dispatch branch in `_fused_attention_metal`
- `mlx-lm/mlx_lm/models/fused_kv_decode_tiled.py` — Phase 1 instrumentation hooks (optional,
  via env var)

**Pre-existing (not modified):**
- `mlx-lm/mlx_lm/models/fused_kv_decode_kernels.py` — left as-is; Phase 3 lives in a new
  file rather than mutating the NPT=4-asserting kernel

## Forecast (not result)

At 8K, current measured: nvfp4+iso = 32.4 ms/step (30.84 tok/s); nvfp4+default = 10.8 ms/step
(92.45 tok/s). Forecast post-Phase-3: **parity is plausible in the optimistic branch,
~1.5–1.7× is the conservative branch.** The saved ms wins are not guaranteed to add linearly
(Phase 2 may amortise differently once Phase 3 removes inter-kernel sync). Treat as
forecast that Phase 4 measurements either validate or refute.

---

## Phase 0 — Clean baseline + persisted attribution evidence

The 4K cross-session numbers from the prior session swung dramatically (default-KV cells
moved +77% at 8K and -47% at 4K despite no code touching their path). Need a clean
reference. **Also need to persist the post-Codex-fix per-kernel attribution** — the saved
4K/8K matrix JSONs only contain end-to-end tok/s, not the per-component breakdown that
proves the V-accum tiled win.

### Task 0.1: Confirm clean system state

**Files:**
- Inspect: system state only (no edits)

- [x] **Step 1: Check swap usage and pages free**

Run:
```bash
sysctl vm.swapusage && vm_stat | head -8
```

Expected: `vm.swapusage` `used` < 5000M (low swap pressure) and Pages free × 16384 > 8 GB.
If swap > 20 GB, restart browser/Slack/Amazon Q Helper to reclaim memory before benchmarking.

- [x] **Step 2: Check thermal state**

Run:
```bash
sudo powermetrics --samplers smc -n 1 -i 1000 2>/dev/null | grep -E "CPU die temperature|GPU die temperature" | head -2
```

Expected: temperatures < 70°C. If above, idle 5 min and recheck.

- [x] **Step 3: List interfering processes**

Run:
```bash
ps aux | sort -k 4 -nr | head -10
```

Confirm no Claude Code subagents, Codex/Gemini delegations, or other Python ML processes
running concurrently. Document anything > 5% memory in the Phase 0 baseline notes.

### Task 0.2: Write the baseline runner script

**Files:**
- Create: `scripts/run_phase0_baseline.sh`

- [x] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Phase 0 baseline: 3 repeats per (T, cell) at 4K/8K/16K/32K with the tiled V-accum.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATE="${DATE:-$(date +%Y-%m-%d)}"
OUT_DIR="$REPO_ROOT/artifacts/phase0_baseline_${DATE}"
mkdir -p "$OUT_DIR"

BASELINE="${BASELINE:-/Users/anthonylui/Models/Qwen3.6-35B-A3B-4bit}"
NVFP4="${NVFP4:-/Users/anthonylui/Models/Qwen3.6-35B-A3B-nvfp4}"

for T in 4096 8192 16384 32768; do
  D=$(( T < 8192 ? 512 : 1024 ))
  for REPEAT in 1 2 3; do
    OUT="$OUT_DIR/matrix_T${T}_d${D}_r${REPEAT}.json"
    echo "=== T=$T D=$D repeat=$REPEAT -> $OUT ==="
    python3 "$REPO_ROOT/scripts/benchmark_nvfp4_isoquant.py" \
      --baseline-model "$BASELINE" \
      --nvfp4-model "$NVFP4" \
      --output "$OUT" \
      --prefill-tokens "$T" \
      --decode-tokens "$D" \
      --isoquant-bits 3 2>&1 | tail -10
  done
done
echo "Baseline complete: $OUT_DIR"
```

- [x] **Step 2: Make executable + run a smoke at T=4096 only**

```bash
chmod +x scripts/run_phase0_baseline.sh
DATE=2026-04-24-smoke python3 scripts/benchmark_nvfp4_isoquant.py \
  --baseline-model /Users/anthonylui/Models/Qwen3.6-35B-A3B-4bit \
  --nvfp4-model /Users/anthonylui/Models/Qwen3.6-35B-A3B-nvfp4 \
  --output /tmp/phase0_smoke.json \
  --prefill-tokens 4096 --decode-tokens 512 --isoquant-bits 3
```

Expected: 4 cells run cleanly, JSON written, no errors. Verifies the runner before
launching the full sweep.

### Task 0.3: Run the full Phase 0 baseline matrix

**Files:**
- Modifies: `artifacts/phase0_baseline_2026-04-24/` (output dir)

- [x] **Step 1: Launch the full sweep**

This is ~3 hours. Run in foreground if you can monitor; otherwise background:

```bash
DATE=2026-04-24 nohup bash scripts/run_phase0_baseline.sh > /tmp/phase0_baseline.log 2>&1 &
echo $! > /tmp/phase0_baseline.pid
```

- [x] **Step 2: Confirm artifacts written**

After completion (~3 hours):

```bash
ls artifacts/phase0_baseline_2026-04-24/ | wc -l
```

Expected: 12 files (4 T values × 3 repeats).

### Task 0.4: Compute and verify variance gate

**Files:**
- Create: `scripts/phase0_variance_check.py`

- [x] **Step 1: Write the variance check script**

```python
"""Phase 0 variance gate: within-cell variance must be < 5% across 3 repeats at T=8K."""
import json
import sys
from pathlib import Path
from statistics import mean, stdev

ART_DIR = Path("artifacts/phase0_baseline_2026-04-24")
T_GATE = 8192
MAX_VARIANCE_PCT = 5.0

results: dict[str, list[float]] = {}
for f in sorted(ART_DIR.glob(f"matrix_T{T_GATE}_*.json")):
    d = json.loads(f.read_text())
    for cell_id, run in d["runs"].items():
        if run.get("status") != "ok":
            continue
        results.setdefault(cell_id, []).append(run["decode_tok_per_s"])

failed = []
print(f"{'cell':<24}  {'mean tok/s':>12}  {'stdev':>8}  {'cv%':>6}  {'gate'}")
for cell, vals in results.items():
    if len(vals) < 2:
        print(f"{cell:<24}  {vals[0]:>12.2f}  {'n/a':>8}  {'n/a':>6}  N=1 SKIP")
        continue
    m = mean(vals)
    s = stdev(vals)
    cv = (s / m) * 100.0 if m > 0 else 0.0
    ok = cv < MAX_VARIANCE_PCT
    if not ok:
        failed.append((cell, cv))
    print(f"{cell:<24}  {m:>12.2f}  {s:>8.3f}  {cv:>5.1f}  {'OK' if ok else 'FAIL'}")

if failed:
    print("\nGATE FAILED:", failed)
    sys.exit(1)
print("\nGATE PASS")
```

- [x] **Step 2: Run the variance check**

```bash
python3 scripts/phase0_variance_check.py
```

Expected: GATE PASS with all 4 cells at < 5% CV at T=8192. If GATE FAILED, find the source
(thermal / swap / contention) and re-run the affected cells before proceeding.

### Task 0.5: Persist post-fix per-kernel attribution (script in version control)

**Files:**
- Create: `scripts/instrument_isoquant_decode.py` (move from `/tmp` so Phase 0 is
  reproducible — never depend on a `/tmp` script for a paper-cited artifact)
- Create: `artifacts/phase0_baseline_2026-04-24/kernel_attribution_4k.json`
- Create: `artifacts/phase0_baseline_2026-04-24/kernel_attribution_8k.json`

- [x] **Step 1: Create the instrumentation script in the repo**

```bash
cat > scripts/instrument_isoquant_decode.py <<'PYEOF'
"""Per-kernel ms-attribution for IsoQuant decode steps.

Wraps the four hot kernels (pack_indices_3bit, fused_qk_dot, fused_value_accum,
fused_value_accum_tiled) plus _apply_inverse_rotation with mx.synchronize-bracketed
timers, runs N decode steps after a configurable prefill, and writes a JSON
attribution artifact.

Usage:
  python3 scripts/instrument_isoquant_decode.py OUT.json [--prefill T] [--decode N]
"""
import argparse
import json
import os
import time
from collections import defaultdict

os.environ.setdefault("ISOQUANT_BITS", "3")

import mlx.core as mx

import mlx_lm.models.fused_kv_decode_kernels as fkdk
import mlx_lm.models.fused_kv_decode_tiled as fkdt
from mlx_lm import load
from mlx_lm.models.cache import finalize_deferred_kv_caches, make_prompt_cache
from mlx_lm.models.mlx_isoquant import IsoQuantKVCache, reset_stats, stats_summary

p = argparse.ArgumentParser()
p.add_argument("out_path")
p.add_argument("--model", default="/Users/anthonylui/Models/Qwen3.6-35B-A3B-4bit")
p.add_argument("--prefill", type=int, default=4096)
p.add_argument("--decode", type=int, default=100)
args = p.parse_args()

times: dict[str, list[float]] = defaultdict(list)


def _wrap(name, fn):
    def wrapped(*a, **kw):
        mx.synchronize()
        t = time.perf_counter()
        out = fn(*a, **kw)
        if isinstance(out, mx.array):
            mx.eval(out)
        elif isinstance(out, (list, tuple)):
            for o in out:
                if isinstance(o, mx.array):
                    mx.eval(o)
        mx.synchronize()
        times[name].append((time.perf_counter() - t) * 1000.0)
        return out
    return wrapped


fkdk.pack_indices_3bit = _wrap("pack_indices_3bit", fkdk.pack_indices_3bit)
fkdk.fused_qk_dot = _wrap("fused_qk_dot", fkdk.fused_qk_dot)
fkdk.fused_value_accum = _wrap("fused_value_accum", fkdk.fused_value_accum)
fkdt.fused_value_accum_tiled = _wrap("fused_value_accum_tiled", fkdt.fused_value_accum_tiled)

_orig_inv = IsoQuantKVCache._apply_inverse_rotation


def _timed_inv(self, *a, **kw):
    mx.synchronize()
    t = time.perf_counter()
    out = _orig_inv(self, *a, **kw)
    mx.eval(out)
    mx.synchronize()
    times["_apply_inverse_rotation"].append((time.perf_counter() - t) * 1000.0)
    return out


IsoQuantKVCache._apply_inverse_rotation = _timed_inv

print(f"Loading model {args.model}...")
model, tok = load(args.model)
cache = make_prompt_cache(model, kv_cache_type="isoquant")

print(f"Running {args.prefill}-token prefill...")
reset_stats()
prompt = mx.array([1] * args.prefill)
mx.synchronize()
t0 = time.perf_counter()
out = model(prompt[None, :], cache=cache)
mx.eval(out)
mx.synchronize()
prefill_ms = (time.perf_counter() - t0) * 1000.0

times.clear()  # discard prefill timings; only measure decode
finalize_deferred_kv_caches(cache)
mx.synchronize()

print(f"Running {args.decode} decode steps...")
y = mx.array([[42]])
mx.synchronize()
t0 = time.perf_counter()
for _ in range(args.decode):
    out = model(y, cache=cache)
    mx.eval(out)
mx.synchronize()
decode_total_ms = (time.perf_counter() - t0) * 1000.0
per_step_ms = decode_total_ms / args.decode

attribution = {
    "model": args.model,
    "T_prefill": args.prefill,
    "decode_steps": args.decode,
    "prefill_ms": prefill_ms,
    "total_decode_ms": decode_total_ms,
    "per_step_ms": per_step_ms,
    "tok_per_s": 1000.0 / per_step_ms,
    "kernels": {
        k: {
            "calls": len(v),
            "total_ms": sum(v),
            "avg_ms": sum(v) / len(v),
            "per_step_ms": sum(v) / args.decode,
            "pct_of_decode": sum(v) / decode_total_ms * 100.0,
        }
        for k, v in times.items() if v
    },
    "global_stats": stats_summary(),
}
os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
json.dump(attribution, open(args.out_path, "w"), indent=2, default=str)
print(f"Wrote {args.out_path}: per_step={per_step_ms:.2f} ms ({attribution['tok_per_s']:.1f} tok/s)")
PYEOF
chmod +x scripts/instrument_isoquant_decode.py
```

- [x] **Step 2: Run at T=4096, persist artifact**

```bash
ISOQUANT_VACCUM_TILE=128 python3 scripts/instrument_isoquant_decode.py \
  artifacts/phase0_baseline_2026-04-24/kernel_attribution_4k.json \
  --prefill 4096 --decode 100
```

Expected: ~60 ms/step total decode; `fused_value_accum_tiled` ≈ 4.5 ms/step,
`pack_indices_3bit` ≈ 8 ms/step. JSON written.

- [x] **Step 3: Run at T=8192, persist artifact**

```bash
ISOQUANT_VACCUM_TILE=128 python3 scripts/instrument_isoquant_decode.py \
  artifacts/phase0_baseline_2026-04-24/kernel_attribution_8k.json \
  --prefill 8192 --decode 100
```

Expected: ~32 ms/step total decode; attribution shows `fused_value_accum_tiled` < 10
ms/step but `pack_indices_3bit` grows toward 21 ms.

- [x] **Step 4: Commit Phase 0 artifacts + scripts**

```bash
git add artifacts/phase0_baseline_2026-04-24/ \
        scripts/run_phase0_baseline.sh \
        scripts/phase0_variance_check.py \
        scripts/instrument_isoquant_decode.py
git commit -m "phase 0: baseline + per-kernel attribution at 4K/8K/16K/32K (3 repeats)"
```

---

## Phase 1 — Bandwidth sanity check

DeepSeek's pre-fusion check. The roofline analysis put the tiled kernel 450× above the
memory-bound floor at small T — that's a big gap, and the size of Phase 3's expected gain
depends on which floor the tiled kernel actually approaches at large T.

### Task 1.1: Write bandwidth instrumentation

**Files:**
- Create: `scripts/instrument_bandwidth_iso.py`

- [x] **Step 1: Write the script**

```python
"""Bandwidth-bound vs execution-bound: characterize tiled fused_value_accum.

Method: for each T, time fused_value_accum_tiled in isolation. Compute bytes touched
(packed V read + attention weights read + partials write) and divide by measured time
to get achieved GB/s. Compare to M4 Max realistic peak (~300 GB/s for this access pattern).
"""
import json
import os
import time

os.environ["ISOQUANT_BITS"] = "3"

import mlx.core as mx
import numpy as np

from mlx_lm.models.fused_kv_decode_kernels import pack_indices_3bit
from mlx_lm.models.fused_kv_decode_tiled import fused_value_accum_tiled

H_KV, H_Q, D = 2, 16, 256
TILE = 128
N_CALLS = 50

def synthetic(T):
    rng = np.random.default_rng(42)
    indices = mx.array(rng.integers(0, 8, size=(H_KV, T, D), dtype=np.uint8))
    norms = mx.array(rng.standard_normal((H_KV, T)).astype(np.float32))
    centroids = mx.array(np.linspace(-1.5, 1.5, 8, dtype=np.float32))
    raw = mx.array(rng.standard_normal((H_Q, T)).astype(np.float32) * 4.0)
    attn = mx.softmax(raw, axis=-1)
    repeats = H_Q // H_KV
    kv_head_map = mx.arange(H_Q, dtype=mx.uint32) // repeats
    V_packed = pack_indices_3bit(indices)
    mx.eval(V_packed, norms, centroids, attn, kv_head_map)
    return V_packed, centroids, norms, attn, kv_head_map

def time_kernel(T):
    V_packed, centroids, norms, attn, kv_head_map = synthetic(T)
    # warmup
    for _ in range(3):
        out = fused_value_accum_tiled(V_packed, centroids, norms, attn, kv_head_map,
                                       H_Q, T, D, tile_size=TILE)
        mx.eval(out)
    mx.synchronize()
    t = time.perf_counter()
    for _ in range(N_CALLS):
        out = fused_value_accum_tiled(V_packed, centroids, norms, attn, kv_head_map,
                                       H_Q, T, D, tile_size=TILE)
        mx.eval(out)
    mx.synchronize()
    return (time.perf_counter() - t) * 1000.0 / N_CALLS

def bytes_touched(T):
    """Per-call bytes touched (read + write).

    Accounting:
    - packed V is keyed by KV head but the kernel grid is over (tile, q_head). Each
      (tile, q_head) TG re-reads its KV-head's packed slice and norms. The hardware
      cache may amortise this, but the *bandwidth touched* is the per-q-head sum.
      Conservatively count H_Q reads (worst case, no cache).
    - norms keyed similarly: H_KV * T * 4 bytes per re-read, H_Q reads worst case.
    - attention weights are per (q_head, T): one read per TG, summed across tiles.
    - partials write: num_tiles * H_Q * D * 4.
    - mx.sum reduction reads partials once and writes (H_Q, D) output.
    """
    num_tiles = (T + TILE - 1) // TILE
    # Per-q-head per-tile reads:
    packed_v_per_qhead = T * (D * 3 // 8)
    norms_per_qhead = T * 4
    attn_w_per_qhead = T * 4
    per_qhead_reads = (packed_v_per_qhead + norms_per_qhead + attn_w_per_qhead) * num_tiles
    total_kernel_reads = per_qhead_reads * H_Q
    partials_write = num_tiles * H_Q * D * 4
    # mx.sum reduction pass: read all partials, write reduced output
    reduction_read = num_tiles * H_Q * D * 4
    reduction_write = H_Q * D * 4
    return total_kernel_reads + partials_write + reduction_read + reduction_write

PEAK_GBS = 300.0   # M4 Max realistic for this access pattern

results = []
for T in [1024, 2048, 4096, 8192, 16384]:
    ms = time_kernel(T)
    bytes_ = bytes_touched(T)
    achieved_gbs = bytes_ / 1e9 / (ms / 1000.0)
    pct_peak = achieved_gbs / PEAK_GBS * 100.0
    results.append({"T": T, "ms": ms, "bytes": bytes_,
                    "achieved_gbs": achieved_gbs, "pct_peak": pct_peak})
    print(f"T={T:>6}  ms={ms:>7.3f}  GB/s={achieved_gbs:>7.2f}  {pct_peak:>5.1f}% peak")

out_path = f"artifacts/phase1_bandwidth/tiled_v_accum_bw.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
json.dump({"peak_gbs": PEAK_GBS, "results": results},
          open(out_path, "w"), indent=2)
print(f"Wrote {out_path}")
```

- [x] **Step 2: Run it**

```bash
python3 scripts/instrument_bandwidth_iso.py
```

Expected: GB/s achieved per T value, plus % of 300 GB/s peak. Likely range based on prior
data: 5-30% of peak at small T (dispatch-bound), trending up at large T.

### Task 1.2: Decision memo for Phase 3 priority

**Files:**
- Create: `artifacts/phase1_bandwidth/phase3_decision_memo.md`

- [x] **Step 1: Write the memo using the rubric**

```markdown
# Phase 3 priority decision

Tiled V-accum bandwidth at T={1024..16384}: see tiled_v_accum_bw.json

Maximum achieved BW: __ GB/s (__% of 300 GB/s peak)

Decision per Phase 1 rubric:
- ≥ 60% of peak: traffic-reduction path → fusion gives upper-end gain. Proceed Phase 3 as planned.
- 20–60%: mixed (dispatch + traffic). Proceed; expect mid-range gain.
- < 20%: instruction-level waste. Investigate scattered centroid-load patterns first;
  consider software-pipelined Kernel C variant before fusing.

CONCLUSION: <one of the three branches>
```

- [x] **Step 2: Commit**

```bash
git add scripts/instrument_bandwidth_iso.py artifacts/phase1_bandwidth/
git commit -m "phase 1: bandwidth characterization of tiled V-accum kernel"
```

---

## Phase 2 — Incremental packed-cache append

The deterministic mid-single-digit-ms win. Replaces the per-decode-step full repack
(`pack_indices_3bit` on the entire stored cache) with an incremental append into a
pre-allocated packed buffer.

### Task 2.1: Write the failing correctness test

**Files:**
- Create: `mlx-lm/tests/test_iso_incremental_pack.py`

- [x] **Step 1: Write the test**

```python
"""Phase 2: incremental packed-cache append must be bit-exact vs full re-pack."""
import os

os.environ["ISOQUANT_BITS"] = "3"

import mlx.core as mx
import numpy as np
import pytest

from mlx_lm.models.fused_kv_decode_kernels import pack_indices_3bit
from mlx_lm.models.mlx_isoquant import IsoQuantKVCache

NUM_HEADS, HEAD_DIM = 2, 256


def _gen_kv(seq_len: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    keys = mx.array(rng.standard_normal((1, NUM_HEADS, seq_len, HEAD_DIM)).astype(np.float32))
    values = mx.array(rng.standard_normal((1, NUM_HEADS, seq_len, HEAD_DIM)).astype(np.float32))
    mx.eval(keys, values)
    return keys, values


def test_incremental_append_matches_rebuild():
    """After 10 decode steps, the incrementally-built packed cache must equal a
    fresh pack_indices_3bit call on the full compressed indices."""
    cache = IsoQuantKVCache(num_heads=NUM_HEADS, head_dim=HEAD_DIM, bit_width=3, layer_idx=0)
    # Prefill 256 tokens
    keys, values = _gen_kv(256)
    cache.update_and_fetch(keys, values)
    cache.finalize_deferred_prefill()

    # 10 decode steps, one new token each
    for step in range(10):
        k_new, v_new = _gen_kv(1, seed=100 + step)
        cache.update_and_fetch(k_new, v_new)

    # The incrementally-maintained packed buffer
    inc_keys = cache._packed_keys_cache
    inc_vals = cache._packed_values_cache
    assert inc_keys is not None and inc_vals is not None, "incremental cache must be populated"

    # Rebuild from scratch via pack_indices_3bit on the stored compressed indices
    ref_keys = pack_indices_3bit(cache.compressed_keys["indices"])
    ref_vals = pack_indices_3bit(cache.compressed_values["indices"])
    mx.eval(inc_keys, inc_vals, ref_keys, ref_vals)

    np.testing.assert_array_equal(np.asarray(inc_keys), np.asarray(ref_keys))
    np.testing.assert_array_equal(np.asarray(inc_vals), np.asarray(ref_vals))


def test_incremental_append_long_sequence():
    """10K total tokens, no drift between incremental and rebuild paths."""
    cache = IsoQuantKVCache(num_heads=NUM_HEADS, head_dim=HEAD_DIM, bit_width=3, layer_idx=0)
    keys, values = _gen_kv(2048)
    cache.update_and_fetch(keys, values)
    cache.finalize_deferred_prefill()
    for step in range(8000):
        k_new, v_new = _gen_kv(1, seed=10000 + step)
        cache.update_and_fetch(k_new, v_new)
    inc_keys = cache._packed_keys_cache
    inc_vals = cache._packed_values_cache
    ref_keys = pack_indices_3bit(cache.compressed_keys["indices"])
    ref_vals = pack_indices_3bit(cache.compressed_values["indices"])
    mx.eval(inc_keys, inc_vals, ref_keys, ref_vals)
    np.testing.assert_array_equal(np.asarray(inc_keys), np.asarray(ref_keys))
    np.testing.assert_array_equal(np.asarray(inc_vals), np.asarray(ref_vals))
```

- [x] **Step 2: Run the tests to verify they FAIL**

```bash
cd mlx-lm && python3 -m pytest tests/test_iso_incremental_pack.py -v
```

Expected: FAIL with `inc_keys is None` (because the current code invalidates the packed
cache on every decode append; there's no incrementally-maintained buffer to compare).

### Task 2.2: Replace invalidation with incremental append in `update_and_fetch`

**Files:**
- Modify: `mlx-lm/mlx_lm/models/mlx_isoquant.py:599-658` (the `update_and_fetch` decode branch)

- [x] **Step 1: Read the current decode branch**

Run:
```bash
sed -n '599,658p' mlx-lm/mlx_lm/models/mlx_isoquant.py
```

Locate the section starting with `# Decode phase (post-finalize): compress incrementally.`
and ending with `self._invalidate_fused_caches()`.

- [x] **Step 2: Replace the invalidation with append-into-packed**

In `mlx-lm/mlx_lm/models/mlx_isoquant.py`, replace the decode-branch ending:

```python
        self.compressed_keys = _concat_compressed(
            self.compressed_keys, new_compressed_keys
        )
        self.compressed_values = _concat_compressed(
            self.compressed_values, new_compressed_vals
        )
        self._invalidate_fused_caches()
```

with:

```python
        self.compressed_keys = _concat_compressed(
            self.compressed_keys, new_compressed_keys
        )
        self.compressed_values = _concat_compressed(
            self.compressed_values, new_compressed_vals
        )
        # Phase 2: incremental packed-cache append. Pack only the new token's indices and
        # concatenate to the existing _packed_*_cache; do NOT invalidate the whole cache.
        from .fused_kv_decode_kernels import pack_indices_3bit

        new_packed_k = pack_indices_3bit(new_compressed_keys["indices"])
        new_packed_v = pack_indices_3bit(new_compressed_vals["indices"])
        if self._packed_keys_cache is None:
            self._packed_keys_cache = new_packed_k
            self._packed_values_cache = new_packed_v
        else:
            self._packed_keys_cache = mx.concatenate(
                [self._packed_keys_cache, new_packed_k], axis=1
            )
            self._packed_values_cache = mx.concatenate(
                [self._packed_values_cache, new_packed_v], axis=1
            )
```

- [x] **Step 3: Update `finalize_deferred_prefill` to populate the packed buffer in bulk**

Locate `finalize_deferred_prefill` (around line 524). After `self.compressed_keys = self._compress_batch(all_keys)` and the corresponding values line, add:

```python
        # Phase 2: pre-populate packed cache once at finalize so decode appends
        # extend rather than rebuild.
        from .fused_kv_decode_kernels import pack_indices_3bit

        self._packed_keys_cache = pack_indices_3bit(self.compressed_keys["indices"])
        self._packed_values_cache = pack_indices_3bit(self.compressed_values["indices"])
```

(immediately before the existing `self._invalidate_fused_caches()` call — and remove that
call since we just populated it.)

- [x] **Step 4: Run tests to verify they PASS**

```bash
cd mlx-lm && python3 -m pytest tests/test_iso_incremental_pack.py -v
```

Expected: both tests PASS (bit-exact equality across 10 and 8000 decode steps).

### Task 2.3: Verify packed_cache_misses counter drops to ~0

**Files:**
- Run only

- [x] **Step 1: Re-run instrumentation, check counter**

```bash
ISOQUANT_VACCUM_TILE=128 python3 scripts/instrument_isoquant_decode.py \
  /tmp/phase2_smoke.json --prefill 4096 --decode 50
```

Expected in the printed `Global IsoQuant stats`: `packed_cache_misses` near 10 (one per
attention layer at finalize), not 2000+ as before. `packed_cache_hit_rate` near 1.0.

If `packed_cache_misses` is still high, the `_invalidate_fused_caches()` call still fires
somewhere — `grep -n "_invalidate_fused_caches" mlx-lm/mlx_lm/models/mlx_isoquant.py` and
ensure only the runtime-shape-reset call at line ~593 remains (decode and finalize callers
are removed by Task 2.2).

### Task 2.4: Phase 2 end-to-end benchmark

**Files:**
- Modifies: `artifacts/phase2_incremental/` (output)

- [x] **Step 1: Re-run the full matrix at 4K and 8K**

```bash
mkdir -p artifacts/phase2_incremental
for T in 4096 8192; do
  D=$(( T < 8192 ? 512 : 1024 ))
  for R in 1 2 3; do
    python3 scripts/benchmark_nvfp4_isoquant.py \
      --baseline-model /Users/anthonylui/Models/Qwen3.6-35B-A3B-4bit \
      --nvfp4-model /Users/anthonylui/Models/Qwen3.6-35B-A3B-nvfp4 \
      --output "artifacts/phase2_incremental/matrix_T${T}_d${D}_r${R}.json" \
      --prefill-tokens "$T" --decode-tokens "$D" --isoquant-bits 3 2>&1 | tail -8
  done
done
```

- [x] **Step 2: Compare to Phase 0 baseline**

Use the phase0_variance_check.py script as a template — write a small comparator that prints:

```
cell                      phase0 mean tok/s    phase2 mean tok/s    delta tok/s    delta %
nvfp4_isoquant            30.84                XX                   YY            ZZ%
baseline_isoquant         22.01                ...
nvfp4_default             92.45 (unchanged)    ...                  (sanity check, expect ~0% change)
baseline_default          ...                  ...                  (sanity check)
```

Expected (forecast): nvfp4_isoquant rises from ~30 tok/s toward ~37-43 tok/s at 8K
(the 9 ms expected savings on a 32 ms baseline = ~28% improvement). Default-KV cells
should be unchanged within variance — if they shift, environmental contamination, not the
code change.

**Gate:** observed delta on nvfp4_isoquant within 50% of predicted ~9 ms saving on the
per-kernel attribution. If smaller, find leakage (likely MLX buffer-allocation amortising
the saved work) before Phase 3.

- [x] **Step 3: Commit**

```bash
git add mlx-lm/mlx_lm/models/mlx_isoquant.py \
        mlx-lm/tests/test_iso_incremental_pack.py \
        artifacts/phase2_incremental/
git commit -m "phase 2: incremental packed-cache append (eliminates per-step repack)"
```

---

## Phase 3 — Fused NPT=8 + T-tile pipeline

Structural cleanup. New single fused kernel handles head_dim=256 (NPT=8) and tiles over T,
replacing the 3-kernel pipeline's dispatch overhead and inter-kernel sync. **Skip this
phase if Phase 1 returned <20% of peak BW** — investigate scattered centroid loads first.

### Task 3.1: Write the design doc

**Files:**
- Create: `docs/superpowers/specs/2026-04-24-fused-npt8-tiled-design.md`

- [x] **Step 1: Write the doc with the kernel signature, threadgroup map, FA2 merge contract**

```markdown
# Fused NPT=8 T-tiled attention kernel — design

## Function signature
fused_attention_npt8_tiled(
    K_packed: (H_kv, T, packed_bytes) uint8,
    V_packed: (H_kv, T, packed_bytes) uint8,
    centroids: (8,) float32,
    k_norms: (H_kv, T) float32,
    v_norms: (H_kv, T) float32,
    q_rot: (H_q, D=256) float32,
    kv_head_map: (H_q,) uint32,
    blocks_t: (H_kv, N_BLOCKS=64, 4, 4) float32,
    scale: float,
    use_hadamard: bool,
    mask: (H_q, T) float32 | None,
    tile_size: int,
) -> (H_q, D=256) float32

## Threadgroup map
Grid:        (32 * num_tiles, H_q, 1)
Threadgroup: (32, 1, 1)
Per TG: handles one (tile, head). 32 threads, NPT=8 (D/32=8 dims per thread).

## Two-pass strategy
Pass 1 (per-tile kernel): each TG produces (m_tile, l_tile, O_tile) per head.
Pass 2 (merge kernel): reduces num_tiles partials per head via FA2 merge formula:
  m   = max(m_a, m_b)
  l   = exp(m_a - m) * l_a + exp(m_b - m) * l_b
  O_d = exp(m_a - m) * O_a_d + exp(m_b - m) * O_b_d
Then divide O by l, apply SO(4) inverse rotation + optional Hadamard butterfly.

## Numerical contract
FP32 throughout merge. Match 3-kernel pipeline output up to FP16 re-association tolerance
(rtol=1e-3, atol=1e-4 per the existing equivalence test pattern).

## Risks
- NPT=8 doubles per-thread register state vs NPT=4. Watch occupancy in Xcode capture.
- Hadamard butterfly currently uses threadgroup memory of size D=256 floats. With 32
  threads that fits. Confirm shared memory budget in the merge kernel.
```

- [x] **Step 2: Commit the spec**

```bash
git add docs/superpowers/specs/2026-04-24-fused-npt8-tiled-design.md
git commit -m "spec: design doc for fused NPT=8 + T-tiled attention kernel"
```

### Task 3.2: Write the failing equivalence test (small fixture first)

**Files:**
- Create: `mlx-lm/tests/test_fused_npt8.py`

- [x] **Step 1: Write the test for d=256, T=256, single head**

**Important:** the reference path uses the **stable, unmodified `fused_value_accum`** (not
the `fused_value_accum_tiled` we built last session). Comparing the new fused NPT=8 kernel
against another recently-introduced kernel weakens fault localization — if both share a
bug, the test passes silently. Comparing against `fused_value_accum` (the original serial
kernel that's been in production for many runs) means a test failure points cleanly at the
new kernel.

```python
"""Phase 3: fused NPT=8 T-tiled kernel must equal the stable 3-kernel pipeline.

The reference uses the original `fused_value_accum` (serial T loop, in production for many
runs) — not the recently-introduced `fused_value_accum_tiled`. Comparing two new kernels to
each other could pass silently if they share a bug.
"""
import os

os.environ["ISOQUANT_BITS"] = "3"

import mlx.core as mx
import numpy as np

from mlx_lm.models.fused_kv_decode_kernels import (
    fused_qk_dot,
    fused_value_accum,  # stable baseline, NOT the tiled variant
    pack_indices_3bit,
)


def _ref_3kernel_stable(K_packed, V_packed, centroids, k_norms, v_norms, q_rot,
                         kv_head_map, num_heads, T, D, scale, mask=None):
    """Reference: original 3-kernel pipeline (stable baseline)."""
    scores = fused_qk_dot(K_packed, centroids, k_norms, q_rot, kv_head_map,
                          num_heads, T, D) * scale
    if mask is not None:
        scores = scores + mask
    attn = mx.softmax(scores, axis=-1)
    # NB: stable serial fused_value_accum — NOT fused_value_accum_tiled
    out_rot = fused_value_accum(V_packed, centroids, v_norms, attn,
                                  kv_head_map, num_heads, T, D)
    return out_rot  # NB: skip inverse rotation for now; test rotated output match


def _synthetic_d256(T, H_kv=2, H_q=16):
    rng = np.random.default_rng(11)
    indices_k = mx.array(rng.integers(0, 8, (H_kv, T, 256), dtype=np.uint8))
    indices_v = mx.array(rng.integers(0, 8, (H_kv, T, 256), dtype=np.uint8))
    norms_k = mx.array(rng.standard_normal((H_kv, T)).astype(np.float32))
    norms_v = mx.array(rng.standard_normal((H_kv, T)).astype(np.float32))
    centroids = mx.array(np.linspace(-1.5, 1.5, 8, dtype=np.float32))
    q_rot = mx.array(rng.standard_normal((H_q, 256)).astype(np.float32))
    kv_head_map = mx.arange(H_q, dtype=mx.uint32) // (H_q // H_kv)
    K_p = pack_indices_3bit(indices_k)
    V_p = pack_indices_3bit(indices_v)
    mx.eval(K_p, V_p, norms_k, norms_v, centroids, q_rot, kv_head_map)
    return K_p, V_p, centroids, norms_k, norms_v, q_rot, kv_head_map


def test_fused_npt8_matches_stable_3kernel_no_inverse_rotation():
    """NPT=8 fused kernel matches the original (non-tiled) 3-kernel reference."""
    from mlx_lm.models.fused_kv_decode_npt8_tiled import fused_attention_npt8_tiled

    T, H_kv, H_q, D = 256, 2, 16, 256
    K_p, V_p, c, nk, nv, q, m = _synthetic_d256(T, H_kv, H_q)
    scale = 1.0 / np.sqrt(D)

    ref = _ref_3kernel_stable(K_p, V_p, c, nk, nv, q, m, H_q, T, D, scale)
    out = fused_attention_npt8_tiled(K_p, V_p, c, nk, nv, q, m,
                                       blocks_t=mx.zeros((H_kv, 64, 4, 4)),
                                       scale=scale, use_hadamard=False,
                                       mask=None, tile_size=128,
                                       num_heads=H_q, seq_len=T, head_dim=D)
    mx.eval(ref, out)

    # NB: ref skips inverse rotation; new kernel must also skip when blocks_t is zeros
    # AND use_hadamard=False (verify by inspecting kernel source).
    np.testing.assert_allclose(np.asarray(ref), np.asarray(out),
                                rtol=1e-3, atol=1e-4)


def test_fused_npt8_also_matches_tiled_v_accum_path():
    """Sanity cross-check: NPT=8 fused must also match the tiled-V-accum path
    (which Phase 2 already validated against the stable serial kernel). Two
    equivalence relations — the new kernel == both. If the stable-baseline test
    above passes but this one fails, fused_value_accum_tiled has drifted from its
    Phase 2 baseline; fix that drift first."""
    from mlx_lm.models.fused_kv_decode_npt8_tiled import fused_attention_npt8_tiled
    from mlx_lm.models.fused_kv_decode_tiled import fused_value_accum_tiled

    T, H_kv, H_q, D = 256, 2, 16, 256
    K_p, V_p, c, nk, nv, q, m = _synthetic_d256(T, H_kv, H_q)
    scale = 1.0 / np.sqrt(D)

    scores = fused_qk_dot(K_p, c, nk, q, m, H_q, T, D) * scale
    attn = mx.softmax(scores, axis=-1)
    ref_tiled = fused_value_accum_tiled(V_p, c, nv, attn, m, H_q, T, D, tile_size=128)

    out = fused_attention_npt8_tiled(K_p, V_p, c, nk, nv, q, m,
                                       blocks_t=mx.zeros((H_kv, 64, 4, 4)),
                                       scale=scale, use_hadamard=False,
                                       mask=None, tile_size=128,
                                       num_heads=H_q, seq_len=T, head_dim=D)
    mx.eval(ref_tiled, out)

    np.testing.assert_allclose(np.asarray(ref_tiled), np.asarray(out),
                                rtol=1e-3, atol=1e-4)
```

- [x] **Step 2: Run to verify FAIL**

```bash
cd mlx-lm && python3 -m pytest tests/test_fused_npt8.py -v
```

Expected: FAIL — `ImportError: fused_kv_decode_npt8_tiled` (module doesn't exist yet).

### Task 3.3: Implement the per-tile main kernel

**Files:**
- Create: `mlx-lm/mlx_lm/models/fused_kv_decode_npt8_tiled.py`

- [x] **Step 1: Write the per-tile main kernel + Python wrapper**

Begin with the per-tile kernel only (skip merge for now; merge in Task 3.4):

```python
"""Phase 3: fused NPT=8 T-tiled attention kernel for head_dim=256.

The existing fully-fused kernel asserts NPT=4 (head_dim=128). This is the NPT=8
generalisation, with T-parallel tiling so the inner T-loop runs across multiple
threadgroups and partial (m, l, O) tuples are merged in a second kernel.

The per-tile kernel is a near-clone of _FULLY_FUSED_ATTENTION_SOURCE in
fused_kv_decode_kernels.py with: NPT=8, T-bounded loop, partials written to
global instead of finalised in-register, inverse rotation deferred to merge.
"""
from __future__ import annotations

import mlx.core as mx

# Per-tile kernel: produces (m_tile, l_tile, O_tile) for one (tile, head).
_PER_TILE_NPT8_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    uint tile_id = threadgroup_position_in_grid.x;
    uint q_head = threadgroup_position_in_grid.y;
    uint kv_head = kv_head_map[q_head];
    uint T = seq_len[0];
    uint T_TILE = tile_size[0];
    uint use_mask = has_mask[0];
    uint num_tiles = (T + T_TILE - 1) / T_TILE;

    uint t_start = tile_id * T_TILE;
    uint t_end = min(t_start + T_TILE, T);

    uint dim_base = lane * NPT;  // NPT=8 for D=256
    uint w = dim_base / 8;
    uint bp_base = dim_base % 8;
    uint w_byte = w * 3;

    float q_r[NPT];
    for (uint i = 0; i < NPT; i++) q_r[i] = q[q_head * D + dim_base + i];

    float m_run = -1e38f;
    float l_run = 0.0f;
    float O_r[NPT];
    for (uint i = 0; i < NPT; i++) O_r[i] = 0.0f;

    uint kv_k_base = kv_head * T * PACKED_WORDS * 3;
    uint kv_v_base = kv_head * T * PACKED_WORDS * 3;
    uint stride_bytes = PACKED_WORDS * 3;

    for (uint t = t_start; t < t_end; t++) {
        uint k_off = kv_k_base + t * stride_bytes + w_byte;
        uint kw = uint(K_packed[k_off]) | (uint(K_packed[k_off+1]) << 8)
                 | (uint(K_packed[k_off+2]) << 16);
        float k_norm = k_norms[kv_head * T + t];

        float partial = 0.0f;
        for (uint i = 0; i < NPT; i++) {
            float k_val = centroids[(kw >> ((bp_base + i) * 3)) & 0x7] * k_norm;
            partial += q_r[i] * k_val;
        }
        float score = simd_sum(partial) * scale_val[0];
        if (use_mask) score += mask_data[q_head * T + t];

        float m_new = max(m_run, score);
        float corr = exp(m_run - m_new);
        float es = exp(score - m_new);

        uint v_off = kv_v_base + t * stride_bytes + w_byte;
        uint vw = uint(V_packed[v_off]) | (uint(V_packed[v_off+1]) << 8)
                 | (uint(V_packed[v_off+2]) << 16);
        float v_norm = v_norms[kv_head * T + t];

        for (uint i = 0; i < NPT; i++) {
            float v_val = centroids[(vw >> ((bp_base + i) * 3)) & 0x7] * v_norm;
            O_r[i] = O_r[i] * corr + es * v_val;
        }
        l_run = l_run * corr + es;
        m_run = m_new;
    }

    // Write per-tile partials: O_tiles[tile_id, head, dim], m_tiles[tile_id, head], l_tiles[...]
    uint base = (tile_id * num_heads_q[0] + q_head) * D + dim_base;
    for (uint i = 0; i < NPT; i++) O_tiles[base + i] = O_r[i];
    if (lane == 0) {
        m_tiles[tile_id * num_heads_q[0] + q_head] = m_run;
        l_tiles[tile_id * num_heads_q[0] + q_head] = l_run;
    }
"""

_kernel_cache: dict[str, object] = {}


def _get_per_tile_kernel():
    if "per_tile_npt8" not in _kernel_cache:
        _kernel_cache["per_tile_npt8"] = mx.fast.metal_kernel(
            name="fused_attn_npt8_per_tile",
            input_names=["K_packed", "V_packed", "centroids", "k_norms", "v_norms",
                          "q", "kv_head_map", "scale_val", "seq_len", "tile_size",
                          "num_heads_q", "mask_data", "has_mask"],
            output_names=["O_tiles", "m_tiles", "l_tiles"],
            source=_PER_TILE_NPT8_SOURCE,
        )
    return _kernel_cache["per_tile_npt8"]
```

- [x] **Step 2: Add the Python wrapper (per-tile only, no merge yet)**

Append to the same file:

```python
def _per_tile_dispatch(K_packed, V_packed, centroids, k_norms, v_norms,
                        q_rot, kv_head_map, scale, mask, num_heads, seq_len,
                        head_dim, tile_size):
    assert head_dim == 256, f"NPT=8 kernel requires head_dim=256, got {head_dim}"
    num_tiles = (seq_len + tile_size - 1) // tile_size
    kernel = _get_per_tile_kernel()
    if mask is not None:
        m = mask.reshape(-1).astype(mx.float32)
        has_mask = mx.array([1], dtype=mx.uint32)
    else:
        m = mx.zeros((1,), dtype=mx.float32)
        has_mask = mx.array([0], dtype=mx.uint32)
    return kernel(
        inputs=[K_packed.reshape(-1), V_packed.reshape(-1), centroids.reshape(-1),
                k_norms.reshape(-1), v_norms.reshape(-1), q_rot.reshape(-1),
                kv_head_map.reshape(-1), mx.array([scale], dtype=mx.float32),
                mx.array([seq_len], dtype=mx.uint32),
                mx.array([tile_size], dtype=mx.uint32),
                mx.array([num_heads], dtype=mx.uint32), m, has_mask],
        template=[("D", head_dim), ("NPT", head_dim // 32),
                   ("PACKED_WORDS", head_dim // 8)],
        output_shapes=[(num_tiles * num_heads * head_dim,),
                        (num_tiles * num_heads,),
                        (num_tiles * num_heads,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
        grid=(num_tiles * 32, num_heads, 1),
        threadgroup=(32, 1, 1),
    )
```

### Task 3.4: Implement the merge kernel + public function

**Files:**
- Modify: `mlx-lm/mlx_lm/models/fused_kv_decode_npt8_tiled.py`

- [x] **Step 1: Add the merge kernel source**

Append:

```python
# Merge kernel: reduces num_tiles partials per head via FA2 formula.
_MERGE_NPT8_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    uint q_head = threadgroup_position_in_grid.x;
    uint NUM_TILES = num_tiles[0];
    uint dim_base = lane * NPT;

    float m_acc = -1e38f;
    float l_acc = 0.0f;
    float O_acc[NPT];
    for (uint i = 0; i < NPT; i++) O_acc[i] = 0.0f;

    for (uint t = 0; t < NUM_TILES; t++) {
        float m_t = m_tiles[t * num_heads_q[0] + q_head];
        float l_t = l_tiles[t * num_heads_q[0] + q_head];
        uint o_base = (t * num_heads_q[0] + q_head) * D + dim_base;

        float m_new = max(m_acc, m_t);
        float corr_acc = exp(m_acc - m_new);
        float corr_t = exp(m_t - m_new);
        for (uint i = 0; i < NPT; i++) {
            O_acc[i] = O_acc[i] * corr_acc + O_tiles[o_base + i] * corr_t;
        }
        l_acc = l_acc * corr_acc + l_t * corr_t;
        m_acc = m_new;
    }

    // Normalize
    float inv_l = (l_acc > 0.0f) ? (1.0f / l_acc) : 0.0f;
    for (uint i = 0; i < NPT; i++) O_acc[i] *= inv_l;

    // Skip inverse rotation for the equivalence test (use_hadamard=false branch).
    // Inverse rotation will be added in Task 3.5.
    for (uint i = 0; i < NPT; i++) {
        output[q_head * D + dim_base + i] = O_acc[i];
    }
"""

def _get_merge_kernel():
    if "merge_npt8" not in _kernel_cache:
        _kernel_cache["merge_npt8"] = mx.fast.metal_kernel(
            name="fused_attn_npt8_merge",
            input_names=["O_tiles", "m_tiles", "l_tiles", "num_tiles", "num_heads_q"],
            output_names=["output"],
            source=_MERGE_NPT8_SOURCE,
        )
    return _kernel_cache["merge_npt8"]


def fused_attention_npt8_tiled(K_packed, V_packed, centroids, k_norms, v_norms,
                                 q_rot, kv_head_map, blocks_t, scale, use_hadamard,
                                 mask, tile_size, num_heads, seq_len, head_dim):
    """Fused NPT=8 T-tiled attention. Returns (H_q, D) float32."""
    O_tiles, m_tiles, l_tiles = _per_tile_dispatch(
        K_packed, V_packed, centroids, k_norms, v_norms, q_rot, kv_head_map,
        scale, mask, num_heads, seq_len, head_dim, tile_size,
    )
    num_tiles = (seq_len + tile_size - 1) // tile_size
    merge = _get_merge_kernel()
    (output,) = merge(
        inputs=[O_tiles, m_tiles, l_tiles,
                mx.array([num_tiles], dtype=mx.uint32),
                mx.array([num_heads], dtype=mx.uint32)],
        template=[("D", head_dim), ("NPT", head_dim // 32)],
        output_shapes=[(num_heads * head_dim,)],
        output_dtypes=[mx.float32],
        grid=(32, num_heads, 1),
        threadgroup=(32, 1, 1),
    )
    return output.reshape(num_heads, head_dim)
```

- [x] **Step 2: Run the failing test from Task 3.2**

```bash
cd mlx-lm && python3 -m pytest tests/test_fused_npt8.py -v
```

Expected: PASS within rtol=1e-3, atol=1e-4. If FAIL, the most common bugs are: (a) grid
launched in threadgroups not threads — check `grid=(num_tiles * 32, ...)` matches the
launch convention; (b) NPT mismatch — assert head_dim=256 and NPT=8; (c) m/l write-once
guard — only `lane == 0` should write per-head m/l, not all 32 lanes.

### Task 3.5: Add inverse rotation to the merge kernel

**Files:**
- Modify: `mlx-lm/mlx_lm/models/fused_kv_decode_npt8_tiled.py` (the merge source)
- Modify: `mlx-lm/tests/test_fused_npt8.py` (extend test to cover inverse rotation)

- [x] **Step 1: Replace the "skip inverse rotation" output write in the merge kernel**

Replace the trailing block in `_MERGE_NPT8_SOURCE`:

```c
    for (uint i = 0; i < NPT; i++) {
        output[q_head * D + dim_base + i] = O_acc[i];
    }
```

with the inverse rotation. NPT=8 means each thread owns **2 consecutive SO(4) blocks**
(blocks of size 4 dims; 8 dims per thread = 2 blocks). For NPT=4 in the existing kernel
each thread owned 1 block at `lane_block_id = lane`. For NPT=8, each thread owns blocks
`lane * 2` and `lane * 2 + 1`:

```c
    // SO(4) inverse rotation: each thread does 2 blocks
    for (uint b_off = 0; b_off < 2; b_off++) {
        uint block_id = lane * 2 + b_off;
        uint bo = (kv_head * N_BLOCKS + block_id) * 16;
        uint o_off = b_off * 4;  // O_acc[0..3] for first block, [4..7] for second
        float r0 = blocks_t[bo+ 0]*O_acc[o_off+0] + blocks_t[bo+ 1]*O_acc[o_off+1]
                 + blocks_t[bo+ 2]*O_acc[o_off+2] + blocks_t[bo+ 3]*O_acc[o_off+3];
        float r1 = blocks_t[bo+ 4]*O_acc[o_off+0] + blocks_t[bo+ 5]*O_acc[o_off+1]
                 + blocks_t[bo+ 6]*O_acc[o_off+2] + blocks_t[bo+ 7]*O_acc[o_off+3];
        float r2 = blocks_t[bo+ 8]*O_acc[o_off+0] + blocks_t[bo+ 9]*O_acc[o_off+1]
                 + blocks_t[bo+10]*O_acc[o_off+2] + blocks_t[bo+11]*O_acc[o_off+3];
        float r3 = blocks_t[bo+12]*O_acc[o_off+0] + blocks_t[bo+13]*O_acc[o_off+1]
                 + blocks_t[bo+14]*O_acc[o_off+2] + blocks_t[bo+15]*O_acc[o_off+3];
        O_acc[o_off+0] = r0; O_acc[o_off+1] = r1; O_acc[o_off+2] = r2; O_acc[o_off+3] = r3;
    }

    if (USE_HADAMARD) {
        threadgroup float sa[D];
        threadgroup float sb[D];
        for (uint i = 0; i < NPT; i++) sa[dim_base + i] = O_acc[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* src = sa;
        threadgroup float* dst = sb;
        for (uint stride = 1; stride < D; stride <<= 1) {
            for (uint i = 0; i < NPT; i++) {
                uint idx = dim_base + i;
                uint partner = idx ^ stride;
                float sv = src[idx];
                float pv = src[partner];
                dst[idx] = ((idx & stride) == 0) ? (sv + pv) : (pv - sv);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            threadgroup float* tmp = src; src = dst; dst = tmp;
        }
        float wht_norm = 1.0f / sqrt((float)D);
        for (uint i = 0; i < NPT; i++) {
            output[q_head * D + dim_base + i] = src[dim_base + i] * wht_norm;
        }
    } else {
        for (uint i = 0; i < NPT; i++) {
            output[q_head * D + dim_base + i] = O_acc[i];
        }
    }
```

Also add `blocks_t`, `kv_head_map` to the merge kernel's input list, plus templates
`USE_HADAMARD` and `N_BLOCKS = D / 4`. Pattern matches existing
`_FULLY_FUSED_ATTENTION_SOURCE` (lines 240-280) for NPT=4; only change is the per-block
loop.

- [x] **Step 2: Extend test to cover inverse rotation case**

In `mlx-lm/tests/test_fused_npt8.py`, add:

```python
def test_fused_npt8_matches_full_pipeline_with_inverse_rotation():
    from mlx_lm.models.fused_kv_decode_npt8_tiled import fused_attention_npt8_tiled
    from mlx_lm.models.mlx_isoquant import IsoQuantKVCache
    # Build a real cache, prefill 256 tokens, then call fused_attention vs the
    # 3-kernel reference path on the same inputs. Assert allclose.
    # ...
```

- [x] **Step 3: Run all NPT=8 tests**

```bash
cd mlx-lm && python3 -m pytest tests/test_fused_npt8.py -v
```

Expected: both tests PASS.

### Task 3.6: Wire into IsoQuantKVCache._fused_attention_metal

**Files:**
- Modify: `mlx-lm/mlx_lm/models/mlx_isoquant.py:786-825` (the `_fused_attention_metal`
  dispatcher and the threshold)

- [x] **Step 1: Add a class-level threshold + dispatch branch**

In `IsoQuantKVCache`, near `_T_TILED_VALUE_ACCUM_THRESHOLD`, add:

```python
    # When True, route head_dim=256 attention through the new fused NPT=8 T-tiled kernel
    # instead of the 3-kernel pipeline. Set to False to fall back for A/B comparison.
    _USE_FUSED_NPT8 = True
```

In `_fused_attention_metal`, after the existing dispatcher chooses between single-kernel
and 3-kernel based on `_SINGLE_KERNEL_T_THRESHOLD`, prepend a check:

```python
        if self._USE_FUSED_NPT8 and D == 256:
            from .fused_kv_decode_npt8_tiled import fused_attention_npt8_tiled
            import os
            try:
                tile_size = int(os.environ.get("FUSED_NPT8_TILE", 128))
            except ValueError:
                tile_size = 128
            return fused_attention_npt8_tiled(
                K_packed=k_packed, V_packed=v_packed, centroids=centroids,
                k_norms=k_norms, v_norms=v_norms, q_rot=q_rot,
                kv_head_map=kv_head_map, blocks_t=self.block_matrices_t,
                scale=scale, use_hadamard=self._use_hadamard, mask=mask,
                tile_size=tile_size, num_heads=H_q, seq_len=T, head_dim=D,
            )
```

- [x] **Step 2: Smoke-test against a real model**

```bash
ISOQUANT_VACCUM_TILE=128 FUSED_NPT8_TILE=128 \
  python3 scripts/instrument_isoquant_decode.py \
  /tmp/phase3_smoke.json --prefill 4096 --decode 50
```

Expected: runs without errors; per-step time should drop further (target <25 ms/step at 4K
context based on the forecast).

### Task 3.7: Phase 3 end-to-end benchmark + occupancy check

**Files:**
- Create: `artifacts/phase3_fused/`

- [x] **Step 1: Run the matrix at 4K, 8K, 16K**

```bash
mkdir -p artifacts/phase3_fused
for T in 4096 8192 16384; do
  D=$(( T < 8192 ? 512 : 1024 ))
  for R in 1 2 3; do
    python3 scripts/benchmark_nvfp4_isoquant.py \
      --baseline-model /Users/anthonylui/Models/Qwen3.6-35B-A3B-4bit \
      --nvfp4-model /Users/anthonylui/Models/Qwen3.6-35B-A3B-nvfp4 \
      --output "artifacts/phase3_fused/matrix_T${T}_d${D}_r${R}.json" \
      --prefill-tokens "$T" --decode-tokens "$D" --isoquant-bits 3 2>&1 | tail -8
  done
done
```

- [x] **Step 2: Compare to Phase 2 baseline**

Use the same comparator script pattern as Task 2.4 step 2.

**Gate:** at 8K, nvfp4_isoquant within 1.7× of nvfp4_default (10.8 ms/step → must be
≤ 18.4 ms/step). If not, residual is either model-compute-adjacent (out of scope) or a
constant-factor tax on scalar quantisation (relevant for Phase 5 Branch C).

- [x] **Step 3: Xcode GPU capture for occupancy**

Run a short Python script under Xcode's Metal Frame Capture (`xcrun xctrace` or Instruments
GUI), capturing one decode step. Save the capture to
`artifacts/phase3_fused/gpu_capture.gputrace`. Verify NPT=8 kernel occupancy is comparable
to the prior tiled V kernel; if dropped >30%, reduce per-thread state and re-test.

- [x] **Step 4: Commit**

```bash
git add mlx-lm/mlx_lm/models/fused_kv_decode_npt8_tiled.py \
        mlx-lm/mlx_lm/models/mlx_isoquant.py \
        mlx-lm/tests/test_fused_npt8.py \
        artifacts/phase3_fused/ \
        docs/superpowers/specs/2026-04-24-fused-npt8-tiled-design.md
git commit -m "phase 3: fused NPT=8 T-tiled attention kernel for head_dim=256"
```

---

## Phase 4 — Scaling validation at 32K

The datapoint that separates this from standard KV-quant work and validates the
depth-reduction model. Most published benchmarks stop at 8K.

### Task 4.1: Run the full matrix at 4K/8K/16K/32K post-Phase-3

**Files:**
- Create: `artifacts/phase4_scaling/`

- [x] **Step 1: Sweep T**

```bash
mkdir -p artifacts/phase4_scaling
for T in 4096 8192 16384 32768; do
  D=$(( T < 8192 ? 512 : 1024 ))
  for R in 1 2 3; do
    python3 scripts/benchmark_nvfp4_isoquant.py \
      --baseline-model /Users/anthonylui/Models/Qwen3.6-35B-A3B-4bit \
      --nvfp4-model /Users/anthonylui/Models/Qwen3.6-35B-A3B-nvfp4 \
      --output "artifacts/phase4_scaling/matrix_T${T}_d${D}_r${R}.json" \
      --prefill-tokens "$T" --decode-tokens "$D" --isoquant-bits 3 2>&1 | tail -5
  done
done
```

### Task 4.2: Plot the speedup curve vs the Brent-bound prediction

**Files:**
- Create: `scripts/run_scaling_validation.py`
- Create: `artifacts/phase4_scaling/scaling_chart.png`

- [x] **Step 1: Write the chart script**

```python
"""Phase 4: tiled-Kernel-C speedup vs T, overlay theoretical Brent bound.

Theoretical: S(T) = T / (B + log2(T / B)), saturates at T/B ≈ P_concurrent ≈ 160.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ART = Path("artifacts/phase4_scaling")
PHASE0 = Path("artifacts/phase0_baseline_2026-04-24")  # serial baseline reference
B = 32  # tile size

def cell_mean(art_dir, cell_id, T):
    vals = []
    for f in art_dir.glob(f"matrix_T{T}_*.json"):
        d = json.loads(f.read_text())
        run = d["runs"].get(cell_id, {})
        if run.get("status") == "ok":
            vals.append(run["decode_tok_per_s"])
    return np.mean(vals) if vals else None

Ts = [4096, 8192, 16384, 32768]
serial_baseline = [cell_mean(PHASE0, "nvfp4_isoquant", t) for t in Ts]  # pre-tiling
post_p3 = [cell_mean(ART, "nvfp4_isoquant", t) for t in Ts]
speedup = [p / s if (p and s) else None for p, s in zip(post_p3, serial_baseline)]

theory = [t / (B + np.log2(t / B)) for t in Ts]
theory_norm = [x / theory[0] * speedup[0] for x in theory]  # rebase to first measured

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(Ts, speedup, "o-", label="measured speedup vs serial baseline")
ax.plot(Ts, theory_norm, "--", label="Brent bound prediction")
ax.set_xlabel("Context T (tokens)")
ax.set_ylabel("Speedup factor")
ax.set_xscale("log", base=2)
ax.set_title("IsoQuant decode scaling: measured vs Brent-bound prediction")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
out = ART / "scaling_chart.png"
fig.savefig(out, dpi=120)
print(f"Wrote {out}")
print(f"Measured speedups: {dict(zip(Ts, speedup))}")
print(f"Saturation expected around T/B ≈ 160 → T ≈ {160 * B}")
```

- [x] **Step 2: Run the chart**

```bash
python3 scripts/run_scaling_validation.py
```

### Task 4.3: PPL regression test at 32K

**Files:**
- Create: `scripts/ppl_32k_isoquant_vs_default.py`
- Create: `artifacts/phase4_scaling/ppl_32k.json`

- [x] **Step 1: Confirm held-out text source exists**

Use **wikitext-103-raw test** (the canonical PPL benchmark, ~245K words). Critical: file
name must match contents — naming the file `wikitext-103-test.txt` and downloading
wikitext-2 silently invalidates any paper claim derived from it.

```bash
# wikitext-103-raw test split (~750KB compressed, plain text after extract).
mkdir -p data
test -f data/wikitext-103-raw-test.txt || (
  curl -L https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/test-00000-of-00001.parquet \
    -o /tmp/wt103.parquet && \
  python3 -c "
import pyarrow.parquet as pq
rows = pq.read_table('/tmp/wt103.parquet').to_pylist()
open('data/wikitext-103-raw-test.txt', 'w').write('\n'.join(r['text'] for r in rows))
" && rm /tmp/wt103.parquet
)
wc -w data/wikitext-103-raw-test.txt
```

Expected: file exists with **> 240K words** (wikitext-103-raw test is ~245K). If the count
is < 100K you've fetched wikitext-2 by accident; delete the file and re-run with the
exact URL above.

- [x] **Step 2: Write the PPL comparator**

```python
"""Phase 4: long-context PPL regression at 32K. Flag divergence > 5%."""
import json, os, sys
os.environ["ISOQUANT_BITS"] = "3"
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODEL = "/Users/anthonylui/Models/Qwen3.6-35B-A3B-nvfp4"
CTX = 32768

def ppl(model, tok, kv_cache_type, text):
    cache = make_prompt_cache(model, kv_cache_type=kv_cache_type)
    ids = tok.encode(text)[:CTX]
    arr = mx.array(ids)
    logits = model(arr[None, :], cache=cache)
    mx.eval(logits)
    # Standard log-prob-of-next-token PPL on the prompt
    log_probs = mx.log_softmax(logits[0, :-1, :], axis=-1)
    targets = arr[1:]
    nll = -mx.take_along_axis(log_probs, targets[:, None], axis=-1).mean()
    return float(mx.exp(nll))

text = open("data/wikitext-103-raw-test.txt").read()
model, tok = load(MODEL)
ppl_default = ppl(model, tok, "default", text)
ppl_iso = ppl(model, tok, "isoquant", text)
divergence_pct = abs(ppl_iso - ppl_default) / ppl_default * 100.0
result = {"ctx": CTX, "ppl_default": ppl_default, "ppl_isoquant": ppl_iso,
          "divergence_pct": divergence_pct, "pass": divergence_pct < 5.0}
os.makedirs("artifacts/phase4_scaling", exist_ok=True)
json.dump(result, open("artifacts/phase4_scaling/ppl_32k.json", "w"), indent=2)
print(json.dumps(result, indent=2))
sys.exit(0 if result["pass"] else 1)
```

- [x] **Step 3: Run + persist**

```bash
python3 scripts/ppl_32k_isoquant_vs_default.py
```

Expected exit 0 with `divergence_pct < 5.0`. If FAIL, the IsoQuant 3-bit codebook has
quality degradation at long context — investigate before claiming Branch A in Phase 5.

### Task 4.4: Commit Phase 4 artifacts + write the validation memo

**Files:**
- Create: `artifacts/phase4_scaling/validation_memo.md`

- [x] **Step 1: Write a one-paragraph memo**

```markdown
# Phase 4 validation memo

Measured speedup curve at T={4K, 8K, 16K, 32K}: see scaling_chart.png.

Brent bound prediction: speedup grows then saturates at T/B ≈ 160 → T ≈ 5120.

Observed: <one of>
  - Confirmed: speedup grows 4K → 16K, flattens at 32K. THEORY VALIDATED.
  - Earlier saturation than predicted: tile size B is wrong. Sweep tile size.
  - Still growing at 32K: P_concurrent is higher than estimated. Revise model.
  - Collapsed at 32K: thermal/memory ceiling. Identify and document.

PPL at 32K: isoquant vs default <within|outside> 5% divergence threshold.
```

- [x] **Step 2: Commit**

```bash
git add scripts/run_scaling_validation.py scripts/ppl_32k_isoquant_vs_default.py \
        artifacts/phase4_scaling/
git commit -m "phase 4: scaling validation at 32K + Brent-bound chart + PPL regression"
```

---

## Phase 5 — Decision gate (no coding)

After Phase 4, pick exactly one of three pre-defined branches.

### Task 5.1: Write the decision memo

**Files:**
- Create: `docs/PHASE5_DECISION_2026-04-27.md`

- [x] **Step 1: Compute the post-Phase-3 measured ms/step at 8K from the matrix**

Use Phase 3 / Phase 4 artifacts: extract nvfp4_isoquant tok/s at 8K, compute ms/step,
compare to nvfp4_default 10.8 ms/step. Compute the ratio.

- [x] **Step 2: Pick the branch**

```markdown
# Phase 5 decision

Measured at 8K: nvfp4_isoquant = ___ ms/step (___ tok/s)
Reference: nvfp4_default = 10.8 ms/step (92.45 tok/s)
Ratio: ___×

BRANCH SELECTED: <A | B | C>

A — parity or better at 8K+: write the paper. Don't touch representation.
    Core contribution: depth-reduction framing, 7.2× kernel gain (V-accum) +
    incremental-pack savings + fused-NPT8 savings, scaling validation at 32K.
    Submit to MLSys or EuroMLSys.

B — small residual gap (1.3–1.7×) at 8K+: paper still strong; honest about gap.
    Discussion flags representation change as future work. Stop at:
    "compression granularity governs decode scaling; scalar QQ is asymptotically
    pinned; alternative granularities are future work." Do NOT invent PPL studies
    you haven't run.

C — large residual gap (>1.7×) at 8K+: investigate representation. Steps:
    (1) PPL study: 4-dim SO(4)-block VQ vs current scalar QQ, matched bit-budget,
        on Qwen3-30B-A3B (small, fast iteration) first.
    (2) Verify histogram-kernel arithmetic against actual bit budget. K=8 at 4-dim
        is 0.75 bits/dim; matching current 3-bits/dim needs K=4096, which breaks
        the histogram-size assumption.
    (3) Only if PPL holds, scope kernel redesign as separate project with own gate.
```

- [x] **Step 2: Commit**

```bash
git add docs/PHASE5_DECISION_2026-04-27.md
git commit -m "phase 5: decision memo - branch C selected"
```

---

## Paper-writing track (parallel from Phase 2)

Don't leave writing until Phase 5. Draft as you go.

- After Phase 2: methodology section (IsoQuant architecture, scalar QQ, SO(4)+WHT, DKV
  constraint, NPT=4 vs NPT=8 architectural constraint)
- After Phase 3: results section through 8K, depth-reduction framing with Brent's theorem,
  roofline analysis from Phase 1
- After Phase 4: scaling figure and the saturation prediction
- After Phase 5: discussion, future work, decision on inclusion of VQ/PQ material

**Keep out of the paper until you have data:**
- Anything about $O(T+d)$ being achievable at this $d_k$
- Any claim that 4-dim block VQ is a free win
- Any throughput forecast that hasn't been measured

---

## Timeline summary

| Phase | Calendar | Risk |
|---|---|---|
| 0. Clean baseline + persisted attribution | 0.5 day | low |
| 1. Bandwidth check | 0.5 day | low |
| 2. Incremental append | 1–2 days | low |
| 3. Fused NPT=8 + T-tiling | 3–5 days | medium |
| 4. 32K validation | 1 day | low |
| 5. Decision gate | 0.5 day | — |
| **Total** | **6–10 working days** | |

**Critical path: Phase 3.** Hard 7-day budget; if blown, ship Phase 2 alone and write paper
against partially-fused result.

## Success criteria

- Kernel C: 32.66 → ~1–3 ms/step at 4K, **10–30× total** vs original serial version
- Decode step at 8K: ~32 ms → ~13–18 ms, i.e. nvfp4+isoquant at 55–75 tok/s
- Gap to nvfp4+default: **1× to 1.5×** acknowledged as constant-factor residual of scalar
  quantisation
- Validated Brent-bound scaling model with empirically-identified saturation point
- Paper draft submittable to EuroMLSys with **measurement, not forecast**, as backbone

## Out-of-scope

- MoE / DeltaNet kernel optimisations (the ~39 ms model-compute floor)
- Kimi MLA + IsoQuant DKV validation (blocked on 128 GB hardware + Kimi checkpoint)
- NVFP4 throughput hardware investigation (M5 Max FP4 path question)
- 1T-class model benchmarks
- Anything in Phase 5 Branch C beyond the PPL study (representation redesign is its own
  project gated on Branch C being selected)

## Review-pattern note (for future multi-LLM loops)

This session's adversarial-review loop produced four reply types with different
signal/noise ratios. Rule of thumb: **any reply whose novel contribution rests on a
derivation gets that derivation independently re-checked before it enters a rider
document. The flattery reply is the most dangerous — it wraps errors in the most
superlatives. The more a reply tells you the idea is publishable, the more carefully check
the equations.**
