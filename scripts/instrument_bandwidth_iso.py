"""Phase 1: bandwidth-bound vs execution-bound characterization of fused_value_accum_tiled.

Method: for each T, time the tiled V kernel in isolation and compute achieved GB/s.
Compare to M4 Max realistic peak (~300 GB/s for this access pattern).

Decision rule for Phase 3 priority:
  ≥ 60% of peak: traffic-reduction gain (upper-end). Proceed Phase 3 as planned.
  20–60%:        mixed (dispatch + traffic). Proceed; expect mid-range gain.
  < 20%:         instruction-level waste. Investigate centroid loads first.

Byte accounting (worst case, no cache amortisation across query heads — see plan
Phase 1 byte model fix):
  packed V re-read per q_head per tile: H_Q * num_tiles * T * (D*3/8) bytes
  norms re-read per q_head per tile:    H_Q * num_tiles * T * 4 bytes
  attn weights read per q_head per tile: H_Q * num_tiles * T * 4 bytes
  per-tile partials write:               num_tiles * H_Q * D * 4 bytes
  reduction read:                        num_tiles * H_Q * D * 4 bytes
  reduction output write:                H_Q * D * 4 bytes
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("ISOQUANT_BITS", "3")

REPO_ROOT = Path(__file__).resolve().parents[1]
MLX_LM_SRC = REPO_ROOT / "mlx-lm"
if str(MLX_LM_SRC) not in sys.path:
    sys.path.insert(0, str(MLX_LM_SRC))

import mlx.core as mx
import numpy as np

from mlx_lm.models.fused_kv_decode_kernels import pack_indices_3bit
from mlx_lm.models.fused_kv_decode_tiled import fused_value_accum_tiled

H_KV, H_Q, D = 2, 16, 256
TILE = 128
N_CALLS = 50
PEAK_GBS = 300.0  # M4 Max realistic peak for this access pattern


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


def time_kernel(T, n_calls=N_CALLS):
    V_packed, centroids, norms, attn, kv_head_map = synthetic(T)
    # warmup
    for _ in range(3):
        out = fused_value_accum_tiled(
            V_packed, centroids, norms, attn, kv_head_map, H_Q, T, D, tile_size=TILE
        )
        mx.eval(out)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_calls):
        out = fused_value_accum_tiled(
            V_packed, centroids, norms, attn, kv_head_map, H_Q, T, D, tile_size=TILE
        )
        mx.eval(out)
    mx.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / n_calls


def bytes_touched(T):
    """Two bounds on per-call DRAM traffic.

    BEST CASE (full cache reuse across q_heads sharing a kv_head): unique-byte
    lower bound. Achievable on M4 Max if L2 keeps the per-tile packed-V slice
    resident across the H_Q/H_KV q_heads sharing its kv_head.

    WORST CASE (no cache, every (tile, q_head) pulls its own copy): upper bound;
    reaching peak BW under this assumption is impossible.

    The truth is between. Reporting both lets the rubric distinguish "near peak
    via cache reuse" from "scattered access defeats cache".
    """
    num_tiles = (T + TILE - 1) // TILE
    # Best case (unique bytes, max cache reuse):
    packed_v_unique = H_KV * T * (D * 3 // 8)
    norms_unique = H_KV * T * 4
    attn_w_unique = H_Q * T * 4  # genuinely per-q_head, no sharing
    partials_write = num_tiles * H_Q * D * 4
    reduction_read = num_tiles * H_Q * D * 4
    reduction_write = H_Q * D * 4
    best = (
        packed_v_unique
        + norms_unique
        + attn_w_unique
        + partials_write
        + reduction_read
        + reduction_write
    )
    # Worst case (every (tile, q_head) re-pulls from DRAM):
    packed_v_worst = H_Q * num_tiles * T * (D * 3 // 8)
    norms_worst = H_Q * num_tiles * T * 4
    attn_w_worst = H_Q * num_tiles * T * 4
    worst = (
        packed_v_worst
        + norms_worst
        + attn_w_worst
        + partials_write
        + reduction_read
        + reduction_write
    )
    return best, worst


def _legacy_unused(T):
    """(kept for reference; not used)"""
    num_tiles = (T + TILE - 1) // TILE
    packed_v_per_qhead = T * (D * 3 // 8)
    norms_per_qhead = T * 4
    attn_w_per_qhead = T * 4
    per_qhead_reads = (
        packed_v_per_qhead + norms_per_qhead + attn_w_per_qhead
    ) * num_tiles
    total_kernel_reads = per_qhead_reads * H_Q
    partials_write = num_tiles * H_Q * D * 4
    reduction_read = num_tiles * H_Q * D * 4
    reduction_write = H_Q * D * 4
    return total_kernel_reads + partials_write + reduction_read + reduction_write


results = []
print(
    f"{'T':>6}  {'ms':>7}  {'best_MB':>8}  {'best_GB/s':>10}  "
    f"{'best_%pk':>9}  {'worst_MB':>9}  {'worst_GB/s':>11}"
)
for T in [1024, 2048, 4096, 8192, 16384]:
    ms = time_kernel(T)
    best_b, worst_b = bytes_touched(T)
    best_gbs = best_b / 1e9 / (ms / 1000.0)
    worst_gbs = worst_b / 1e9 / (ms / 1000.0)
    best_pct = best_gbs / PEAK_GBS * 100.0
    worst_pct = worst_gbs / PEAK_GBS * 100.0
    results.append(
        {
            "T": T,
            "ms_per_call": ms,
            "bytes_best": best_b,
            "bytes_worst": worst_b,
            "achieved_gbs_best": best_gbs,
            "achieved_gbs_worst": worst_gbs,
            "pct_peak_best": best_pct,
            "pct_peak_worst": worst_pct,
        }
    )
    print(
        f"{T:>6}  {ms:>7.3f}  {best_b / 1e6:>8.2f}  {best_gbs:>10.2f}  "
        f"{best_pct:>8.1f}%  {worst_b / 1e6:>9.2f}  {worst_gbs:>11.2f}"
    )

# Use the BEST-CASE achieved pct_peak as the realistic ceiling — if even the
# best-case bound (max possible cache reuse) is well below peak, the kernel
# isn't bandwidth-bound; it's execution/dispatch-bound.
max_best_pct = max(r["pct_peak_best"] for r in results)
if max_best_pct >= 60:
    decision = (
        f"Traffic-reduction path (best-case {max_best_pct:.1f}% ≥ 60% peak): "
        f"Phase 3 fusion gives upper-end gain."
    )
elif max_best_pct >= 20:
    decision = (
        f"Mixed (best-case {max_best_pct:.1f}% in 20-60%): Phase 3 gives "
        f"mid-range gain. Proceed."
    )
else:
    decision = (
        f"Execution/dispatch-bound (best-case {max_best_pct:.1f}% < 20% peak): "
        f"the kernel is not memory-throughput-limited even under perfect cache "
        f"reuse. Phase 3 fusion's upside from traffic reduction is small. "
        f"Investigate centroid loads / dispatch overhead / software pipelining "
        f"BEFORE committing to the NPT=8 fusion."
    )

print(f"\nMax best-case achieved: {max_best_pct:.1f}% of {PEAK_GBS} GB/s peak")
print(f"DECISION: {decision}")

out_path = "artifacts/phase1_bandwidth/tiled_v_accum_bw.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
json.dump(
    {
        "peak_gbs": PEAK_GBS,
        "results": results,
        "max_pct_peak_best": max_best_pct,
        "decision": decision,
    },
    open(out_path, "w"),
    indent=2,
)
print(f"Wrote {out_path}")
