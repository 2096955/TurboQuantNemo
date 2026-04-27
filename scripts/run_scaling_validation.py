#!/usr/bin/env python3
"""Phase 4: tiled-Kernel-C speedup vs T, overlay theoretical Brent bound.

Theoretical: S(T) = T / (B + log2(T / B)), saturates at T/B ~ P_concurrent ~ 160.

Reads Phase 4 matrix results and Phase 0 baseline, computes speedup of
nvfp4+isoquant (post-fused-NPT8) over Phase 0 nvfp4+isoquant (pre-fusion),
and plots against the Brent-bound prediction.
"""

import json
import sys
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PHASE4_DIR = Path("artifacts/phase4_scaling")
PHASE0_DIR = Path("artifacts/phase0_baseline_2026-04-24")
B = 32  # tile size


def cell_mean_tps(art_dir: Path, cell_id: str, T: int) -> float | None:
    vals = cell_tps_values(art_dir, cell_id, T)
    return mean(vals) if vals else None


def cell_tps_values(art_dir: Path, cell_id: str, T: int) -> list[float]:
    vals = []
    for f in sorted(art_dir.glob(f"matrix_T{T}_*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        runs = d.get("runs", {})
        run = runs.get(cell_id, {})
        if run.get("status") == "ok":
            tps = run.get("decode_tok_per_s") or run.get("decode_tps")
            if tps:
                vals.append(float(tps))
    return vals


def cell_tps_stats(art_dir: Path, cell_id: str, T: int) -> dict:
    vals = cell_tps_values(art_dir, cell_id, T)
    if not vals:
        return {"n": 0, "values": [], "mean": None, "stdev": None, "cv_pct": None}
    avg = mean(vals)
    sd = stdev(vals) if len(vals) > 1 else 0.0
    return {
        "n": len(vals),
        "values": vals,
        "mean": avg,
        "stdev": sd,
        "cv_pct": (sd / avg * 100.0) if avg else None,
    }


def cell_mean_ms(art_dir: Path, cell_id: str, T: int) -> float | None:
    tps = cell_mean_tps(art_dir, cell_id, T)
    if tps and tps > 0:
        return 1000.0 / tps
    return None


def main():
    Ts = [4096, 8192, 16384, 32768]
    cell = "nvfp4_isoquant"
    cell_default = "nvfp4_default"

    phase0_tps = [cell_mean_tps(PHASE0_DIR, cell, t) for t in Ts]
    phase4_tps = [cell_mean_tps(PHASE4_DIR, cell, t) for t in Ts]
    default_tps = [cell_mean_tps(PHASE4_DIR, cell_default, t) for t in Ts]

    phase0_ms = [cell_mean_ms(PHASE0_DIR, cell, t) for t in Ts]
    phase4_ms = [cell_mean_ms(PHASE4_DIR, cell, t) for t in Ts]
    default_ms = [cell_mean_ms(PHASE4_DIR, cell_default, t) for t in Ts]

    print(
        f"{'T':>6}  {'P0 tok/s':>10}  {'P4 tok/s':>10}  {'speedup':>8}  {'P4 ms':>8}  {'def ms':>8}  {'ratio':>6}"
    )
    print("-" * 75)

    speedups = []
    ratios = []
    stats = {}
    for i, T in enumerate(Ts):
        p0 = phase0_tps[i]
        p4 = phase4_tps[i]
        dm = default_ms[i]
        p4m = phase4_ms[i]
        s = p4 / p0 if (p0 and p4) else None
        r = p4m / dm if (p4m and dm) else None
        speedups.append(s)
        ratios.append(r)
        print(
            f"{T:>6}  {p0 or 0:>10.2f}  {p4 or 0:>10.2f}  {s or 0:>7.2f}x  {p4m or 0:>7.1f}  {dm or 0:>7.1f}  {r or 0:>5.2f}x"
        )
        stats[str(T)] = {
            "phase0_nvfp4_isoquant": cell_tps_stats(PHASE0_DIR, cell, T),
            "phase4_nvfp4_default": cell_tps_stats(PHASE4_DIR, cell_default, T),
            "phase4_nvfp4_isoquant": cell_tps_stats(PHASE4_DIR, cell, T),
        }

    valid = [(T, s) for T, s in zip(Ts, speedups) if s is not None]
    if not valid:
        print("\nNo valid data to plot. Check Phase 0 and Phase 4 artifacts.")
        sys.exit(1)

    valid_Ts = [x[0] for x in valid]
    valid_speedups = [x[1] for x in valid]

    theory = [t / (B + np.log2(t / B)) for t in valid_Ts]
    if valid_speedups[0] and valid_speedups[0] > 0:
        theory_norm = [x / theory[0] * valid_speedups[0] for x in theory]
    else:
        theory_norm = theory

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(
        valid_Ts, valid_speedups, "o-", label="Measured speedup (P4/P0)", linewidth=2
    )
    ax1.plot(
        valid_Ts, theory_norm, "s--", label="Brent bound (normalized)", linewidth=1.5
    )
    ax1.set_xlabel("Context T (tokens)")
    ax1.set_ylabel("Speedup factor vs Phase 0 baseline")
    ax1.set_xscale("log", base=2)
    ax1.set_title("IsoQuant decode scaling: measured vs Brent bound")
    ax1.legend()
    ax1.grid(alpha=0.3)

    valid_ratios = [(T, r) for T, r in zip(Ts, ratios) if r is not None]
    if valid_ratios:
        rTs = [x[0] for x in valid_ratios]
        rVals = [x[1] for x in valid_ratios]
        ax2.plot(
            rTs,
            rVals,
            "D-",
            color="red",
            label="nvfp4+iso / nvfp4+default",
            linewidth=2,
        )
        ax2.axhline(y=1.0, color="green", linestyle=":", label="Parity", alpha=0.7)
        ax2.axhline(
            y=1.7, color="orange", linestyle=":", label="1.7x threshold", alpha=0.7
        )
        ax2.set_xlabel("Context T (tokens)")
        ax2.set_ylabel("ms/step ratio (iso / default)")
        ax2.set_xscale("log", base=2)
        ax2.set_title("Gap to default KV (lower = better)")
        ax2.legend()
        ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = PHASE4_DIR / "scaling_chart.png"
    fig.savefig(out, dpi=150)
    print(f"\nWrote {out}")

    saturation_T = 160 * B
    print(f"Brent saturation expected at T/B ~ 160 -> T ~ {saturation_T}")
    print(f"Measured speedups: {dict(zip(Ts, speedups))}")
    print(f"Gap ratios (iso/default ms): {dict(zip(Ts, ratios))}")

    summary = {
        "Ts": Ts,
        "phase0_tps": phase0_tps,
        "phase4_tps": phase4_tps,
        "default_tps": default_tps,
        "speedups": speedups,
        "gap_ratios": ratios,
        "stats": stats,
        "brent_B": B,
        "saturation_T": saturation_T,
    }
    json.dump(
        summary, open(PHASE4_DIR / "scaling_summary.json", "w"), indent=2, default=str
    )
    print(f"Wrote {PHASE4_DIR / 'scaling_summary.json'}")


if __name__ == "__main__":
    main()
