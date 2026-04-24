"""Phase 0: per-T coefficient of variation for each matrix cell (3 repeats).

Reads matrix_T{T}_d{D}_r{r}.json from the Phase 0 artifact directory; D is
512 for T < 8192, else 1024.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev

DEFAULT_ART_DIR = Path("artifacts/phase0_baseline_2026-04-24")
CELLS = [
    "baseline_default",
    "baseline_isoquant",
    "nvfp4_default",
    "nvfp4_isoquant",
]


def d_for_t(t: int) -> int:
    return 512 if t < 8192 else 1024


def print_per_t_table(art_dir: Path, t_values: list[int] | None = None) -> int:
    """Print per-T CV table. Returns 0; prints errors to stderr for missing files."""
    if t_values is None:
        t_values = [4096, 8192, 16384, 32768]
    print(
        f"Source: {art_dir.resolve()}\n"
        f"Per-T per-cell: mean decode_tok_per_s, stdev, CV% across r=1..3\n"
    )
    print(
        f"{'T':>6}  {'cell':<24}  "
        f"{'r1':>8}  {'r2':>8}  {'r3':>8}  {'mean':>8}  {'CV%':>6}"
    )
    for t in t_values:
        d = d_for_t(t)
        for cell in CELLS:
            vals: list[float] = []
            for r in (1, 2, 3):
                f = art_dir / f"matrix_T{t}_d{d}_r{r}.json"
                if not f.is_file():
                    print(
                        f"{t:>6}  {cell:<24}  MISSING  {f.name}",
                        file=sys.stderr,
                    )
                    continue
                dct = json.loads(f.read_text(encoding="utf-8"))
                run = dct.get("runs", {}).get(cell, {})
                if run.get("status") != "ok" or "decode_tok_per_s" not in run:
                    print(
                        f"{t:>6}  {cell:<24}  STATUS_NOT_OK  {f.name}",
                        file=sys.stderr,
                    )
                    continue
                vals.append(float(run["decode_tok_per_s"]))
            if len(vals) < 2:
                print(f"{t:>6}  {cell:<24}  (insufficient ok runs)")
                continue
            m = mean(vals)
            s = stdev(vals) if len(vals) > 1 else 0.0
            cv = (s / m) * 100.0 if m > 0 else 0.0
            v1, v2, v3 = (vals + [float("nan")] * 3)[:3]
            print(
                f"{t:>6}  {cell:<24}  {v1:>8.2f}  {v2:>8.2f}  "
                f"{v3:>8.2f}  {m:>8.2f}  {cv:>5.1f}"
            )
        print()
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-T CV% for each cell across 3 repeats of the Phase 0 matrix"
    )
    p.add_argument(
        "--art-dir",
        type=Path,
        default=DEFAULT_ART_DIR,
        help="Directory containing matrix_*.json (default: %(default)s)",
    )
    p.add_argument(
        "--t-values",
        type=int,
        nargs="*",
        default=[4096, 8192, 16384, 32768],
        help="Prefill T values to summarize",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    art_dir: Path = args.art_dir
    if not art_dir.is_dir():
        print(f"ERROR: artifact directory not found: {art_dir}", file=sys.stderr)
        return 2
    return print_per_t_table(art_dir, list(args.t_values))


if __name__ == "__main__":
    raise SystemExit(main())
