"""Phase 0 variance gate: within-cell variance must be < 5% across repeats at gate T.

Default: T_gate = 8192 (plan Task 0.4). Use --per-t to print full per-T table first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from phase0_matrix_cv_summary import print_per_t_table

DEFAULT_ART_DIR = Path("artifacts/phase0_baseline_2026-04-24")
MAX_VARIANCE_PCT = 5.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 0 repeatability: CV% of decode_tok_per_s at gate T"
    )
    p.add_argument(
        "--art-dir",
        type=Path,
        default=DEFAULT_ART_DIR,
        help="Matrix JSON directory (default: %(default)s)",
    )
    p.add_argument(
        "--gate-t",
        type=int,
        default=8192,
        help="Prefill T for strict gate (default: %(default)s, plan 0.4)",
    )
    p.add_argument(
        "--per-t",
        action="store_true",
        help="Print per-T CV table for all standard T, then run gate at --gate-t",
    )
    p.add_argument(
        "--max-cv",
        type=float,
        default=MAX_VARIANCE_PCT,
        help="Max allowed CV%% for PASS (default: %(default)s)",
    )
    return p.parse_args()


def run_gate(art_dir: Path, t_gate: int, max_cv: float) -> int:
    results: dict[str, list[float]] = {}
    d = 512 if t_gate < 8192 else 1024
    for f in sorted(art_dir.glob(f"matrix_T{t_gate}_d{d}_r*.json")):
        dct = json.loads(f.read_text(encoding="utf-8"))
        for cell_id, run in dct.get("runs", {}).items():
            if run.get("status") != "ok":
                continue
            results.setdefault(cell_id, []).append(float(run["decode_tok_per_s"]))

    failed: list[tuple[str, float]] = []
    print(
        f"Gate: T={t_gate}  art_dir={art_dir.resolve()}\n"
        f"{'cell':<24}  {'mean tok/s':>12}  {'stdev':>8}  "
        f"{'cv%':>6}  gate (<{max_cv}%)"
    )
    for cell, vals in sorted(results.items()):
        if len(vals) < 2:
            only = vals[0] if vals else 0.0
            print(f"{cell:<24}  {only:>12.2f}  {'n/a':>8}  {'n/a':>6}  N<2 SKIP")
            continue
        m = mean(vals)
        s = stdev(vals)
        cv = (s / m) * 100.0 if m > 0 else 0.0
        ok = cv < max_cv
        if not ok:
            failed.append((cell, cv))
        print(
            f"{cell:<24}  {m:>12.2f}  {s:>8.3f}  {cv:>5.1f}  {'OK' if ok else 'FAIL'}"
        )

    if failed:
        print("\nGATE FAILED:", failed)
        return 1
    print("\nGATE PASS")
    return 0


def main() -> int:
    args = parse_args()
    art_dir: Path = args.art_dir
    if not art_dir.is_dir():
        print(f"ERROR: {art_dir} is not a directory", file=sys.stderr)
        return 2

    if args.per_t:
        print("=== per-T CV (all standard T) ===\n")
        print_per_t_table(art_dir)
        print("=== 8K-style gate (configurable) ===\n")

    return run_gate(art_dir, args.gate_t, args.max_cv)


if __name__ == "__main__":
    raise SystemExit(main())
