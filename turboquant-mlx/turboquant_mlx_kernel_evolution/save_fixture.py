#!/usr/bin/env python3
"""Save a deterministic closed-loop fixture for evolution / CI (optional)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from turboquant_benchmark_suite import build_closed_loop_fixture, save_fixture_npz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output", type=Path, default=Path("fixtures/turboquant_score_closed_loop.npz")
    )
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--seq-kv", type=int, default=512)
    ap.add_argument("--num-queries", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    fx = build_closed_loop_fixture(
        head_dim=args.head_dim,
        seq_kv=args.seq_kv,
        num_queries=args.num_queries,
        bit_width=args.bits,
        codebook_dir=str(_ROOT / "codebooks"),
        seed=args.seed,
    )
    save_fixture_npz(fx, args.output)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
