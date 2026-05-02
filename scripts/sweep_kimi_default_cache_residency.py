#!/usr/bin/env python3
"""Sweep `max_resident_experts` on Kimi K2.x with default MLA cache only (arm L0).

Produces the roadmap artifact:
  artifacts/kimi_k26_residency/default_cache_sweep.json

Reuses the measurement protocol from `ab_kimi_layered_stack.py` (L0: default cache,
no observer/predictor/cliques, NPT16 off).

Example:
  PYTHONPATH=mlx-lm python scripts/sweep_kimi_default_cache_residency.py \\
    --model /Volumes/Samsung9904tb/Kimi-K2.6 \\
    --sweep-values 64,128,200,400,800 \\
    --output artifacts/kimi_k26_residency/default_cache_sweep.json

Assumptions:
  Importing ``ab_kimi_layered_stack`` mutates ``sys.path`` (adds mlx-lm); keep that stable.
  ``best_max_resident_experts`` / ``best_tok_per_s`` maximize tok/s among successful runs only;
  they ignore RAM envelope and expert hit-rate tradeoffs — inspect ``runs`` for decisions.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_kimi_ab():
    path = Path(__file__).resolve().parent / "ab_kimi_layered_stack.py"
    spec = importlib.util.spec_from_file_location("kimi_layered_ab", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    kab = _load_kimi_ab()
    ARM_L0 = kab.ARM_DEFS["L0"]

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/Volumes/Samsung9904tb/Kimi-K2.6")
    p.add_argument(
        "--sweep-values",
        default="64,128,200,400,800,1200",
        help="Comma-separated max_resident_experts values (ascending recommended).",
    )
    p.add_argument("--prompt-tokens", type=int, default=32)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--decode-steps", type=int, default=30)
    p.add_argument(
        "--output",
        default=str(
            REPO_ROOT / "artifacts/kimi_k26_residency/default_cache_sweep.json"
        ),
    )
    args = p.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    vals: list[int] = []
    for part in args.sweep_values.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(int(part, 10))
        except ValueError:
            print(f"ERROR: not an integer: {part!r}", flush=True)
            sys.exit(2)

    if not vals:
        print("ERROR: empty --sweep-values", flush=True)
        sys.exit(2)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prev_npt16 = os.environ.get("ISOQUANT_USE_NPT16_FUSED")
    os.environ["ISOQUANT_USE_NPT16_FUSED"] = "0"

    runs: dict[str, dict] = {}
    model = tokenizer = None

    print(f"\nDefault-cache residency sweep (L0): values={vals}", flush=True)

    for mr in vals:
        tag = str(mr)
        print(f"\n========== max_resident_experts={mr} ==========", flush=True)
        cfg = kab.build_model_config(ARM_L0, mr, None)
        if model is not None:
            kab.teardown(model, tokenizer)
            model, tokenizer = None, None
        try:
            model, tokenizer = kab.load_model(args.model, cfg)
            r = kab.run_arm(
                model,
                f"L0_r{mr}",
                ARM_L0,
                prompt_tokens=args.prompt_tokens,
                warmup=args.warmup,
                decode_steps=args.decode_steps,
            )
            kab.print_summary(f"L0_r{mr}", r)
            r["max_resident_experts"] = mr
            runs[tag] = r
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"  FAILED: {err}", flush=True)
            runs[tag] = {
                "error": err,
                "max_resident_experts": mr,
            }
        finally:
            if model is not None:
                kab.teardown(model, tokenizer)
                model, tokenizer = None, None

    if prev_npt16 is None:
        os.environ.pop("ISOQUANT_USE_NPT16_FUSED", None)
    else:
        os.environ["ISOQUANT_USE_NPT16_FUSED"] = prev_npt16

    scored = [(k, v) for k, v in runs.items() if "error" not in v]
    best_key = None
    best_tok = -1.0
    for k, v in scored:
        tps = float(v.get("tok_per_s", 0.0))
        if tps > best_tok:
            best_tok = tps
            best_key = k

    payload = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "arm": "L0",
        "model": args.model,
        "sweep_values": vals,
        "prompt_tokens": args.prompt_tokens,
        "warmup": args.warmup,
        "decode_steps": args.decode_steps,
        "runs": runs,
        "best_max_resident_experts": int(best_key) if best_key is not None else None,
        "best_tok_per_s": best_tok if best_key is not None else None,
        "notes": (
            "Higher max_resident_experts increases hit rate but also RAM pressure; "
            "OOM entries are recorded as error runs. Prefer clean Metal state before sweeping."
        ),
    }
    try:
        out_path.write_text(json.dumps(payload, indent=2))
    except TypeError as e:
        print(
            f"ERROR: JSON serialization failed (non-JSON-safe value in run results): {e}",
            flush=True,
        )
        sys.exit(1)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
