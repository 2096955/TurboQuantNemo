#!/usr/bin/env python3
"""Run MoE offload benchmark twice: TurboQuant vs IsoQuant (same seed/profile).

Writes a single JSON with both runs for pathway IsoQuant vs TurboQuant comparison.
Requires a local MLX model path (e.g. Qwen3-30B-A3B-4bit, Gemma 4, Nemotron).

Example:
  python scripts/compare_pathway_kv_modes.py \\
    --model /path/to/Qwen3-30B-A3B-4bit \\
    --output results/qwen3_kv_isoquant_vs_turboquant.json \\
    --expert-offload --turboquant-bits 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Import sibling benchmark module
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark_moe_offload import run_benchmark  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="TurboQuant vs IsoQuant paired benchmark")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--profile", type=str, choices=["A", "B"], default="A")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--expert-offload", action="store_true")
    p.add_argument("--max-resident-experts", type=int, default=None)
    p.add_argument("--expert-offload-dir", type=str, default=None)
    p.add_argument("--prefill-step-size", type=int, default=64)
    p.add_argument("--turboquant-bits", type=int, default=3)
    p.add_argument("--use-predictor", action="store_true")
    p.add_argument("--use-dedekimi-observer", action="store_true")
    p.add_argument("--task-expert-cliques-file", type=str, default=None)
    args = p.parse_args()

    os.environ["TURBOQUANT_BITS"] = str(args.turboquant_bits)

    task_expert_cliques = None
    if args.task_expert_cliques_file:
        task_expert_cliques = json.loads(
            Path(args.task_expert_cliques_file).read_text(encoding="utf-8")
        )

    prefill_tokens, decode_tokens = (512, 128) if args.profile == "A" else (1024, 256)

    out: dict = {
        "model": args.model,
        "turboquant_bits": args.turboquant_bits,
        "profile": args.profile,
        "runs": {},
    }

    for name, kv in (("turboquant", "turboquant"), ("isoquant", "isoquant")):
        metrics, gate_failed = run_benchmark(
            args.model,
            profile=args.profile,
            prefill_tokens=prefill_tokens,
            prefill_step_size=args.prefill_step_size,
            decode_tokens=decode_tokens,
            seed=args.seed,
            kv_cache_type=kv,
            expert_offload=args.expert_offload,
            max_resident_experts=args.max_resident_experts,
            expert_offload_dir=args.expert_offload_dir,
            memory_mode=None,
            strict_gates=False,
            split_decode_timing=False,
            warm_second_pass=False,
            use_predictor=args.use_predictor,
            use_dedekimi_observer=args.use_dedekimi_observer,
            task_expert_cliques=task_expert_cliques,
        )
        out["runs"][name] = metrics
        if gate_failed:
            out["runs"][name]["_soft_gate_failed"] = True

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
