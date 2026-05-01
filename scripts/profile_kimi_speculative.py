#!/usr/bin/env python3
"""Self-speculative decode smoke test on Kimi K2.6.

Loads the 4-bit Kimi as target and the 2-bit-experts Kimi as draft.
Runs speculative_generate_step from mlx_lm.generate. Compares output
to a baseline greedy run with the target alone. Records throughput
and acceptance rate.

Memory note: both models in RAM simultaneously is the major risk.
Each uses expert_offload, so resident expert memory is bounded by
max_resident_experts. Set draft's max_resident lower than target's
to avoid evicting target's experts.

Usage:
    python3 scripts/profile_kimi_speculative.py \\
        --target /Volumes/Samsung9904tb/Kimi-K2.6 \\
        --draft /Volumes/Samsung9904tb/Kimi-K2.6-2bit-experts \\
        --num-draft-tokens 4 \\
        --max-tokens 64 \\
        --output artifacts/kimi_k26_speculative/speculative_smoke.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MLX_LM_ROOT = REPO_ROOT / "mlx-lm"
if MLX_LM_ROOT.exists() and str(MLX_LM_ROOT) not in sys.path:
    sys.path.insert(0, str(MLX_LM_ROOT))


SMOKE_PROMPTS = [
    "The capital of France is",
    "Explain the concept of attention in transformers in one sentence.",
]


def run_baseline(target_path: str, max_tokens: int, max_resident: int):
    """Greedy generation with the target model alone."""
    from mlx_lm import load, generate

    model, tokenizer = load(
        target_path,
        model_config={
            "expert_offload": True,
            "max_resident_experts": max_resident,
        },
    )
    results = []
    for prompt in SMOKE_PROMPTS:
        t0 = time.time()
        out = generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False
        )
        elapsed = time.time() - t0
        tok_count = len(tokenizer.encode(out)) - len(tokenizer.encode(prompt))
        results.append(
            {
                "prompt": prompt,
                "output": out,
                "elapsed_s": round(elapsed, 2),
                "approx_tok_count": tok_count,
                "tok_per_s": round(max(tok_count, 1) / elapsed, 3),
            }
        )
        print(
            f"  baseline [{prompt[:40]}] {elapsed:.1f}s, ~{tok_count} tok, "
            f"{results[-1]['tok_per_s']} tok/s",
            flush=True,
        )
    del model, tokenizer
    gc.collect()
    return results


def run_speculative(
    target_path: str,
    draft_path: str,
    num_draft: int,
    max_tokens: int,
    max_resident_target: int,
    max_resident_draft: int,
):
    """Speculative decode with target + draft."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import speculative_generate_step

    print(f"  Loading target: {target_path}", flush=True)
    target, tokenizer = load(
        target_path,
        model_config={
            "expert_offload": True,
            "max_resident_experts": max_resident_target,
        },
    )
    print(f"  Loading draft: {draft_path}", flush=True)
    draft, _ = load(
        draft_path,
        model_config={
            "expert_offload": True,
            "max_resident_experts": max_resident_draft,
        },
    )

    results = []
    for prompt in SMOKE_PROMPTS:
        prompt_ids = mx.array(tokenizer.encode(prompt))
        accepted_count = 0
        n_calls = 0
        out_tokens = []
        t0 = time.time()
        for token, _logprobs, accepted_flag in speculative_generate_step(
            prompt_ids,
            target,
            draft,
            num_draft_tokens=num_draft,
            max_tokens=max_tokens,
        ):
            out_tokens.append(int(token))
            n_calls += 1
            if accepted_flag:
                accepted_count += 1
            if len(out_tokens) >= max_tokens:
                break
        elapsed = time.time() - t0
        out_text = tokenizer.decode(out_tokens)
        tok_count = len(out_tokens)
        results.append(
            {
                "prompt": prompt,
                "output": out_text,
                "elapsed_s": round(elapsed, 2),
                "tok_count": tok_count,
                "tok_per_s": round(tok_count / elapsed, 3),
                "n_accepted": accepted_count,
                "n_total": n_calls,
                "acceptance_rate": round(accepted_count / max(n_calls, 1), 3),
            }
        )
        print(
            f"  spec [{prompt[:40]}] {elapsed:.1f}s, {tok_count} tok, "
            f"{results[-1]['tok_per_s']} tok/s, accept={results[-1]['acceptance_rate']}",
            flush=True,
        )

    del target, draft, tokenizer
    gc.collect()
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--num-draft-tokens", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--max-resident-target", type=int, default=1500)
    p.add_argument("--max-resident-draft", type=int, default=500)
    p.add_argument(
        "--output",
        default="artifacts/kimi_k26_speculative/speculative_smoke.json",
    )
    p.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline run (assume baseline numbers from prior runs)",
    )
    args = p.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target": args.target,
        "draft": args.draft,
        "num_draft_tokens": args.num_draft_tokens,
        "max_tokens": args.max_tokens,
        "max_resident_target": args.max_resident_target,
        "max_resident_draft": args.max_resident_draft,
    }

    if not args.skip_baseline:
        print("\n=== Baseline (target only) ===")
        payload["baseline"] = run_baseline(
            args.target, args.max_tokens, args.max_resident_target
        )

    print("\n=== Speculative (target + draft) ===")
    payload["speculative"] = run_speculative(
        args.target,
        args.draft,
        args.num_draft_tokens,
        args.max_tokens,
        args.max_resident_target,
        args.max_resident_draft,
    )

    if "baseline" in payload:
        bl_med = float(np.median([r["tok_per_s"] for r in payload["baseline"]]))
        sp_med = float(np.median([r["tok_per_s"] for r in payload["speculative"]]))
        accept_med = float(
            np.median([r["acceptance_rate"] for r in payload["speculative"]])
        )
        payload["summary"] = {
            "baseline_tok_per_s_median": round(bl_med, 3),
            "speculative_tok_per_s_median": round(sp_med, 3),
            "speedup": round(sp_med / max(bl_med, 1e-6), 2),
            "acceptance_rate_median": round(accept_med, 3),
        }
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for k, v in payload["summary"].items():
            print(f"  {k}: {v}")

    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
