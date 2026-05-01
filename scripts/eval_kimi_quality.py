#!/usr/bin/env python3
"""Quality eval for Kimi quantization variants.

Uses the Phase 4 prompts (3 simple factual prompts) plus a longer-form
generation, both at temperature=0 (greedy) for deterministic comparison.

Outputs a JSON artifact with per-prompt outputs, prefix-match lengths,
and a coarse pass/fail gate.

Usage:
    python3 scripts/eval_kimi_quality.py \\
        --reference /Volumes/Samsung9904tb/Kimi-K2.6 \\
        --variant /Volumes/Samsung9904tb/Kimi-K2.6-2bit-experts \\
        --max-tokens 64 \\
        --output artifacts/kimi_k26_speculative/quality_2bit_vs_4bit.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MLX_LM_ROOT = REPO_ROOT / "mlx-lm"
if MLX_LM_ROOT.exists() and str(MLX_LM_ROOT) not in sys.path:
    sys.path.insert(0, str(MLX_LM_ROOT))


PHASE_4_PROMPTS = [
    "What is 2+2?",
    "The capital of France is",
    "Explain the concept of attention in transformers in one sentence.",
]


def repetition_ratio(text: str) -> float:
    """Bigram repetition ratio. >0.3 indicates collapse."""
    words = text.lower().split()
    if len(words) < 10:
        return 0.0
    bigrams = [f"{words[j]} {words[j + 1]}" for j in range(len(words) - 1)]
    return 1.0 - len(set(bigrams)) / max(len(bigrams), 1)


def run_one(model_path: str, max_tokens: int, max_resident_experts: int) -> list[dict]:
    """Load model, run all prompts, return per-prompt outputs."""
    from mlx_lm import load, generate

    print(f"  Loading {model_path}...", flush=True)
    t0 = time.time()
    model, tokenizer = load(
        model_path,
        model_config={
            "expert_offload": True,
            "max_resident_experts": max_resident_experts,
        },
    )
    load_s = time.time() - t0
    print(f"  Loaded in {load_s:.1f}s", flush=True)

    results = []
    for prompt in PHASE_4_PROMPTS:
        t0 = time.time()
        out = generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False
        )
        elapsed = time.time() - t0
        results.append(
            {
                "prompt": prompt,
                "output": out,
                "elapsed_s": round(elapsed, 2),
                "char_count": len(out),
                "repetition_ratio": round(repetition_ratio(out), 4),
            }
        )
        print(
            f"  [{prompt[:40]}...] {elapsed:.1f}s, "
            f"{len(out)} chars, rep={results[-1]['repetition_ratio']}",
            flush=True,
        )

    del model, tokenizer
    gc.collect()
    return results


def prefix_match_chars(a: str, b: str) -> int:
    """Length of the longest matching prefix in characters."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True, help="4-bit reference Kimi path")
    p.add_argument("--variant", required=True, help="Quantized variant Kimi path")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--max-resident-experts", type=int, default=2000)
    p.add_argument(
        "--output",
        default="artifacts/kimi_k26_speculative/quality_eval.json",
    )
    args = p.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Reference: {args.reference} ===")
    ref_results = run_one(args.reference, args.max_tokens, args.max_resident_experts)

    print(f"\n=== Variant: {args.variant} ===")
    var_results = run_one(args.variant, args.max_tokens, args.max_resident_experts)

    comparisons = []
    for ref, var in zip(ref_results, var_results):
        prefix = prefix_match_chars(ref["output"], var["output"])
        comparisons.append(
            {
                "prompt": ref["prompt"],
                "ref_chars": ref["char_count"],
                "var_chars": var["char_count"],
                "prefix_match_chars": prefix,
                "exact_match": ref["output"] == var["output"],
                "ref_repetition": ref["repetition_ratio"],
                "var_repetition": var["repetition_ratio"],
                "ref_output": ref["output"],
                "var_output": var["output"],
            }
        )

    n = len(comparisons)
    exact_matches = sum(1 for c in comparisons if c["exact_match"])
    avg_prefix = sum(c["prefix_match_chars"] for c in comparisons) / n
    avg_ref_chars = sum(c["ref_chars"] for c in comparisons) / n
    avg_prefix_ratio = avg_prefix / max(avg_ref_chars, 1)
    high_rep = sum(1 for c in comparisons if c["var_repetition"] > 0.3)

    payload = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reference": args.reference,
        "variant": args.variant,
        "max_tokens": args.max_tokens,
        "comparisons": comparisons,
        "summary": {
            "n_prompts": n,
            "exact_matches": exact_matches,
            "avg_prefix_match_chars": round(avg_prefix, 1),
            "avg_prefix_match_ratio": round(avg_prefix_ratio, 3),
            "variant_high_repetition_count": high_rep,
            "passes_quality_gate": (high_rep == 0 and avg_prefix_ratio >= 0.20),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Exact matches: {exact_matches}/{n}")
    print(f"  Avg prefix match: {avg_prefix:.1f} chars ({avg_prefix_ratio:.1%} of ref)")
    print(f"  High-repetition variant outputs: {high_rep}/{n}")
    print(
        f"  Quality gate: {'PASS' if payload['summary']['passes_quality_gate'] else 'FAIL'}"
    )
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
