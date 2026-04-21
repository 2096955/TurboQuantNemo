#!/usr/bin/env python3
"""Bit-exactness invariance check for mlx_lm.generate.

Runs the same prompt twice through `mlx_lm.generate` with greedy decoding and
a fixed seed, then asserts the two outputs are identical. Catches
nondeterminism that async expert prefetch (Phase 1) could introduce.

Usage:
    # Single config
    python scripts/invariance_check.py --model <path-or-hf-id> [--prompt TEXT] [--max-tokens N]

    # Standard three-config suite (per Phase 0.6 spec)
    python scripts/invariance_check.py --config-suite

Exits 0 if all configs produce identical output across both runs; 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


CANONICAL_PROMPT = "The capital of France is"
DEFAULT_MAX_TOKENS = 50

STANDARD_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "dense_llama_3_2_3b",
        "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "extra_args": [],
    },
    {
        "name": "moe_qwen36_35b_a3b",
        "model": str(Path.home() / "Models" / "qwen3.6-35b-a3b-mixed"),
        "extra_args": [],
    },
    {
        "name": "moe_qwen36_35b_a3b_offload",
        "model": str(Path.home() / "Models" / "qwen3.6-35b-a3b-mixed"),
        "extra_args": ["--expert-offload", "--max-resident-experts", "2048"],
    },
]


def output_hash(text: str) -> str:
    """Return sha256 hex digest of text. Used for fast equality check."""
    return hashlib.sha256(text.encode()).hexdigest()


def _diff_first_chars(a: str, b: str) -> dict[str, Any]:
    """Find first character index where two strings differ.

    Returns:
        Empty dict if strings identical.
        Dict with first_diff_index + context if they differ at some position.
        Dict with length_mismatch if one is a prefix of the other.
    """
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            return {
                "first_diff_index": i,
                "context_a": a[max(0, i - 20) : i + 20],
                "context_b": b[max(0, i - 20) : i + 20],
            }
    if len(a) != len(b):
        return {"length_mismatch": [len(a), len(b)]}
    return {}


def run_generation(
    model: str, prompt: str, max_tokens: int, extra_args: list[str]
) -> str:
    """Run `mlx_lm.generate` with deterministic settings; return stdout."""
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.generate",
        "--model",
        model,
        "--prompt",
        prompt,
        "--max-tokens",
        str(max_tokens),
        "--temp",
        "0.0",
        "--seed",
        "42",
        *extra_args,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return proc.stdout


def check_invariance(
    model: str,
    prompt: str,
    max_tokens: int,
    extra_args: list[str],
    config_name: str,
) -> tuple[bool, dict[str, Any]]:
    """Run generation twice; return (passed, diagnostics)."""
    out1 = run_generation(model, prompt, max_tokens, extra_args)
    out2 = run_generation(model, prompt, max_tokens, extra_args)
    h1 = output_hash(out1)
    h2 = output_hash(out2)
    passed = h1 == h2
    diag: dict[str, Any] = {
        "config": config_name,
        "passed": passed,
        "hash_run_1": h1,
        "hash_run_2": h2,
        "len_run_1": len(out1),
        "len_run_2": len(out2),
    }
    if not passed:
        diag["diff_chars"] = _diff_first_chars(out1, out2)
    return passed, diag


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", help="Model path (HF id or local path)")
    parser.add_argument("--prompt", default=CANONICAL_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--extra-args",
        nargs="*",
        default=[],
        help="Extra args passed through to mlx_lm.generate",
    )
    parser.add_argument(
        "--config-suite",
        action="store_true",
        help="Run all three standard configs from STANDARD_CONFIGS",
    )
    parser.add_argument("--json", help="Path to write JSON results")
    args = parser.parse_args()

    if args.config_suite:
        results = []
        all_passed = True
        for config in STANDARD_CONFIGS:
            print(f"=== {config['name']} ===")
            passed, diag = check_invariance(
                config["model"],
                args.prompt,
                args.max_tokens,
                config["extra_args"],
                config["name"],
            )
            results.append(diag)
            all_passed = all_passed and passed
            print(f"  {'PASS' if passed else 'FAIL'}")
            if not passed:
                print(f"  diff: {diag.get('diff_chars')}")
        if args.json:
            Path(args.json).write_text(json.dumps(results, indent=2))
        sys.exit(0 if all_passed else 1)
    else:
        if not args.model:
            parser.error("--model required when not using --config-suite")
        passed, diag = check_invariance(
            args.model,
            args.prompt,
            args.max_tokens,
            args.extra_args,
            args.model,
        )
        print(json.dumps(diag, indent=2))
        if args.json:
            Path(args.json).write_text(json.dumps(diag, indent=2))
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
