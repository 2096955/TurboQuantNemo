"""Run the same 12-prompt quality suite against Ollama Q8_0 baseline.

Uses the Ollama /api/generate endpoint with deterministic settings
(temperature 0, seed 42) and applies the same pass/fail criteria
from eval_quality_gate.py.
"""

import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

# Import the same prompt suite and checking logic
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_quality_gate import (
    prompt_suite,
)


def run_ollama_prompt(model: str, prompt_text: str, max_tokens: int = 512) -> dict:
    """Run a single prompt through Ollama API."""
    payload = {
        "model": model,
        "prompt": prompt_text,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "seed": 42,
            "num_predict": max_tokens,
        },
    }

    start = time.time()
    proc = subprocess.run(
        [
            "curl",
            "-s",
            "http://localhost:11434/api/generate",
            "-d",
            json.dumps(payload),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    elapsed = time.time() - start

    if proc.returncode != 0:
        return {"error": proc.stderr, "elapsed": elapsed}

    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON: {proc.stdout[:200]}", "elapsed": elapsed}

    response = result.get("response", "")
    total_duration = result.get("total_duration", 0)
    eval_count = result.get("eval_count", 0)
    eval_duration = result.get("eval_duration", 1)

    tok_per_sec = eval_count / (eval_duration / 1e9) if eval_duration > 0 else 0

    return {
        "response": response,
        "elapsed": elapsed,
        "eval_count": eval_count,
        "tok_per_sec": round(tok_per_sec, 2),
        "total_duration_s": round(total_duration / 1e9, 2) if total_duration else None,
    }


def strip_thinking(text: str) -> str:
    """Strip <think>...</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def check_repetition(text: str, threshold: float = 0.35) -> bool:
    """Return True if text has excessive repetition (same as eval_quality_gate)."""
    words = text.lower().split()
    if len(words) < 10:
        return False
    counts = Counter(words)
    # Check if any non-trivial word dominates
    for word, count in counts.most_common(10):
        if len(word) <= 2:
            continue
        ratio = count / len(words)
        if ratio > threshold:
            return True
    return False


def evaluate_prompt(prompt: dict, response_text: str) -> dict:
    """Apply the same pass/fail criteria as eval_quality_gate."""
    clean = strip_thinking(response_text)
    result = {
        "name": prompt["name"],
        "passed": True,
        "checks": [],
    }

    # Min tokens check
    min_tokens = prompt.get("min_tokens", 10)
    skip_min = prompt.get("skip_min_tokens", False)
    token_count = len(clean.split())
    if not skip_min and token_count < min_tokens:
        result["passed"] = False
        result["checks"].append(f"FAIL: {token_count} tokens < min {min_tokens}")
    else:
        result["checks"].append(f"OK: {token_count} tokens (min {min_tokens})")

    # Expected substrings
    for sub in prompt.get("expected_substr", []):
        if sub.lower() in clean.lower():
            result["checks"].append(f"OK: contains '{sub}'")
        else:
            result["passed"] = False
            result["checks"].append(f"FAIL: missing '{sub}'")

    # Repetition check
    if check_repetition(clean):
        result["passed"] = False
        result["checks"].append("FAIL: excessive repetition detected")
    else:
        result["checks"].append("OK: no degenerate repetition")

    return result


def main():
    model = "qwen3.6-35b-a3b:q8_0"
    prompts = prompt_suite("all", harness_version="v2")

    print(f"Running {len(prompts)} prompts against Ollama {model}")
    print("Settings: temp=0.0, seed=42, max_tokens=512")
    print("=" * 60)

    results = []
    passed_count = 0

    for i, prompt in enumerate(prompts):
        max_tokens = prompt.get("max_tokens", 512)
        print(f"\n[{i + 1}/{len(prompts)}] {prompt['name']}...")

        ollama_result = run_ollama_prompt(model, prompt["text"], max_tokens=max_tokens)

        if "error" in ollama_result:
            print(f"  ERROR: {ollama_result['error'][:200]}")
            results.append(
                {
                    "name": prompt["name"],
                    "passed": False,
                    "error": ollama_result["error"][:500],
                }
            )
            continue

        response = ollama_result["response"]
        eval_result = evaluate_prompt(prompt, response)
        eval_result["tok_per_sec"] = ollama_result["tok_per_sec"]
        eval_result["response_preview"] = strip_thinking(response)[:200]

        if eval_result["passed"]:
            passed_count += 1
            print(f"  PASS ({ollama_result['tok_per_sec']} tok/s)")
        else:
            print(f"  FAIL ({ollama_result['tok_per_sec']} tok/s)")
            for check in eval_result["checks"]:
                if "FAIL" in check:
                    print(f"    {check}")

        results.append(eval_result)

    print("\n" + "=" * 60)
    print(f"Result: {passed_count}/{len(prompts)} passed")

    artifact = {
        "model": model,
        "config": "A (Q8_0 GGUF on Ollama)",
        "settings": {"temperature": 0.0, "seed": 42, "max_tokens": 512},
        "passed": passed_count,
        "total": len(prompts),
        "results": results,
    }

    out_path = (
        Path(__file__).resolve().parent.parent
        / "results"
        / "qwen36_q8_baseline_quality.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
