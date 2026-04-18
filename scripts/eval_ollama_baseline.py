"""Run the same 12-prompt quality suite against Ollama Q8_0 baseline.

Uses the Ollama /api/chat endpoint (chat template, thinking disabled)
with deterministic settings and the EXACT same evaluate_response()
scorer from eval_quality_gate.py — no reimplemented checker.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_quality_gate import (
    evaluate_response,
    prompt_suite,
    strip_thinking,
)


def run_ollama_chat(model: str, prompt_text: str, max_tokens: int = 512) -> dict:
    """Run a single prompt through Ollama /api/chat (chat template)."""
    import subprocess

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "stream": False,
        "think": False,
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
            "http://localhost:11434/api/chat",
            "-d",
            json.dumps(payload),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    elapsed = time.time() - start

    if proc.returncode != 0:
        return {"error": proc.stderr, "elapsed": elapsed}

    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON: {proc.stdout[:200]}", "elapsed": elapsed}

    message = result.get("message", {})
    response = message.get("content", "")
    eval_count = result.get("eval_count", 0)
    eval_duration = result.get("eval_duration", 1)
    tok_per_sec = eval_count / (eval_duration / 1e9) if eval_duration > 0 else 0

    return {
        "response": response,
        "elapsed": elapsed,
        "eval_count": eval_count,
        "tok_per_sec": round(tok_per_sec, 2),
    }


def main():
    model = "qwen3.6-35b-a3b:q8_0"
    prompts = prompt_suite("all", harness_version="v2")

    print(f"Running {len(prompts)} prompts against Ollama {model}")
    print("Endpoint: /api/chat (chat template, think=false)")
    print("Settings: temp=0.0, seed=42, max_tokens=512")
    print("Scorer: eval_quality_gate.evaluate_response (identical to MLX gate)")
    print("=" * 60)

    results = []
    passed_count = 0

    for i, prompt in enumerate(prompts):
        max_tokens = prompt.get("max_tokens", 512)
        print(f"\n[{i + 1}/{len(prompts)}] {prompt['name']}...")

        ollama_result = run_ollama_chat(model, prompt["text"], max_tokens=max_tokens)

        if "error" in ollama_result:
            print(f"  ERROR: {ollama_result['error'][:200]}")
            results.append(
                {
                    "name": prompt["name"],
                    "passed": False,
                    "failures": [f"Ollama error: {ollama_result['error'][:200]}"],
                    "response": "",
                }
            )
            continue

        raw_response = ollama_result["response"]
        passed, failures = evaluate_response(prompt, raw_response)

        entry = {
            "name": prompt["name"],
            "passed": passed,
            "failures": failures,
            "response": raw_response,
            "response_words": len(strip_thinking(raw_response).split()),
            "tok_per_sec": ollama_result["tok_per_sec"],
            "latency_s": round(ollama_result["elapsed"], 3),
        }

        if passed:
            passed_count += 1
            print(f"  PASS ({ollama_result['tok_per_sec']} tok/s)")
        else:
            print(f"  FAIL ({ollama_result['tok_per_sec']} tok/s)")
            for f in failures:
                print(f"    {f}")

        results.append(entry)

    print("\n" + "=" * 60)
    print(f"Result: {passed_count}/{len(prompts)} passed")

    artifact = {
        "model": model,
        "config": "A (Q8_0 GGUF on Ollama)",
        "endpoint": "/api/chat",
        "think": False,
        "scorer": "eval_quality_gate.evaluate_response",
        "settings": {"temperature": 0.0, "seed": 42, "max_tokens": 512},
        "n_pass": passed_count,
        "n_total": len(prompts),
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
