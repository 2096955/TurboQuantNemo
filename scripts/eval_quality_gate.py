"""Quality gate for Nemotron-H MoE checkpoints.

Runs a fixed prompt suite with deterministic generation settings and
automated pass/fail criteria. Exit code 0 = all checks pass; 1 = at
least one failure.

Checks:
  - Minimum response length (not truncated / empty)
  - No degenerate repetition (the primary failure mode from pre-fix runs)
  - Expected answer present where applicable
  - Optional: malformed fenced code blocks (unbalanced ```)
  - JSON output and baseline comparison for release regression tracking

Set PYTHONHASHSEED=0 for maximum reproducibility across processes.
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

_RUNTIME = None


def _runtime():
    global _RUNTIME
    if _RUNTIME is None:
        import mlx.core as mx
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler

        _RUNTIME = (mx, generate, load, make_sampler)
    return _RUNTIME


PROMPTS = [
    {
        "name": "Code Generation",
        "text": "Write a Python function to compute the nth Fibonacci number.",
        "min_tokens": 30,
        "expected_substr": ["def ", "fibonacci", "return"],
        "description": "Must produce a Python function definition",
    },
    {
        "name": "Instruction Following",
        "text": "Explain the theory of relativity to a 5 year old.",
        "min_tokens": 30,
        "expected_substr": [],
        "description": "Must produce a coherent multi-sentence explanation",
    },
    {
        "name": "Basic Reasoning",
        "text": "If I have 3 apples and eat 1, how many apples do I have left? Think step by step.",
        "min_tokens": 15,
        "expected_substr": ["2"],
        "description": "Must arrive at the answer 2",
    },
    {
        "name": "List Generation",
        "text": "List 5 programming languages and one strength of each.",
        "min_tokens": 30,
        "expected_substr": [],
        "description": "Must produce a structured list",
    },
    {
        "name": "Math",
        "text": "What is 17 * 24? Show your work.",
        "min_tokens": 10,
        "expected_substr": ["408"],
        "description": "Must compute 408",
    },
]

# Heavier coding-oriented tasks (use --suite coding or all).
CODING_PROMPTS = [
    {
        "name": "Bugfix (off-by-one)",
        "text": (
            "Here is buggy Python:\n\n"
            "def slice_last_n(items, n):\n"
            "    return items[len(items) - n : len(items) - 1]\n\n"
            "Fix the bug so it returns the last n items. Output only the corrected function."
        ),
        "min_tokens": 25,
        "expected_substr": ["return", "items"],
        "description": "Must return a corrected function using the list tail",
    },
    {
        "name": "Traceback interpretation",
        "text": (
            "Explain what this Python error means and how to fix it:\n"
            "TypeError: unsupported operand type(s) for +: 'int' and 'str'"
        ),
        "min_tokens": 40,
        "expected_substr": ["str", "int"],
        "description": "Must mention type mismatch and conversion/casting",
    },
    {
        "name": "Multi-file style refactor",
        "text": (
            "You have module a.py with `def fetch(): ...` and b.py importing it. "
            "Describe concrete steps to rename `fetch` to `load_data` across both files "
            "and what to run to verify (tests or grep). Be specific."
        ),
        "min_tokens": 50,
        "expected_substr": ["rename", "import"],
        "description": "Must give actionable rename/refactor steps",
    },
    {
        "name": "Write a minimal test",
        "text": (
            "Write a pytest test for a function `add(a,b)` that asserts add(2,3)==5. "
            "Output only the test code."
        ),
        "min_tokens": 15,
        "expected_substr": ["def ", "assert", "add"],
        "description": "Must output pytest-style test with assert",
    },
    {
        "name": "Implement LRU cache get",
        "text": (
            "Implement a class method `get(self, key)` for an LRU cache with a fixed capacity. "
            "On miss return None; on hit move key to most-recent. "
            "Assume `self._order` and `self._data` exist."
        ),
        "min_tokens": 20,
        "expected_substr": ["key", "return", "None"],
        "description": "Must implement method-body cache miss/hit logic",
    },
    {
        "name": "Race condition explanation",
        "text": (
            "Explain a classic race between two threads incrementing a shared counter without a lock, "
            "and fix it with one Python snippet using threading.Lock."
        ),
        "min_tokens": 45,
        "expected_substr": ["lock", "thread"],
        "description": "Must mention lock and threading",
    },
]


def prompt_suite(name: str) -> list[dict[str, Any]]:
    n = name.lower().strip()
    if n == "default":
        return list(PROMPTS)
    if n == "coding":
        return list(CODING_PROMPTS)
    if n == "all":
        return list(PROMPTS) + list(CODING_PROMPTS)
    raise ValueError(f"Unknown suite {name!r}")


def detect_malformed_fences(text: str) -> bool:
    """True if ``` fences look unbalanced (odd count)."""
    return text.count("```") % 2 != 0


def detect_repetition(text: str, window: int = 20, threshold: float = 0.4) -> bool:
    """Return True if the text contains degenerate repetition.

    Checks whether any sliding window of `window` words repeats more than
    `threshold` fraction of the total word count.
    """
    words = text.split()
    if len(words) < window * 2:
        return False
    for i in range(len(words) - window):
        pattern = " ".join(words[i : i + window])
        count = text.count(pattern)
        if count >= 3:
            return True
    sentences = re.split(r"[.!?\n]", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) >= 4:
        counts = Counter(sentences)
        most_common_count = counts.most_common(1)[0][1]
        if most_common_count >= 3 and most_common_count / len(sentences) > threshold:
            return True
    return False


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output for evaluation."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"</think>\s*", "", cleaned)
    return cleaned.strip()


def build_model_config(
    expert_offload: bool, max_resident_experts: int
) -> dict[str, Any]:
    model_config: dict[str, Any] = {}
    if expert_offload:
        model_config["expert_offload"] = True
        model_config["max_resident_experts"] = max_resident_experts
    return model_config


def render_prompt(tokenizer, prompt_text: str, system_prompt: str | None) -> list[int]:
    if getattr(tokenizer, "has_chat_template", False):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt_text})
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    print("Warning: chat_template not found, using raw text encoding.")
    return tokenizer.encode(prompt_text)


def generate_response(
    model,
    tokenizer,
    prompt_text: str,
    *,
    system_prompt: str | None,
    max_tokens: int,
    temp: float,
) -> tuple[str, float]:
    _, generate, _, make_sampler = _runtime()
    sampler = make_sampler(temp)
    prompt = render_prompt(tokenizer, prompt_text, system_prompt)
    t0 = time.perf_counter()
    response = generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=False,
    )
    return response.strip(), time.perf_counter() - t0


def evaluate_response(
    prompt_data: dict,
    raw_response: str,
    *,
    default_max_word_repeat_ratio: float = 0.35,
) -> tuple[bool, list[str]]:
    """Evaluate a response against quality criteria. Returns (passed, list of failure reasons)."""
    failures = []
    response = strip_thinking(raw_response)
    response_lower = response.lower()
    words = response.split()

    min_tokens = prompt_data.get("min_tokens", 15)
    if len(words) < min_tokens:
        failures.append(f"Too short: {len(words)} words (min {min_tokens})")

    max_words = prompt_data.get("max_words")
    if max_words is not None and len(words) > max_words:
        failures.append(f"Too long: {len(words)} words (max {max_words})")

    for substr in prompt_data.get("expected_substr", []):
        if substr.lower() not in response_lower:
            failures.append(f"Missing expected substring: {substr!r}")

    if detect_repetition(raw_response):
        failures.append("Degenerate repetition detected")

    # Stricter: single-token dominance (common degenerate mode).
    if words:
        top = Counter(words).most_common(1)[0][1]
        max_ratio = prompt_data.get(
            "max_word_repeat_ratio", default_max_word_repeat_ratio
        )
        if top / len(words) > max_ratio:
            failures.append(
                f"Word repetition ratio too high: {top / len(words):.2f} (max {max_ratio})"
            )

    if prompt_data.get("check_fences", True) and "```" in raw_response:
        if detect_malformed_fences(raw_response):
            failures.append("Malformed fenced code blocks (unbalanced ```)")

    return len(failures) == 0, failures


def run_gate(
    *,
    model_path: str,
    expert_offload: bool,
    max_resident_experts: int,
    system_prompt: str | None,
    max_tokens: int,
    seed: int,
    temp: float,
    prompts: list[dict[str, Any]],
    suite_name: str,
    strict: bool,
) -> tuple[bool, dict[str, Any]]:
    mx, _, load, _ = _runtime()
    mx.random.seed(seed)
    repeat_ratio = 0.22 if strict else 0.35
    print(
        f"Loading model from {model_path} "
        f"(expert_offload={expert_offload}, seed={seed}, temp={temp})..."
    )
    model, tokenizer = load(
        model_path,
        model_config=build_model_config(expert_offload, max_resident_experts),
    )

    print(f"\n--- Quality Gate ({len(prompts)} prompts) ---\n")
    if system_prompt:
        print(f"System prompt: {system_prompt}\n")

    all_passed = True
    results = []

    for prompt_data in prompts:
        print(f"Task: {prompt_data['name']}")
        print(f"Prompt: {prompt_data['text']}")
        print(f"Criteria: {prompt_data['description']}")
        print("-" * 40)

        response, elapsed = generate_response(
            model,
            tokenizer,
            prompt_data["text"],
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temp=temp,
        )
        print(f"Response ({elapsed:.2f}s):\n{response}\n")

        passed, failures = evaluate_response(
            prompt_data, response, default_max_word_repeat_ratio=repeat_ratio
        )
        status = "PASS" if passed else "FAIL"
        print(f"Result: {status}")
        if failures:
            for f in failures:
                print(f"  - {f}")
        print("=" * 80 + "\n")

        results.append((prompt_data["name"], passed, failures, elapsed))
        if not passed:
            all_passed = False

    print("--- Summary ---")
    for name, passed, failures, elapsed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name} ({elapsed:.2f}s)")
    n_pass = sum(1 for _, p, _, _ in results if p)
    print(f"\n{n_pass}/{len(results)} passed")

    if all_passed:
        print("\nQuality gate: PASSED")
    else:
        print("\nQuality gate: FAILED")

    artifact = {
        "version": 2,
        "model_path": model_path,
        "expert_offload": expert_offload,
        "max_resident_experts": max_resident_experts,
        "seed": seed,
        "temp": temp,
        "max_tokens": max_tokens,
        "suite": suite_name,
        "strict": strict,
        "max_word_repeat_ratio_default": repeat_ratio,
        "n_pass": n_pass,
        "n_total": len(results),
        "results": [
            {
                "name": name,
                "passed": passed,
                "failures": failures,
                "latency_s": round(elapsed, 4),
            }
            for name, passed, failures, elapsed in results
        ],
    }
    return all_passed, artifact


def load_baseline(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def baseline_regressed(current: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """Return human-readable regression messages if a previously passing task now fails."""
    by_name = {r["name"]: r for r in baseline.get("results", [])}
    bad: list[str] = []
    for r in current.get("results", []):
        name = r["name"]
        prev = by_name.get(name)
        if prev and prev.get("passed") and not r.get("passed"):
            bad.append(f"Regression: {name!r} passed in baseline but failed now")
    return bad


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quality gate for Nemotron-H MoE checkpoints"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model under evaluation.",
    )
    parser.add_argument(
        "--expert-offload",
        action="store_true",
        help="Enable expert offload.",
    )
    parser.add_argument("--max-resident-experts", type=int, default=4000)
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Optional system prompt passed through the model chat template.",
    )
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible generation.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy/deterministic).",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="default",
        choices=["default", "coding", "all"],
        help="Prompt suite: default (smoke), coding (extra tasks), or all.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Write run artifact (pass/fail per task) to this path.",
    )
    parser.add_argument(
        "--compare-baseline",
        type=str,
        default=None,
        help="JSON from a prior run; fail if any task that passed then fails now.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Tighter word-level repetition gate (default max ratio 0.22 vs 0.35).",
    )
    args = parser.parse_args()

    try:
        prompts = prompt_suite(args.suite)
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(2)

    passed, artifact = run_gate(
        model_path=args.model,
        expert_offload=args.expert_offload,
        max_resident_experts=args.max_resident_experts,
        system_prompt=args.system_prompt,
        max_tokens=args.max_tokens,
        seed=args.seed,
        temp=args.temp,
        prompts=prompts,
        suite_name=args.suite,
        strict=args.strict,
    )

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(artifact, f, indent=2)
        print(f"Wrote artifact: {out}")

    if args.compare_baseline:
        baseline = load_baseline(Path(args.compare_baseline))
        regs = baseline_regressed(artifact, baseline)
        if regs:
            for line in regs:
                print(line, file=sys.stderr)
            sys.exit(1)

    sys.exit(0 if passed else 1)
