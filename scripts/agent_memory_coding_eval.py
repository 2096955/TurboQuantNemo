"""Agent-memory coding evaluator for KV backend comparisons.

This harness evaluates the practical coding path the user cares about:
multi-step coding tasks with explicit context resets and a persisted
external memory note. It is not a raw long-context benchmark. The model
must continue work across turns using only the saved scratch memory plus
the current step instruction.

What it measures:
  1. Step completion quality on multi-step coding tasks.
  2. State retention via scratch-memory reuse after each context reset.
  3. Repair efficiency when a validator-guided retry is allowed.
  4. Latency, decode-hit rate, and peak memory for the run.

The memory note is derived only from the model's own prior outputs:
imports, classes, function names, and a short code skeleton. No oracle
task data is injected into the saved memory.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

_RUNTIME = None


def _run_text(cmd: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _swap_used_mb() -> float | None:
    out = _run_text(["sysctl", "vm.swapusage"])
    if not out or "used =" not in out:
        return None
    try:
        value = out.split("used =", 1)[1].strip().split()[0]
        if value.endswith("M"):
            return round(float(value[:-1]), 2)
        if value.endswith("G"):
            return round(float(value[:-1]) * 1024.0, 2)
    except Exception:
        return None
    return None


def _process_rss_mb() -> float | None:
    out = _run_text(["ps", "-o", "rss=", "-p", str(os.getpid())])
    if not out:
        return None
    try:
        return round(float(out.strip()) / 1024.0, 2)
    except ValueError:
        return None


def _vm_pageouts() -> int | None:
    out = _run_text(["vm_stat"])
    if not out:
        return None
    for line in out.splitlines():
        if "Pages paged out" in line:
            try:
                return int(line.split(":")[1].strip().rstrip("."))
            except Exception:
                return None
    return None


def _system_snapshot() -> dict[str, Any]:
    return {
        "process_rss_mb": _process_rss_mb(),
        "swap_used_mb": _swap_used_mb(),
        "vm_pages_paged_out": _vm_pageouts(),
    }


def _delta(after: float | int | None, before: float | int | None):
    if after is None or before is None:
        return None
    return round(after - before, 2)


def _runtime():
    global _RUNTIME
    if _RUNTIME is None:
        import mlx.core as mx
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler

        _RUNTIME = (mx, generate, load, make_sampler)
    return _RUNTIME


def build_model_config(
    expert_offload: bool,
    max_resident_experts: int | None,
    *,
    max_cached_shards: int | None = None,
    use_predictor: bool = False,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if expert_offload:
        cfg["expert_offload"] = True
        if max_resident_experts is not None:
            cfg["max_resident_experts"] = max_resident_experts
        if max_cached_shards is not None:
            cfg["max_cached_shards"] = max_cached_shards
    if use_predictor:
        cfg["use_predictor"] = True
    return cfg


def render_prompt(tokenizer, text: str, system_prompt: str | None = None) -> list[int]:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})
        kwargs = {"tokenize": True, "add_generation_prompt": True}
        try:
            kwargs["enable_thinking"] = False
        except Exception:
            pass
        return tokenizer.apply_chat_template(messages, **kwargs)
    return tokenizer.encode(text)


def strip_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return re.sub(r"</think>\s*", "", cleaned).strip()


def detect_repetition(text: str, threshold: float = 0.4) -> bool:
    words = text.split()
    if len(words) < 10:
        return False
    top = Counter(words).most_common(1)[0][1]
    return top / len(words) > threshold


def generate_response(
    model,
    tokenizer,
    prompt_text: str,
    *,
    system_prompt: str | None,
    max_tokens: int,
    temp: float,
    kv_cache_type: str = "default",
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
        kv_cache_type=kv_cache_type,
    )
    return response.strip(), time.perf_counter() - t0


def get_offload_stats(model) -> dict[str, Any] | None:
    mgr = getattr(model, "expert_offload_manager", None)
    if mgr is None:
        return None
    return mgr.stats_summary()


def reset_offload_stats(model) -> None:
    mgr = getattr(model, "expert_offload_manager", None)
    if mgr is None:
        return
    with mgr._lock:
        mgr.hits = 0
        mgr.misses = 0
        mgr.evictions = 0
        mgr.prefill_evals = 0
        mgr.decode_evals = 0
        mgr.prefill_hits = 0
        mgr.prefill_misses = 0
        mgr.decode_hits = 0
        mgr.decode_misses = 0
        mgr.load_time_ms_total = 0.0
        mgr.load_count = 0
        mgr.prefetch_requests = 0
        mgr.prefetch_already_cached = 0


def flush_offload_cache(model) -> None:
    mgr = getattr(model, "expert_offload_manager", None)
    if mgr is None:
        return
    with mgr._cond:
        mgr._cache.clear()
        mgr._lru.clear()


def extract_python_payload(raw_response: str) -> str:
    text = strip_thinking(raw_response).strip()
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if fenced:
        return max((block.strip() for block in fenced), key=len, default="")

    text = re.sub(r"^```(?:python)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def unique_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def extract_code_features(text: str) -> dict[str, Any]:
    code = extract_python_payload(text)
    imports: list[str] = []
    classes: list[str] = []
    functions: list[str] = []
    tests: list[str] = []
    skeleton_lines: list[str] = []

    for line in code.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^(from\s+\S+\s+import\s+.+|import\s+.+)", stripped):
            imports.append(stripped)
            skeleton_lines.append(stripped)
            continue
        class_match = re.match(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)\b", stripped)
        if class_match:
            name = class_match.group(1)
            classes.append(name)
            skeleton_lines.append(stripped)
            continue
        fn_match = re.match(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\b", stripped)
        if fn_match:
            name = fn_match.group(1)
            functions.append(name)
            if name.startswith("test_"):
                tests.append(name)
            skeleton_lines.append(stripped)
            continue
        if stripped.startswith("@"):
            skeleton_lines.append(stripped)

    symbols = unique_keep_order(classes + functions)
    return {
        "code": code,
        "imports": unique_keep_order(imports),
        "classes": unique_keep_order(classes),
        "functions": unique_keep_order(functions),
        "tests": unique_keep_order(tests),
        "symbols": symbols,
        "skeleton_lines": unique_keep_order(skeleton_lines)[:14],
    }


def merge_memory_features(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
) -> dict[str, Any]:
    if previous is None:
        return current
    return {
        "code": current.get("code", ""),
        "imports": unique_keep_order(previous.get("imports", []) + current["imports"]),
        "classes": unique_keep_order(previous.get("classes", []) + current["classes"]),
        "functions": unique_keep_order(
            previous.get("functions", []) + current["functions"]
        ),
        "tests": unique_keep_order(previous.get("tests", []) + current["tests"]),
        "symbols": unique_keep_order(previous.get("symbols", []) + current["symbols"]),
        "skeleton_lines": unique_keep_order(
            previous.get("skeleton_lines", []) + current["skeleton_lines"]
        )[:18],
    }


def format_memory_note(features: dict[str, Any] | None, *, max_chars: int) -> str:
    if not features:
        return "Saved external memory: none."

    lines = ["Saved external memory from prior turns:"]
    if features.get("imports"):
        lines.append("- imports: " + ", ".join(features["imports"][:6]))
    if features.get("classes"):
        lines.append("- classes: " + ", ".join(features["classes"][:6]))
    if features.get("functions"):
        lines.append("- functions: " + ", ".join(features["functions"][:10]))
    if features.get("tests"):
        lines.append("- tests: " + ", ".join(features["tests"][:8]))
    if features.get("skeleton_lines"):
        lines.append("- code skeleton:")
        for line in features["skeleton_lines"][:12]:
            lines.append(f"  {line}")

    note = "\n".join(lines).strip()
    if len(note) <= max_chars:
        return note
    return note[: max_chars - 3].rstrip() + "..."


def validate_python_syntax(code: str) -> str | None:
    if not code.strip():
        return "Empty code payload"
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg} (line {exc.lineno})"
    except Exception as exc:
        return f"Parse error: {type(exc).__name__}: {exc}"
    return None


def has_stub_patterns(code: str) -> list[str]:
    failures: list[str] = []
    if "TODO" in code or "todo" in code:
        failures.append("Stub marker detected: TODO")
    if "NotImplementedError" in code:
        failures.append("Stub marker detected: NotImplementedError")
    if re.search(r"(^|\s)\.\.\.(\s|$)", code):
        failures.append("Stub marker detected: ellipsis")
    return failures


def syntax_error_line_number(failures: list[str]) -> int | None:
    for failure in failures:
        match = re.search(r"line (\d+)\)?", failure)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
    return None


def build_code_excerpt(code: str, *, line_no: int | None, radius: int = 2) -> str:
    if not code.strip():
        return ""
    lines = code.splitlines()
    if not lines:
        return ""

    if line_no is None or line_no < 1 or line_no > len(lines):
        excerpt_lines = lines[: min(len(lines), 24)]
        return "\n".join(
            f"{idx + 1:>3}: {line}" for idx, line in enumerate(excerpt_lines)
        )

    start = max(1, line_no - radius)
    end = min(len(lines), line_no + radius)
    excerpt = []
    for idx in range(start, end + 1):
        excerpt.append(f"{idx:>3}: {lines[idx - 1]}")
    return "\n".join(excerpt)


def symbol_present(code: str, symbol: str) -> bool:
    return re.search(rf"\b{re.escape(symbol)}\b", code) is not None


def evaluate_step_output(step: dict[str, Any], raw_response: str) -> dict[str, Any]:
    features = extract_code_features(raw_response)
    code = features["code"]
    failures: list[str] = []

    if detect_repetition(strip_thinking(raw_response)):
        failures.append("Degenerate repetition detected")

    min_words = step.get("min_words", 20)
    if len(code.split()) < min_words:
        failures.append(f"Too short: {len(code.split())} words (min {min_words})")

    if step.get("require_parseable_python", True):
        syntax_error = validate_python_syntax(code)
        if syntax_error:
            failures.append(syntax_error)

    failures.extend(has_stub_patterns(code))

    for required in step.get("required_import_substrings", []):
        if required not in code:
            failures.append(f"Missing required import/content: {required!r}")

    for symbol in step.get("expected_symbols", []):
        if not symbol_present(code, symbol):
            failures.append(f"Missing expected symbol: {symbol!r}")

    for symbol in step.get("must_preserve_symbols", []):
        if not symbol_present(code, symbol):
            failures.append(f"Did not preserve prior symbol: {symbol!r}")

    required_tests = step.get("required_test_names", [])
    for test_name in required_tests:
        if test_name not in features["tests"]:
            failures.append(f"Missing required test function: {test_name!r}")

    min_tests = step.get("min_test_count")
    if min_tests is not None and len(features["tests"]) < min_tests:
        failures.append(
            f"Too few pytest tests: {len(features['tests'])} found (min {min_tests})"
        )

    preserve_symbols = step.get("must_preserve_symbols", [])
    preserved = [s for s in preserve_symbols if symbol_present(code, s)]
    preserve_hit_rate = (
        round(len(preserved) / len(preserve_symbols), 4) if preserve_symbols else None
    )

    return {
        "passed": not failures,
        "failures": failures,
        "code_word_count": len(code.split()),
        "features": features,
        "preserve_hit_rate": preserve_hit_rate,
        "preserved_symbols": preserved,
    }


TASKS = [
    {
        "name": "csv_pipeline_module",
        "description": "Multi-step data pipeline module with tests.",
        "steps": [
            {
                "name": "parse_rows",
                "prompt": (
                    "Write a complete Python module that defines exactly one function "
                    "`parse_rows(text: str) -> list[dict]`. The input is CSV with columns "
                    "`user,team,active,score`. Use `csv.DictReader` and `io.StringIO`. "
                    "Coerce `active` to bool, coerce `score` to int, and raise `ValueError` "
                    "for a non-integer score. Output only the complete module."
                ),
                "expected_symbols": ["parse_rows"],
                "required_import_substrings": ["import csv", "import io"],
                "min_words": 45,
            },
            {
                "name": "filter_and_summarize",
                "prompt": (
                    "Continue the same module after a context reset. Keep `parse_rows`. "
                    "Add `filter_active(rows)` and `summarize_by_team(rows)`. "
                    "`summarize_by_team` must return `dict[str, int]` with per-team "
                    "score totals for active rows only. Output the complete module."
                ),
                "expected_symbols": ["parse_rows", "filter_active", "summarize_by_team"],
                "must_preserve_symbols": ["parse_rows"],
                "min_words": 70,
            },
            {
                "name": "tests",
                "prompt": (
                    "Continue the same module after a context reset. Keep all existing "
                    "functions. Add pytest tests named `test_parse_rows_bad_score`, "
                    "`test_filter_active_only_active`, and `test_summarize_by_team_totals`. "
                    "Output the complete module."
                ),
                "expected_symbols": [
                    "parse_rows",
                    "filter_active",
                    "summarize_by_team",
                    "pytest",
                ],
                "must_preserve_symbols": [
                    "parse_rows",
                    "filter_active",
                    "summarize_by_team",
                ],
                "required_test_names": [
                    "test_parse_rows_bad_score",
                    "test_filter_active_only_active",
                    "test_summarize_by_team_totals",
                ],
                "required_import_substrings": ["import pytest"],
                "min_test_count": 3,
                "min_words": 120,
            },
        ],
    },
    {
        "name": "merge_sort_module",
        "description": "Incrementally build merge sort with tests.",
        "steps": [
            {
                "name": "merge_helper",
                "prompt": (
                    "Write a complete Python module that defines "
                    "`merge_sorted(left: list[int], right: list[int]) -> list[int]`. "
                    "It should merge two already-sorted integer lists. Output only "
                    "the complete module."
                ),
                "expected_symbols": ["merge_sorted"],
                "min_words": 35,
            },
            {
                "name": "merge_sort",
                "prompt": (
                    "Continue the same module after a context reset. Keep `merge_sorted`. "
                    "Add `merge_sort(values: list[int]) -> list[int]` that uses the helper "
                    "recursively. Output the complete module."
                ),
                "expected_symbols": ["merge_sorted", "merge_sort"],
                "must_preserve_symbols": ["merge_sorted"],
                "min_words": 55,
            },
            {
                "name": "tests_and_docs",
                "prompt": (
                    "Continue the same module after a context reset. Keep both functions. "
                    "Add short docstrings and pytest tests named `test_merge_sort_empty`, "
                    "`test_merge_sort_duplicates`, and `test_merge_sort_sorted_input`. "
                    "Output the complete module."
                ),
                "expected_symbols": ["merge_sorted", "merge_sort", "pytest"],
                "must_preserve_symbols": ["merge_sorted", "merge_sort"],
                "required_test_names": [
                    "test_merge_sort_empty",
                    "test_merge_sort_duplicates",
                    "test_merge_sort_sorted_input",
                ],
                "required_import_substrings": ["import pytest"],
                "min_test_count": 3,
                "min_words": 100,
            },
        ],
    },
    {
        "name": "lru_cache_module",
        "description": "Stateful class extension with tests after context resets.",
        "steps": [
            {
                "name": "base_cache",
                "prompt": (
                    "Write a complete Python module that defines `class LRUCache` using "
                    "`collections.OrderedDict`. It must support `get(self, key)` and "
                    "`put(self, key, value)` for a fixed capacity cache with eviction. "
                    "Output only the complete module."
                ),
                "expected_symbols": ["LRUCache", "get", "put", "OrderedDict"],
                "required_import_substrings": ["OrderedDict"],
                "min_words": 65,
            },
            {
                "name": "stats_and_delete",
                "prompt": (
                    "Continue the same module after a context reset. Keep `LRUCache`, "
                    "`get`, and `put`. Add `delete(self, key)` plus hit/miss/eviction "
                    "counters as attributes named `hits`, `misses`, and `evictions`. "
                    "Output the complete module."
                ),
                "expected_symbols": [
                    "LRUCache",
                    "get",
                    "put",
                    "delete",
                    "hits",
                    "misses",
                    "evictions",
                ],
                "must_preserve_symbols": ["LRUCache", "get", "put"],
                "min_words": 90,
            },
            {
                "name": "cache_tests",
                "prompt": (
                    "Continue the same module after a context reset. Keep the full cache "
                    "implementation. Add pytest tests named `test_lru_eviction`, "
                    "`test_lru_counters`, and `test_lru_delete`. Output the complete module."
                ),
                "expected_symbols": [
                    "LRUCache",
                    "get",
                    "put",
                    "delete",
                    "pytest",
                ],
                "must_preserve_symbols": ["LRUCache", "get", "put", "delete"],
                "required_test_names": [
                    "test_lru_eviction",
                    "test_lru_counters",
                    "test_lru_delete",
                ],
                "required_import_substrings": ["import pytest"],
                "min_test_count": 3,
                "min_words": 130,
            },
        ],
    },
]


TASK_INDEX = {task["name"]: task for task in TASKS}


DEFAULT_SYSTEM_PROMPT = (
    "You are continuing a Python coding task after a hard context reset. "
    "Use the saved external memory if provided. Preserve prior symbols "
    "named in the memory unless the current step explicitly asks to remove them. "
    "Keep the module compact and complete. Use short docstrings, concise tests, "
    "and no explanation outside the module."
)


def build_step_prompt(
    task: dict[str, Any],
    step: dict[str, Any],
    *,
    memory_mode: str,
    memory_note: str,
    prior_failures: list[str] | None,
    prior_attempt_code: str | None,
    prior_attempt_excerpt: str | None,
) -> str:
    sections = [
        f"Task: {task['name']}",
        f"Task objective: {task['description']}",
        "The previous full conversation is unavailable.",
    ]

    if memory_mode == "scratchpad":
        sections.append(memory_note or "Saved external memory: none.")
    else:
        sections.append(
            "Saved external memory is unavailable for this run. Reconstruct the work from the current step only."
        )

    if prior_failures:
        sections.append("Validator feedback from the previous attempt:")
        sections.extend(f"- {failure}" for failure in prior_failures)
        if any(
            "was never closed" in failure
            or "unexpected EOF" in failure
            or "EOF while scanning" in failure
            for failure in prior_failures
        ):
            sections.append(
                "The previous module appears truncated or unfinished. Rewrite it more compactly so the full module fits."
            )
        if prior_attempt_excerpt:
            sections.append(
                "Relevant excerpt from your previous module attempt (line-numbered):"
            )
            sections.append(f"```python\n{prior_attempt_excerpt}\n```")
        elif prior_attempt_code:
            sections.append("Your previous module attempt:")
            sections.append(f"```python\n{prior_attempt_code}\n```")
        sections.append(
            "Fix the validator issues by editing that module, not by switching formats. "
            "Return one complete parseable Python module with all required preserved symbols intact."
        )

    sections.append(f"Current step ({step['name']}): {step['prompt']}")
    sections.append(
        "Keep the module compact: minimal blank lines, short docstrings, and concise pytest cases."
    )
    sections.append(
        "Return only the complete Python module. Do not include markdown, explanations, TODO markers, ellipses, or placeholders."
    )
    return "\n\n".join(sections)


def run_task(
    model,
    tokenizer,
    task: dict[str, Any],
    *,
    memory_mode: str,
    memory_note_chars: int,
    repair_attempts: int,
    max_tokens: int,
    temp: float,
    kv_cache_type: str,
    system_prompt: str | None,
    max_step_latency_s: float | None,
    max_task_latency_s: float | None,
) -> dict[str, Any]:
    task_results: list[dict[str, Any]] = []
    memory_features: dict[str, Any] | None = None
    task_started_at = time.perf_counter()

    for step in task["steps"]:
        flush_offload_cache(model)
        reset_offload_stats(model)

        saved_memory = (
            format_memory_note(memory_features, max_chars=memory_note_chars)
            if memory_mode == "scratchpad"
            else ""
        )
        prior_failures: list[str] | None = None
        prior_attempt_code: str | None = None
        prior_attempt_excerpt: str | None = None
        best_attempt: dict[str, Any] | None = None
        cumulative_step_latency_s = 0.0

        for attempt_idx in range(1, repair_attempts + 2):
            prompt_text = build_step_prompt(
                task,
                step,
                memory_mode=memory_mode,
                memory_note=saved_memory,
                prior_failures=prior_failures,
                prior_attempt_code=prior_attempt_code,
                prior_attempt_excerpt=prior_attempt_excerpt,
            )
            response, latency = generate_response(
                model,
                tokenizer,
                prompt_text,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temp=temp,
                kv_cache_type=kv_cache_type,
            )
            stats = get_offload_stats(model) or {}
            evaluation = evaluate_step_output(step, response)
            cumulative_step_latency_s += latency
            attempt = {
                "attempt": attempt_idx,
                "latency_s": round(latency, 2),
                "decode_hit_rate": round(stats.get("decode_hit_rate", 0), 4),
                "response_len_words": len(strip_thinking(response).split()),
                "passed": evaluation["passed"],
                "failures": evaluation["failures"],
                "preserve_hit_rate": evaluation["preserve_hit_rate"],
                "features": {
                    "imports": evaluation["features"]["imports"],
                    "classes": evaluation["features"]["classes"],
                    "functions": evaluation["features"]["functions"],
                    "tests": evaluation["features"]["tests"],
                },
                "code_preview": evaluation["features"]["code"][:400],
            }
            best_attempt = attempt
            if evaluation["passed"]:
                memory_features = merge_memory_features(
                    memory_features, evaluation["features"]
                )
                break

            prior_failures = evaluation["failures"]
            prior_attempt_code = evaluation["features"]["code"]
            prior_attempt_excerpt = build_code_excerpt(
                evaluation["features"]["code"],
                line_no=syntax_error_line_number(evaluation["failures"]),
            )
            flush_offload_cache(model)
            reset_offload_stats(model)

        assert best_attempt is not None
        step_failures = list(best_attempt["failures"])
        if (
            max_step_latency_s is not None
            and cumulative_step_latency_s > max_step_latency_s
        ):
            step_failures.append(
                f"Step latency budget exceeded: {cumulative_step_latency_s:.2f}s > {max_step_latency_s:.2f}s"
            )
        step_entry = {
            "step": step["name"],
            "expected_symbols": step.get("expected_symbols", []),
            "must_preserve_symbols": step.get("must_preserve_symbols", []),
            "memory_mode": memory_mode,
            "saved_memory_note": saved_memory if memory_mode == "scratchpad" else None,
            "passed": best_attempt["passed"] and not step_failures,
            "attempts_used": best_attempt["attempt"],
            "attempts_allowed": repair_attempts + 1,
            "latency_s": round(cumulative_step_latency_s, 2),
            "final_attempt_latency_s": best_attempt["latency_s"],
            "decode_hit_rate": best_attempt["decode_hit_rate"],
            "preserve_hit_rate": best_attempt["preserve_hit_rate"],
            "failures": step_failures,
            "features": best_attempt["features"],
            "code_preview": best_attempt["code_preview"],
        }
        task_results.append(step_entry)

        if (
            max_task_latency_s is not None
            and (time.perf_counter() - task_started_at) > max_task_latency_s
        ):
            break

    n_passed = sum(1 for step in task_results if step["passed"])
    preserve_rates = [
        step["preserve_hit_rate"]
        for step in task_results
        if step["preserve_hit_rate"] is not None
    ]
    attempt_total = sum(step["attempts_used"] for step in task_results)
    total_latency_s = round(sum(step["latency_s"] for step in task_results), 2)
    time_budget_ok = (
        total_latency_s <= max_task_latency_s if max_task_latency_s is not None else None
    )
    aborted_for_time = (
        max_task_latency_s is not None
        and len(task_results) < len(task["steps"])
        and total_latency_s > max_task_latency_s
    )
    return {
        "name": task["name"],
        "description": task["description"],
        "steps": task_results,
        "summary": {
            "n_passed": n_passed,
            "n_total": len(task_results),
            "n_expected_total": len(task["steps"]),
            "pass_rate": round(n_passed / max(len(task_results), 1), 4),
            "avg_preserve_hit_rate": round(
                sum(preserve_rates) / max(len(preserve_rates), 1), 4
            )
            if preserve_rates
            else None,
            "total_attempts_used": attempt_total,
            "repair_turns_used": attempt_total - len(task_results),
            "total_latency_s": total_latency_s,
            "max_task_latency_s": max_task_latency_s,
            "time_budget_ok": time_budget_ok,
            "aborted_for_time": aborted_for_time,
        },
    }


def run_eval(
    *,
    model_path: str,
    expert_offload: bool,
    max_resident_experts: int | None,
    max_cached_shards: int | None,
    system_prompt: str | None,
    max_tokens: int,
    seed: int,
    temp: float,
    kv_cache_type: str,
    use_predictor: bool,
    memory_mode: str,
    memory_note_chars: int,
    repair_attempts: int,
    tasks: list[dict[str, Any]],
    memory_limit_mb: float | None = None,
    max_step_latency_s: float | None = None,
    max_task_latency_s: float | None = None,
    max_run_latency_s: float | None = None,
) -> tuple[bool, dict[str, Any]]:
    mx, _, load, _ = _runtime()
    mx.random.seed(seed)
    run_started_at = time.perf_counter()

    old_mem_limit = None
    if memory_limit_mb is not None:
        old_mem_limit = mx.set_memory_limit(int(memory_limit_mb * 1024 * 1024))
        print(f"Memory limit set to {memory_limit_mb:.0f} MB")
    mx.reset_peak_memory()
    system_before_load = _system_snapshot()

    print(
        f"Loading model from {model_path} "
        f"(expert_offload={expert_offload}, kv_cache_type={kv_cache_type}, "
        f"use_predictor={use_predictor}, memory_mode={memory_mode})..."
    )
    load_started_at = time.perf_counter()
    model, tokenizer = load(
        model_path,
        model_config=build_model_config(
            expert_offload,
            max_resident_experts,
            max_cached_shards=max_cached_shards,
            use_predictor=use_predictor,
        ),
    )
    load_latency_s = round(time.perf_counter() - load_started_at, 2)
    system_after_load = _system_snapshot()

    all_results: list[dict[str, Any]] = []
    for task in tasks:
        print(f"\n=== Agent Memory Task: {task['name']} ===")
        task_result = run_task(
            model,
            tokenizer,
            task,
            memory_mode=memory_mode,
            memory_note_chars=memory_note_chars,
            repair_attempts=repair_attempts,
            max_tokens=max_tokens,
            temp=temp,
            kv_cache_type=kv_cache_type,
            system_prompt=system_prompt,
            max_step_latency_s=max_step_latency_s,
            max_task_latency_s=max_task_latency_s,
        )
        all_results.append(task_result)
        summary = task_result["summary"]
        print(
            f"  pass_rate={summary['pass_rate']:.0%} "
            f"avg_preserve={summary['avg_preserve_hit_rate']} "
            f"repair_turns={summary['repair_turns_used']}"
        )

    peak_memory_mb = round(mx.get_peak_memory() / (1024 * 1024), 1)
    system_at_end = _system_snapshot()
    fits_cap = peak_memory_mb <= memory_limit_mb if memory_limit_mb is not None else None
    total_run_latency_s = round(time.perf_counter() - run_started_at, 2)
    run_latency_ok = (
        total_run_latency_s <= max_run_latency_s if max_run_latency_s is not None else None
    )

    task_passes = [task["summary"]["n_passed"] == task["summary"]["n_total"] for task in all_results]
    overall_passed = (
        all(task_passes)
        and (fits_cap is not False)
        and (run_latency_ok is not False)
    )
    avg_task_pass = sum(
        task["summary"]["pass_rate"] for task in all_results
    ) / max(len(all_results), 1)
    avg_preserve = [
        task["summary"]["avg_preserve_hit_rate"]
        for task in all_results
        if task["summary"]["avg_preserve_hit_rate"] is not None
    ]
    total_repairs = sum(task["summary"]["repair_turns_used"] for task in all_results)

    if old_mem_limit is not None:
        mx.set_memory_limit(old_mem_limit)

    artifact = {
        "version": 1,
        "model_path": model_path,
        "expert_offload": expert_offload,
        "max_resident_experts": max_resident_experts,
        "max_cached_shards": max_cached_shards,
        "kv_cache_type": kv_cache_type,
        "use_predictor": use_predictor,
        "memory_mode": memory_mode,
        "memory_note_chars": memory_note_chars,
        "repair_attempts": repair_attempts,
        "max_step_latency_s": max_step_latency_s,
        "max_task_latency_s": max_task_latency_s,
        "max_run_latency_s": max_run_latency_s,
        "seed": seed,
        "temp": temp,
        "max_tokens": max_tokens,
        "memory": {
            "peak_mb": peak_memory_mb,
            "cap_mb": memory_limit_mb,
            "capped": memory_limit_mb is not None,
            "fits_cap": fits_cap,
        },
        "system_prompt": system_prompt,
        "system_before_load": system_before_load,
        "system_after_load": system_after_load,
        "tasks": all_results,
        "summary": {
            "n_tasks": len(all_results),
            "n_tasks_fully_passed": sum(1 for passed in task_passes if passed),
            "avg_task_pass_rate": round(avg_task_pass, 4),
            "avg_preserve_hit_rate": round(
                sum(avg_preserve) / max(len(avg_preserve), 1), 4
            )
            if avg_preserve
            else None,
            "total_repair_turns_used": total_repairs,
            "load_latency_s": load_latency_s,
            "total_run_latency_s": total_run_latency_s,
            "run_latency_ok": run_latency_ok,
            "overall_passed": overall_passed,
        },
        "system_at_end": system_at_end,
        "system_deltas": {
            "rss_mb_load_delta": _delta(
                system_after_load.get("process_rss_mb"),
                system_before_load.get("process_rss_mb"),
            ),
            "rss_mb_total_delta": _delta(
                system_at_end.get("process_rss_mb"),
                system_before_load.get("process_rss_mb"),
            ),
            "swap_used_mb_delta": _delta(
                system_at_end.get("swap_used_mb"),
                system_before_load.get("swap_used_mb"),
            ),
            "vm_pages_paged_out_delta": _delta(
                system_at_end.get("vm_pages_paged_out"),
                system_before_load.get("vm_pages_paged_out"),
            ),
        },
    }
    return overall_passed, artifact


def list_tasks_json() -> list[dict[str, Any]]:
    return [
        {
            "name": task["name"],
            "description": task["description"],
            "steps": [step["name"] for step in task["steps"]],
        }
        for task in TASKS
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Agent-memory coding evaluator for KV backend comparisons"
    )
    parser.add_argument("--model", type=str, help="Path to the model under evaluation.")
    parser.add_argument("--task", type=str, default="all", help="Task name or 'all'.")
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print available tasks as JSON and exit.",
    )
    parser.add_argument("--expert-offload", action="store_true")
    parser.add_argument("--max-resident-experts", type=int, default=None)
    parser.add_argument("--max-cached-shards", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--kv-cache-type", type=str, default="default")
    parser.add_argument(
        "--memory-mode",
        type=str,
        default="scratchpad",
        choices=["scratchpad", "none"],
    )
    parser.add_argument("--memory-note-chars", type=int, default=1200)
    parser.add_argument("--repair-attempts", type=int, default=1)
    parser.add_argument("--max-step-latency-s", type=float, default=None)
    parser.add_argument("--max-task-latency-s", type=float, default=None)
    parser.add_argument("--max-run-latency-s", type=float, default=None)
    parser.add_argument("--memory-limit-mb", type=float, default=None)
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--use-predictor", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    if args.list_tasks:
        print(json.dumps(list_tasks_json(), indent=2))
        return

    if not args.model:
        parser.error("--model is required unless --list-tasks is used")

    if args.task == "all":
        tasks = TASKS
    else:
        if args.task not in TASK_INDEX:
            parser.error(f"Unknown task {args.task!r}")
        tasks = [TASK_INDEX[args.task]]

    passed, artifact = run_eval(
        model_path=args.model,
        expert_offload=args.expert_offload,
        max_resident_experts=args.max_resident_experts,
        max_cached_shards=args.max_cached_shards,
        system_prompt=args.system_prompt,
        max_tokens=args.max_tokens,
        seed=args.seed,
        temp=args.temp,
        kv_cache_type=args.kv_cache_type,
        use_predictor=args.use_predictor,
        memory_mode=args.memory_mode,
        memory_note_chars=args.memory_note_chars,
        repair_attempts=args.repair_attempts,
        tasks=tasks,
        memory_limit_mb=args.memory_limit_mb,
        max_step_latency_s=args.max_step_latency_s,
        max_task_latency_s=args.max_task_latency_s,
        max_run_latency_s=args.max_run_latency_s,
    )

    print("\n=== Summary ===")
    print(json.dumps(artifact["summary"], indent=2))
    if artifact["memory"]["capped"]:
        print(
            f"Memory: peak={artifact['memory']['peak_mb']:.0f} MB / "
            f"cap={artifact['memory']['cap_mb']:.0f} MB -> "
            f"{'PASS' if artifact['memory']['fits_cap'] else 'FAIL'}"
        )

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(artifact, f, indent=2)
        print(f"Wrote artifact: {out}")

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
