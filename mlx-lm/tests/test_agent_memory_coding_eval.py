import importlib.util
from pathlib import Path


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "agent_memory_coding_eval.py"
    )
    spec = importlib.util.spec_from_file_location("agent_memory_coding_eval", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


mod = _load_module()


def test_extract_python_payload_prefers_fenced_block():
    raw = """
Some explanation.

```python
import math

def square(x: int) -> int:
    return x * x
```
"""
    code = mod.extract_python_payload(raw)
    assert "def square" in code
    assert "Some explanation" not in code


def test_extract_code_features_collects_symbols_and_tests():
    raw = """
```python
import pytest
from typing import List

class Helper:
    pass

def merge_sorted(left: list[int], right: list[int]) -> list[int]:
    return left + right

def test_merge_sorted_simple():
    assert merge_sorted([1], [2]) == [1, 2]
```
"""
    features = mod.extract_code_features(raw)
    assert "import pytest" in features["imports"]
    assert "Helper" in features["classes"]
    assert "merge_sorted" in features["functions"]
    assert "test_merge_sorted_simple" in features["tests"]
    assert "merge_sorted" in features["symbols"]


def test_merge_memory_features_keeps_prior_symbols():
    prev = {
        "code": "",
        "imports": ["import csv"],
        "classes": [],
        "functions": ["parse_rows"],
        "tests": [],
        "symbols": ["parse_rows"],
        "skeleton_lines": ["import csv", "def parse_rows(text: str):"],
    }
    current = {
        "code": "",
        "imports": ["import pytest"],
        "classes": [],
        "functions": ["filter_active"],
        "tests": ["test_filter_active_only_active"],
        "symbols": ["filter_active"],
        "skeleton_lines": [
            "import pytest",
            "def filter_active(rows):",
            "def test_filter_active_only_active():",
        ],
    }
    merged = mod.merge_memory_features(prev, current)
    assert merged["imports"] == ["import csv", "import pytest"]
    assert merged["functions"] == ["parse_rows", "filter_active"]
    assert merged["tests"] == ["test_filter_active_only_active"]
    assert "parse_rows" in merged["symbols"]


def test_evaluate_step_output_passes_valid_python_module():
    step = {
        "expected_symbols": ["parse_rows", "filter_active"],
        "must_preserve_symbols": ["parse_rows"],
        "required_import_substrings": ["import csv"],
        "required_test_names": ["test_filter_active_only_active"],
        "min_test_count": 1,
        "min_words": 20,
    }
    raw = """
```python
import csv
import pytest

def parse_rows(text: str) -> list[dict]:
    return []

def filter_active(rows):
    return rows

def test_filter_active_only_active():
    assert filter_active([1]) == [1]
```
"""
    result = mod.evaluate_step_output(step, raw)
    assert result["passed"] is True
    assert result["preserve_hit_rate"] == 1.0


def test_evaluate_step_output_fails_missing_preserved_symbol_and_parse_error():
    step = {
        "expected_symbols": ["merge_sorted", "merge_sort"],
        "must_preserve_symbols": ["merge_sorted"],
        "min_words": 5,
    }
    raw = """
```python
def merge_sort(values)
    return values
```
"""
    result = mod.evaluate_step_output(step, raw)
    assert result["passed"] is False
    assert any("Did not preserve prior symbol" in failure for failure in result["failures"])
    assert any("SyntaxError" in failure for failure in result["failures"])


def test_evaluate_step_output_fails_stub_markers():
    step = {
        "expected_symbols": ["merge_sorted"],
        "min_words": 5,
    }
    raw = """
```python
def merge_sorted(left, right):
    raise NotImplementedError("todo")
```
"""
    result = mod.evaluate_step_output(step, raw)
    assert result["passed"] is False
    assert any("Stub marker detected" in failure for failure in result["failures"])


def test_build_step_prompt_uses_memory_note_and_feedback():
    task = mod.TASK_INDEX["merge_sort_module"]
    step = task["steps"][1]
    prompt = mod.build_step_prompt(
        task,
        step,
        memory_mode="scratchpad",
        memory_note="Saved external memory from prior turns:\n- functions: merge_sorted",
        prior_failures=["Missing expected symbol: 'merge_sort'"],
        prior_attempt_code="def merge_sorted(left, right):\n    return left + right\n",
        prior_attempt_excerpt="  1: def merge_sorted(left, right):\n  2:     return left + right",
    )
    assert "Saved external memory" in prompt
    assert "Missing expected symbol" in prompt
    assert "merge_sorted" in prompt
    assert "Relevant excerpt from your previous module attempt" in prompt
    assert "Keep the module compact" in prompt


def test_build_code_excerpt_centers_on_syntax_line():
    code = "\n".join(
        [
            "def merge_sorted(left, right):",
            "    return left + right",
            "",
            "def merge_sort(values)",
            "    return values",
            "",
            "def test_merge_sort_empty():",
            "    assert merge_sort([]) == []",
        ]
    )
    excerpt = mod.build_code_excerpt(code, line_no=4, radius=1)
    assert "  3:" in excerpt
    assert "  4: def merge_sort(values)" in excerpt
    assert "  5:     return values" in excerpt
