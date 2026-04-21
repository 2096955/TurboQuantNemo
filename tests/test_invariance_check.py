"""Unit tests for scripts/invariance_check.py helpers."""

import sys
from pathlib import Path

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import invariance_check as ic


class TestOutputHash:
    def test_identical_strings_hash_equal(self):
        assert ic.output_hash("hello world") == ic.output_hash("hello world")

    def test_different_strings_hash_differ(self):
        assert ic.output_hash("hello") != ic.output_hash("world")

    def test_empty_string_hashes(self):
        assert isinstance(ic.output_hash(""), str)
        assert len(ic.output_hash("")) == 64  # sha256 hex


class TestDiffFirstChars:
    def test_identical_inputs_returns_empty_dict(self):
        assert ic._diff_first_chars("abc", "abc") == {}

    def test_returns_first_diff_index(self):
        result = ic._diff_first_chars("abcXdef", "abcYdef")
        assert result["first_diff_index"] == 3

    def test_includes_context_around_diff(self):
        result = ic._diff_first_chars("0123456789Xabcdefghij", "0123456789Yabcdefghij")
        assert "0123456789X" in result["context_a"]
        assert "0123456789Y" in result["context_b"]

    def test_length_mismatch_when_one_is_prefix(self):
        result = ic._diff_first_chars("abc", "abcdef")
        assert result["length_mismatch"] == [3, 6]


class TestExtractGeneratedText:
    def test_extracts_text_between_delimiters(self):
        stdout = (
            "Fetching files: 100%\n"
            "==========\n"
            "Paris.\n"
            "==========\n"
            "Prompt: 40 tokens, 268.751 tokens-per-sec\n"
            "Peak memory: 1.840 GB\n"
        )
        assert ic._extract_generated_text(stdout) == "Paris."

    def test_strips_timing_footer_invariant_to_perf_jitter(self):
        run1 = "==========\nHello\n==========\nPrompt: 5 tokens, 268 tps\n"
        run2 = "==========\nHello\n==========\nPrompt: 5 tokens, 497 tps\n"
        assert ic._extract_generated_text(run1) == ic._extract_generated_text(run2)

    def test_raises_when_delimiters_missing(self):
        import pytest

        with pytest.raises(ValueError, match="missing expected"):
            ic._extract_generated_text("just plain text, no delimiters")
