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
