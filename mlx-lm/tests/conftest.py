# Copyright © 2024 Apple Inc.
"""Test-wide defaults: 5-minute ceiling per test when pytest-timeout is installed."""

from __future__ import annotations

import importlib.util

import pytest

if importlib.util.find_spec("pytest_timeout") is None:
    raise ImportError(
        "mlx-lm tests require pytest-timeout (5 min ceiling per test). "
        "From mlx-lm/: pip install -e '.[test]'"
    )


def pytest_collection_modifyitems(config, items) -> None:
    mark = pytest.mark.timeout(300)
    for item in items:
        item.add_marker(mark)
