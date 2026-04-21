#!/usr/bin/env python3
"""Unit tests for compare_mojo_vs_mlx.py logic (no matplotlib required)."""

import sys
import os

# Add scripts to path
sys.path.insert(0, os.path.dirname(__file__))

# Mock matplotlib to allow import without installation
import types

sys.modules["matplotlib"] = types.ModuleType("matplotlib")
sys.modules["matplotlib.pyplot"] = types.ModuleType("matplotlib.pyplot")
sys.modules["matplotlib.patches"] = types.ModuleType("matplotlib.patches")
sys.modules["matplotlib"].rcParams = {"update": lambda x: None}

# Now we can import our module
from compare_mojo_vs_mlx import (
    KernelResult,
    match_kernels,
    cohens_d,
    geometric_mean_speedup,
)


def test_match_kernels():
    """Test kernel matching logic."""
    mlx_results = [
        KernelResult(
            "mlx", "matmul", "1024x1024x1024", "float16", 100, 100, 5, 95, 105
        ),
        KernelResult("mlx", "softmax", "8x48x128", "float16", 50, 50, 2, 48, 52),
        KernelResult("mlx", "rope", "1x48x128x128", "float16", 30, 30, 1, 29, 31),
    ]

    mojo_results = [
        KernelResult("mojo", "matmul", "1024x1024x1024", "float16", 80, 80, 4, 76, 84),
        KernelResult("mojo", "softmax", "8x48x128", "float16", 45, 45, 2, 43, 47),
        KernelResult("mojo", "different", "1x1x1", "float16", 10, 10, 1, 9, 11),
    ]

    matches = match_kernels(mlx_results, mojo_results)

    assert len(matches) == 2, f"Expected 2 matches, got {len(matches)}"
    assert matches[0][0].name == "matmul"
    assert matches[1][0].name == "softmax"

    print("✓ test_match_kernels passed")


def test_cohens_d():
    """Test Cohen's d calculation."""
    # Equal groups should give d ≈ 0
    d = cohens_d(100, 10, 20, 100, 10, 20)
    assert abs(d) < 0.01, f"Equal groups should have d≈0, got {d}"

    # MLX slower (higher mean) should give positive d
    d = cohens_d(100, 10, 20, 80, 10, 20)
    assert d > 0, f"MLX slower should give positive d, got {d}"

    # MLX faster (lower mean) should give negative d
    d = cohens_d(80, 10, 20, 100, 10, 20)
    assert d < 0, f"MLX faster should give negative d, got {d}"

    # Zero std should not crash
    d = cohens_d(100, 0, 20, 100, 0, 20)
    assert d == 0.0

    print("✓ test_cohens_d passed")


def test_geometric_mean_speedup():
    """Test geometric mean calculation."""
    # All 1.0 should give 1.0
    gm = geometric_mean_speedup([1.0, 1.0, 1.0])
    assert abs(gm - 1.0) < 0.01

    # 2x and 0.5x should give ~1.0
    gm = geometric_mean_speedup([2.0, 0.5])
    assert abs(gm - 1.0) < 0.01

    # Consistent 2x speedup
    gm = geometric_mean_speedup([2.0, 2.0, 2.0])
    assert abs(gm - 2.0) < 0.01

    # Empty list
    gm = geometric_mean_speedup([])
    assert gm == 1.0

    print("✓ test_geometric_mean_speedup passed")


def test_parse_kernel_results():
    """Test JSON parsing."""
    from compare_mojo_vs_mlx import parse_kernel_results

    sample_json = {
        "framework": "mlx",
        "kernels": [
            {
                "name": "matmul",
                "shape": "1024x1024x1024",
                "dtype": "float16",
                "statistics": {
                    "median_us": 123.4,
                    "mean_us": 125.0,
                    "std_us": 5.0,
                    "ci95_bca": [120.0, 130.0],
                },
                "throughput": {
                    "tflops": 17.5,
                    "gbs": 1200.0,
                    "roofline_pct": 85.2,
                },
            }
        ],
    }

    results = parse_kernel_results(sample_json)
    assert len(results) == 1
    assert results[0].framework == "mlx"
    assert results[0].name == "matmul"
    assert results[0].median_us == 123.4
    assert results[0].tflops == 17.5

    print("✓ test_parse_kernel_results passed")


if __name__ == "__main__":
    print("Running compare_mojo_vs_mlx unit tests...")
    test_match_kernels()
    test_cohens_d()
    test_geometric_mean_speedup()
    test_parse_kernel_results()
    print("\n✅ All tests passed!")
