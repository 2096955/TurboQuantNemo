#!/usr/bin/env python3
"""Compare Mojo vs MLX kernel benchmark results.

Reads JSON outputs from both frameworks + roofline calibration.
Produces LaTeX tables, publication-quality charts, and summary JSON.

Usage:
    python scripts/compare_mojo_vs_mlx.py \
      --mlx results/mlx_kernels.json \
      --mojo mojo-bench/results/mojo_kernels.json \
      --roofline results/roofline_m4max.json \
      --output-dir results/comparison/
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update(
    {
        "font.size": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "serif",
    }
)

# Kernel invocation counts per decode step, used only to build an illustrative
# kernel-mix chart. Without end-to-end decode measurements and representative
# per-model shapes, these weights should not be interpreted as wall-clock decode
# fractions.
INVOCATIONS_PER_DECODE = {
    "matmul": 6,  # Q, K, V, O, up, down
    "softmax": 1,
    "rope": 1,
    "rotate_forward": 1,
    "rotate_inverse": 1,
    "kv_compress": 1,
    "fused_attention": 1,
}


@dataclass
class KernelResult:
    """Unified kernel result representation."""

    framework: str
    name: str
    shape: str
    dtype: str
    median_us: float
    mean_us: float
    std_us: float
    ci95_lo: float
    ci95_hi: float
    n_iterations: int = 20
    tflops: float | None = None
    gbs: float | None = None
    roofline_pct: float | None = None
    flops_total: int | None = None
    bytes_accessed: int | None = None


# ---------------------------------------------------------------------------
# Load & match
# ---------------------------------------------------------------------------


def load_results(path: str) -> dict[str, Any]:
    """Load benchmark JSON results.

    Supports three formats:
    - Single aggregated JSON file with a "kernels" list (MLX output)
    - Single per-shape JSON file with "kernel" key (single Mojo result)
    - Directory of per-shape JSON files (Mojo output directory)
    """
    if os.path.isdir(path):
        # Mojo writes one JSON per kernel/shape — aggregate them
        entries = []
        framework = "mojo"
        framework_version = ""
        for fname in sorted(os.listdir(path)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(path, fname)) as f:
                entry = json.load(f)
            framework = entry.get("framework", framework)
            framework_version = entry.get("framework_version", framework_version)
            entries.append(entry)
        return {
            "framework": framework,
            "framework_version": framework_version,
            "kernels": entries,
        }
    with open(path) as f:
        data = json.load(f)
    # Single Mojo per-shape file (has "kernel" key, no "kernels" list)
    if "kernel" in data and "kernels" not in data:
        return {
            "framework": data.get("framework", "mojo"),
            "framework_version": data.get("framework_version", ""),
            "kernels": [data],
        }
    return data


def _normalize_kernel_entry(k: dict[str, Any]) -> tuple[str, str, dict, dict]:
    """Extract (name, shape, stats_dict, throughput_dict) from a kernel entry.

    Handles both MLX format (name/shape/stats) and Mojo format (kernel/shape/stats).
    """
    name = k.get("name") or k.get("kernel", "unknown")
    shape = k.get("shape", "")
    # Both MLX and Mojo write "stats"; parser previously had a typo ("statistics")
    stats = k.get("stats") or k.get("statistics", {})
    throughput = k.get("throughput", {})
    return name, shape, stats, throughput


def parse_kernel_results(data: dict[str, Any]) -> list[KernelResult]:
    """Parse kernel results from framework JSON into KernelResult objects."""
    framework = data["framework"]
    results = []

    for k in data.get("kernels", []):
        name, shape, stats, throughput = _normalize_kernel_entry(k)

        # Extract CI bounds
        ci95_bca = stats.get(
            "ci95_bca", [stats.get("p5_us", 0), stats.get("p95_us", 0)]
        )
        ci95_lo = ci95_bca[0] if len(ci95_bca) > 0 else stats.get("p5_us", 0)
        ci95_hi = ci95_bca[1] if len(ci95_bca) > 1 else stats.get("p95_us", 0)

        results.append(
            KernelResult(
                framework=framework,
                name=name,
                shape=shape,
                dtype=k.get("dtype", "float32"),
                median_us=stats["median_us"],
                mean_us=stats["mean_us"],
                std_us=stats["std_us"],
                ci95_lo=ci95_lo,
                ci95_hi=ci95_hi,
                n_iterations=stats.get("n_iterations", 20),
                tflops=throughput.get("tflops"),
                gbs=throughput.get("gbs"),
                roofline_pct=throughput.get("roofline_pct"),
                flops_total=throughput.get("flops_total"),
                bytes_accessed=throughput.get("bytes_accessed"),
            )
        )

    return results


def classify_kernel_family(name: str) -> str:
    """Normalize a benchmark kernel name into a family used for aggregation."""
    for family in INVOCATIONS_PER_DECODE:
        if name == family or name.startswith(f"{family}_") or family in name:
            return family
    return name


def match_kernels(
    mlx_results: list[KernelResult], mojo_results: list[KernelResult]
) -> list[tuple[KernelResult, KernelResult]]:
    """Find kernels with matching names and shapes across both frameworks.

    Returns list of (mlx_result, mojo_result) pairs.
    """
    matches: list[tuple[KernelResult, KernelResult]] = []

    # First pass: exact match (name + shape + dtype)
    unmatched_mlx: list[KernelResult] = []
    used_mojo: set[int] = set()
    for mlx in mlx_results:
        found = False
        for i, mojo in enumerate(mojo_results):
            if i in used_mojo:
                continue
            if (
                mlx.name == mojo.name
                and mlx.shape == mojo.shape
                and mlx.dtype == mojo.dtype
            ):
                matches.append((mlx, mojo))
                used_mojo.add(i)
                found = True
                break
        if not found:
            unmatched_mlx.append(mlx)

    # Second pass: relaxed match (name + shape), tolerating dtype mismatch.
    # This is necessary because some benchmark outputs are fp16 vs fp32 across
    # frameworks while still representing the same kernel/shape workload.
    for mlx in unmatched_mlx:
        for i, mojo in enumerate(mojo_results):
            if i in used_mojo:
                continue
            if mlx.name == mojo.name and mlx.shape == mojo.shape:
                matches.append((mlx, mojo))
                used_mojo.add(i)
                break

    return matches


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def cohens_d(
    g1_mean: float, g1_std: float, g1_n: int, g2_mean: float, g2_std: float, g2_n: int
) -> float:
    """Cohen's d effect size for two groups.

    Positive d means g1 > g2 (MLX slower if g1=MLX).
    """
    if g1_std == 0 and g2_std == 0:
        return 0.0

    pooled_std = math.sqrt(
        ((g1_n - 1) * g1_std**2 + (g2_n - 1) * g2_std**2) / (g1_n + g2_n - 2)
    )
    if pooled_std == 0:
        return 0.0

    return (g1_mean - g2_mean) / pooled_std


def geometric_mean_ratio(values: list[float]) -> float:
    """Geometric mean of positive ratios."""
    if not values:
        return 1.0

    # Use log-space to avoid overflow
    log_product = sum(math.log(max(v, 1e-10)) for v in values)
    return math.exp(log_product / len(values))


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------


def generate_latex_tables(
    matches: list[tuple[KernelResult, KernelResult]], output_dir: Path
) -> None:
    """Generate LaTeX tables per kernel category."""

    # Group by kernel name prefix
    categories = {}
    for mlx, mojo in matches:
        category = mlx.name.split("_")[0] if "_" in mlx.name else mlx.name
        if category not in categories:
            categories[category] = []
        categories[category].append((mlx, mojo))

    for category, pairs in categories.items():
        lines = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(f"\\caption{{{category.title()} Kernel Comparison}}")
        lines.append(r"\begin{tabular}{lrrrrr}")
        lines.append(r"\toprule")
        lines.append(
            r"Shape & MLX ($\mu$s) & Mojo ($\mu$s) & MLX TFLOPS & Mojo TFLOPS & MLX Speedup \\"
        )
        lines.append(r"\midrule")

        for mlx, mojo in sorted(pairs, key=lambda p: p[0].shape):
            mlx_tflops = mlx.tflops if mlx.tflops else 0
            mojo_tflops = mojo.tflops if mojo.tflops else 0
            mlx_speedup = mojo.median_us / mlx.median_us if mlx.median_us > 0 else 0

            lines.append(
                f"{mlx.shape} & {mlx.median_us:.1f} & {mojo.median_us:.1f} & "
                f"{mlx_tflops:.2f} & {mojo_tflops:.2f} & {mlx_speedup:.2f}x \\\\"
            )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        out_path = output_dir / f"table_{category}.tex"
        with open(out_path, "w") as f:
            f.write("\n".join(lines))

        print(f"Wrote LaTeX table: {out_path}")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def generate_roofline_plot(
    matches: list[tuple[KernelResult, KernelResult]],
    roofline_data: dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate roofline plot with both frameworks."""

    fig, ax = plt.subplots(figsize=(10, 6))

    peak_tflops = roofline_data["calibration"]["peak_fp16_tflops"]
    peak_bw_gbs = roofline_data["calibration"]["peak_memory_bandwidth_gbs"]

    # Compute arithmetic intensities
    mlx_ai = []
    mlx_tflops = []
    mojo_ai = []
    mojo_tflops = []

    for mlx, mojo in matches:
        if mlx.flops_total and mlx.bytes_accessed and mlx.bytes_accessed > 0:
            ai = mlx.flops_total / mlx.bytes_accessed
            mlx_ai.append(ai)
            mlx_tflops.append(mlx.tflops if mlx.tflops else 0)

        if mojo.flops_total and mojo.bytes_accessed and mojo.bytes_accessed > 0:
            ai = mojo.flops_total / mojo.bytes_accessed
            mojo_ai.append(ai)
            mojo_tflops.append(mojo.tflops if mojo.tflops else 0)

    # Plot roofline ceiling
    ai_range = np.logspace(-2, 3, 100)
    roofline_ceiling = np.minimum(peak_tflops, peak_bw_gbs * ai_range / 1000)
    ax.plot(
        ai_range,
        roofline_ceiling,
        "k--",
        linewidth=2,
        label="Hardware Ceiling",
        alpha=0.7,
    )

    # Plot kernel results
    if mlx_ai:
        ax.scatter(
            mlx_ai, mlx_tflops, c="blue", marker="o", s=50, alpha=0.7, label="MLX"
        )
    if mojo_ai:
        ax.scatter(
            mojo_ai, mojo_tflops, c="red", marker="^", s=50, alpha=0.7, label="Mojo"
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=12)
    ax.set_ylabel("Throughput (TFLOPS)", fontsize=12)
    ax.set_title("Roofline Plot: MLX vs Mojo", fontsize=14, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out_path = output_dir / "roofline_plot.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Wrote roofline plot: {out_path}")


def generate_roofline_fraction_bars(
    matches: list[tuple[KernelResult, KernelResult]], output_dir: Path
) -> None:
    """Generate side-by-side bar chart of % of peak achieved."""

    # Extract data
    labels = []
    mlx_pcts = []
    mojo_pcts = []

    for mlx, mojo in matches[:15]:  # Limit to first 15 for readability
        labels.append(f"{mlx.name}\n{mlx.shape}")
        mlx_pcts.append(mlx.roofline_pct if mlx.roofline_pct else 0)
        mojo_pcts.append(mojo.roofline_pct if mojo.roofline_pct else 0)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, mlx_pcts, width, label="MLX", color="blue", alpha=0.7)
    ax.bar(x + width / 2, mojo_pcts, width, label="Mojo", color="red", alpha=0.7)

    ax.set_ylabel("% of Peak Performance", fontsize=12)
    ax.set_title(
        "Roofline Efficiency: Achieved % of Peak", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "roofline_fraction_bars.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Wrote roofline fraction bars: {out_path}")


def generate_throughput_bars(
    matches: list[tuple[KernelResult, KernelResult]], output_dir: Path
) -> None:
    """Generate absolute throughput bar charts per kernel."""

    # Group by kernel name
    by_kernel = {}
    for mlx, mojo in matches:
        if mlx.name not in by_kernel:
            by_kernel[mlx.name] = []
        by_kernel[mlx.name].append((mlx, mojo))

    for kernel_name, pairs in by_kernel.items():
        if not pairs:
            continue

        labels = []
        mlx_throughputs = []
        mojo_throughputs = []

        for mlx, mojo in pairs[:10]:  # Max 10 per chart
            labels.append(mlx.shape)
            mlx_throughputs.append(mlx.tflops if mlx.tflops else 0)
            mojo_throughputs.append(mojo.tflops if mojo.tflops else 0)

        if not labels:
            continue

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(
            x - width / 2, mlx_throughputs, width, label="MLX", color="blue", alpha=0.7
        )
        ax.bar(
            x + width / 2, mojo_throughputs, width, label="Mojo", color="red", alpha=0.7
        )

        ax.set_ylabel("Throughput (TFLOPS)", fontsize=12)
        ax.set_title(
            f"{kernel_name}: Absolute Throughput", fontsize=14, fontweight="bold"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        safe_name = kernel_name.replace("/", "_")
        out_path = output_dir / f"throughput_bars_{safe_name}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Wrote throughput bars: {out_path}")


def generate_throughput_lines(
    matches: list[tuple[KernelResult, KernelResult]], output_dir: Path
) -> None:
    """Generate line plots of throughput vs matrix size for standard sweeps."""

    # Focus on matmul square sweeps (512, 1024, 2048, 4096, 8192)
    square_sizes = [
        "512x512x512",
        "1024x1024x1024",
        "2048x2048x2048",
        "4096x4096x4096",
        "8192x8192x8192",
    ]

    mlx_sweep = []
    mojo_sweep = []
    sizes = []

    for mlx, mojo in matches:
        if mlx.name == "matmul" and mlx.shape in square_sizes:
            size = int(mlx.shape.split("x")[0])
            sizes.append(size)
            mlx_sweep.append(mlx.tflops if mlx.tflops else 0)
            mojo_sweep.append(mojo.tflops if mojo.tflops else 0)

    if sizes:
        # Sort by size
        sorted_data = sorted(zip(sizes, mlx_sweep, mojo_sweep))
        sizes, mlx_sweep, mojo_sweep = zip(*sorted_data)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            sizes, mlx_sweep, "o-", color="blue", linewidth=2, markersize=8, label="MLX"
        )
        ax.plot(
            sizes,
            mojo_sweep,
            "^-",
            color="red",
            linewidth=2,
            markersize=8,
            label="Mojo",
        )

        ax.set_xlabel("Matrix Size (NxNxN)", fontsize=12)
        ax.set_ylabel("Throughput (TFLOPS)", fontsize=12)
        ax.set_title("MatMul Throughput vs Matrix Size", fontsize=14, fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        out_path = output_dir / "throughput_lines_matmul.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Wrote throughput lines: {out_path}")


def generate_heatmap(
    matches: list[tuple[KernelResult, KernelResult]], output_dir: Path
) -> None:
    """Generate heatmap of log2(MLX_time/Mojo_time)."""

    # Group by kernel type
    by_kernel = {}
    for mlx, mojo in matches:
        if mlx.name not in by_kernel:
            by_kernel[mlx.name] = []
        by_kernel[mlx.name].append((mlx, mojo))

    # Build matrix
    kernel_names = list(by_kernel.keys())[:15]  # Limit to 15 kernels
    max_shapes = max(len(by_kernel[k]) for k in kernel_names)

    matrix = np.zeros((len(kernel_names), max_shapes))
    shape_labels = []

    for i, kernel in enumerate(kernel_names):
        pairs = by_kernel[kernel][:max_shapes]
        for j, (mlx, mojo) in enumerate(pairs):
            if mojo.median_us > 0:
                ratio = mlx.median_us / mojo.median_us
                matrix[i, j] = math.log2(ratio)
            if i == 0:
                shape_labels.append(mlx.shape[:20])  # Truncate long shapes

    # Pad shape labels
    while len(shape_labels) < max_shapes:
        shape_labels.append("")

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)

    ax.set_xticks(np.arange(max_shapes))
    ax.set_yticks(np.arange(len(kernel_names)))
    ax.set_xticklabels(shape_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(kernel_names, fontsize=10)

    ax.set_xlabel("Shape", fontsize=12)
    ax.set_ylabel("Kernel", fontsize=12)
    ax.set_title(
        "Time Ratio: log2(MLX_time/Mojo_time)\nPositive=MLX slower, Negative=MLX faster",
        fontsize=14,
        fontweight="bold",
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("log2(MLX time / Mojo time)", fontsize=10)

    plt.tight_layout()
    out_path = output_dir / "heatmap_performance_ratio.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Wrote heatmap: {out_path}")


def generate_energy_chart(
    matches: list[tuple[KernelResult, KernelResult]],
    roofline_data: dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate TFLOPS/W comparison if framework-specific power data exists."""

    power_by_framework = roofline_data.get("power_baseline_by_framework", {})
    mlx_watts = power_by_framework.get("mlx")
    mojo_watts = power_by_framework.get("mojo")

    if not mlx_watts or not mojo_watts:
        print(
            "Skipping energy chart: need framework-specific power baselines for MLX and Mojo"
        )
        return

    # Compute energy efficiency
    labels = []
    mlx_efficiency = []
    mojo_efficiency = []

    for mlx, mojo in matches[:15]:
        labels.append(f"{mlx.name}\n{mlx.shape}")
        mlx_tflops = mlx.tflops if mlx.tflops else 0
        mojo_tflops = mojo.tflops if mojo.tflops else 0

        mlx_efficiency.append(mlx_tflops / mlx_watts if mlx_watts > 0 else 0)
        mojo_efficiency.append(mojo_tflops / mojo_watts if mojo_watts > 0 else 0)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, mlx_efficiency, width, label="MLX", color="blue", alpha=0.7)
    ax.bar(x + width / 2, mojo_efficiency, width, label="Mojo", color="red", alpha=0.7)

    ax.set_ylabel("TFLOPS/W", fontsize=12)
    ax.set_title(
        f"Energy Efficiency (MLX {mlx_watts:.1f}W, Mojo {mojo_watts:.1f}W)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "energy_efficiency.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Wrote energy chart: {out_path}")


def build_decode_kernel_mix(
    matches: list[tuple[KernelResult, KernelResult]]
) -> dict[str, Any]:
    """Build an illustrative kernel-mix summary from matched microbenchmarks."""
    per_kernel_samples: dict[str, dict[str, list[float]]] = {}

    for mlx, mojo in matches:
        family = classify_kernel_family(mlx.name)
        if family not in INVOCATIONS_PER_DECODE:
            continue
        if family not in per_kernel_samples:
            per_kernel_samples[family] = {"mlx": [], "mojo": []}
        per_kernel_samples[family]["mlx"].append(mlx.median_us)
        per_kernel_samples[family]["mojo"].append(mojo.median_us)

    if not per_kernel_samples:
        return {
            "available": False,
            "kind": "illustrative_kernel_mix",
            "note": "No supported kernel families were present in the matched benchmark set.",
        }

    representative_latency_us = {}
    mlx_weighted = {}
    mojo_weighted = {}

    for family, samples in per_kernel_samples.items():
        mlx_rep = geometric_mean_ratio(samples["mlx"])
        mojo_rep = geometric_mean_ratio(samples["mojo"])
        representative_latency_us[family] = {
            "mlx_us": round(mlx_rep, 3),
            "mojo_us": round(mojo_rep, 3),
        }
        weight = INVOCATIONS_PER_DECODE[family]
        mlx_weighted[family] = mlx_rep * weight
        mojo_weighted[family] = mojo_rep * weight

    mlx_total = sum(mlx_weighted.values())
    mojo_total = sum(mojo_weighted.values())

    return {
        "available": True,
        "kind": "illustrative_kernel_mix",
        "has_end_to_end_decode_reference": False,
        "note": (
            "Percentages are derived from representative per-kernel geometric-mean "
            "latencies multiplied by decode-step invocation counts across the "
            "matched microbenchmarks. They are not fractions of wall-clock decode time."
        ),
        "representative_latency_us": representative_latency_us,
        "mlx_relative_mix_pct": {
            family: round(100 * value / mlx_total, 2) if mlx_total > 0 else 0.0
            for family, value in mlx_weighted.items()
        },
        "mojo_relative_mix_pct": {
            family: round(100 * value / mojo_total, 2) if mojo_total > 0 else 0.0
            for family, value in mojo_weighted.items()
        },
    }


def generate_decode_attribution(
    matches: list[tuple[KernelResult, KernelResult]], output_dir: Path
) -> dict[str, Any]:
    """Generate an illustrative kernel-mix chart for the matched benchmark set."""
    kernel_mix = build_decode_kernel_mix(matches)
    if not kernel_mix.get("available"):
        print("Skipping decode attribution chart: no supported kernel families matched")
        return kernel_mix

    kernels = list(kernel_mix["mlx_relative_mix_pct"].keys())
    mlx_pcts = [kernel_mix["mlx_relative_mix_pct"][k] for k in kernels]
    mojo_pcts = [kernel_mix["mojo_relative_mix_pct"][k] for k in kernels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    bottom_mlx = 0.0
    for i, kernel in enumerate(kernels):
        ax1.bar("MLX", mlx_pcts[i], bottom=bottom_mlx, label=kernel)
        bottom_mlx += mlx_pcts[i]

    bottom_mojo = 0.0
    for i, kernel in enumerate(kernels):
        ax2.bar("Mojo", mojo_pcts[i], bottom=bottom_mojo, label=kernel)
        bottom_mojo += mojo_pcts[i]

    for ax in [ax1, ax2]:
        ax.set_ylabel("% of weighted matched-kernel cost", fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    ax1.set_title("MLX Illustrative Kernel Mix", fontsize=12, fontweight="bold")
    ax2.set_title("Mojo Illustrative Kernel Mix", fontsize=12, fontweight="bold")

    fig.text(
        0.5,
        0.02,
        kernel_mix["note"],
        ha="center",
        fontsize=9,
        color="dimgray",
    )

    plt.tight_layout(rect=(0, 0.06, 1, 1))
    out_path = output_dir / "decode_time_attribution.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Wrote kernel-mix chart: {out_path}")

    return kernel_mix


# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------


def write_summary_json(
    matches: list[tuple[KernelResult, KernelResult]],
    decode_kernel_mix: dict[str, Any],
    output_dir: Path,
) -> None:
    """Write summary JSON with explicit time-ratio and speedup semantics."""

    time_ratios = []
    cohens_d_results = {}

    for mlx, mojo in matches:
        if mojo.median_us > 0:
            time_ratio = mlx.median_us / mojo.median_us
            time_ratios.append(time_ratio)

            # Cohen's d using actual iteration counts from adaptive benchmarking
            d = cohens_d(
                mlx.mean_us,
                mlx.std_us,
                mlx.n_iterations,
                mojo.mean_us,
                mojo.std_us,
                mojo.n_iterations,
            )
            cohens_d_results[f"{mlx.name}_{mlx.shape}"] = round(d, 3)

    geo_time_ratio = geometric_mean_ratio(time_ratios)
    mlx_speedup = 1 / geo_time_ratio if geo_time_ratio > 0 else 0.0

    # Per-kernel time ratio (aggregate by kernel name)
    per_kernel = {}
    for mlx, mojo in matches:
        if mlx.name not in per_kernel:
            per_kernel[mlx.name] = []
        if mojo.median_us > 0:
            per_kernel[mlx.name].append(mlx.median_us / mojo.median_us)

    per_kernel_time_ratio = {
        k: round(geometric_mean_ratio(v), 3) for k, v in per_kernel.items()
    }
    per_kernel_mlx_speedup = {
        k: round(1 / ratio, 3) if ratio > 0 else 0.0
        for k, ratio in per_kernel_time_ratio.items()
    }
    exact_dtype_matches = sum(1 for mlx, mojo in matches if mlx.dtype == mojo.dtype)
    cross_dtype_matches = len(matches) - exact_dtype_matches

    summary = {
        "matching": {
            "policy": "exact_dtype_then_name_shape_fallback",
            "exact_dtype_matches": exact_dtype_matches,
            "cross_dtype_matches": cross_dtype_matches,
            "all_matches_cross_dtype": cross_dtype_matches == len(matches),
            "note": (
                "Cross-dtype matches compare the same kernel name and shape but can "
                "reflect different precisions across frameworks."
            ),
        },
        "geometric_mean_time_ratio_mlx_over_mojo": round(geo_time_ratio, 3),
        "geometric_mean_mlx_speedup_over_mojo": round(mlx_speedup, 3),
        "per_kernel_time_ratio_mlx_over_mojo": per_kernel_time_ratio,
        "per_kernel_mlx_speedup_over_mojo": per_kernel_mlx_speedup,
        "per_kernel_cohens_d": cohens_d_results,
        "decode_kernel_mix": decode_kernel_mix,
        "total_comparisons": len(matches),
    }

    out_path = output_dir / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote summary: {out_path}")
    print(f"\nGeometric mean time ratio (MLX/Mojo): {geo_time_ratio:.3f}x")
    print(f"Geometric mean MLX speedup over Mojo: {mlx_speedup:.3f}x")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compare Mojo vs MLX kernel benchmark results"
    )
    parser.add_argument(
        "--mlx",
        type=str,
        required=True,
        help="Path to MLX benchmark JSON",
    )
    parser.add_argument(
        "--mojo",
        type=str,
        required=True,
        help="Path to Mojo benchmark JSON file or results directory",
    )
    parser.add_argument(
        "--roofline",
        type=str,
        required=True,
        help="Path to roofline calibration JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/comparison/",
        help="Output directory for charts and tables",
    )
    args = parser.parse_args()

    # Load data
    print("Loading benchmark results...")
    mlx_data = load_results(args.mlx)
    mojo_data = load_results(args.mojo)
    roofline_data = load_results(args.roofline)

    mlx_results = parse_kernel_results(mlx_data)
    mojo_results = parse_kernel_results(mojo_data)

    print(f"Loaded {len(mlx_results)} MLX results, {len(mojo_results)} Mojo results")

    # Match kernels
    matches = match_kernels(mlx_results, mojo_results)
    print(f"Matched {len(matches)} common kernels")

    if not matches:
        print("ERROR: No matching kernels found between MLX and Mojo results")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate outputs
    print("\nGenerating LaTeX tables...")
    generate_latex_tables(matches, output_dir)

    print("\nGenerating charts...")
    generate_roofline_plot(matches, roofline_data, output_dir)
    generate_roofline_fraction_bars(matches, output_dir)
    generate_throughput_bars(matches, output_dir)
    generate_throughput_lines(matches, output_dir)
    generate_heatmap(matches, output_dir)
    generate_energy_chart(matches, roofline_data, output_dir)

    print("\nGenerating decode attribution...")
    decode_attribution = generate_decode_attribution(matches, output_dir)

    print("\nGenerating summary...")
    write_summary_json(matches, decode_attribution, output_dir)

    print(f"\nAll outputs written to {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
