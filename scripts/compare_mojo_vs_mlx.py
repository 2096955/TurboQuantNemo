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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
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

# Nemotron-H 120B layer count and invocations-per-decode-step for decode time attribution
NEMOTRON_LAYERS = 80
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
    tflops: float | None = None
    gbs: float | None = None
    roofline_pct: float | None = None
    flops_total: int | None = None
    bytes_accessed: int | None = None


# ---------------------------------------------------------------------------
# Load & match
# ---------------------------------------------------------------------------


def load_results(path: str) -> dict[str, Any]:
    """Load benchmark JSON results."""
    with open(path) as f:
        return json.load(f)


def parse_kernel_results(data: dict[str, Any]) -> list[KernelResult]:
    """Parse kernel results from framework JSON into KernelResult objects."""
    framework = data["framework"]
    results = []

    for k in data.get("kernels", []):
        stats = k["statistics"]
        throughput = k.get("throughput", {})

        # Extract CI bounds
        ci95_bca = stats.get(
            "ci95_bca", [stats.get("p5_us", 0), stats.get("p95_us", 0)]
        )
        ci95_lo = ci95_bca[0] if len(ci95_bca) > 0 else stats.get("p5_us", 0)
        ci95_hi = ci95_bca[1] if len(ci95_bca) > 1 else stats.get("p95_us", 0)

        results.append(
            KernelResult(
                framework=framework,
                name=k["name"],
                shape=k["shape"],
                dtype=k["dtype"],
                median_us=stats["median_us"],
                mean_us=stats["mean_us"],
                std_us=stats["std_us"],
                ci95_lo=ci95_lo,
                ci95_hi=ci95_hi,
                tflops=throughput.get("tflops"),
                gbs=throughput.get("gbs"),
                roofline_pct=throughput.get("roofline_pct"),
                flops_total=throughput.get("flops_total"),
                bytes_accessed=throughput.get("bytes_accessed"),
            )
        )

    return results


def match_kernels(
    mlx_results: list[KernelResult], mojo_results: list[KernelResult]
) -> list[tuple[KernelResult, KernelResult]]:
    """Find kernels with matching names and shapes across both frameworks.

    Returns list of (mlx_result, mojo_result) pairs.
    """
    matches = []

    for mlx in mlx_results:
        mlx_key = (mlx.name, mlx.shape, mlx.dtype)
        for mojo in mojo_results:
            mojo_key = (mojo.name, mojo.shape, mojo.dtype)
            if mlx_key == mojo_key:
                matches.append((mlx, mojo))
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


def geometric_mean_speedup(speedups: list[float]) -> float:
    """Geometric mean of speedup ratios."""
    if not speedups:
        return 1.0

    # Use log-space to avoid overflow
    log_product = sum(math.log(max(s, 1e-10)) for s in speedups)
    return math.exp(log_product / len(speedups))


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
            r"Shape & MLX ($\mu$s) & Mojo ($\mu$s) & MLX TFLOPS & Mojo TFLOPS & Speedup \\"
        )
        lines.append(r"\midrule")

        for mlx, mojo in sorted(pairs, key=lambda p: p[0].shape):
            mlx_tflops = mlx.tflops if mlx.tflops else 0
            mojo_tflops = mojo.tflops if mojo.tflops else 0
            speedup = mlx.median_us / mojo.median_us if mojo.median_us > 0 else 0

            lines.append(
                f"{mlx.shape} & {mlx.median_us:.1f} & {mojo.median_us:.1f} & "
                f"{mlx_tflops:.2f} & {mojo_tflops:.2f} & {speedup:.2f}x \\\\"
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
    """Generate heatmap of log2(MLX_time/Mojo_time).

    Zero = equal, positive = Mojo faster, negative = MLX faster.
    """

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
        "Performance Ratio: log2(MLX/Mojo)\nPositive=Mojo faster, Negative=MLX faster",
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
    """Generate TFLOPS/W comparison (if power data available)."""

    # Check if power data exists
    power_baseline = roofline_data.get("power_baseline", {})
    avg_watts = power_baseline.get("avg_package_watts") or power_baseline.get(
        "avg_gpu_watts"
    )

    if not avg_watts:
        print("Skipping energy chart: no power baseline data")
        return

    # Compute energy efficiency
    labels = []
    mlx_efficiency = []
    mojo_efficiency = []

    for mlx, mojo in matches[:15]:
        labels.append(f"{mlx.name}\n{mlx.shape}")
        mlx_tflops = mlx.tflops if mlx.tflops else 0
        mojo_tflops = mojo.tflops if mojo.tflops else 0

        mlx_efficiency.append(mlx_tflops / avg_watts if avg_watts > 0 else 0)
        mojo_efficiency.append(mojo_tflops / avg_watts if avg_watts > 0 else 0)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, mlx_efficiency, width, label="MLX", color="blue", alpha=0.7)
    ax.bar(x + width / 2, mojo_efficiency, width, label="Mojo", color="red", alpha=0.7)

    ax.set_ylabel("TFLOPS/W", fontsize=12)
    ax.set_title(
        f"Energy Efficiency (baseline: {avg_watts:.1f}W)",
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


def generate_decode_attribution(
    matches: list[tuple[KernelResult, KernelResult]], output_dir: Path
) -> dict[str, Any]:
    """Generate decode time attribution stacked bar chart.

    Returns attribution data for summary JSON.
    """

    # Aggregate by kernel name
    kernel_contributions = {}

    for mlx, mojo in matches:
        # Extract base kernel name
        base_name = mlx.name
        for k in INVOCATIONS_PER_DECODE:
            if k in base_name:
                base_name = k
                break

        if base_name not in INVOCATIONS_PER_DECODE:
            continue

        invocations = INVOCATIONS_PER_DECODE[base_name]
        layers = NEMOTRON_LAYERS

        mlx_contrib = mlx.median_us * invocations * layers
        mojo_contrib = mojo.median_us * invocations * layers

        if base_name not in kernel_contributions:
            kernel_contributions[base_name] = {"mlx": 0, "mojo": 0}

        kernel_contributions[base_name]["mlx"] += mlx_contrib
        kernel_contributions[base_name]["mojo"] += mojo_contrib

    # Build stacked bar chart
    kernels = list(kernel_contributions.keys())
    mlx_times = [kernel_contributions[k]["mlx"] for k in kernels]
    mojo_times = [kernel_contributions[k]["mojo"] for k in kernels]

    mlx_total = sum(mlx_times)
    mojo_total = sum(mojo_times)

    # Normalize to percentages
    mlx_pcts = [100 * t / mlx_total if mlx_total > 0 else 0 for t in mlx_times]
    mojo_pcts = [100 * t / mojo_total if mojo_total > 0 else 0 for t in mojo_times]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # MLX stacked bar
    bottom_mlx = 0
    for i, kernel in enumerate(kernels):
        ax1.bar("MLX", mlx_pcts[i], bottom=bottom_mlx, label=kernel)
        bottom_mlx += mlx_pcts[i]

    # Mojo stacked bar
    bottom_mojo = 0
    for i, kernel in enumerate(kernels):
        ax2.bar("Mojo", mojo_pcts[i], bottom=bottom_mojo, label=kernel)
        bottom_mojo += mojo_pcts[i]

    for ax in [ax1, ax2]:
        ax.set_ylabel("% of Total Kernel Time", fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    ax1.set_title(
        f"MLX Decode Attribution\nTotal: {mlx_total / 1000:.1f}ms",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_title(
        f"Mojo Decode Attribution\nTotal: {mojo_total / 1000:.1f}ms",
        fontsize=12,
        fontweight="bold",
    )

    # Add warning if below 50%
    if mlx_total < 50000 or mojo_total < 50000:  # < 50ms
        fig.text(
            0.5,
            0.02,
            "WARNING: Kernel time may be <50% of actual decode time (theoretical lower bound)",
            ha="center",
            fontsize=10,
            color="red",
            fontweight="bold",
        )

    plt.tight_layout()
    out_path = output_dir / "decode_time_attribution.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Wrote decode attribution: {out_path}")

    return {
        "mlx_total_kernel_time_us": mlx_total,
        "mojo_total_kernel_time_us": mojo_total,
        "fraction_of_decode_mlx": min(
            mlx_total / 100000, 1.0
        ),  # Assume 100ms decode = 100%
        "fraction_of_decode_mojo": min(mojo_total / 100000, 1.0),
        "below_50_pct": mlx_total < 50000 or mojo_total < 50000,
        "per_kernel": {
            k: {
                "mlx_us": kernel_contributions[k]["mlx"],
                "mojo_us": kernel_contributions[k]["mojo"],
            }
            for k in kernels
        },
    }


# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------


def write_summary_json(
    matches: list[tuple[KernelResult, KernelResult]],
    decode_attribution: dict[str, Any],
    output_dir: Path,
) -> None:
    """Write summary JSON with geometric mean speedup and Cohen's d."""

    speedups = []
    cohens_d_results = {}

    for mlx, mojo in matches:
        if mojo.median_us > 0:
            speedup = mlx.median_us / mojo.median_us
            speedups.append(speedup)

            # Cohen's d (assuming n=20 iterations for both)
            d = cohens_d(mlx.mean_us, mlx.std_us, 20, mojo.mean_us, mojo.std_us, 20)
            cohens_d_results[f"{mlx.name}_{mlx.shape}"] = round(d, 3)

    geo_mean = geometric_mean_speedup(speedups)

    # Per-kernel speedup (aggregate by kernel name)
    per_kernel = {}
    for mlx, mojo in matches:
        if mlx.name not in per_kernel:
            per_kernel[mlx.name] = []
        if mojo.median_us > 0:
            per_kernel[mlx.name].append(mlx.median_us / mojo.median_us)

    per_kernel_speedup = {
        k: round(geometric_mean_speedup(v), 3) for k, v in per_kernel.items()
    }

    summary = {
        "geometric_mean_speedup": round(geo_mean, 3),
        "per_kernel_speedup": per_kernel_speedup,
        "per_kernel_cohens_d": cohens_d_results,
        "decode_attribution": decode_attribution,
        "total_comparisons": len(matches),
    }

    out_path = output_dir / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote summary: {out_path}")
    print(f"\nGeometric mean speedup (MLX/Mojo): {geo_mean:.3f}x")


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
        help="Path to Mojo benchmark JSON",
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
