# Benchmark Comparison Guide

This guide explains how to use `scripts/compare_mojo_vs_mlx.py` to generate publication-quality comparison charts and tables.

## Prerequisites

Install required Python packages:

```bash
pip install -r requirements-benchmark.txt
```

Or install individually:

```bash
pip install numpy scipy matplotlib
```

## Running the Comparison

### 1. Generate benchmark results

First, run both MLX and Mojo benchmarks:

```bash
# MLX benchmarks
python scripts/benchmark_mlx_kernels.py --output results/mlx_kernels.json

# Roofline calibration
python scripts/roofline_calibrate.py --output results/roofline_m4max.json

# Mojo benchmarks (when available)
cd mojo-bench
pixi run bench-all  # Writes to mojo-bench/results/mojo_kernels.json
cd ..
```

### 2. Run the comparison

```bash
python scripts/compare_mojo_vs_mlx.py \
  --mlx results/mlx_kernels.json \
  --mojo mojo-bench/results/mojo_kernels.json \
  --roofline results/roofline_m4max.json \
  --output-dir results/comparison/
```

## Outputs

The comparison script generates:

### LaTeX Tables

One table per kernel category (e.g., `table_matmul.tex`, `table_softmax.tex`) with:
- Shape configurations
- MLX and Mojo latencies
- Throughput (TFLOPS)
- Speedup ratios

Example:

```latex
\begin{table}[htbp]
\centering
\caption{MatMul Kernel Comparison}
\begin{tabular}{lrrrrr}
\toprule
Shape & MLX (μs) & Mojo (μs) & MLX TFLOPS & Mojo TFLOPS & Speedup \\
\midrule
1024x1024x1024 & 123.4 & 98.2 & 17.5 & 22.0 & 1.26x \\
...
\bottomrule
\end{tabular}
\end{table}
```

### Charts (300 DPI PNG)

1. **Roofline Plot** (`roofline_plot.png`)
   - Both frameworks plotted on arithmetic intensity vs throughput
   - Hardware ceiling line from calibration data
   - Log-log scale

2. **Roofline Fraction Bars** (`roofline_fraction_bars.png`)
   - Side-by-side comparison of % of peak performance achieved
   - Per kernel/shape

3. **Throughput Bars** (`throughput_bars_<kernel>.png`)
   - Absolute throughput (TFLOPS) per kernel
   - Separate chart for each kernel type

4. **Throughput Lines** (`throughput_lines_matmul.png`)
   - MatMul throughput vs matrix size
   - For standard square sweep (512, 1024, 2048, 4096, 8192)

5. **Heatmap** (`heatmap_performance_ratio.png`)
   - log₂(MLX_time / Mojo_time)
   - Zero = equal, positive = Mojo faster, negative = MLX faster
   - Color-coded: red (Mojo faster) to blue (MLX faster)

6. **Energy Efficiency** (`energy_efficiency.png`)
   - TFLOPS/W comparison (if power baseline available)

7. **Decode Time Attribution** (`decode_time_attribution.png`)
   - Stacked bar showing each kernel's fraction of decode time
   - For Nemotron-H 120B (80 layers)
   - Includes warning if <50% of actual decode time

### Summary JSON

`summary.json` contains:

```json
{
  "geometric_mean_speedup": 1.23,
  "per_kernel_speedup": {
    "matmul": 1.15,
    "softmax": 0.95,
    ...
  },
  "per_kernel_cohens_d": {
    "matmul_1024x1024x1024": 0.8,
    ...
  },
  "decode_attribution": {
    "mlx_total_kernel_time_us": 45000,
    "mojo_total_kernel_time_us": 38000,
    "fraction_of_decode_mlx": 0.45,
    "fraction_of_decode_mojo": 0.38,
    "below_50_pct": true
  },
  "total_comparisons": 42
}
```

## Statistical Measures

### Geometric Mean Speedup

The script uses **geometric mean** (not arithmetic mean) for speedup ratios because:
- Arithmetic mean is biased by outliers
- Geometric mean is symmetric: GM([2x, 0.5x]) = 1.0
- Standard practice in performance comparisons

### Cohen's d Effect Size

Cohen's d quantifies the practical significance of performance differences:
- |d| < 0.2: negligible
- 0.2 ≤ |d| < 0.5: small
- 0.5 ≤ |d| < 0.8: medium
- |d| ≥ 0.8: large

Positive d means MLX is slower, negative d means MLX is faster.

### 95% BCa Confidence Intervals

Error bars on all charts represent 95% BCa (bias-corrected and accelerated) bootstrap confidence intervals from the benchmark data.

## Decode Time Attribution

The script estimates what fraction of a full decode step each kernel represents:

**Invocations per decode step:**
- matmul: 6 (Q, K, V, O, up, down)
- softmax: 1
- rope: 1
- rotate_forward: 1
- rotate_inverse: 1
- kv_compress: 1
- fused_attention: 1

**Model:** Nemotron-H 120B (80 layers)

**Important:** These are theoretical lower bounds. Actual decode time includes:
- Framework overhead
- Memory allocation/deallocation
- Data movement
- Kernel launch overhead
- CPU-GPU synchronization

If the sum of kernel times is <50% of a realistic decode time, the script will flag this prominently.

## Troubleshooting

### Missing dependencies

```bash
pip install numpy scipy matplotlib
```

### No matching kernels

If you see "ERROR: No matching kernels found", ensure:
- Both JSON files follow the same schema (see `scripts/benchmark_mlx_kernels.py` output)
- Kernel names and shapes match exactly
- The `kernels` array is populated in both files

### Missing power data

If power baseline is unavailable in roofline calibration, the energy efficiency chart will be skipped with a message.

## Integration with Papers

LaTeX tables can be directly included in papers:

```latex
\input{results/comparison/table_matmul.tex}
```

Charts are 300 DPI PNG, suitable for publication. Convert to PDF if needed:

```bash
convert -density 300 results/comparison/roofline_plot.png results/comparison/roofline_plot.pdf
```

## Example Workflow

```bash
# 1. Run benchmarks
python scripts/benchmark_mlx_kernels.py --output results/mlx_kernels.json
python scripts/roofline_calibrate.py --output results/roofline_m4max.json

# 2. (When Mojo benchmarks are available)
cd mojo-bench && pixi run bench-all && cd ..

# 3. Compare
python scripts/compare_mojo_vs_mlx.py \
  --mlx results/mlx_kernels.json \
  --mojo mojo-bench/results/mojo_kernels.json \
  --roofline results/roofline_m4max.json \
  --output-dir results/comparison/

# 4. View results
open results/comparison/roofline_plot.png
cat results/comparison/summary.json
```

## Testing

Unit tests for the comparison logic:

```bash
python scripts/test_compare_mojo_vs_mlx.py
```

This tests:
- Kernel matching logic
- Cohen's d calculation
- Geometric mean speedup
- JSON parsing

## References

- Methodology: `docs/superpowers/specs/2026-04-13-mojo-vs-mlx-kernel-benchmark-design.md`
- MLX benchmarks: `scripts/benchmark_mlx_kernels.py`
- Roofline calibration: `scripts/roofline_calibrate.py`
- Paper draft: `docs/FROM_ATTENTION_TO_CONSUMER_HARDWARE.md`
