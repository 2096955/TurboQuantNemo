# GPU Profiling Protocol: Xcode Instruments Metal System Trace

## Purpose

This document provides a standardized procedure for profiling Mojo and MLX kernels using Xcode Instruments Metal System Trace on Apple Silicon M4 Max. Use this protocol to attribute performance differences, validate optimization claims, and diagnose bottlenecks beyond aggregate throughput metrics.

---

## 1. Prerequisites

### Software
- **Xcode 16.0 or later** (Instruments is included with Xcode)
- **macOS 14.0+** (for Metal System Trace support on Apple Silicon)
- **Mojo SDK** (with pixi environment for `mojo-bench/`)
- **Python 3.11+** with MLX installed

### Hardware
- **Apple Silicon M4 Max** (or equivalent Apple GPU)
- **AC power supply** (not battery) for stable thermal conditions
- **Clear GPU load** (close other apps, especially web browsers and GPU-intensive tools)

### Benchmark Artifacts
- Main benchmark results must be available: `results/comparison/summary.json`
- MLX benchmark script: `scripts/benchmark_mlx_kernels.py`
- Mojo benchmark directory: `mojo-bench/` with pixi environment configured
- Each kernel has a standalone profiling target

---

## 2. Kernel Selection

### Step 1: Identify Performance Gaps

After running the main benchmark suite:

```bash
python scripts/benchmark_mlx_kernels.py  # MLX all-kernels run
cd mojo-bench && pixi run bench-all      # Mojo all-kernels run
```

Parse `results/comparison/summary.json` to identify the **top-3 kernels by performance gap**:

```python
import json
import math

with open("results/comparison/summary.json") as f:
    summary = json.load(f)

# Use per-kernel time ratio from the comparison summary.
# ratio < 1.0 => MLX faster, ratio > 1.0 => Mojo faster
ratios = summary["per_kernel_time_ratio_mlx_over_mojo"]

# Top 3: largest multiplicative gap in either direction
top_3 = sorted(
    ratios.items(),
    key=lambda x: abs(math.log2(x[1])) if x[1] > 0 else float("inf"),
    reverse=True,
)[:3]
print("Top 3 kernels by performance gap:")
for kernel_name, ratio in top_3:
    direction = "MLX faster" if ratio < 1.0 else "Mojo faster"
    print(f"  {kernel_name}: {ratio:.2f}x time ratio ({direction})")
```

### Step 2: Document Selection Rationale

Record in your profiling session log:
- Which kernels were selected and why (e.g., "Largest MLX advantage: 2.8x on `gemm_4096_4096_4096`")
- Aggregate throughput for each (will help interpret absolute vs relative metrics)

---

## 3. Profiling Procedure

### 3.1 Open Instruments

```bash
# Open Instruments
open /Applications/Xcode.app/Contents/Applications/Instruments.app

# Alternatively, from Xcode menu:
# Xcode → Open Developer Tool → Instruments
```

### 3.2 Create a New Trace

1. **Select Template**: Choose **"Metal System Trace"** (under GPU category)
2. **Target Selection**: You will configure this per framework in steps below

### 3.3 Profile MLX Kernel

#### Step A: Configure Instruments Target

1. In Instruments, click the **"Choose Target"** dropdown
2. Select **"Attach to Process"** → search for and select your active Python process, or leave blank to attach on launch
3. Click the **record circle** button (or press Cmd+R)

#### Step B: Launch MLX Benchmark

In a separate terminal, run the single-kernel benchmark:

```bash
# Profile kernel: <kernel_name> (e.g., gemm_4096_4096_4096)
python scripts/benchmark_mlx_kernels.py --kernel <kernel_name> --iterations 10
```

Instruments will capture the GPU trace while the benchmark runs. Allow the process to complete.

#### Step C: Stop Recording

When the process finishes, stop the trace in Instruments (press Cmd+R again or click stop).

#### Step D: Repeat 3 Times

Record **three independent trace runs** for the same kernel. Instruments will create separate traces (e.g., `MLX_<kernel>_run1.trace`, `MLX_<kernel>_run2.trace`, `MLX_<kernel>_run3.trace`).

Between runs:
- Wait 10 seconds for GPU to cool
- Note the timestamp and any system activity in your log
- Keep system conditions consistent (same background load, AC power)

### 3.4 Profile Mojo Kernel

#### Step A: Configure Instruments Target

1. In Instruments, click **"Choose Target"** → **"Attach to Process"**
2. You will attach after launching the binary

#### Step B: Build & Launch Mojo Benchmark

In a separate terminal:

```bash
cd mojo-bench

# Build the specific kernel benchmark (adjust name as needed)
pixi run bench-<kernel_name> --build-only

# Note the binary path printed to stdout (typically: build/bench-<kernel_name>)
```

#### Step C: Attach and Profile

In Instruments:
1. Click the **record button** to start recording
2. Immediately in the terminal, run:
   ```bash
   ./build/bench-<kernel_name> --iterations 10
   ```
3. Instruments will attach to the process and capture the trace

#### Step D: Stop Recording

When the binary finishes, stop the trace in Instruments.

#### Step E: Repeat 3 Times

Capture three independent traces for the same Mojo kernel.

---

## 4. Extracting Metrics from Traces

### 4.1 Within Instruments

For each trace run:

1. **Open the trace file** (double-click in Instruments' file list)
2. **Locate the Metal GPU view**: In the timeline, expand **"GPU"** section
3. **Select a kernel dispatch**: Click on a dispatch row to highlight it
4. **Open Inspector panel**: View → Show Inspector (Cmd+Option+I)
5. **Record the following metrics** from the Inspector:

   | Metric | Instruments Location | Notes |
   |--------|----------------------|-------|
   | **ALU Utilization %** | GPU → ALU Pipeline → Utilization % | Measure when a kernel is running (expand dispatch) |
   | **Memory Bandwidth %** | GPU → Memory → Bandwidth Utilization % | Percentage of peak GPU memory bandwidth used |
   | **GPU Occupancy (Waves/EU)** | GPU → Occupancy | Waves per Execution Unit (lower = understaffed) |
   | **Memory Stalls** | GPU → Stall Cycles → Memory Stalls | Cycles waiting for memory loads/stores |
   | **ALU Stalls** | GPU → Stall Cycles → ALU Stalls | Cycles where ALU pipes are blocked |
   | **Control Flow Stalls** | GPU → Stall Cycles → Control Flow Stalls | Cycles blocked by branches or barriers |

### 4.2 Export and Aggregate

For each kernel:

- Record median of the **three runs** for each metric
- Round percentages to nearest 1%
- Ignore outlier runs (more than 2σ from mean) — explain in notes if excluded

---

## 5. Side-by-Side Comparison

### 5.1 Compare MLX vs Mojo for Same Kernel

Create a comparison table:

```markdown
### Kernel: gemm_4096_4096_4096

| Metric | MLX Median | Mojo Median | Difference |
|--------|-----------|-----------|-----------|
| ALU Util (%) | 72 | 64 | -8 |
| Memory BW (%) | 48 | 55 | +7 |
| Occupancy (waves/EU) | 6.2 | 5.1 | -1.1 |
| Memory Stalls (%) | 18 | 22 | +4 |
| ALU Stalls (%) | 5 | 8 | +3 |
| Control Flow Stalls (%) | 2 | 2 | — |

**Interpretation**: MLX achieves higher ALU utilization and lower memory stalls, suggesting better data reuse (tiling, cache efficiency). Mojo has higher memory bandwidth, indicating less optimal access patterns.
```

### 5.2 Check Dispatch Patterns

1. In Instruments, zoom into the GPU timeline for each kernel
2. Observe:
   - **Dispatch frequency**: How often does each command buffer submit?
   - **Kernel duration**: How long does a single kernel run (typically 1–100 ms for large GEMMs)?
   - **Gap between dispatches**: Any idle GPU time?
3. Record any significant differences in the comparison log

---

## 6. Interpretation Guide

Use these heuristics to diagnose bottlenecks:

### High ALU Utilization (>75%) / Low Memory BW (<50%)
- **Likely**: Compute-bound (good for heavy mathematical kernels)
- **Check**: Is the math operation itself the bottleneck? (E.g., GEMM with small matrix should be compute-bound)
- **Action**: Profile further with ALU pipeline breakdown; look for math-specific stalls

### Low ALU (<50%) / High Memory BW (>70%)
- **Likely**: Memory-bound (data access is the bottleneck)
- **Check**: Memory access patterns (stride, bank conflicts, cache misses)
- **Action**: Optimize tiling, prefetch strategy, or data layout; reduce stall cycles

### Low ALU (<40%) / Low Memory BW (<40%)
- **Likely**: Stalled pipeline (synchronization, occupancy, or control flow)
- **Check**: Stall cycle breakdown; GPU occupancy
- **Action**: Increase parallelism (more threadgroups), reduce synchronization barriers, or simplify branches

### High ALU (>75%) / High Memory BW (>70%)
- **Likely**: Near-optimal utilization
- **Action**: Minor tweaks only; diminishing returns on further optimization

### Occupancy Analysis
- **< 4 waves/EU**: Severely underutilized (each EU has max ~10 waves)
- **4–6 waves/EU**: Moderate utilization (acceptable for memory-limited kernels)
- **> 6 waves/EU**: Good occupancy (should see low idle time)

---

## 7. Reporting Template

For each profiled kernel, fill out this template:

### Template

```markdown
## Kernel Profiling Report: <kernel_name>

### Environment
- **Date**: YYYY-MM-DD
- **System**: Apple Silicon M4 Max, macOS version
- **Power**: AC (yes/no)
- **Background Load**: (describe any other processes)

### Selection Rationale
- **Performance Gap**: MLX vs Mojo speedup ratio
- **Why Selected**: (e.g., "Largest slowdown in Mojo")

### MLX Results (3 runs, median reported)
| Run | ALU (%) | BW (%) | Occupancy (w/EU) | Mem Stalls | ALU Stalls | CF Stalls | Notes |
|-----|---------|--------|------------------|-----------|-----------|-----------|-------|
| 1   | 72      | 48     | 6.2              | 18%       | 5%        | 2%        |       |
| 2   | 71      | 49     | 6.1              | 19%       | 5%        | 2%        |       |
| 3   | 73      | 47     | 6.3              | 17%       | 5%        | 2%        |       |
| **Median** | **72** | **48** | **6.2** | **18%** | **5%** | **2%** |       |

### Mojo Results (3 runs, median reported)
| Run | ALU (%) | BW (%) | Occupancy (w/EU) | Mem Stalls | ALU Stalls | CF Stalls | Notes |
|-----|---------|--------|------------------|-----------|-----------|-----------|-------|
| 1   | 64      | 55     | 5.1              | 22%       | 8%        | 2%        |       |
| 2   | 65      | 54     | 5.2              | 21%       | 8%        | 2%        |       |
| 3   | 63      | 56     | 5.0              | 23%       | 8%        | 2%        |       |
| **Median** | **64** | **55** | **5.1** | **22%** | **8%** | **2%** |       |

### Comparison
| Metric | MLX | Mojo | Delta | Favorable |
|--------|-----|------|-------|-----------|
| ALU Util (%) | 72 | 64 | -8 | MLX |
| Memory BW (%) | 48 | 55 | +7 | Mojo (lower is better for BW pressure) |
| Occupancy (w/EU) | 6.2 | 5.1 | -1.1 | MLX |
| Mem Stalls (%) | 18 | 22 | +4 | MLX |
| ALU Stalls (%) | 5 | 8 | +3 | MLX |

### Interpretation

**Bottleneck**: Memory-bound (high BW utilization, stall cycles driven by memory)

**Key Finding**: MLX better data reuse (lower BW utilization, fewer memory stalls). Mojo's higher memory bandwidth suggests less efficient access patterns (possible issues: poor tiling, unaligned reads, or cache misses).

**Recommendation**: Investigate Mojo kernel's memory access pattern; consider:
1. Memory layout (row-major vs column-major)
2. Tiling strategy (tile size impacts cache reuse)
3. Coalescing (ensure contiguous memory access)

### Trace Files
- `MLX_<kernel>_run1.trace` – Size: X MB
- `MLX_<kernel>_run2.trace` – Size: Y MB
- `MLX_<kernel>_run3.trace` – Size: Z MB
- `Mojo_<kernel>_run1.trace` – Size: X MB
- `Mojo_<kernel>_run2.trace` – Size: Y MB
- `Mojo_<kernel>_run3.trace` – Size: Z MB
```

---

## 8. Known Limitations

### Measurement Overhead
- **Instruments adds 5–15% overhead** to GPU metrics. Traces capture empirical performance, not peak theoretical.
- Compare relative metrics (ratios) rather than absolute percentages when comparing Instruments results to theoretical models.

### GPU-Side Metrics Only
- Metal System Trace captures **GPU-only** metrics. CPU-side metrics (host dispatch overhead, kernel launch latency) are not included.
- For end-to-end performance, also measure with `time(1)` or the benchmark's built-in throughput counter.

### MLX Python Wrapper Noise
- Running `python scripts/benchmark_mlx_kernels.py` includes Python startup, MPS (Metal Performance Shaders) initialization, and framework overhead.
- **Mitigation**: Use the `--kernel <name>` filter to profile only the target kernel and minimize startup noise.
- If profiling multiple kernels, expect the first run to be slower (one-time MPS cache warmup).

### Mojo Tier 3 Metal Dispatch Patterns
- Mojo's Metal code generation (Tier 3) may differ from native hand-optimized Metal, resulting in different dispatch patterns, register pressure, and occupancy.
- Do not directly compare Mojo dispatch efficiency to production hand-tuned Metal kernels.

### Profiling on AC Power
- **Always profile on AC power** (not battery). Battery thermal throttling can artificially degrade metrics.
- Ensure GPU fans are not obstructed and thermal conditions are stable (no external heat sources).

### Occupancy Caveats
- High occupancy does not guarantee performance; memory latency may still stall waves.
- Low occupancy with high ALU utilization may indicate scalar or cache-efficient compute.

### Trace File Size
- Large traces (100+ MB) may take time to load in Instruments. Keep iteration counts reasonable (10 iterations is a good balance).

---

## 9. Quick Reference: Instruments Shortcuts

| Action | Shortcut |
|--------|----------|
| Start/Stop recording | Cmd+R |
| Open Inspector panel | Cmd+Option+I |
| Zoom timeline in | Scroll up / trackpad pinch |
| Zoom timeline out | Scroll down / trackpad pinch |
| Pause/Resume playback | Space |
| Jump to next dispatch | Right arrow |
| Jump to previous dispatch | Left arrow |
| Export trace | File → Export Trace (`.trace` bundle) |
| View GPU memory | GPU → Memory section in timeline |
| Compare two traces | Window → Comparison Window (Cmd+Option+C) |

---

## 10. Checklist for Profiling Session

- [ ] System on AC power, fans clear
- [ ] Background processes closed (especially browsers, GPU tools)
- [ ] Top-3 kernels identified from `results/comparison/summary.json`
- [ ] Xcode 16+ and Instruments available
- [ ] MLX benchmark script tested (runs without errors on single kernel)
- [ ] Mojo pixi environment set up and builds without errors
- [ ] Three independent trace runs collected for each kernel (MLX + Mojo)
- [ ] Metrics recorded in profiling template (median of 3 runs)
- [ ] Trace files saved with descriptive names and timestamps
- [ ] Comparison table filled out with deltas and interpretation
- [ ] Bottleneck identified and recommendation documented
- [ ] Trace files archived (optional: `.trace` files can be large; document path for reference)

---

## 11. Troubleshooting

### Issue: Instruments Fails to Attach to Process
- **Solution**: Ensure the Python or Mojo process is running when Instruments tries to attach. Click the record button *before* launching the benchmark.
- **Alt**: Use the **"All Processes"** target and filter by name in the timeline view.

### Issue: No GPU Metrics Appearing in Trace
- **Solution**: Metal System Trace requires GPU activity. Ensure the benchmark is actually running (not just loading). Check that the system is using the integrated GPU (not an external display).
- **Check**: `system_profiler SPDisplaysDataType | grep -i metal`

### Issue: High Variance Between Runs (>10% difference in metrics)
- **Solution**: Ensure system is at thermal equilibrium. Wait 30–60 seconds between runs, and close other GPU-heavy apps.
- **Alternative**: Increase iteration count (use `--iterations 20` or higher) to average out per-frame variance.

### Issue: Trace File Cannot Be Opened
- **Solution**: The `.trace` bundle is large and may be incomplete if the process was killed. Re-run the profiling, ensuring the benchmark completes normally.

---

## 12. Next Steps After Profiling

1. **Document findings**: Fill out the profiling report template for each kernel
2. **Commit traces**: Optionally save `.trace` files to `results/profiling/` with dated filenames
3. **Create issue**: If a bottleneck is identified, file a GitHub issue with:
   - Profiling report summary
   - Interpretation and recommended optimization
   - Link to trace files (if archived)
4. **Iterate**: Profile again after implementing an optimization to measure the impact
5. **Archive baseline**: If this is the first profiling session, save these results as the baseline for future comparisons

---

## 13. References

- [Xcode Instruments User Guide](https://developer.apple.com/documentation/xcode/instruments) – Official Apple documentation
- [Metal Programming Guide](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf) – GPU architecture and capabilities
- [MLX Documentation](https://ml-explore.github.io/mlx/) – MLX framework reference
- [Mojo Language Documentation](https://docs.modular.com/mojo/) – Mojo SDK and Metal interop
