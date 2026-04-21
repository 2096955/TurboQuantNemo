from collections import List
from std.math import sqrt, log
from std.time import perf_counter_ns

comptime MAX_ITERS = 500
comptime BATCH_SIZE = 25
comptime WARMUP_ITERS = 10
comptime BOOTSTRAP_SAMPLES = 10_000
comptime CI_TARGET_PCT = 2.0  # Stop when CI width < 2% of median
comptime DW_THRESHOLD = 1.5   # Flag if Durbin-Watson < 1.5


@value
struct BenchResult:
    var mean_us: Float64
    var median_us: Float64
    var std_us: Float64
    var p5_us: Float64
    var p95_us: Float64
    var p99_us: Float64
    var ci95_lo: Float64
    var ci95_hi: Float64
    var n_iterations: Int
    var dw_statistic: Float64
    var timings_us: List[Float64]
    var ci_converged: Bool
    var ci_width_pct: Float64


def sort_float_list(inout arr: List[Float64]):
    """Simple insertion sort for benchmark result lists."""
    for i in range(1, len(arr)):
        var key = arr[i]
        var j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


def percentile(sorted_arr: List[Float64], pct: Float64) -> Float64:
    """Get percentile from sorted array using linear interpolation."""
    var n = len(sorted_arr)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_arr[0]

    var pos = Float64(n - 1) * pct / 100.0
    var floor_idx = Int(pos)
    var ceil_idx = floor_idx + 1

    if ceil_idx >= n:
        return sorted_arr[n - 1]

    var frac = pos - Float64(floor_idx)
    return sorted_arr[floor_idx] + frac * (sorted_arr[ceil_idx] - sorted_arr[floor_idx])


def compute_mean(arr: List[Float64]) -> Float64:
    var s: Float64 = 0.0
    for i in range(len(arr)):
        s += arr[i]
    return s / Float64(len(arr))


def compute_std(arr: List[Float64], mean: Float64) -> Float64:
    var s: Float64 = 0.0
    for i in range(len(arr)):
        var d = arr[i] - mean
        s += d * d
    return sqrt(s / Float64(len(arr) - 1)) if len(arr) > 1 else 0.0


def durbin_watson(timings: List[Float64]) -> Float64:
    """Compute Durbin-Watson statistic for serial autocorrelation detection."""
    if len(timings) < 3:
        return 2.0  # No autocorrelation assumed
    var sum_sq_diff: Float64 = 0.0
    var sum_sq_resid: Float64 = 0.0
    var mean = compute_mean(timings)
    for i in range(len(timings)):
        var resid = timings[i] - mean
        sum_sq_resid += resid * resid
        if i > 0:
            var diff = (timings[i] - mean) - (timings[i-1] - mean)
            sum_sq_diff += diff * diff
    return sum_sq_diff / sum_sq_resid if sum_sq_resid > 0 else 2.0


def percentile_bootstrap_ci(timings: List[Float64], n_samples: Int = BOOTSTRAP_SAMPLES, seed: Int = 42) -> Tuple[Float64, Float64]:
    """Percentile bootstrap 95% CI.

    This is a simple percentile bootstrap implementation. For production use,
    proper BCa (Bias-Corrected and Accelerated) bootstrap should be implemented
    with bias-correction factor (z0) and acceleration parameter (α) to adjust
    the percentile positions based on empirical distribution characteristics.

    TODO: Implement full BCa bootstrap with z0 and α calculation.
    """
    var n = len(timings)
    if n < 5:
        var lo = timings[0] if n > 0 else 0.0
        var hi = timings[n - 1] if n > 0 else 0.0
        return (lo, hi)

    var boot_means = List[Float64]()
    # Simple LCG for reproducibility (no stdlib rand in Mojo yet)
    var rng_state = seed
    for _ in range(n_samples):
        var s: Float64 = 0.0
        for _ in range(n):
            rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
            var idx = rng_state % n
            s += timings[idx]
        boot_means.append(s / Float64(n))

    sort_float_list(boot_means)
    var lo_idx = Int(Float64(n_samples) * 0.025)
    var hi_idx = Int(Float64(n_samples) * 0.975)
    return (boot_means[lo_idx], boot_means[hi_idx])


def adaptive_bench[bench_fn: fn() -> Float64]() -> BenchResult:
    """Run adaptive benchmark with automatic convergence detection.

    Runs warmup iterations, then batches of BATCH_SIZE iterations until:
    - CI width < 2% of median (converged), or
    - MAX_ITERS reached (not converged)

    After convergence or timeout, computes final CI with full 10k bootstrap samples.
    """
    var timings = List[Float64]()

    # Warmup phase
    for _ in range(WARMUP_ITERS):
        _ = bench_fn()

    # Adaptive iteration phase
    var ci_converged = False
    var total_iters = 0

    while total_iters < MAX_ITERS:
        # Run a batch
        for _ in range(BATCH_SIZE):
            var t = bench_fn()
            timings.append(t)
            total_iters += 1

        # Check convergence with quick 1000-sample bootstrap
        if len(timings) >= 10:
            var sorted_timings = timings
            sort_float_list(sorted_timings)
            var med = percentile(sorted_timings, 50.0)

            var ci_quick = percentile_bootstrap_ci(timings, n_samples=1000)
            var ci_width = ci_quick[1] - ci_quick[0]
            var ci_width_pct = (ci_width / med) * 100.0 if med > 0 else 100.0

            if ci_width_pct < CI_TARGET_PCT:
                ci_converged = True
                break

    # Compute final statistics with full bootstrap
    var sorted_timings = timings
    sort_float_list(sorted_timings)

    var mean = compute_mean(timings)
    var median = percentile(sorted_timings, 50.0)
    var std = compute_std(timings, mean)
    var p5 = percentile(sorted_timings, 5.0)
    var p95 = percentile(sorted_timings, 95.0)
    var p99 = percentile(sorted_timings, 99.0)

    var ci_final = percentile_bootstrap_ci(timings, n_samples=BOOTSTRAP_SAMPLES)
    var ci_width = ci_final[1] - ci_final[0]
    var ci_width_pct = (ci_width / median) * 100.0 if median > 0 else 100.0

    var dw = durbin_watson(timings)

    return BenchResult(
        mean_us=mean,
        median_us=median,
        std_us=std,
        p5_us=p5,
        p95_us=p95,
        p99_us=p99,
        ci95_lo=ci_final[0],
        ci95_hi=ci_final[1],
        n_iterations=total_iters,
        dw_statistic=dw,
        timings_us=timings,
        ci_converged=ci_converged,
        ci_width_pct=ci_width_pct
    )
