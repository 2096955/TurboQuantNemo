from collections import List
from math import sqrt, log
from time import perf_counter_ns

alias MAX_ITERS = 500
alias BATCH_SIZE = 25
alias WARMUP_ITERS = 10
alias BOOTSTRAP_SAMPLES = 10_000
alias CI_TARGET_PCT = 2.0  # Stop when CI width < 2% of median
alias DW_THRESHOLD = 1.5   # Flag if Durbin-Watson < 1.5


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


fn sort_float_list(inout arr: List[Float64]):
    """Simple insertion sort for benchmark result lists."""
    for i in range(1, len(arr)):
        var key = arr[i]
        var j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


fn percentile(sorted_arr: List[Float64], pct: Float64) -> Float64:
    """Get percentile from sorted array."""
    var idx = Int(Float64(len(sorted_arr) - 1) * pct / 100.0)
    return sorted_arr[idx]


fn compute_mean(arr: List[Float64]) -> Float64:
    var s: Float64 = 0.0
    for i in range(len(arr)):
        s += arr[i]
    return s / Float64(len(arr))


fn compute_std(arr: List[Float64], mean: Float64) -> Float64:
    var s: Float64 = 0.0
    for i in range(len(arr)):
        var d = arr[i] - mean
        s += d * d
    return sqrt(s / Float64(len(arr) - 1)) if len(arr) > 1 else 0.0


fn durbin_watson(timings: List[Float64]) -> Float64:
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


fn bca_bootstrap_ci(timings: List[Float64], seed: Int = 42) -> Tuple[Float64, Float64]:
    """BCa bootstrap 95% CI with 10k samples.

    Simplified implementation — production version should use proper
    BCa bias-correction and acceleration. This implements percentile
    bootstrap as a baseline; Codex review should verify BCa correctness.
    """
    var n = len(timings)
    if n < 5:
        var lo = timings[0] if n > 0 else 0.0
        var hi = timings[n - 1] if n > 0 else 0.0
        return (lo, hi)

    var boot_means = List[Float64]()
    # Simple LCG for reproducibility (no stdlib rand in Mojo yet)
    var rng_state = seed
    for _ in range(BOOTSTRAP_SAMPLES):
        var s: Float64 = 0.0
        for _ in range(n):
            rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
            var idx = rng_state % n
            s += timings[idx]
        boot_means.append(s / Float64(n))

    sort_float_list(boot_means)
    var lo_idx = Int(Float64(BOOTSTRAP_SAMPLES) * 0.025)
    var hi_idx = Int(Float64(BOOTSTRAP_SAMPLES) * 0.975)
    return (boot_means[lo_idx], boot_means[hi_idx])
