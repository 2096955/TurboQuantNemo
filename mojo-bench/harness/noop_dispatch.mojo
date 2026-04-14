from gpu.host import DeviceContext
from time import perf_counter_ns

fn measure_noop_dispatch(warmup: Int = 100, iters: Int = 500) -> Float64:
    """Measure GPU dispatch overhead in microseconds."""
    var ctx = DeviceContext()

    # Warmup
    for _ in range(warmup):
        ctx.synchronize()

    # Measure
    var start = perf_counter_ns()
    for _ in range(iters):
        ctx.synchronize()
    var elapsed = perf_counter_ns() - start

    return Float64(elapsed) / Float64(iters) / 1000.0  # ns -> us


fn main():
    var overhead_us = measure_noop_dispatch()
    print("Dispatch overhead:", overhead_us, "us")
