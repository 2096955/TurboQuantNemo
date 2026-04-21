from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from layout import LayoutTensor, Layout
from std.time import perf_counter_ns
from std.sys import has_accelerator

comptime NOOP_SIZE = 1
comptime noop_layout = Layout.row_major(NOOP_SIZE)

def gpu_noop_kernel(
    dummy: LayoutTensor[DType.float32, noop_layout, MutAnyOrigin],
):
    """Kernel that does nothing. Measures pure dispatch overhead."""
    pass

def measure_noop_dispatch(warmup: Int = 100, iters: Int = 500) raises -> Float64:
    """Measure GPU dispatch + sync overhead in microseconds."""
    var ctx = DeviceContext()

    dev_dummy = ctx.enqueue_create_buffer[DType.float32](NOOP_SIZE)
    dummy_tensor = LayoutTensor[DType.float32, noop_layout](dev_dummy)

    # Warmup
    for _ in range(warmup):
        ctx.enqueue_function[gpu_noop_kernel, gpu_noop_kernel](
            dummy_tensor,
            grid_dim=1, block_dim=1,
        )
        ctx.synchronize()

    # Measure
    var start = perf_counter_ns()
    for _ in range(iters):
        ctx.enqueue_function[gpu_noop_kernel, gpu_noop_kernel](
            dummy_tensor,
            grid_dim=1, block_dim=1,
        )
        ctx.synchronize()
    var elapsed = perf_counter_ns() - start

    return Float64(elapsed) / Float64(iters) / 1000.0


def main() raises:
    comptime if not has_accelerator():
        print("No compatible GPU found — cannot measure dispatch overhead")
    else:
        var overhead_us = measure_noop_dispatch()
        print("Dispatch overhead:", overhead_us, "us")
