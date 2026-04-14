"""Softmax benchmark for Mojo GPU with numerically stable implementation."""
from std.math import ceildiv, exp
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from layout import LayoutTensor, Layout
from std.time import perf_counter_ns
from std.random import seed, random_float64

comptime WARMUP = 10
comptime ITERS = 50
comptime FP32 = DType.float32

# Softmax attention config
comptime NUM_HEADS = 48
comptime HEAD_DIM = 128

# Layout must cover the largest benchmarked dense attention matrix.
# S=8192: 1 * 48 * 8192 * 8192 = 3,221,225,472 elements (~12.9 GB FP32)
# S=32768 excluded: dense (1, 48, 32768, 32768) requires 206 GB FP32 —
# exceeds any current Apple Silicon config. Real models at 32K+ use
# flash attention, not dense softmax.
comptime MAX_ELEMENTS = 3221225472  # covers up to seq_len=8192
comptime flat_layout = Layout.row_major(MAX_ELEMENTS)


def gpu_softmax_kernel(
    input: LayoutTensor[DType.float32, flat_layout, MutAnyOrigin],
    output: LayoutTensor[DType.float32, flat_layout, MutAnyOrigin],
    seq_len: Int,
):
    """Row-wise softmax kernel with numerical stability (max-subtract).

    Input/Output shape: [batch, num_heads, seq_len, seq_len]
    Each thread processes one row of the attention matrix.
    """
    var row_idx = block_idx.x * block_dim.x + thread_idx.x
    var total_rows = 1 * NUM_HEADS * seq_len  # batch=1

    if row_idx >= total_rows:
        return

    var row_start = row_idx * seq_len

    # Pass 1: Find max for numerical stability
    var max_val: Float32 = rebind[Float32](input[row_start])
    for col in range(1, seq_len):
        var val = rebind[Float32](input[row_start + col])
        if val > max_val:
            max_val = val

    # Pass 2: Compute exp(x - max) and sum
    var sum: Float32 = exp(rebind[Float32](input[row_start]) - max_val)
    for col in range(1, seq_len):
        sum += exp(rebind[Float32](input[row_start + col]) - max_val)

    # Pass 3: Normalize and write
    for col in range(seq_len):
        output[row_start + col] = exp(rebind[Float32](input[row_start + col]) - max_val) / sum


def bench_softmax_shape(
    ctx: DeviceContext, seq_len: Int, shape_name: String
) raises -> Float64:
    """Benchmark softmax for given sequence length. Returns elapsed time in microseconds."""

    var batch = 1
    var total_elements = batch * NUM_HEADS * seq_len * seq_len

    # Guard: ensure shape fits the flat layout
    if total_elements > MAX_ELEMENTS:
        print("SKIP:", shape_name, "- exceeds MAX_ELEMENTS (", total_elements, ">", MAX_ELEMENTS, ")")
        return -1.0

    # Allocate host buffer via context
    host_input = ctx.enqueue_create_host_buffer[DType.float32](total_elements)
    ctx.synchronize()

    # Initialize with random logits (seed=42)
    seed(42)
    for i in range(total_elements):
        host_input[i] = Float32(random_float64() * 20.0 - 10.0)

    # Create device buffers
    dev_input = ctx.enqueue_create_buffer[DType.float32](total_elements)
    dev_output = ctx.enqueue_create_buffer[DType.float32](total_elements)

    # Copy to device
    ctx.enqueue_copy(dst_buf=dev_input, src_buf=host_input)
    ctx.synchronize()

    # Grid configuration: one thread per row
    var total_rows = batch * NUM_HEADS * seq_len
    var threads_per_block = 256
    var blocks = ceildiv(total_rows, threads_per_block)

    # Wrap device buffers in LayoutTensors
    input_tensor = LayoutTensor[DType.float32, flat_layout](dev_input)
    output_tensor = LayoutTensor[DType.float32, flat_layout](dev_output)

    # Warmup
    for _ in range(WARMUP):
        ctx.enqueue_function[gpu_softmax_kernel, gpu_softmax_kernel](
            input_tensor, output_tensor, seq_len,
            grid_dim=blocks, block_dim=threads_per_block,
        )
        ctx.synchronize()

    # Timed iterations
    var start = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[gpu_softmax_kernel, gpu_softmax_kernel](
            input_tensor, output_tensor, seq_len,
            grid_dim=blocks, block_dim=threads_per_block,
        )
        ctx.synchronize()
    var elapsed = perf_counter_ns() - start

    return Float64(elapsed) / Float64(ITERS) / 1000.0


def compute_bandwidth(total_elements: Int, dtype_size: Int, elapsed_us: Float64) -> Float64:
    """Compute memory bandwidth in GB/s.

    Softmax does one read and one write per element.
    GB/s = (2 * elements * sizeof(dtype)) / elapsed_s / 1e9
    """
    var bytes = Float64(2 * total_elements * dtype_size)
    var elapsed_s = elapsed_us / 1_000_000.0
    return bytes / elapsed_s / 1_000_000_000.0


def main() raises:
    comptime if not has_accelerator():
        print("No compatible GPU found")
    else:
        print("=== Mojo GPU Softmax Benchmark ===")
        print("Config: batch=1, heads=", NUM_HEADS, ", head_dim=", HEAD_DIM)
        print()

        ctx = DeviceContext()

        # Sequence length sweep per benchmark spec Section 3.2
        # S=32768 excluded: dense (1,48,S,S) at S=32768 requires 206 GB FP32 —
        # exceeds any current Apple Silicon. Flash attention benchmarks for
        # longer contexts are left to future work.
        var seq_lengths = List[Int]()
        seq_lengths.append(128)
        seq_lengths.append(512)
        seq_lengths.append(2048)
        seq_lengths.append(8192)

        var names = List[String]()
        names.append("seq_128")
        names.append("seq_512")
        names.append("seq_2048")
        names.append("seq_8192")

        # Benchmark FP32
        print("--- FP32 Softmax ---")
        for i in range(len(seq_lengths)):
            var S = seq_lengths[i]
            var name = names[i]
            var total_elements = 1 * NUM_HEADS * S * S

            var elapsed_us = bench_softmax_shape(ctx, S, name)
            if elapsed_us < 0.0:
                print()
                continue

            var gbs = compute_bandwidth(total_elements, 4, elapsed_us)  # FP32 = 4 bytes

            print(name, ":")
            print("  Shape: (1,", NUM_HEADS, ",", S, ",", S, ")")
            print("  Elements:", total_elements)
            print("  Time:", elapsed_us, "us")
            print("  Bandwidth:", gbs, "GB/s")
            print()

        print("=== Softmax Benchmark Complete ===")
        print("NOTE: Results should be post-processed with adaptive_bench harness")
        print("      and written to JSON for comparison with MLX.")
