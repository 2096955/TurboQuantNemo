"""Softmax benchmark for Mojo GPU with numerically stable implementation."""
from gpu.host import DeviceContext
from gpu import thread_idx, block_idx, block_dim, barrier
from layout import LayoutTensor, Layout
from memory import UnsafePointer
from time import perf_counter_ns
from random import seed, random_float64
from math import exp

alias WARMUP = 10
alias ITERS = 50
alias FP32 = DType.float32
alias FP16 = DType.float16

# Softmax attention config
alias NUM_HEADS = 48
alias HEAD_DIM = 128


fn gpu_softmax_kernel[
    seq_len: Int,
    dtype: DType,
    layout: Layout
](
    input: LayoutTensor[mut=False, dtype=dtype, layout=layout],
    output: LayoutTensor[mut=True, dtype=dtype, layout=layout]
):
    """Row-wise softmax kernel with numerical stability (max-subtract).

    Input/Output shape: [batch, num_heads, seq_len, seq_len]
    Each thread processes one row of the attention matrix.
    Uses two-pass algorithm:
    1. Find max value in row
    2. Compute exp(x - max) and sum
    3. Normalize by sum
    """
    # Flatten to [batch * num_heads * seq_len, seq_len] conceptually
    # Each thread handles one row
    var row_idx = block_idx.x * block_dim.x + thread_idx.x
    var total_rows = 1 * NUM_HEADS * seq_len  # batch=1

    if row_idx >= total_rows:
        return

    var row_start = row_idx * seq_len

    # Pass 1: Find max for numerical stability
    var max_val: Float32 = -3.402823e38  # -FLT_MAX
    for col in range(seq_len):
        var val = Float32(input[row_start + col])
        if val > max_val:
            max_val = val

    # Pass 2: Compute exp(x - max) and sum
    var sum: Float32 = 0.0
    for col in range(seq_len):
        var val = Float32(input[row_start + col])
        var exp_val = exp(val - max_val)
        sum += exp_val

    # Pass 3: Normalize and write
    for col in range(seq_len):
        var val = Float32(input[row_start + col])
        var exp_val = exp(val - max_val)
        var normalized = exp_val / sum
        if dtype == DType.float16:
            output[row_start + col] = Float16(normalized)
        else:
            output[row_start + col] = normalized


fn bench_softmax_shape[
    dtype: DType
](ctx: DeviceContext, seq_len: Int, shape_name: String) -> Float64:
    """Benchmark softmax for given sequence length. Returns elapsed time in microseconds."""

    # Input shape: [1, 48, S, S]
    var batch = 1
    var total_elements = batch * NUM_HEADS * seq_len * seq_len

    # Allocate host buffers
    var host_input = UnsafePointer[Float32].alloc(total_elements)
    var host_output = UnsafePointer[Float32].alloc(total_elements)

    # Initialize with random logits (seed=42)
    seed(42)
    for i in range(total_elements):
        # Simulate attention logits in range [-10, 10]
        host_input[i] = Float32(random_float64() * 20.0 - 10.0)

    # Create device buffers
    var dev_input = ctx.enqueue_create_buffer[dtype](total_elements)
    var dev_output = ctx.enqueue_create_buffer[dtype](total_elements)

    # Copy to device
    if dtype == DType.float16:
        var host_input_fp16 = UnsafePointer[Float16].alloc(total_elements)
        for i in range(total_elements):
            host_input_fp16[i] = Float16(host_input[i])
        ctx.enqueue_copy(dev_input, host_input_fp16, total_elements)
        host_input_fp16.free()
    else:
        ctx.enqueue_copy(dev_input, host_input, total_elements)

    ctx.synchronize()

    # Grid configuration: one thread per row
    var total_rows = batch * NUM_HEADS * seq_len
    var threads_per_block = 256
    var blocks = (total_rows + threads_per_block - 1) // threads_per_block

    # Warmup
    for _ in range(WARMUP):
        ctx.enqueue_function[gpu_softmax_kernel[
            seq_len, dtype,
            Layout.row_major(total_elements)
        ]](
            dev_input.as_tensor(), dev_output.as_tensor(),
            grid_dim=(blocks,), block_dim=(threads_per_block,)
        )
        ctx.synchronize()

    # Timed iterations
    var start = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[gpu_softmax_kernel[
            seq_len, dtype,
            Layout.row_major(total_elements)
        ]](
            dev_input.as_tensor(), dev_output.as_tensor(),
            grid_dim=(blocks,), block_dim=(threads_per_block,)
        )
        ctx.synchronize()
    var elapsed = perf_counter_ns() - start

    # Cleanup
    host_input.free()
    host_output.free()

    return Float64(elapsed) / Float64(ITERS) / 1000.0


fn compute_bandwidth(total_elements: Int, dtype_size: Int, elapsed_us: Float64) -> Float64:
    """Compute memory bandwidth in GB/s.

    Softmax does one read and one write per element.
    GB/s = (2 * elements * sizeof(dtype)) / elapsed_s / 1e9
    """
    var bytes = Float64(2 * total_elements * dtype_size)
    var elapsed_s = elapsed_us / 1_000_000.0
    return bytes / elapsed_s / 1_000_000_000.0


fn main() raises:
    print("=== Mojo GPU Softmax Benchmark ===")
    print("Config: batch=1, heads=", NUM_HEADS, ", head_dim=", HEAD_DIM)
    print()

    var ctx = DeviceContext()

    # Sequence length sweep
    var seq_lengths = List[Int]()
    seq_lengths.append(128)
    seq_lengths.append(512)
    seq_lengths.append(2048)
    seq_lengths.append(8192)
    seq_lengths.append(32768)

    var names = List[String]()
    names.append("seq_128")
    names.append("seq_512")
    names.append("seq_2048")
    names.append("seq_8192")
    names.append("seq_32768")

    # Benchmark FP32
    print("--- FP32 Softmax ---")
    for i in range(len(seq_lengths)):
        var S = seq_lengths[i]
        var name = names[i]
        var total_elements = 1 * NUM_HEADS * S * S

        var elapsed_us = bench_softmax_shape[FP32](ctx, S, name)
        var gbs = compute_bandwidth(total_elements, 4, elapsed_us)  # FP32 = 4 bytes

        print(name, ":")
        print("  Shape: (1,", NUM_HEADS, ",", S, ",", S, ")")
        print("  Elements:", total_elements)
        print("  Time:", elapsed_us, "us")
        print("  Bandwidth:", gbs, "GB/s")
        print()

    # Benchmark FP16
    print("--- FP16 Softmax ---")
    for i in range(len(seq_lengths)):
        var S = seq_lengths[i]
        var name = names[i]
        var total_elements = 1 * NUM_HEADS * S * S

        var elapsed_us = bench_softmax_shape[FP16](ctx, S, name)
        var gbs = compute_bandwidth(total_elements, 2, elapsed_us)  # FP16 = 2 bytes

        print(name, ":")
        print("  Shape: (1,", NUM_HEADS, ",", S, ",", S, ")")
        print("  Elements:", total_elements)
        print("  Time:", elapsed_us, "us")
        print("  Bandwidth:", gbs, "GB/s")
        print()

    print("=== Softmax Benchmark Complete ===")
    print("NOTE: Results should be post-processed with adaptive_bench harness")
    print("      and written to JSON for comparison with MLX.")
