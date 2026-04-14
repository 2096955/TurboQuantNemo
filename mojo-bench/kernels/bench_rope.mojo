"""Rotary Position Embeddings (RoPE) benchmark for Mojo GPU."""
from gpu.host import DeviceContext
from gpu import thread_idx, block_idx, block_dim, barrier
from layout import LayoutTensor, Layout
from memory import UnsafePointer
from time import perf_counter_ns
from random import seed, random_float64
from math import sin, cos, pow

alias WARMUP = 10
alias ITERS = 50
alias FP32 = DType.float32
alias FP16 = DType.float16

# RoPE config matching Qwen/LLaMA models
alias NUM_HEADS = 48
alias HEAD_DIM = 128
alias ROPE_BASE = 10000.0


fn precompute_freqs_cis(head_dim: Int, seq_len: Int, base: Float64) -> (UnsafePointer[Float32], UnsafePointer[Float32]):
    """Precompute sin/cos frequency tables for RoPE.

    Returns (cos_table, sin_table) of size [seq_len, head_dim/2].
    Each position gets rotary frequencies for pairs of dimensions.
    """
    var half_dim = head_dim // 2
    var table_size = seq_len * half_dim

    var cos_table = UnsafePointer[Float32].alloc(table_size)
    var sin_table = UnsafePointer[Float32].alloc(table_size)

    # Compute frequencies: freq[i] = 1.0 / (base^(2i/head_dim))
    for pos in range(seq_len):
        for i in range(half_dim):
            var freq = 1.0 / pow(base, Float64(2 * i) / Float64(head_dim))
            var angle = Float64(pos) * freq

            var idx = pos * half_dim + i
            cos_table[idx] = Float32(cos(angle))
            sin_table[idx] = Float32(sin(angle))

    return (cos_table, sin_table)


fn gpu_rope_kernel[
    seq_len: Int,
    head_dim: Int,
    dtype: DType,
    layout_input: Layout,
    layout_cos: Layout,
    layout_sin: Layout,
    layout_output: Layout
](
    input: LayoutTensor[mut=False, dtype=dtype, layout=layout_input],
    cos_table: LayoutTensor[mut=False, dtype=dtype, layout=layout_cos],
    sin_table: LayoutTensor[mut=False, dtype=dtype, layout=layout_sin],
    output: LayoutTensor[mut=True, dtype=dtype, layout=layout_output]
):
    """Apply Rotary Position Embeddings.

    Input shape: [batch, num_heads, seq_len, head_dim]
    cos/sin tables: [seq_len, head_dim/2]

    For each position and dimension pair (2i, 2i+1):
    out[2i]   = input[2i]   * cos[i] - input[2i+1] * sin[i]
    out[2i+1] = input[2i+1] * cos[i] + input[2i]   * sin[i]
    """
    var tid = block_idx.x * block_dim.x + thread_idx.x

    # Flatten: [batch * num_heads * seq_len * head_dim]
    var total_elements = 1 * NUM_HEADS * seq_len * head_dim

    if tid >= total_elements:
        return

    # Decode position in input tensor
    var head_idx = (tid // head_dim) % NUM_HEADS
    var pos = (tid // head_dim) // NUM_HEADS
    var dim = tid % head_dim

    # RoPE only applies to pairs
    if dim % 2 == 1:
        # Odd dimension: handled by even dimension's pair computation
        # But we still need to compute it
        var pair_dim = dim - 1
        var freq_idx = pos * (head_dim // 2) + (pair_dim // 2)

        var cos_val = Float32(cos_table[freq_idx])
        var sin_val = Float32(sin_table[freq_idx])

        var even_val = Float32(input[tid - 1])  # input[2i]
        var odd_val = Float32(input[tid])       # input[2i+1]

        # out[2i+1] = input[2i+1] * cos + input[2i] * sin
        var result = odd_val * cos_val + even_val * sin_val

        if dtype == DType.float16:
            output[tid] = Float16(result)
        else:
            output[tid] = result
    else:
        # Even dimension: compute both pair elements
        var freq_idx = pos * (head_dim // 2) + (dim // 2)

        var cos_val = Float32(cos_table[freq_idx])
        var sin_val = Float32(sin_table[freq_idx])

        var even_val = Float32(input[tid])      # input[2i]
        var odd_val = Float32(input[tid + 1])   # input[2i+1]

        # out[2i] = input[2i] * cos - input[2i+1] * sin
        var result = even_val * cos_val - odd_val * sin_val

        if dtype == DType.float16:
            output[tid] = Float16(result)
        else:
            output[tid] = result


fn bench_rope_shape[
    dtype: DType
](ctx: DeviceContext, seq_len: Int, shape_name: String) -> Float64:
    """Benchmark RoPE for given sequence length. Returns elapsed time in microseconds."""

    # Input shape: [1, 48, S, 128]
    var batch = 1
    var total_elements = batch * NUM_HEADS * seq_len * HEAD_DIM
    var table_size = seq_len * (HEAD_DIM // 2)

    # Precompute frequency tables on host
    var (host_cos, host_sin) = precompute_freqs_cis(HEAD_DIM, seq_len, ROPE_BASE)

    # Allocate input/output
    var host_input = UnsafePointer[Float32].alloc(total_elements)
    var host_output = UnsafePointer[Float32].alloc(total_elements)

    # Initialize input with random activations (seed=42)
    seed(42)
    for i in range(total_elements):
        host_input[i] = Float32(random_float64() * 2.0 - 1.0)  # [-1, 1]

    # Create device buffers
    var dev_input = ctx.enqueue_create_buffer[dtype](total_elements)
    var dev_output = ctx.enqueue_create_buffer[dtype](total_elements)
    var dev_cos = ctx.enqueue_create_buffer[dtype](table_size)
    var dev_sin = ctx.enqueue_create_buffer[dtype](table_size)

    # Copy to device
    if dtype == DType.float16:
        var host_input_fp16 = UnsafePointer[Float16].alloc(total_elements)
        var host_cos_fp16 = UnsafePointer[Float16].alloc(table_size)
        var host_sin_fp16 = UnsafePointer[Float16].alloc(table_size)

        for i in range(total_elements):
            host_input_fp16[i] = Float16(host_input[i])
        for i in range(table_size):
            host_cos_fp16[i] = Float16(host_cos[i])
            host_sin_fp16[i] = Float16(host_sin[i])

        ctx.enqueue_copy(dev_input, host_input_fp16, total_elements)
        ctx.enqueue_copy(dev_cos, host_cos_fp16, table_size)
        ctx.enqueue_copy(dev_sin, host_sin_fp16, table_size)

        host_input_fp16.free()
        host_cos_fp16.free()
        host_sin_fp16.free()
    else:
        ctx.enqueue_copy(dev_input, host_input, total_elements)
        ctx.enqueue_copy(dev_cos, host_cos, table_size)
        ctx.enqueue_copy(dev_sin, host_sin, table_size)

    ctx.synchronize()

    # Grid configuration
    var threads_per_block = 256
    var blocks = (total_elements + threads_per_block - 1) // threads_per_block

    # Warmup
    for _ in range(WARMUP):
        ctx.enqueue_function[gpu_rope_kernel[
            seq_len, HEAD_DIM, dtype,
            Layout.row_major(total_elements),
            Layout.row_major(table_size),
            Layout.row_major(table_size),
            Layout.row_major(total_elements)
        ]](
            dev_input.as_tensor(),
            dev_cos.as_tensor(),
            dev_sin.as_tensor(),
            dev_output.as_tensor(),
            grid_dim=(blocks,), block_dim=(threads_per_block,)
        )
        ctx.synchronize()

    # Timed iterations
    var start = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[gpu_rope_kernel[
            seq_len, HEAD_DIM, dtype,
            Layout.row_major(total_elements),
            Layout.row_major(table_size),
            Layout.row_major(table_size),
            Layout.row_major(total_elements)
        ]](
            dev_input.as_tensor(),
            dev_cos.as_tensor(),
            dev_sin.as_tensor(),
            dev_output.as_tensor(),
            grid_dim=(blocks,), block_dim=(threads_per_block,)
        )
        ctx.synchronize()
    var elapsed = perf_counter_ns() - start

    # Cleanup
    host_input.free()
    host_output.free()
    host_cos.free()
    host_sin.free()

    return Float64(elapsed) / Float64(ITERS) / 1000.0


fn compute_bandwidth(total_elements: Int, dtype_size: Int, elapsed_us: Float64) -> Float64:
    """Compute memory bandwidth in GB/s.

    RoPE reads: input + cos + sin
    RoPE writes: output
    Total: 4 arrays, but cos/sin are smaller (seq_len * head_dim/2)
    Approximate as 2 * input_size for simplicity (dominant term)
    """
    var bytes = Float64(2 * total_elements * dtype_size)
    var elapsed_s = elapsed_us / 1_000_000.0
    return bytes / elapsed_s / 1_000_000_000.0


fn main() raises:
    print("=== Mojo GPU RoPE Benchmark ===")
    print("Config: batch=1, heads=", NUM_HEADS, ", head_dim=", HEAD_DIM)
    print("RoPE base:", ROPE_BASE)
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
    print("--- FP32 RoPE ---")
    for i in range(len(seq_lengths)):
        var S = seq_lengths[i]
        var name = names[i]
        var total_elements = 1 * NUM_HEADS * S * HEAD_DIM

        var elapsed_us = bench_rope_shape[FP32](ctx, S, name)
        var gbs = compute_bandwidth(total_elements, 4, elapsed_us)

        print(name, ":")
        print("  Shape: (1,", NUM_HEADS, ",", S, ",", HEAD_DIM, ")")
        print("  Elements:", total_elements)
        print("  Time:", elapsed_us, "us")
        print("  Bandwidth:", gbs, "GB/s")
        print()

    # Benchmark FP16
    print("--- FP16 RoPE ---")
    for i in range(len(seq_lengths)):
        var S = seq_lengths[i]
        var name = names[i]
        var total_elements = 1 * NUM_HEADS * S * HEAD_DIM

        var elapsed_us = bench_rope_shape[FP16](ctx, S, name)
        var gbs = compute_bandwidth(total_elements, 2, elapsed_us)

        print(name, ":")
        print("  Shape: (1,", NUM_HEADS, ",", S, ",", HEAD_DIM, ")")
        print("  Elements:", total_elements)
        print("  Time:", elapsed_us, "us")
        print("  Bandwidth:", gbs, "GB/s")
        print()

    print("=== RoPE Benchmark Complete ===")
    print("NOTE: Results should be post-processed with adaptive_bench harness")
    print("      and written to JSON for comparison with MLX.")
