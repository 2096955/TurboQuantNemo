"""Rotary Position Embeddings (RoPE) benchmark for Mojo GPU."""
from std.math import ceildiv, sin, cos, pow
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from layout import LayoutTensor, Layout
from std.time import perf_counter_ns
from std.random import seed, random_float64

comptime WARMUP = 10
comptime ITERS = 50
comptime FP32 = DType.float32

# RoPE config matching Qwen/LLaMA models
comptime NUM_HEADS = 48
comptime HEAD_DIM = 128
comptime ROPE_BASE = 10000.0

# Maximum flat layout for input tensor
# Largest shape: [1, 48, 32768, 128] = 201,326,592 elements
comptime MAX_INPUT_ELEMENTS = 201326592
comptime flat_input_layout = Layout.row_major(MAX_INPUT_ELEMENTS)

# Maximum flat layout for cos/sin tables
# Largest: 32768 * 64 = 2,097,152
comptime MAX_TABLE_ELEMENTS = 2097152
comptime flat_table_layout = Layout.row_major(MAX_TABLE_ELEMENTS)


def gpu_rope_kernel(
    input: LayoutTensor[DType.float32, flat_input_layout, MutAnyOrigin],
    cos_table: LayoutTensor[DType.float32, flat_table_layout, MutAnyOrigin],
    sin_table: LayoutTensor[DType.float32, flat_table_layout, MutAnyOrigin],
    output: LayoutTensor[DType.float32, flat_input_layout, MutAnyOrigin],
    seq_len: Int,
    head_dim: Int,
):
    """Apply Rotary Position Embeddings.

    Input shape: [batch, num_heads, seq_len, head_dim]
    cos/sin tables: [seq_len, head_dim/2]

    For each position and dimension pair (2i, 2i+1):
    out[2i]   = input[2i]   * cos[i] - input[2i+1] * sin[i]
    out[2i+1] = input[2i+1] * cos[i] + input[2i]   * sin[i]

    Each thread processes one dimension PAIR to avoid race conditions.
    """
    var tid = block_idx.x * block_dim.x + thread_idx.x

    var total_pairs = 1 * NUM_HEADS * seq_len * (head_dim // 2)

    if tid >= total_pairs:
        return

    # Decode position in input tensor for this pair
    var pair_idx_in_head = tid % (head_dim // 2)
    var head_idx = (tid // (head_dim // 2)) % NUM_HEADS
    var pos = (tid // (head_dim // 2)) // NUM_HEADS

    # Calculate indices for the dimension pair (2i, 2i+1)
    var even_idx = (pos * NUM_HEADS + head_idx) * head_dim + (pair_idx_in_head * 2)
    var odd_idx = even_idx + 1

    # Get frequency table index
    var freq_idx = pos * (head_dim // 2) + pair_idx_in_head

    var cos_val = rebind[Float32](cos_table[freq_idx])
    var sin_val = rebind[Float32](sin_table[freq_idx])

    # Read BOTH input values BEFORE writing ANY output
    var x0 = rebind[Float32](input[even_idx])
    var x1 = rebind[Float32](input[odd_idx])

    # Compute both output values
    output[even_idx] = x0 * cos_val - x1 * sin_val
    output[odd_idx] = x0 * sin_val + x1 * cos_val


def bench_rope_shape(
    ctx: DeviceContext, seq_len: Int, shape_name: String
) raises -> Float64:
    """Benchmark RoPE for given sequence length. Returns elapsed time in microseconds."""

    var batch = 1
    var total_elements = batch * NUM_HEADS * seq_len * HEAD_DIM
    var table_size = seq_len * (HEAD_DIM // 2)

    # Precompute frequency tables on host using managed buffers
    host_cos = ctx.enqueue_create_host_buffer[DType.float32](table_size)
    host_sin = ctx.enqueue_create_host_buffer[DType.float32](table_size)
    ctx.synchronize()

    var half_dim = HEAD_DIM // 2
    for pos in range(seq_len):
        for i in range(half_dim):
            var freq = 1.0 / pow(Float64(ROPE_BASE), Float64(2 * i) / Float64(HEAD_DIM))
            var angle = Float64(pos) * freq
            var idx = pos * half_dim + i
            host_cos[idx] = Float32(cos(angle))
            host_sin[idx] = Float32(sin(angle))

    # Allocate input
    host_input = ctx.enqueue_create_host_buffer[DType.float32](total_elements)
    ctx.synchronize()

    # Initialize input with random activations (seed=42)
    seed(42)
    for i in range(total_elements):
        host_input[i] = Float32(random_float64() * 2.0 - 1.0)

    # Create device buffers
    dev_input = ctx.enqueue_create_buffer[DType.float32](total_elements)
    dev_output = ctx.enqueue_create_buffer[DType.float32](total_elements)
    dev_cos = ctx.enqueue_create_buffer[DType.float32](table_size)
    dev_sin = ctx.enqueue_create_buffer[DType.float32](table_size)

    # Copy to device
    ctx.enqueue_copy(dst_buf=dev_input, src_buf=host_input)
    ctx.enqueue_copy(dst_buf=dev_cos, src_buf=host_cos)
    ctx.enqueue_copy(dst_buf=dev_sin, src_buf=host_sin)
    ctx.synchronize()

    # Grid configuration: one thread per dimension PAIR
    var total_pairs = total_elements // 2
    var threads_per_block = 256
    var blocks = ceildiv(total_pairs, threads_per_block)

    # Wrap device buffers in LayoutTensors
    input_tensor = LayoutTensor[DType.float32, flat_input_layout](dev_input)
    output_tensor = LayoutTensor[DType.float32, flat_input_layout](dev_output)
    cos_tensor = LayoutTensor[DType.float32, flat_table_layout](dev_cos)
    sin_tensor = LayoutTensor[DType.float32, flat_table_layout](dev_sin)

    # Warmup
    for _ in range(WARMUP):
        ctx.enqueue_function[gpu_rope_kernel, gpu_rope_kernel](
            input_tensor, cos_tensor, sin_tensor, output_tensor,
            seq_len, HEAD_DIM,
            grid_dim=blocks, block_dim=threads_per_block,
        )
        ctx.synchronize()

    # Timed iterations
    var start = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[gpu_rope_kernel, gpu_rope_kernel](
            input_tensor, cos_tensor, sin_tensor, output_tensor,
            seq_len, HEAD_DIM,
            grid_dim=blocks, block_dim=threads_per_block,
        )
        ctx.synchronize()
    var elapsed = perf_counter_ns() - start

    return Float64(elapsed) / Float64(ITERS) / 1000.0


def compute_bandwidth(total_elements: Int, dtype_size: Int, elapsed_us: Float64) -> Float64:
    """Compute memory bandwidth in GB/s.

    RoPE reads: input + cos + sin
    RoPE writes: output
    Approximate as 2 * input_size for simplicity (dominant term).
    """
    var bytes = Float64(2 * total_elements * dtype_size)
    var elapsed_s = elapsed_us / 1_000_000.0
    return bytes / elapsed_s / 1_000_000_000.0


def main() raises:
    comptime if not has_accelerator():
        print("No compatible GPU found")
    else:
        print("=== Mojo GPU RoPE Benchmark ===")
        print("Config: batch=1, heads=", NUM_HEADS, ", head_dim=", HEAD_DIM)
        print("RoPE base:", ROPE_BASE)
        print()

        ctx = DeviceContext()

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

            var elapsed_us = bench_rope_shape(ctx, S, name)
            var gbs = compute_bandwidth(total_elements, 4, elapsed_us)

            print(name, ":")
            print("  Shape: (1,", NUM_HEADS, ",", S, ",", HEAD_DIM, ")")
            print("  Elements:", total_elements)
            print("  Time:", elapsed_us, "us")
            print("  Bandwidth:", gbs, "GB/s")
            print()

        print("=== RoPE Benchmark Complete ===")
        print("NOTE: Results should be post-processed with adaptive_bench harness")
        print("      and written to JSON for comparison with MLX.")
