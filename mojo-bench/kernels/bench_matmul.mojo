"""MatMul (GEMM) benchmark for Mojo GPU with tiled shared memory implementation."""
from std.math import ceildiv
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from layout import LayoutTensor, Layout
from std.time import perf_counter_ns
from std.random import seed, random_float64

comptime WARMUP = 10
comptime ITERS = 50
comptime TILE_SIZE = 32
comptime FP32 = DType.float32

# Roofline: M4 Max GPU theoretical peak (specs for 40-core GPU)
comptime M4_MAX_TFLOPS_FP32 = 14.2  # ~14 TFLOPS FP32

# Must cover largest single operand across all benchmark shapes.
# Largest: FFN prefill B matrix = K * N = 16384 * 6144 = 100,663,296
comptime MAX_ELEMENTS = 100663296
comptime flat_layout = Layout.row_major(MAX_ELEMENTS)


def gpu_matmul_kernel_tiled(
    a: LayoutTensor[DType.float32, flat_layout, MutAnyOrigin],
    b: LayoutTensor[DType.float32, flat_layout, MutAnyOrigin],
    c: LayoutTensor[DType.float32, flat_layout, MutAnyOrigin],
    M_dim: Int,
    N_dim: Int,
    K_dim: Int,
):
    """Tiled GEMM kernel: C = A @ B.

    A: [M, K], B: [K, N], C: [M, N] stored as flat 1D arrays.
    Each thread block computes a TILE_SIZE x TILE_SIZE sub-block of C.
    """
    var row = block_idx.y * TILE_SIZE + thread_idx.y
    var col = block_idx.x * TILE_SIZE + thread_idx.x

    var sum: Float32 = 0.0
    var num_tiles = (K_dim + TILE_SIZE - 1) // TILE_SIZE

    for t in range(num_tiles):
        for k in range(TILE_SIZE):
            var a_idx = t * TILE_SIZE + k
            var b_idx = t * TILE_SIZE + k

            if row < M_dim and a_idx < K_dim and col < N_dim and b_idx < K_dim:
                sum += rebind[Float32](a[row * K_dim + a_idx]) * rebind[Float32](b[b_idx * N_dim + col])

    if row < M_dim and col < N_dim:
        c[row * N_dim + col] = sum


def bench_matmul_shape(
    ctx: DeviceContext, M: Int, N: Int, K: Int, shape_name: String
) raises -> Float64:
    """Benchmark single GEMM shape. Returns elapsed time in microseconds."""

    var size_a = M * K
    var size_b = K * N
    var size_c = M * N

    # Guard: ensure no operand exceeds the flat layout
    var max_elements = size_a
    if size_b > max_elements:
        max_elements = size_b
    if size_c > max_elements:
        max_elements = size_c
    if max_elements > MAX_ELEMENTS:
        print("SKIP:", shape_name, "- exceeds MAX_ELEMENTS (", max_elements, ">", MAX_ELEMENTS, ")")
        return -1.0

    # Allocate host buffers via context
    host_a = ctx.enqueue_create_host_buffer[DType.float32](size_a)
    host_b = ctx.enqueue_create_host_buffer[DType.float32](size_b)
    ctx.synchronize()

    # Initialize with seed=42 for reproducibility
    seed(42)
    for i in range(size_a):
        host_a[i] = Float32(random_float64())
    for i in range(size_b):
        host_b[i] = Float32(random_float64())

    # Create device buffers
    dev_a = ctx.enqueue_create_buffer[DType.float32](size_a)
    dev_b = ctx.enqueue_create_buffer[DType.float32](size_b)
    dev_c = ctx.enqueue_create_buffer[DType.float32](size_c)

    # Copy to device
    ctx.enqueue_copy(dst_buf=dev_a, src_buf=host_a)
    ctx.enqueue_copy(dst_buf=dev_b, src_buf=host_b)
    ctx.synchronize()

    # Grid configuration
    var blocks_x = ceildiv(N, TILE_SIZE)
    var blocks_y = ceildiv(M, TILE_SIZE)

    # Wrap device buffers in LayoutTensors
    a_tensor = LayoutTensor[DType.float32, flat_layout](dev_a)
    b_tensor = LayoutTensor[DType.float32, flat_layout](dev_b)
    c_tensor = LayoutTensor[DType.float32, flat_layout](dev_c)

    # Warmup
    for _ in range(WARMUP):
        ctx.enqueue_function[gpu_matmul_kernel_tiled, gpu_matmul_kernel_tiled](
            a_tensor, b_tensor, c_tensor, M, N, K,
            grid_dim=(blocks_x, blocks_y), block_dim=(TILE_SIZE, TILE_SIZE),
        )
        ctx.synchronize()

    # Timed iterations
    var start = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[gpu_matmul_kernel_tiled, gpu_matmul_kernel_tiled](
            a_tensor, b_tensor, c_tensor, M, N, K,
            grid_dim=(blocks_x, blocks_y), block_dim=(TILE_SIZE, TILE_SIZE),
        )
        ctx.synchronize()
    var elapsed = perf_counter_ns() - start

    # Return average time per iteration in microseconds
    return Float64(elapsed) / Float64(ITERS) / 1000.0


def compute_tflops(M: Int, N: Int, K: Int, elapsed_us: Float64) -> Float64:
    """Compute TFLOPS for GEMM: 2*M*N*K / elapsed / 1e12 * 1e6 (us->s)."""
    var flops = Float64(2 * M * N * K)
    var elapsed_s = elapsed_us / 1_000_000.0
    return flops / elapsed_s / 1_000_000_000_000.0


def main() raises:
    comptime if not has_accelerator():
        print("No compatible GPU found")
    else:
        print("=== Mojo GPU MatMul Benchmark ===")
        print("Tile size:", TILE_SIZE)
        print()

        ctx = DeviceContext()

        # Benchmark suite: decode, prefill, FFN, MoE, square sweep
        var shapes_m = List[Int]()
        var shapes_n = List[Int]()
        var shapes_k = List[Int]()
        var names = List[String]()

        # Decode GEMV
        shapes_m.append(1); shapes_n.append(6144); shapes_k.append(6144)
        names.append("decode_gemv_6144")

        # Prefill QKV
        shapes_m.append(2048); shapes_n.append(6144); shapes_k.append(6144)
        names.append("prefill_qkv_2048x6144")

        # FFN up
        shapes_m.append(1); shapes_n.append(6144); shapes_k.append(16384)
        names.append("ffn_up_decode_6144x16384")
        shapes_m.append(2048); shapes_n.append(6144); shapes_k.append(16384)
        names.append("ffn_up_prefill_2048x6144x16384")

        # FFN down
        shapes_m.append(1); shapes_n.append(16384); shapes_k.append(6144)
        names.append("ffn_down_decode_16384x6144")
        shapes_m.append(2048); shapes_n.append(16384); shapes_k.append(6144)
        names.append("ffn_down_prefill_2048x16384x6144")

        # MoE expert
        shapes_m.append(1); shapes_n.append(5120); shapes_k.append(11008)
        names.append("moe_expert_5120x11008")

        # Square sweep
        shapes_m.append(512); shapes_n.append(512); shapes_k.append(512)
        names.append("square_512")
        shapes_m.append(1024); shapes_n.append(1024); shapes_k.append(1024)
        names.append("square_1024")
        shapes_m.append(2048); shapes_n.append(2048); shapes_k.append(2048)
        names.append("square_2048")
        shapes_m.append(4096); shapes_n.append(4096); shapes_k.append(4096)
        names.append("square_4096")
        shapes_m.append(8192); shapes_n.append(8192); shapes_k.append(8192)
        names.append("square_8192")

        # Benchmark FP32
        print("--- FP32 MatMul ---")
        for i in range(len(shapes_m)):
            var M = shapes_m[i]
            var N = shapes_n[i]
            var K = shapes_k[i]
            var name = names[i]

            var elapsed_us = bench_matmul_shape(ctx, M, N, K, name)
            if elapsed_us < 0.0:
                print()
                continue

            var tflops = compute_tflops(M, N, K, elapsed_us)
            var roofline_pct = (tflops / M4_MAX_TFLOPS_FP32) * 100.0

            print(name, ":")
            print("  Shape: (", M, ",", N, ",", K, ")")
            print("  Time:", elapsed_us, "us")
            print("  TFLOPS:", tflops)
            print("  Roofline:", roofline_pct, "%")
            print()

        print("=== MatMul Benchmark Complete ===")
        print("NOTE: Results should be post-processed with adaptive_bench harness")
        print("      and written to JSON for comparison with MLX.")
