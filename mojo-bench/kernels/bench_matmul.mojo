"""MatMul (GEMM) benchmark for Mojo GPU with tiled shared memory implementation."""
from gpu.host import DeviceContext
from gpu import thread_idx, block_idx, block_dim, barrier
from layout import LayoutTensor, Layout
from memory import UnsafePointer
from time import perf_counter_ns
from random import seed, random_float64

# Import harness functions (manual inclusion due to Mojo module system)
# Harness functions would be imported here in production - stats and output utilities

alias WARMUP = 10
alias ITERS = 50
alias TILE_SIZE = 32
alias FP16 = DType.float16
alias FP32 = DType.float32

# Roofline: M4 Max GPU theoretical peak (specs for 40-core GPU)
alias M4_MAX_TFLOPS_FP32 = 14.2  # ~14 TFLOPS FP32
alias M4_MAX_TFLOPS_FP16 = 28.4  # ~28 TFLOPS FP16


fn gpu_matmul_kernel_tiled[
    M: Int,
    N: Int,
    K: Int,
    dtype: DType,
    layout_a: Layout,
    layout_b: Layout,
    layout_c: Layout
](
    a: LayoutTensor[mut=False, dtype=dtype, layout=layout_a],
    b: LayoutTensor[mut=False, dtype=dtype, layout=layout_b],
    c: LayoutTensor[mut=True, dtype=dtype, layout=layout_c]
):
    """Tiled GEMM kernel with shared memory: C = A @ B

    A: [M, K], B: [K, N], C: [M, N]
    Each thread block computes a TILE_SIZE x TILE_SIZE sub-block of C.
    Uses shared memory for A and B tiles to reduce global memory traffic.
    """
    # Thread block coordinates in output matrix
    var row = block_idx.y * TILE_SIZE + thread_idx.y
    var col = block_idx.x * TILE_SIZE + thread_idx.x

    # Accumulator for this thread's output element
    var sum: Float32 = 0.0

    # Number of tiles along K dimension
    var num_tiles = (K + TILE_SIZE - 1) // TILE_SIZE

    # TODO: Shared memory allocation in Mojo GPU
    # In production Mojo Metal backend, this would use:
    # var tile_a = shared_memory[TILE_SIZE * TILE_SIZE, dtype]()
    # var tile_b = shared_memory[TILE_SIZE * TILE_SIZE, dtype]()
    #
    # Current Mojo GPU API (Tier 3) may not have shared memory exposed yet.
    # For now, we implement the tiled pattern without shared memory optimization.
    # This will still produce correct results but with lower performance.

    for t in range(num_tiles):
        # Load tile from A: a[row, t*TILE_SIZE + thread_idx.x]
        # Load tile from B: b[t*TILE_SIZE + thread_idx.y, col]
        var a_col = t * TILE_SIZE + thread_idx.x
        var b_row = t * TILE_SIZE + thread_idx.y

        # Compute contribution from this tile
        # In shared memory version, we'd load the full tile cooperatively,
        # sync, then compute. Here we do direct global loads.
        for k in range(TILE_SIZE):
            var a_idx = t * TILE_SIZE + k
            var b_idx = t * TILE_SIZE + k

            if row < M and a_idx < K and col < N and b_idx < K:
                # A[row, a_idx] @ B[b_idx, col]
                var a_val = Float32(a[row * K + a_idx])
                var b_val = Float32(b[b_idx * N + col])
                sum += a_val * b_val

    # Write result
    if row < M and col < N:
        if dtype == DType.float16:
            c[row * N + col] = Float16(sum)
        else:
            c[row * N + col] = sum


fn bench_matmul_shape[
    dtype: DType
](ctx: DeviceContext, M: Int, N: Int, K: Int, shape_name: String) -> Float64:
    """Benchmark single GEMM shape. Returns elapsed time in microseconds."""

    # Allocate host buffers
    var size_a = M * K
    var size_b = K * N
    var size_c = M * N

    var host_a = UnsafePointer[Float32].alloc(size_a)
    var host_b = UnsafePointer[Float32].alloc(size_b)
    var host_c = UnsafePointer[Float32].alloc(size_c)

    # Initialize with seed=42 for reproducibility
    seed(42)
    for i in range(size_a):
        host_a[i] = Float32(random_float64())
    for i in range(size_b):
        host_b[i] = Float32(random_float64())
    for i in range(size_c):
        host_c[i] = 0.0

    # Create device buffers
    var dev_a = ctx.enqueue_create_buffer[dtype](size_a)
    var dev_b = ctx.enqueue_create_buffer[dtype](size_b)
    var dev_c = ctx.enqueue_create_buffer[dtype](size_c)

    # Copy to device
    if dtype == DType.float16:
        var host_a_fp16 = UnsafePointer[Float16].alloc(size_a)
        var host_b_fp16 = UnsafePointer[Float16].alloc(size_b)
        for i in range(size_a):
            host_a_fp16[i] = Float16(host_a[i])
        for i in range(size_b):
            host_b_fp16[i] = Float16(host_b[i])
        ctx.enqueue_copy(dev_a, host_a_fp16, size_a)
        ctx.enqueue_copy(dev_b, host_b_fp16, size_b)
        host_a_fp16.free()
        host_b_fp16.free()
    else:
        ctx.enqueue_copy(dev_a, host_a, size_a)
        ctx.enqueue_copy(dev_b, host_b, size_b)

    ctx.synchronize()

    # Grid configuration
    var blocks_x = (N + TILE_SIZE - 1) // TILE_SIZE
    var blocks_y = (M + TILE_SIZE - 1) // TILE_SIZE

    # Warmup
    for _ in range(WARMUP):
        ctx.enqueue_function[gpu_matmul_kernel_tiled[
            M, N, K, dtype,
            Layout.row_major(M, K),
            Layout.row_major(K, N),
            Layout.row_major(M, N)
        ]](
            dev_a.as_tensor(), dev_b.as_tensor(), dev_c.as_tensor(),
            grid_dim=(blocks_x, blocks_y), block_dim=(TILE_SIZE, TILE_SIZE)
        )
        ctx.synchronize()

    # Timed iterations
    var start = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[gpu_matmul_kernel_tiled[
            M, N, K, dtype,
            Layout.row_major(M, K),
            Layout.row_major(K, N),
            Layout.row_major(M, N)
        ]](
            dev_a.as_tensor(), dev_b.as_tensor(), dev_c.as_tensor(),
            grid_dim=(blocks_x, blocks_y), block_dim=(TILE_SIZE, TILE_SIZE)
        )
        ctx.synchronize()
    var elapsed = perf_counter_ns() - start

    # Cleanup
    host_a.free()
    host_b.free()
    host_c.free()

    # Return average time per iteration in microseconds
    return Float64(elapsed) / Float64(ITERS) / 1000.0


fn compute_tflops(M: Int, N: Int, K: Int, elapsed_us: Float64) -> Float64:
    """Compute TFLOPS for GEMM: 2*M*N*K / elapsed / 1e12 * 1e6 (us->s)."""
    var flops = Float64(2 * M * N * K)
    var elapsed_s = elapsed_us / 1_000_000.0
    return flops / elapsed_s / 1_000_000_000_000.0


fn main() raises:
    print("=== Mojo GPU MatMul Benchmark ===")
    print("Tile size:", TILE_SIZE)
    print()

    var ctx = DeviceContext()

    # Benchmark suite: decode, prefill, FFN, MoE, square sweep
    # Format: (M, N, K, name)
    alias NUM_SHAPES = 14

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

        var elapsed_us = bench_matmul_shape[FP32](ctx, M, N, K, name)
        var tflops = compute_tflops(M, N, K, elapsed_us)
        var roofline_pct = (tflops / M4_MAX_TFLOPS_FP32) * 100.0

        print(name, ":")
        print("  Shape: (", M, ",", N, ",", K, ")")
        print("  Time:", elapsed_us, "us")
        print("  TFLOPS:", tflops)
        print("  Roofline:", roofline_pct, "%")
        print()

    # Benchmark FP16
    print("--- FP16 MatMul ---")
    for i in range(len(shapes_m)):
        var M = shapes_m[i]
        var N = shapes_n[i]
        var K = shapes_k[i]
        var name = names[i]

        var elapsed_us = bench_matmul_shape[FP16](ctx, M, N, K, name)
        var tflops = compute_tflops(M, N, K, elapsed_us)
        var roofline_pct = (tflops / M4_MAX_TFLOPS_FP16) * 100.0

        print(name, ":")
        print("  Shape: (", M, ",", N, ",", K, ")")
        print("  Time:", elapsed_us, "us")
        print("  TFLOPS:", tflops)
        print("  Roofline:", roofline_pct, "%")
        print()

    print("=== MatMul Benchmark Complete ===")
    print("NOTE: Results should be post-processed with adaptive_bench harness")
    print("      and written to JSON for comparison with MLX.")
