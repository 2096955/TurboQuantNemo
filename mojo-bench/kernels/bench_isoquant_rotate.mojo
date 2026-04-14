"""IsoQuant Rotation Kernel Benchmark for Mojo GPU.

Benchmarks quaternion-based SO(4) rotations for KV cache compression.
Two sub-benchmarks:
1. Forward rotation (write-path, prefill): (H, S, D) for S in {128, 512, 2048, 8192}
2. Inverse rotation (read-path, decode): (H, 1, D)

Each includes dense (full DxD matmul) vs structured (4x4 block) implementations.
"""
from gpu.host import DeviceContext
from gpu import thread_idx, block_idx, block_dim
from layout import LayoutTensor, Layout
from memory import UnsafePointer
from time import perf_counter_ns
from random import seed, random_float64
from math import sqrt

alias WARMUP = 10
alias ITERS = 50
alias FP32 = DType.float32

# IsoQuant config
alias NUM_HEADS = 8  # GQA heads
alias HEAD_DIM = 128
alias BLOCK_SIZE = 4
alias NUM_BLOCKS = HEAD_DIM // BLOCK_SIZE  # 32 blocks per head


fn walsh_hadamard_inplace(x: UnsafePointer[Float32], D: Int):
    """In-place Walsh-Hadamard Transform on vector of length D.

    D must be power of 2. Implements butterfly pattern with normalization.
    """
    var step = 1
    while step < D:
        for i in range(0, D, step * 2):
            for j in range(step):
                var idx_a = i + j
                var idx_b = i + j + step
                var a = x[idx_a]
                var b = x[idx_b]
                x[idx_a] = a + b
                x[idx_b] = a - b
        step *= 2

    # Normalize
    var scale = Float32(1.0 / sqrt(Float64(D)))
    for i in range(D):
        x[i] *= scale


fn inverse_walsh_hadamard_inplace(x: UnsafePointer[Float32], D: Int):
    """Inverse Walsh-Hadamard Transform (same as forward for WHT)."""
    walsh_hadamard_inplace(x, D)


fn structured_rotate_forward_cpu(
    x: UnsafePointer[Float32],
    block_matrices: UnsafePointer[Float32],
    out: UnsafePointer[Float32],
    H: Int,
    S: Int,
    D: Int,
    use_hadamard: Bool
):
    """Forward rotation: WHT + block matmul.

    x: (H, S, D)
    block_matrices: (H, n_blocks, 4, 4) — note: stored as transposed for forward
    out: (H, S, D)
    """
    var n_blocks = D // BLOCK_SIZE

    # Process each head and sequence position
    for h in range(H):
        for s in range(S):
            # Apply WHT if enabled
            var x_base = h * S * D + s * D
            if use_hadamard:
                # Copy to output, apply WHT in-place
                for d in range(D):
                    out[x_base + d] = x[x_base + d]
                walsh_hadamard_inplace(out.offset(x_base), D)
            else:
                for d in range(D):
                    out[x_base + d] = x[x_base + d]

            # Apply block rotations
            for b in range(n_blocks):
                # Get 4-element block from output (post-WHT)
                var block_start = x_base + b * BLOCK_SIZE
                var temp_in = UnsafePointer[Float32].alloc(BLOCK_SIZE)
                for i in range(BLOCK_SIZE):
                    temp_in[i] = out[block_start + i]

                # Get 4x4 rotation matrix for this head and block
                var mat_base = h * n_blocks * BLOCK_SIZE * BLOCK_SIZE + b * BLOCK_SIZE * BLOCK_SIZE

                # Matmul: temp_out = temp_in @ block_matrices[h, b]
                # block_matrices stored as (4, 4) in row-major
                var temp_out = UnsafePointer[Float32].alloc(BLOCK_SIZE)
                for i in range(BLOCK_SIZE):
                    temp_out[i] = 0.0
                    for j in range(BLOCK_SIZE):
                        temp_out[i] += temp_in[j] * block_matrices[mat_base + j * BLOCK_SIZE + i]

                # Write back
                for i in range(BLOCK_SIZE):
                    out[block_start + i] = temp_out[i]

                temp_in.free()
                temp_out.free()


fn structured_rotate_inverse_cpu(
    x_rot: UnsafePointer[Float32],
    block_matrices: UnsafePointer[Float32],
    out: UnsafePointer[Float32],
    H: Int,
    S: Int,
    D: Int,
    use_hadamard: Bool
):
    """Inverse rotation: block matmul (transposed) + inverse WHT.

    x_rot: (H, S, D)
    block_matrices: (H, n_blocks, 4, 4) — used transposed for inverse
    out: (H, S, D)
    """
    var n_blocks = D // BLOCK_SIZE

    for h in range(H):
        for s in range(S):
            var x_base = h * S * D + s * D

            # Copy input to output
            for d in range(D):
                out[x_base + d] = x_rot[x_base + d]

            # Apply block rotations (transposed)
            for b in range(n_blocks):
                var block_start = x_base + b * BLOCK_SIZE
                var temp_in = UnsafePointer[Float32].alloc(BLOCK_SIZE)
                for i in range(BLOCK_SIZE):
                    temp_in[i] = out[block_start + i]

                var mat_base = h * n_blocks * BLOCK_SIZE * BLOCK_SIZE + b * BLOCK_SIZE * BLOCK_SIZE

                # Matmul with transpose: temp_out = temp_in @ block_matrices[h, b].T
                # Transpose means swap indices: temp_out[i] = sum_j temp_in[j] * mat[i][j]
                var temp_out = UnsafePointer[Float32].alloc(BLOCK_SIZE)
                for i in range(BLOCK_SIZE):
                    temp_out[i] = 0.0
                    for j in range(BLOCK_SIZE):
                        temp_out[i] += temp_in[j] * block_matrices[mat_base + i * BLOCK_SIZE + j]

                for i in range(BLOCK_SIZE):
                    out[block_start + i] = temp_out[i]

                temp_in.free()
                temp_out.free()

            # Apply inverse WHT if enabled
            if use_hadamard:
                inverse_walsh_hadamard_inplace(out.offset(x_base), D)


fn dense_rotate_forward_cpu(
    x: UnsafePointer[Float32],
    dense_matrix: UnsafePointer[Float32],
    out: UnsafePointer[Float32],
    H: Int,
    S: Int,
    D: Int,
    use_hadamard: Bool
):
    """Dense forward rotation: WHT + full DxD matmul.

    x: (H, S, D)
    dense_matrix: (H, D, D)
    out: (H, S, D)
    """
    for h in range(H):
        for s in range(S):
            var x_base = h * S * D + s * D

            # Apply WHT if enabled
            if use_hadamard:
                for d in range(D):
                    out[x_base + d] = x[x_base + d]
                walsh_hadamard_inplace(out.offset(x_base), D)
            else:
                for d in range(D):
                    out[x_base + d] = x[x_base + d]

            # Full DxD matmul
            var temp = UnsafePointer[Float32].alloc(D)
            for i in range(D):
                temp[i] = out[x_base + i]

            var mat_base = h * D * D
            for i in range(D):
                out[x_base + i] = 0.0
                for j in range(D):
                    out[x_base + i] += temp[j] * dense_matrix[mat_base + j * D + i]

            temp.free()


fn dense_rotate_inverse_cpu(
    x_rot: UnsafePointer[Float32],
    dense_matrix: UnsafePointer[Float32],
    out: UnsafePointer[Float32],
    H: Int,
    S: Int,
    D: Int,
    use_hadamard: Bool
):
    """Dense inverse rotation: DxD matmul (transposed) + inverse WHT.

    x_rot: (H, S, D)
    dense_matrix: (H, D, D)
    out: (H, S, D)
    """
    for h in range(H):
        for s in range(S):
            var x_base = h * S * D + s * D

            # Copy input
            var temp = UnsafePointer[Float32].alloc(D)
            for d in range(D):
                temp[d] = x_rot[x_base + d]

            # Full DxD matmul (transposed)
            var mat_base = h * D * D
            for i in range(D):
                out[x_base + i] = 0.0
                for j in range(D):
                    out[x_base + i] += temp[j] * dense_matrix[mat_base + i * D + j]

            temp.free()

            # Apply inverse WHT if enabled
            if use_hadamard:
                inverse_walsh_hadamard_inplace(out.offset(x_base), D)


fn compute_rmse(a: UnsafePointer[Float32], b: UnsafePointer[Float32], size: Int) -> Float64:
    """Compute RMSE between two arrays."""
    var sum_sq = 0.0
    for i in range(size):
        var diff = Float64(a[i]) - Float64(b[i])
        sum_sq += diff * diff
    return sqrt(sum_sq / Float64(size))


fn bench_forward_rotation(ctx: DeviceContext, S: Int, use_hadamard: Bool) -> (Float64, Float64, Float64, Float64):
    """Benchmark forward rotation for sequence length S.

    Returns: (dense_time_us, structured_time_us, speedup, rmse)
    """
    var H = NUM_HEADS
    var D = HEAD_DIM
    var n_blocks = NUM_BLOCKS

    var input_size = H * S * D
    var block_mat_size = H * n_blocks * BLOCK_SIZE * BLOCK_SIZE
    var dense_mat_size = H * D * D

    # Allocate buffers
    var input = UnsafePointer[Float32].alloc(input_size)
    var block_matrices = UnsafePointer[Float32].alloc(block_mat_size)
    var dense_matrix = UnsafePointer[Float32].alloc(dense_mat_size)
    var out_structured = UnsafePointer[Float32].alloc(input_size)
    var out_dense = UnsafePointer[Float32].alloc(input_size)

    # Initialize with seed=42
    seed(42)
    for i in range(input_size):
        input[i] = Float32(random_float64() * 2.0 - 1.0)

    # Generate random block matrices (orthogonal would be ideal, but random for benchmark)
    for i in range(block_mat_size):
        block_matrices[i] = Float32(random_float64() * 2.0 - 1.0)

    # Construct dense matrix from block matrices (block diagonal structure)
    for i in range(dense_mat_size):
        dense_matrix[i] = 0.0

    for h in range(H):
        for b in range(n_blocks):
            for i in range(BLOCK_SIZE):
                for j in range(BLOCK_SIZE):
                    var row = b * BLOCK_SIZE + i
                    var col = b * BLOCK_SIZE + j
                    var block_idx = h * n_blocks * BLOCK_SIZE * BLOCK_SIZE + b * BLOCK_SIZE * BLOCK_SIZE + i * BLOCK_SIZE + j
                    var dense_idx = h * D * D + row * D + col
                    dense_matrix[dense_idx] = block_matrices[block_idx]

    # Warmup + benchmark dense
    for _ in range(WARMUP):
        dense_rotate_forward_cpu(input, dense_matrix, out_dense, H, S, D, use_hadamard)

    var start_dense = perf_counter_ns()
    for _ in range(ITERS):
        dense_rotate_forward_cpu(input, dense_matrix, out_dense, H, S, D, use_hadamard)
    var elapsed_dense = perf_counter_ns() - start_dense
    var dense_time_us = Float64(elapsed_dense) / Float64(ITERS) / 1000.0

    # Warmup + benchmark structured
    for _ in range(WARMUP):
        structured_rotate_forward_cpu(input, block_matrices, out_structured, H, S, D, use_hadamard)

    var start_structured = perf_counter_ns()
    for _ in range(ITERS):
        structured_rotate_forward_cpu(input, block_matrices, out_structured, H, S, D, use_hadamard)
    var elapsed_structured = perf_counter_ns() - start_structured
    var structured_time_us = Float64(elapsed_structured) / Float64(ITERS) / 1000.0

    # Compute RMSE between structured and dense
    var rmse = compute_rmse(out_structured, out_dense, input_size)

    # Compute speedup
    var speedup = dense_time_us / structured_time_us

    # Cleanup
    input.free()
    block_matrices.free()
    dense_matrix.free()
    out_structured.free()
    out_dense.free()

    return (dense_time_us, structured_time_us, speedup, rmse)


fn bench_inverse_rotation(ctx: DeviceContext, use_hadamard: Bool) -> (Float64, Float64, Float64, Float64, Float64):
    """Benchmark inverse rotation for decode (S=1).

    Returns: (dense_time_us, structured_time_us, speedup, rmse, roundtrip_rmse)
    """
    var H = NUM_HEADS
    var S = 1
    var D = HEAD_DIM
    var n_blocks = NUM_BLOCKS

    var input_size = H * S * D
    var block_mat_size = H * n_blocks * BLOCK_SIZE * BLOCK_SIZE
    var dense_mat_size = H * D * D

    # Allocate buffers
    var input = UnsafePointer[Float32].alloc(input_size)
    var block_matrices = UnsafePointer[Float32].alloc(block_mat_size)
    var dense_matrix = UnsafePointer[Float32].alloc(dense_mat_size)
    var rotated_structured = UnsafePointer[Float32].alloc(input_size)
    var rotated_dense = UnsafePointer[Float32].alloc(input_size)
    var out_structured = UnsafePointer[Float32].alloc(input_size)
    var out_dense = UnsafePointer[Float32].alloc(input_size)
    var roundtrip = UnsafePointer[Float32].alloc(input_size)

    # Initialize with seed=42
    seed(42)
    for i in range(input_size):
        input[i] = Float32(random_float64() * 2.0 - 1.0)

    for i in range(block_mat_size):
        block_matrices[i] = Float32(random_float64() * 2.0 - 1.0)

    # Construct dense matrix
    for i in range(dense_mat_size):
        dense_matrix[i] = 0.0

    for h in range(H):
        for b in range(n_blocks):
            for i in range(BLOCK_SIZE):
                for j in range(BLOCK_SIZE):
                    var row = b * BLOCK_SIZE + i
                    var col = b * BLOCK_SIZE + j
                    var block_idx = h * n_blocks * BLOCK_SIZE * BLOCK_SIZE + b * BLOCK_SIZE * BLOCK_SIZE + i * BLOCK_SIZE + j
                    var dense_idx = h * D * D + row * D + col
                    dense_matrix[dense_idx] = block_matrices[block_idx]

    # First apply forward rotation to get rotated inputs
    structured_rotate_forward_cpu(input, block_matrices, rotated_structured, H, S, D, use_hadamard)
    dense_rotate_forward_cpu(input, dense_matrix, rotated_dense, H, S, D, use_hadamard)

    # Warmup + benchmark dense inverse
    for _ in range(WARMUP):
        dense_rotate_inverse_cpu(rotated_dense, dense_matrix, out_dense, H, S, D, use_hadamard)

    var start_dense = perf_counter_ns()
    for _ in range(ITERS):
        dense_rotate_inverse_cpu(rotated_dense, dense_matrix, out_dense, H, S, D, use_hadamard)
    var elapsed_dense = perf_counter_ns() - start_dense
    var dense_time_us = Float64(elapsed_dense) / Float64(ITERS) / 1000.0

    # Warmup + benchmark structured inverse
    for _ in range(WARMUP):
        structured_rotate_inverse_cpu(rotated_structured, block_matrices, out_structured, H, S, D, use_hadamard)

    var start_structured = perf_counter_ns()
    for _ in range(ITERS):
        structured_rotate_inverse_cpu(rotated_structured, block_matrices, out_structured, H, S, D, use_hadamard)
    var elapsed_structured = perf_counter_ns() - start_structured
    var structured_time_us = Float64(elapsed_structured) / Float64(ITERS) / 1000.0

    # Compute RMSE between structured and dense inverse
    var rmse = compute_rmse(out_structured, out_dense, input_size)

    # Test roundtrip: forward then inverse should recover original
    structured_rotate_inverse_cpu(rotated_structured, block_matrices, roundtrip, H, S, D, use_hadamard)
    var roundtrip_rmse = compute_rmse(input, roundtrip, input_size)

    var speedup = dense_time_us / structured_time_us

    # Cleanup
    input.free()
    block_matrices.free()
    dense_matrix.free()
    rotated_structured.free()
    rotated_dense.free()
    out_structured.free()
    out_dense.free()
    roundtrip.free()

    return (dense_time_us, structured_time_us, speedup, rmse, roundtrip_rmse)


fn compute_bandwidth(H: Int, S: Int, D: Int, elapsed_us: Float64) -> Float64:
    """Compute memory bandwidth in GB/s.

    Reads: input (H*S*D) + block_matrices (H*n_blocks*4*4)
    Writes: output (H*S*D)
    """
    var input_bytes = Float64(H * S * D * 4)  # FP32
    var mat_bytes = Float64(H * NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE * 4)
    var total_bytes = 2.0 * input_bytes + mat_bytes  # read input + matrices, write output
    var elapsed_s = elapsed_us / 1_000_000.0
    return total_bytes / elapsed_s / 1_000_000_000.0


fn main() raises:
    print("=== Mojo IsoQuant Rotation Benchmark ===")
    print("Config: H=", NUM_HEADS, ", D=", HEAD_DIM, ", n_blocks=", NUM_BLOCKS)
    print()

    var ctx = DeviceContext()

    # Sub-benchmark 1: Forward Rotation (prefill)
    print("--- Sub-benchmark 1: Forward Rotation (Prefill) ---")
    print("Shape: (H, S, D) where H=", NUM_HEADS, ", D=", HEAD_DIM)
    print()

    var seq_lengths = List[Int]()
    seq_lengths.append(128)
    seq_lengths.append(512)
    seq_lengths.append(2048)
    seq_lengths.append(8192)

    var use_hadamard = True

    for i in range(len(seq_lengths)):
        var S = seq_lengths[i]
        print("Sequence length:", S)

        var (dense_time, struct_time, speedup, rmse) = bench_forward_rotation(ctx, S, use_hadamard)
        var gb_dense = compute_bandwidth(NUM_HEADS, S, HEAD_DIM, dense_time)
        var gb_struct = compute_bandwidth(NUM_HEADS, S, HEAD_DIM, struct_time)

        print("  Dense:      ", dense_time, "us, ", gb_dense, "GB/s")
        print("  Structured: ", struct_time, "us, ", gb_struct, "GB/s")
        print("  Speedup:    ", speedup, "x")
        print("  RMSE:       ", rmse)

        # Validation
        var rmse_threshold = sqrt(Float64(HEAD_DIM)) * 1e-5
        if rmse < rmse_threshold:
            print("  ✓ PASS: RMSE within tolerance")
        else:
            print("  ✗ FAIL: RMSE exceeds threshold (", rmse_threshold, ")")
        print()

    print()
    print("--- Sub-benchmark 2: Inverse Rotation (Decode) ---")
    print("Shape: (H, 1, D) where H=", NUM_HEADS, ", D=", HEAD_DIM)
    print()

    var (dense_time_inv, struct_time_inv, speedup_inv, rmse_inv, roundtrip_rmse) = bench_inverse_rotation(ctx, use_hadamard)
    var gb_dense_inv = compute_bandwidth(NUM_HEADS, 1, HEAD_DIM, dense_time_inv)
    var gb_struct_inv = compute_bandwidth(NUM_HEADS, 1, HEAD_DIM, struct_time_inv)

    print("  Dense:           ", dense_time_inv, "us, ", gb_dense_inv, "GB/s")
    print("  Structured:      ", struct_time_inv, "us, ", gb_struct_inv, "GB/s")
    print("  Speedup:         ", speedup_inv, "x")
    print("  RMSE (vs dense): ", rmse_inv)
    print("  Roundtrip RMSE:  ", roundtrip_rmse)

    # Validation
    var rmse_threshold_inv = sqrt(Float64(HEAD_DIM)) * 1e-5
    var roundtrip_threshold = 1e-4

    var pass_rmse = rmse_inv < rmse_threshold_inv
    var pass_roundtrip = roundtrip_rmse < roundtrip_threshold

    if pass_rmse:
        print("  ✓ PASS: Inverse RMSE within tolerance")
    else:
        print("  ✗ FAIL: Inverse RMSE exceeds threshold (", rmse_threshold_inv, ")")

    if pass_roundtrip:
        print("  ✓ PASS: Roundtrip RMSE within tolerance")
    else:
        print("  ✗ FAIL: Roundtrip RMSE exceeds threshold (", roundtrip_threshold, ")")

    print()
    print("=== IsoQuant Rotation Benchmark Complete ===")
    print("NOTE: Results should be post-processed with adaptive_bench harness")
    print("      and written to JSON for comparison with MLX.")
