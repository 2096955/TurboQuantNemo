"""IsoQuant Rotation Kernel Benchmark for Mojo GPU.

Benchmarks quaternion-based SO(4) rotations for KV cache compression.
Two sub-benchmarks:
1. Forward rotation (write-path, prefill): (H, S, D) for S in {128, 512, 2048, 8192}
2. Inverse rotation (read-path, decode): (H, 1, D)

Each includes dense (full DxD matmul) vs structured (4x4 block) implementations.
"""
from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from layout import LayoutTensor, Layout
from std.memory import UnsafePointer
from std.time import perf_counter_ns
from std.random import seed, random_float64
from std.math import sqrt

comptime WARMUP = 10
comptime ITERS = 50
comptime FP32 = DType.float32

# IsoQuant config
comptime NUM_HEADS = 8  # GQA heads
comptime HEAD_DIM = 128
comptime BLOCK_SIZE = 4
comptime NUM_BLOCKS = HEAD_DIM // BLOCK_SIZE  # 32 blocks per head


def walsh_hadamard_inplace(x: UnsafePointer[Float32, MutAnyOrigin], D: Int):
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


def inverse_walsh_hadamard_inplace(x: UnsafePointer[Float32, MutAnyOrigin], D: Int):
    """Inverse Walsh-Hadamard Transform (same as forward for WHT)."""
    walsh_hadamard_inplace(x, D)


def generate_so4_from_quaternion(mat: UnsafePointer[Float32, MutAnyOrigin], offset: Int):
    """Generate a proper SO(4) left-isoclinic rotation matrix from a random unit quaternion.

    Constructs a 4x4 orthogonal matrix from quaternion (a, b, c, d) using left-isoclinic form:
        L = [[ a, -b, -c, -d],
             [ b,  a, -d,  c],
             [ c,  d,  a, -b],
             [ d, -c,  b,  a]]

    This matrix satisfies L^T @ L = I (orthogonal) when quaternion is normalized.

    Args:
        mat: Pointer to matrix storage
        offset: Base offset for this 4x4 block (in flat array)
    """
    # Generate 4 random values
    var a = Float32(random_float64() * 2.0 - 1.0)
    var b = Float32(random_float64() * 2.0 - 1.0)
    var c = Float32(random_float64() * 2.0 - 1.0)
    var d = Float32(random_float64() * 2.0 - 1.0)

    # Normalize to unit quaternion
    var norm = sqrt(Float64(a*a + b*b + c*c + d*d))
    a = Float32(Float64(a) / norm)
    b = Float32(Float64(b) / norm)
    c = Float32(Float64(c) / norm)
    d = Float32(Float64(d) / norm)

    # Construct left-isoclinic SO(4) matrix (row-major storage)
    # Row 0: [ a, -b, -c, -d]
    mat[offset + 0*4 + 0] = a
    mat[offset + 0*4 + 1] = -b
    mat[offset + 0*4 + 2] = -c
    mat[offset + 0*4 + 3] = -d

    # Row 1: [ b,  a, -d,  c]
    mat[offset + 1*4 + 0] = b
    mat[offset + 1*4 + 1] = a
    mat[offset + 1*4 + 2] = -d
    mat[offset + 1*4 + 3] = c

    # Row 2: [ c,  d,  a, -b]
    mat[offset + 2*4 + 0] = c
    mat[offset + 2*4 + 1] = d
    mat[offset + 2*4 + 2] = a
    mat[offset + 2*4 + 3] = -b

    # Row 3: [ d, -c,  b,  a]
    mat[offset + 3*4 + 0] = d
    mat[offset + 3*4 + 1] = -c
    mat[offset + 3*4 + 2] = b
    mat[offset + 3*4 + 3] = a


def structured_rotate_forward_cpu(
    x: UnsafePointer[Float32, MutAnyOrigin],
    block_matrices: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    H: Int,
    S: Int,
    D: Int,
    use_hadamard: Bool
):
    """Forward rotation: WHT + block matmul.

    x: (H, S, D)
    block_matrices: (H, n_blocks, 4, 4) -- stored as transposed for forward.
    dst: (H, S, D).
    """
    var n_blocks = D // BLOCK_SIZE

    # Pre-allocate scratch buffers outside hot loops
    var temp_in_list = List[Float32](capacity=BLOCK_SIZE)
    var temp_out_list = List[Float32](capacity=BLOCK_SIZE)
    for _ in range(BLOCK_SIZE):
        temp_in_list.append(0.0)
        temp_out_list.append(0.0)
    var temp_in = temp_in_list.unsafe_ptr()
    var temp_out = temp_out_list.unsafe_ptr()

    # Process each head and sequence position
    for h in range(H):
        for s in range(S):
            # Apply WHT if enabled
            var x_base = h * S * D + s * D
            if use_hadamard:
                # Copy to output, apply WHT in-place
                for d in range(D):
                    dst[x_base + d] = x[x_base + d]
                walsh_hadamard_inplace(dst + x_base, D)
            else:
                for d in range(D):
                    dst[x_base + d] = x[x_base + d]

            # Apply block rotations
            for b in range(n_blocks):
                # Get 4-element block from output (post-WHT)
                var block_start = x_base + b * BLOCK_SIZE
                for i in range(BLOCK_SIZE):
                    temp_in[i] = dst[block_start + i]

                # Get 4x4 rotation matrix for this head and block
                var mat_base = h * n_blocks * BLOCK_SIZE * BLOCK_SIZE + b * BLOCK_SIZE * BLOCK_SIZE

                # Matmul: temp_out = temp_in @ block_matrices[h, b]
                for i in range(BLOCK_SIZE):
                    temp_out[i] = 0.0
                    for j in range(BLOCK_SIZE):
                        temp_out[i] += temp_in[j] * block_matrices[mat_base + j * BLOCK_SIZE + i]

                # Write back
                for i in range(BLOCK_SIZE):
                    dst[block_start + i] = temp_out[i]


def structured_rotate_inverse_cpu(
    x_rot: UnsafePointer[Float32, MutAnyOrigin],
    block_matrices: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    H: Int,
    S: Int,
    D: Int,
    use_hadamard: Bool
):
    """Inverse rotation: block matmul (transposed) + inverse WHT.

    x_rot: (H, S, D).
    block_matrices: (H, n_blocks, 4, 4) -- used transposed for inverse.
    dst: (H, S, D).
    """
    var n_blocks = D // BLOCK_SIZE

    # Pre-allocate scratch buffers outside hot loops
    var temp_in_list = List[Float32](capacity=BLOCK_SIZE)
    var temp_out_list = List[Float32](capacity=BLOCK_SIZE)
    for _ in range(BLOCK_SIZE):
        temp_in_list.append(0.0)
        temp_out_list.append(0.0)
    var temp_in = temp_in_list.unsafe_ptr()
    var temp_out = temp_out_list.unsafe_ptr()

    for h in range(H):
        for s in range(S):
            var x_base = h * S * D + s * D

            # Copy input to output
            for d in range(D):
                dst[x_base + d] = x_rot[x_base + d]

            # Apply block rotations (transposed)
            for b in range(n_blocks):
                var block_start = x_base + b * BLOCK_SIZE
                for i in range(BLOCK_SIZE):
                    temp_in[i] = dst[block_start + i]

                var mat_base = h * n_blocks * BLOCK_SIZE * BLOCK_SIZE + b * BLOCK_SIZE * BLOCK_SIZE

                # Matmul with transpose: temp_out = temp_in @ block_matrices[h, b].T
                # Transpose means swap indices: temp_out[i] = sum_j temp_in[j] * mat[i][j]
                for i in range(BLOCK_SIZE):
                    temp_out[i] = 0.0
                    for j in range(BLOCK_SIZE):
                        temp_out[i] += temp_in[j] * block_matrices[mat_base + i * BLOCK_SIZE + j]

                for i in range(BLOCK_SIZE):
                    dst[block_start + i] = temp_out[i]

            # Apply inverse WHT if enabled
            if use_hadamard:
                inverse_walsh_hadamard_inplace(dst + x_base, D)


def dense_rotate_forward_cpu(
    x: UnsafePointer[Float32, MutAnyOrigin],
    dense_matrix: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    H: Int,
    S: Int,
    D: Int,
    use_hadamard: Bool
):
    """Dense forward rotation: WHT + full DxD matmul.

    x: (H, S, D).
    dense_matrix: (H, D, D).
    dst: (H, S, D).
    """
    # Pre-allocate scratch buffer outside hot loops
    var temp_list = List[Float32](capacity=D)
    for _ in range(D):
        temp_list.append(0.0)
    var temp = temp_list.unsafe_ptr()

    for h in range(H):
        for s in range(S):
            var x_base = h * S * D + s * D

            # Apply WHT if enabled
            if use_hadamard:
                for d in range(D):
                    dst[x_base + d] = x[x_base + d]
                walsh_hadamard_inplace(dst + x_base, D)
            else:
                for d in range(D):
                    dst[x_base + d] = x[x_base + d]

            # Full DxD matmul
            for i in range(D):
                temp[i] = dst[x_base + i]

            var mat_base = h * D * D
            for i in range(D):
                dst[x_base + i] = 0.0
                for j in range(D):
                    dst[x_base + i] += temp[j] * dense_matrix[mat_base + j * D + i]


def dense_rotate_inverse_cpu(
    x_rot: UnsafePointer[Float32, MutAnyOrigin],
    dense_matrix: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    H: Int,
    S: Int,
    D: Int,
    use_hadamard: Bool
):
    """Dense inverse rotation: DxD matmul (transposed) + inverse WHT.

    x_rot: (H, S, D).
    dense_matrix: (H, D, D).
    dst: (H, S, D).
    """
    # Pre-allocate scratch buffer outside hot loops
    var temp_list = List[Float32](capacity=D)
    for _ in range(D):
        temp_list.append(0.0)
    var temp = temp_list.unsafe_ptr()

    for h in range(H):
        for s in range(S):
            var x_base = h * S * D + s * D

            # Copy input
            for d in range(D):
                temp[d] = x_rot[x_base + d]

            # Full DxD matmul (transposed)
            var mat_base = h * D * D
            for i in range(D):
                dst[x_base + i] = 0.0
                for j in range(D):
                    dst[x_base + i] += temp[j] * dense_matrix[mat_base + i * D + j]

            # Apply inverse WHT if enabled
            if use_hadamard:
                inverse_walsh_hadamard_inplace(dst + x_base, D)


def compute_rmse(a: UnsafePointer[Float32, MutAnyOrigin], b: UnsafePointer[Float32, MutAnyOrigin], size: Int) -> Float64:
    """Compute RMSE between two arrays."""
    var sum_sq = 0.0
    for i in range(size):
        var diff = Float64(a[i]) - Float64(b[i])
        sum_sq += diff * diff
    return sqrt(sum_sq / Float64(size))


def bench_forward_rotation(ctx: DeviceContext, S: Int, use_hadamard: Bool) -> Tuple[Float64, Float64, Float64, Float64]:
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
    var input_list = List[Float32](capacity=input_size)
    for _ in range(input_size):
        input_list.append(0.0)
    var input = input_list.unsafe_ptr()
    var block_matrices_list = List[Float32](capacity=block_mat_size)
    for _ in range(block_mat_size):
        block_matrices_list.append(0.0)
    var block_matrices = block_matrices_list.unsafe_ptr()
    var dense_matrix_list = List[Float32](capacity=dense_mat_size)
    for _ in range(dense_mat_size):
        dense_matrix_list.append(0.0)
    var dense_matrix = dense_matrix_list.unsafe_ptr()
    var out_structured_list = List[Float32](capacity=input_size)
    for _ in range(input_size):
        out_structured_list.append(0.0)
    var out_structured = out_structured_list.unsafe_ptr()
    var out_dense_list = List[Float32](capacity=input_size)
    for _ in range(input_size):
        out_dense_list.append(0.0)
    var out_dense = out_dense_list.unsafe_ptr()

    # Initialize with seed=42
    seed(42)
    for i in range(input_size):
        input[i] = Float32(random_float64() * 2.0 - 1.0)

    # Generate proper SO(4) rotation matrices from quaternions (one per block)
    for h in range(H):
        for b in range(n_blocks):
            var block_offset = h * n_blocks * BLOCK_SIZE * BLOCK_SIZE + b * BLOCK_SIZE * BLOCK_SIZE
            generate_so4_from_quaternion(block_matrices, block_offset)

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

    return Tuple(dense_time_us, structured_time_us, speedup, rmse)


def bench_inverse_rotation(ctx: DeviceContext, use_hadamard: Bool) -> Tuple[Float64, Float64, Float64, Float64, Float64]:
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
    var input_list = List[Float32](capacity=input_size)
    for _ in range(input_size):
        input_list.append(0.0)
    var input = input_list.unsafe_ptr()
    var block_matrices_list = List[Float32](capacity=block_mat_size)
    for _ in range(block_mat_size):
        block_matrices_list.append(0.0)
    var block_matrices = block_matrices_list.unsafe_ptr()
    var dense_matrix_list = List[Float32](capacity=dense_mat_size)
    for _ in range(dense_mat_size):
        dense_matrix_list.append(0.0)
    var dense_matrix = dense_matrix_list.unsafe_ptr()
    var rotated_structured_list = List[Float32](capacity=input_size)
    for _ in range(input_size):
        rotated_structured_list.append(0.0)
    var rotated_structured = rotated_structured_list.unsafe_ptr()
    var rotated_dense_list = List[Float32](capacity=input_size)
    for _ in range(input_size):
        rotated_dense_list.append(0.0)
    var rotated_dense = rotated_dense_list.unsafe_ptr()
    var out_structured_list = List[Float32](capacity=input_size)
    for _ in range(input_size):
        out_structured_list.append(0.0)
    var out_structured = out_structured_list.unsafe_ptr()
    var out_dense_list = List[Float32](capacity=input_size)
    for _ in range(input_size):
        out_dense_list.append(0.0)
    var out_dense = out_dense_list.unsafe_ptr()
    var roundtrip_list = List[Float32](capacity=input_size)
    for _ in range(input_size):
        roundtrip_list.append(0.0)
    var roundtrip = roundtrip_list.unsafe_ptr()

    # Initialize with seed=42
    seed(42)
    for i in range(input_size):
        input[i] = Float32(random_float64() * 2.0 - 1.0)

    # Generate proper SO(4) rotation matrices from quaternions
    for h in range(H):
        for b in range(n_blocks):
            var block_offset = h * n_blocks * BLOCK_SIZE * BLOCK_SIZE + b * BLOCK_SIZE * BLOCK_SIZE
            generate_so4_from_quaternion(block_matrices, block_offset)

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

    return Tuple(dense_time_us, structured_time_us, speedup, rmse, roundtrip_rmse)


def compute_bandwidth(H: Int, S: Int, D: Int, elapsed_us: Float64, use_hadamard: Bool) -> Float64:
    """Compute memory bandwidth in GB/s.

    Reads: input (H*S*D) + block_matrices (H*n_blocks*4*4) + WHT passes
    Writes: output (H*S*D)

    WHT performs log2(D) passes, each reading and writing D elements per sequence position.
    """
    var input_bytes = Float64(H * S * D * 4)  # FP32
    var mat_bytes = Float64(H * NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE * 4)
    var total_bytes = 2.0 * input_bytes + mat_bytes  # read input + matrices, write output

    # Add WHT bandwidth: log2(D) passes * D reads per (H, S) position
    if use_hadamard:
        var log2_D = 0
        var D_temp = D
        while D_temp > 1:
            log2_D += 1
            D_temp = D_temp // 2
        var wht_bytes = Float64(log2_D * H * S * D * 4)  # FP32 reads per pass
        total_bytes += wht_bytes

    var elapsed_s = elapsed_us / 1_000_000.0
    return total_bytes / elapsed_s / 1_000_000_000.0


def main() raises:
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

        var fwd_result = bench_forward_rotation(ctx, S, use_hadamard)
        var dense_time = fwd_result[0]
        var struct_time = fwd_result[1]
        var speedup = fwd_result[2]
        var rmse = fwd_result[3]
        var gb_dense = compute_bandwidth(NUM_HEADS, S, HEAD_DIM, dense_time, use_hadamard)
        var gb_struct = compute_bandwidth(NUM_HEADS, S, HEAD_DIM, struct_time, use_hadamard)

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

    var inv_result = bench_inverse_rotation(ctx, use_hadamard)
    var dense_time_inv = inv_result[0]
    var struct_time_inv = inv_result[1]
    var speedup_inv = inv_result[2]
    var rmse_inv = inv_result[3]
    var roundtrip_rmse = inv_result[4]
    var gb_dense_inv = compute_bandwidth(NUM_HEADS, 1, HEAD_DIM, dense_time_inv, use_hadamard)
    var gb_struct_inv = compute_bandwidth(NUM_HEADS, 1, HEAD_DIM, struct_time_inv, use_hadamard)

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
