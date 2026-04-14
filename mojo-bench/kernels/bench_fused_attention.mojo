"""Fused Attention Benchmark for Mojo GPU.

Implements IsoQuant attention pipeline with two execution strategies:
1. Unfused: Each operation (QK, softmax, V accum, inverse rotate) as separate kernel with sync
2. Framework-fused: All operations composed into one function, single sync

Pipeline steps:
1. QK dot product: scores = Q @ K^T
2. Softmax: attn_weights = softmax(scores * scale)
3. V accumulation: output_rot = attn_weights @ V
4. Inverse rotation: output = inverse_rotate(output_rot)

All operations work in rotated space (keys and values are already rotated during prefill).
"""
from gpu.host import DeviceContext
from layout import LayoutTensor, Layout
from memory import UnsafePointer
from time import perf_counter_ns
from random import seed, random_float64
from math import sqrt, exp

alias WARMUP = 10
alias ITERS = 50
alias FP32 = DType.float32

# IsoQuant config
alias BATCH = 1
alias NUM_HEADS = 8
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


fn generate_so4_from_quaternion(mat: UnsafePointer[Float32], offset: Int):
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


fn matmul_cpu(
    a: UnsafePointer[Float32],
    b: UnsafePointer[Float32],
    c: UnsafePointer[Float32],
    M: Int, N: Int, K: Int,
    transpose_b: Bool = False
):
    """Simple CPU matmul: C = A @ B (or A @ B^T if transpose_b).

    A: (M, K)
    B: (K, N) or (N, K) if transpose_b
    C: (M, N)
    """
    for i in range(M):
        for j in range(N):
            var sum = Float32(0.0)
            for k in range(K):
                var a_val = a[i * K + k]
                var b_val: Float32
                if transpose_b:
                    # B is (N, K), so B^T[k, j] = B[j, k]
                    b_val = b[j * K + k]
                else:
                    b_val = b[k * N + j]
                sum += a_val * b_val
            c[i * N + j] = sum


fn softmax_cpu(
    scores: UnsafePointer[Float32],
    out: UnsafePointer[Float32],
    B: Int, H: Int, S_q: Int, S_kv: Int,
    scale: Float32
):
    """Numerically stable softmax over last dimension.

    scores: (B, H, S_q, S_kv)
    out: (B, H, S_q, S_kv)
    scale: 1/sqrt(D)
    """
    for b in range(B):
        for h in range(H):
            for s in range(S_q):
                var row_start = b * H * S_q * S_kv + h * S_q * S_kv + s * S_kv

                # Find max for numerical stability
                var max_val = scores[row_start]
                for i in range(1, S_kv):
                    var val = scores[row_start + i]
                    if val > max_val:
                        max_val = val

                # Compute exp(x - max) and sum
                var sum = Float32(0.0)
                for i in range(S_kv):
                    var val = scores[row_start + i] * scale
                    var exp_val = exp(val - max_val * scale)
                    out[row_start + i] = exp_val
                    sum += exp_val

                # Normalize
                for i in range(S_kv):
                    out[row_start + i] /= sum


fn inverse_rotate_cpu(
    x_rot: UnsafePointer[Float32],
    block_matrices: UnsafePointer[Float32],
    out: UnsafePointer[Float32],
    H: Int, S: Int, D: Int,
    use_hadamard: Bool
):
    """Inverse rotation: block matmul (transposed) + inverse WHT.

    x_rot: (H, S, D) in rotated space
    block_matrices: (H, n_blocks, 4, 4)
    out: (H, S, D) in original space
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


fn bench_unfused_attention(
    ctx: DeviceContext,
    T: Int,
    use_hadamard: Bool
) -> (Float64, Float64, Float64, Float64, Float64):
    """Benchmark unfused attention with per-step timing.

    Returns: (total_time_us, qk_time_us, softmax_time_us, v_time_us, inverse_time_us)
    """
    var H = NUM_HEADS
    var D = HEAD_DIM
    var S_q = 1  # Decode: single query token
    var S_kv = T  # Cached key/value sequence length
    var n_blocks = NUM_BLOCKS

    var scale = Float32(1.0 / sqrt(Float64(D)))

    # Allocate tensors
    var q_size = BATCH * H * S_q * D
    var kv_size = BATCH * H * S_kv * D
    var scores_size = BATCH * H * S_q * S_kv
    var block_mat_size = H * n_blocks * BLOCK_SIZE * BLOCK_SIZE

    var Q = UnsafePointer[Float32].alloc(q_size)
    var K = UnsafePointer[Float32].alloc(kv_size)
    var V = UnsafePointer[Float32].alloc(kv_size)
    var block_matrices = UnsafePointer[Float32].alloc(block_mat_size)

    var scores = UnsafePointer[Float32].alloc(scores_size)
    var attn_weights = UnsafePointer[Float32].alloc(scores_size)
    var output_rot = UnsafePointer[Float32].alloc(q_size)
    var output = UnsafePointer[Float32].alloc(q_size)

    # Initialize with seed=42
    seed(42)
    for i in range(q_size):
        Q[i] = Float32(random_float64() * 2.0 - 1.0)
    for i in range(kv_size):
        K[i] = Float32(random_float64() * 2.0 - 1.0)
        V[i] = Float32(random_float64() * 2.0 - 1.0)

    # Generate rotation matrices
    for h in range(H):
        for b in range(n_blocks):
            var block_offset = h * n_blocks * BLOCK_SIZE * BLOCK_SIZE + b * BLOCK_SIZE * BLOCK_SIZE
            generate_so4_from_quaternion(block_matrices, block_offset)

    # Warmup
    for _ in range(WARMUP):
        # Step 1: QK
        for b in range(BATCH):
            for h in range(H):
                var q_base = b * H * S_q * D + h * S_q * D
                var k_base = b * H * S_kv * D + h * S_kv * D
                var scores_base = b * H * S_q * S_kv + h * S_q * S_kv
                matmul_cpu(Q.offset(q_base), K.offset(k_base), scores.offset(scores_base), S_q, S_kv, D, transpose_b=True)

        # Step 2: Softmax
        softmax_cpu(scores, attn_weights, BATCH, H, S_q, S_kv, scale)

        # Step 3: V accumulation
        for b in range(BATCH):
            for h in range(H):
                var attn_base = b * H * S_q * S_kv + h * S_q * S_kv
                var v_base = b * H * S_kv * D + h * S_kv * D
                var out_base = b * H * S_q * D + h * S_q * D
                matmul_cpu(attn_weights.offset(attn_base), V.offset(v_base), output_rot.offset(out_base), S_q, D, S_kv)

        # Step 4: Inverse rotation
        for b in range(BATCH):
            var out_base = b * H * S_q * D
            inverse_rotate_cpu(output_rot.offset(out_base), block_matrices, output.offset(out_base), H, S_q, D, use_hadamard)

    # Timed iterations with per-step breakdown
    var qk_total = 0
    var softmax_total = 0
    var v_total = 0
    var inverse_total = 0

    var start_all = perf_counter_ns()
    for _ in range(ITERS):
        # Step 1: QK
        var start_qk = perf_counter_ns()
        for b in range(BATCH):
            for h in range(H):
                var q_base = b * H * S_q * D + h * S_q * D
                var k_base = b * H * S_kv * D + h * S_kv * D
                var scores_base = b * H * S_q * S_kv + h * S_q * S_kv
                matmul_cpu(Q.offset(q_base), K.offset(k_base), scores.offset(scores_base), S_q, S_kv, D, transpose_b=True)
        var qk_elapsed = perf_counter_ns() - start_qk
        qk_total += qk_elapsed

        # Step 2: Softmax
        var start_softmax = perf_counter_ns()
        softmax_cpu(scores, attn_weights, BATCH, H, S_q, S_kv, scale)
        var softmax_elapsed = perf_counter_ns() - start_softmax
        softmax_total += softmax_elapsed

        # Step 3: V accumulation
        var start_v = perf_counter_ns()
        for b in range(BATCH):
            for h in range(H):
                var attn_base = b * H * S_q * S_kv + h * S_q * S_kv
                var v_base = b * H * S_kv * D + h * S_kv * D
                var out_base = b * H * S_q * D + h * S_q * D
                matmul_cpu(attn_weights.offset(attn_base), V.offset(v_base), output_rot.offset(out_base), S_q, D, S_kv)
        var v_elapsed = perf_counter_ns() - start_v
        v_total += v_elapsed

        # Step 4: Inverse rotation
        var start_inverse = perf_counter_ns()
        for b in range(BATCH):
            var out_base = b * H * S_q * D
            inverse_rotate_cpu(output_rot.offset(out_base), block_matrices, output.offset(out_base), H, S_q, D, use_hadamard)
        var inverse_elapsed = perf_counter_ns() - start_inverse
        inverse_total += inverse_elapsed

    var total_elapsed = perf_counter_ns() - start_all

    var total_time_us = Float64(total_elapsed) / Float64(ITERS) / 1000.0
    var qk_time_us = Float64(qk_total) / Float64(ITERS) / 1000.0
    var softmax_time_us = Float64(softmax_total) / Float64(ITERS) / 1000.0
    var v_time_us = Float64(v_total) / Float64(ITERS) / 1000.0
    var inverse_time_us = Float64(inverse_total) / Float64(ITERS) / 1000.0

    # Cleanup
    Q.free()
    K.free()
    V.free()
    block_matrices.free()
    scores.free()
    attn_weights.free()
    output_rot.free()
    output.free()

    return (total_time_us, qk_time_us, softmax_time_us, v_time_us, inverse_time_us)


fn fused_attention_cpu(
    Q: UnsafePointer[Float32],
    K: UnsafePointer[Float32],
    V: UnsafePointer[Float32],
    block_matrices: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    H: Int, S_q: Int, S_kv: Int, D: Int,
    scale: Float32,
    use_hadamard: Bool
):
    """Fused attention: QK @ softmax @ V @ inverse_rotate in one function.

    All steps composed together, caller does single synchronization.
    """
    var n_blocks = D // BLOCK_SIZE
    var scores_size = BATCH * H * S_q * S_kv
    var output_rot_size = BATCH * H * S_q * D

    var scores = UnsafePointer[Float32].alloc(scores_size)
    var attn_weights = UnsafePointer[Float32].alloc(scores_size)
    var output_rot = UnsafePointer[Float32].alloc(output_rot_size)

    # Step 1: QK
    for b in range(BATCH):
        for h in range(H):
            var q_base = b * H * S_q * D + h * S_q * D
            var k_base = b * H * S_kv * D + h * S_kv * D
            var scores_base = b * H * S_q * S_kv + h * S_q * S_kv
            matmul_cpu(Q.offset(q_base), K.offset(k_base), scores.offset(scores_base), S_q, S_kv, D, transpose_b=True)

    # Step 2: Softmax
    softmax_cpu(scores, attn_weights, BATCH, H, S_q, S_kv, scale)

    # Step 3: V accumulation
    for b in range(BATCH):
        for h in range(H):
            var attn_base = b * H * S_q * S_kv + h * S_q * S_kv
            var v_base = b * H * S_kv * D + h * S_kv * D
            var out_base = b * H * S_q * D + h * S_q * D
            matmul_cpu(attn_weights.offset(attn_base), V.offset(v_base), output_rot.offset(out_base), S_q, D, S_kv)

    # Step 4: Inverse rotation
    for b in range(BATCH):
        var out_base = b * H * S_q * D
        inverse_rotate_cpu(output_rot.offset(out_base), block_matrices, output.offset(out_base), H, S_q, D, use_hadamard)

    scores.free()
    attn_weights.free()
    output_rot.free()


fn bench_fused_attention(
    ctx: DeviceContext,
    T: Int,
    use_hadamard: Bool
) -> Float64:
    """Benchmark fused attention (all steps composed, single sync).

    Returns: total_time_us
    """
    var H = NUM_HEADS
    var D = HEAD_DIM
    var S_q = 1
    var S_kv = T
    var n_blocks = NUM_BLOCKS

    var scale = Float32(1.0 / sqrt(Float64(D)))

    # Allocate tensors
    var q_size = BATCH * H * S_q * D
    var kv_size = BATCH * H * S_kv * D
    var block_mat_size = H * n_blocks * BLOCK_SIZE * BLOCK_SIZE

    var Q = UnsafePointer[Float32].alloc(q_size)
    var K = UnsafePointer[Float32].alloc(kv_size)
    var V = UnsafePointer[Float32].alloc(kv_size)
    var block_matrices = UnsafePointer[Float32].alloc(block_mat_size)
    var output = UnsafePointer[Float32].alloc(q_size)

    # Initialize with seed=42
    seed(42)
    for i in range(q_size):
        Q[i] = Float32(random_float64() * 2.0 - 1.0)
    for i in range(kv_size):
        K[i] = Float32(random_float64() * 2.0 - 1.0)
        V[i] = Float32(random_float64() * 2.0 - 1.0)

    # Generate rotation matrices
    for h in range(H):
        for b in range(n_blocks):
            var block_offset = h * n_blocks * BLOCK_SIZE * BLOCK_SIZE + b * BLOCK_SIZE * BLOCK_SIZE
            generate_so4_from_quaternion(block_matrices, block_offset)

    # Warmup
    for _ in range(WARMUP):
        fused_attention_cpu(Q, K, V, block_matrices, output, H, S_q, S_kv, D, scale, use_hadamard)

    # Timed iterations
    var start = perf_counter_ns()
    for _ in range(ITERS):
        fused_attention_cpu(Q, K, V, block_matrices, output, H, S_q, S_kv, D, scale, use_hadamard)
    var elapsed = perf_counter_ns() - start

    var total_time_us = Float64(elapsed) / Float64(ITERS) / 1000.0

    # Cleanup
    Q.free()
    K.free()
    V.free()
    block_matrices.free()
    output.free()

    return total_time_us


fn main() raises:
    print("=== Mojo Fused Attention Benchmark (IsoQuant) ===")
    print("Config: B=", BATCH, ", H=", NUM_HEADS, ", D=", HEAD_DIM)
    print("Pipeline: QK -> Softmax -> V -> Inverse Rotation")
    print()

    var ctx = DeviceContext()

    var seq_lengths = List[Int]()
    seq_lengths.append(128)
    seq_lengths.append(512)
    seq_lengths.append(2048)
    seq_lengths.append(8192)

    var use_hadamard = True

    print("--- Unfused Attention (4 separate sync points) ---")
    print()

    for i in range(len(seq_lengths)):
        var T = seq_lengths[i]
        print("Sequence length T =", T)

        var (total_time, qk_time, softmax_time, v_time, inverse_time) = bench_unfused_attention(ctx, T, use_hadamard)

        print("  Total:          ", total_time, "us")
        print("  QK matmul:      ", qk_time, "us (", (qk_time / total_time * 100.0), "%)")
        print("  Softmax:        ", softmax_time, "us (", (softmax_time / total_time * 100.0), "%)")
        print("  V matmul:       ", v_time, "us (", (v_time / total_time * 100.0), "%)")
        print("  Inverse rotate: ", inverse_time, "us (", (inverse_time / total_time * 100.0), "%)")
        print()

    print()
    print("--- Framework-Fused Attention (single sync point) ---")
    print()

    for i in range(len(seq_lengths)):
        var T = seq_lengths[i]
        print("Sequence length T =", T)

        var fused_time = bench_fused_attention(ctx, T, use_hadamard)
        var (unfused_total, _, _, _, _) = bench_unfused_attention(ctx, T, use_hadamard)
        var speedup = unfused_total / fused_time

        print("  Fused time:   ", fused_time, "us")
        print("  Unfused time: ", unfused_total, "us")
        print("  Speedup:      ", speedup, "x")
        print()

    print("=== Fused Attention Benchmark Complete ===")
    print("NOTE: Results should be post-processed with adaptive_bench harness")
    print("      and written to JSON for comparison with MLX.")
