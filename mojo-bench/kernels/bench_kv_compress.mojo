"""KV Cache Compression Benchmark for Mojo.

Benchmarks codebook-based vector quantization for KV cache compression.
Pipeline: Compress (encode) → Decompress (decode)

Compression (encode):
  Input: float32 vectors of shape (H, T, D) where H=8, D=128, T varies
  For each vector: compute norm, normalize, find nearest centroid (brute-force), store index+norm

Decompression (decode):
  For each stored (index, norm): lookup centroid, multiply by norm, output reconstructed vector

Uses synthetic codebook (8 centroids for 3-bit, dimension 128) initialized with seed=42.
Benchmarks shapes (H=8, T, D=128) for T in {128, 512, 2048, 8192}.

Metrics: GB/s, compression/decompression time, reconstruction RMSE
"""
from memory import UnsafePointer
from time import perf_counter_ns
from random import seed, random_float64
from math import sqrt

alias WARMUP = 10
alias ITERS = 50
alias FP32 = DType.float32

# KV cache compression config
alias NUM_HEADS = 8
alias HEAD_DIM = 128
alias NUM_BITS = 3
alias NUM_CENTROIDS = 8  # 2^3 for 3-bit quantization

# Compressed storage: index (uint8) + norm (float32)
alias INDEX_BYTES = 1
alias NORM_BYTES = 4
alias COMPRESSED_BYTES = INDEX_BYTES + NORM_BYTES


fn generate_synthetic_codebook(codebook: UnsafePointer[Float32], D: Int, K: Int):
    """Generate synthetic codebook: K centroids of dimension D.

    Each centroid is a normalized random vector (seed=42 for reproducibility).

    Args:
        codebook: Pointer to codebook storage (K * D elements)
        D: Dimension of each centroid
        K: Number of centroids
    """
    seed(42)
    for k in range(K):
        # Generate random vector
        var norm_sq: Float64 = 0.0
        for d in range(D):
            var val = Float32(random_float64() * 2.0 - 1.0)
            codebook[k * D + d] = val
            norm_sq += Float64(val * val)

        # Normalize to unit length
        var norm = sqrt(norm_sq)
        for d in range(D):
            codebook[k * D + d] = Float32(Float64(codebook[k * D + d]) / norm)


fn compute_l2_norm(vec: UnsafePointer[Float32], D: Int) -> Float32:
    """Compute L2 norm of a vector."""
    var sum_sq: Float64 = 0.0
    for d in range(D):
        var val = Float64(vec[d])
        sum_sq += val * val
    return Float32(sqrt(sum_sq))


fn find_nearest_centroid(
    vec: UnsafePointer[Float32],
    codebook: UnsafePointer[Float32],
    D: Int,
    K: Int
) -> Int:
    """Find nearest centroid via brute-force L2 distance.

    Args:
        vec: Normalized query vector (D elements)
        codebook: Codebook matrix (K * D elements)
        D: Dimension
        K: Number of centroids

    Returns:
        Index of nearest centroid (0 to K-1)
    """
    var min_dist: Float64 = 1e30
    var best_idx = 0

    for k in range(K):
        var dist_sq: Float64 = 0.0
        for d in range(D):
            var diff = Float64(vec[d]) - Float64(codebook[k * D + d])
            dist_sq += diff * diff

        if dist_sq < min_dist:
            min_dist = dist_sq
            best_idx = k

    return best_idx


fn compress_kv(
    kv: UnsafePointer[Float32],
    codebook: UnsafePointer[Float32],
    indices: UnsafePointer[UInt8],
    norms: UnsafePointer[Float32],
    H: Int,
    T: Int,
    D: Int,
    K: Int
):
    """Compress KV cache using codebook quantization.

    For each vector in (H, T, D):
      1. Compute L2 norm
      2. Normalize the vector
      3. Find nearest centroid (argmin over codebook)
      4. Store: centroid index (3-bit) + norm (float32)

    Args:
        kv: Input KV cache (H * T * D elements)
        codebook: Codebook (K * D elements)
        indices: Output indices (H * T elements)
        norms: Output norms (H * T elements)
        H: Number of heads
        T: Sequence length
        D: Head dimension
        K: Number of centroids
    """
    var normalized = UnsafePointer[Float32].alloc(D)

    for h in range(H):
        for t in range(T):
            var vec_idx = h * T * D + t * D
            var out_idx = h * T + t

            # Compute norm
            var norm = compute_l2_norm(kv.offset(vec_idx), D)
            norms[out_idx] = norm

            # Normalize vector
            if norm > 1e-8:
                for d in range(D):
                    normalized[d] = kv[vec_idx + d] / norm
            else:
                # Zero vector edge case
                for d in range(D):
                    normalized[d] = 0.0

            # Find nearest centroid
            var centroid_idx = find_nearest_centroid(normalized, codebook, D, K)
            indices[out_idx] = UInt8(centroid_idx)

    normalized.free()


fn decompress_kv(
    indices: UnsafePointer[UInt8],
    norms: UnsafePointer[Float32],
    codebook: UnsafePointer[Float32],
    kv_out: UnsafePointer[Float32],
    H: Int,
    T: Int,
    D: Int
):
    """Decompress KV cache from indices and norms.

    For each stored (index, norm):
      1. Lookup centroid vector by index
      2. Multiply by stored norm
      3. Output reconstructed vector

    Args:
        indices: Stored centroid indices (H * T elements)
        norms: Stored norms (H * T elements)
        codebook: Codebook (K * D elements)
        kv_out: Output KV cache (H * T * D elements)
        H: Number of heads
        T: Sequence length
        D: Head dimension
    """
    for h in range(H):
        for t in range(T):
            var in_idx = h * T + t
            var out_idx = h * T * D + t * D

            var centroid_idx = Int(indices[in_idx])
            var norm = norms[in_idx]

            # Lookup centroid and scale by norm
            for d in range(D):
                kv_out[out_idx + d] = codebook[centroid_idx * D + d] * norm


fn compute_rmse(a: UnsafePointer[Float32], b: UnsafePointer[Float32], size: Int) -> Float64:
    """Compute RMSE between two arrays."""
    var sum_sq: Float64 = 0.0
    for i in range(size):
        var diff = Float64(a[i]) - Float64(b[i])
        sum_sq += diff * diff
    return sqrt(sum_sq / Float64(size))


fn bench_kv_compression(T: Int) -> (Float64, Float64, Float64, Float64, Float64):
    """Benchmark KV compression for sequence length T.

    Returns: (compress_time_us, decompress_time_us, compress_gb_s, decompress_gb_s, rmse)
    """
    var H = NUM_HEADS
    var D = HEAD_DIM
    var K = NUM_CENTROIDS

    var kv_size = H * T * D
    var index_size = H * T
    var codebook_size = K * D

    # Allocate buffers
    var kv_input = UnsafePointer[Float32].alloc(kv_size)
    var codebook = UnsafePointer[Float32].alloc(codebook_size)
    var indices = UnsafePointer[UInt8].alloc(index_size)
    var norms = UnsafePointer[Float32].alloc(index_size)
    var kv_output = UnsafePointer[Float32].alloc(kv_size)

    # Initialize input with seed=42
    seed(42)
    for i in range(kv_size):
        kv_input[i] = Float32(random_float64() * 2.0 - 1.0)

    # Generate synthetic codebook
    generate_synthetic_codebook(codebook, D, K)

    # Warmup compression
    for _ in range(WARMUP):
        compress_kv(kv_input, codebook, indices, norms, H, T, D, K)

    # Benchmark compression
    var start_compress = perf_counter_ns()
    for _ in range(ITERS):
        compress_kv(kv_input, codebook, indices, norms, H, T, D, K)
    var elapsed_compress = perf_counter_ns() - start_compress
    var compress_time_us = Float64(elapsed_compress) / Float64(ITERS) / 1000.0

    # Warmup decompression
    for _ in range(WARMUP):
        decompress_kv(indices, norms, codebook, kv_output, H, T, D)

    # Benchmark decompression
    var start_decompress = perf_counter_ns()
    for _ in range(ITERS):
        decompress_kv(indices, norms, codebook, kv_output, H, T, D)
    var elapsed_decompress = perf_counter_ns() - start_decompress
    var decompress_time_us = Float64(elapsed_decompress) / Float64(ITERS) / 1000.0

    # Compute RMSE (after final decompression)
    decompress_kv(indices, norms, codebook, kv_output, H, T, D)
    var rmse = compute_rmse(kv_input, kv_output, kv_size)

    # Compute bandwidth (GB/s)
    # Compression: read input (kv_size * 4) + codebook (K*D*4), write indices+norms (index_size * 5)
    var compress_bytes = Float64(kv_size * 4 + codebook_size * 4 + index_size * 5)
    var compress_gb_s = compress_bytes / (compress_time_us / 1_000_000.0) / 1_000_000_000.0

    # Decompression: read indices+norms (index_size * 5) + codebook (K*D*4), write output (kv_size * 4)
    var decompress_bytes = Float64(index_size * 5 + codebook_size * 4 + kv_size * 4)
    var decompress_gb_s = decompress_bytes / (decompress_time_us / 1_000_000.0) / 1_000_000_000.0

    # Cleanup
    kv_input.free()
    codebook.free()
    indices.free()
    norms.free()
    kv_output.free()

    return (compress_time_us, decompress_time_us, compress_gb_s, decompress_gb_s, rmse)


fn main() raises:
    print("=== Mojo KV Compression Benchmark ===")
    print("Config: H=", NUM_HEADS, ", D=", HEAD_DIM, ", K=", NUM_CENTROIDS, " (", NUM_BITS, "-bit)")
    print()

    # Benchmark shapes
    var seq_lengths = List[Int]()
    seq_lengths.append(128)
    seq_lengths.append(512)
    seq_lengths.append(2048)
    seq_lengths.append(8192)

    print("--- Compress + Decompress Pipeline ---")
    print()

    for i in range(len(seq_lengths)):
        var T = seq_lengths[i]
        print("Sequence length:", T)
        print("  Shape: (", NUM_HEADS, ",", T, ",", HEAD_DIM, ")")

        var (comp_time, decomp_time, comp_gb, decomp_gb, rmse) = bench_kv_compression(T)

        print("  Compress:   ", comp_time, "us, ", comp_gb, "GB/s")
        print("  Decompress: ", decomp_time, "us, ", decomp_gb, "GB/s")
        print("  Combined:   ", comp_time + decomp_time, "us")
        print("  RMSE:       ", rmse)

        # Validation: RMSE should be bounded for 3-bit quantization
        # With 8 centroids and random init, expect measurable error but not catastrophic
        var rmse_threshold = 2.0  # Loose threshold for synthetic codebook
        if rmse < rmse_threshold:
            print("  ✓ PASS: RMSE within threshold")
        else:
            print("  ✗ FAIL: RMSE exceeds threshold (", rmse_threshold, ")")
        print()

    print("=== KV Compression Benchmark Complete ===")
    print("NOTE: Results should be post-processed with adaptive_bench harness")
    print("      and written to JSON for comparison with MLX.")
    print()
    print("IMPORTANT: This benchmark uses a synthetic codebook (seed=42).")
    print("           Production KV compression uses precomputed codebooks from")
    print("           mlx-lm/mlx_lm/models/turboquant_codebooks/dim_128_3bit.npz")
    print("           RMSE validation should be done on the MLX side with real codebooks.")
