"""
TurboQuant Codebook Precompute (Phase 1)
=========================================
Computes Lloyd-Max optimal scalar quantization codebooks for the coordinate
distribution that arises after random orthogonal rotation of unit vectors.

After rotation, each coordinate of a d-dimensional unit vector follows
approximately N(0, 1/d). The Lloyd-Max algorithm finds centroids that
minimize MSE for this distribution.

Usage:
    python codebook_precompute.py

Output:
    codebooks/dim_{d}_{b}bit.npz  (for each dimension and bit-width)
    Each file contains:
      - centroids: shape (2^b,) - reconstruction values
      - boundaries: shape (2^b - 1,) - decision boundaries between centroids
      - sigma: scalar - the standard deviation used (1/sqrt(d))
"""

import os
import numpy as np
from scipy import integrate
from scipy.stats import norm


def gaussian_pdf(x: float, sigma: float) -> float:
    """Gaussian PDF with mean 0 and given sigma."""
    return norm.pdf(x, loc=0, scale=sigma)


def gaussian_cdf(x: float, sigma: float) -> float:
    """Gaussian CDF with mean 0 and given sigma."""
    return norm.cdf(x, loc=0, scale=sigma)


def conditional_expectation(a: float, b: float, sigma: float) -> float:
    """
    Compute E[X | a < X < b] for X ~ N(0, sigma^2).

    Uses the identity:
        E[X | a < X < b] = sigma^2 * (phi(a/sigma) - phi(b/sigma))
                           / (Phi(b/sigma) - Phi(a/sigma))

    where phi is the standard normal PDF and Phi is the standard normal CDF.
    """
    a_norm = a / sigma
    b_norm = b / sigma

    prob = norm.cdf(b_norm) - norm.cdf(a_norm)
    if prob < 1e-15:
        return (a + b) / 2.0

    numerator = sigma * (norm.pdf(a_norm) - norm.pdf(b_norm))
    return numerator / prob


def lloyd_max(num_levels: int, sigma: float, max_iter: int = 200,
              tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """
    Lloyd-Max optimal scalar quantizer for N(0, sigma^2).

    Args:
        num_levels: Number of quantization levels (2^bits).
        sigma: Standard deviation of the Gaussian.
        max_iter: Maximum iterations.
        tol: Convergence tolerance on centroid movement.

    Returns:
        centroids: shape (num_levels,) - sorted reconstruction values.
        boundaries: shape (num_levels - 1,) - decision boundaries.
    """
    # Initialise centroids uniformly over [-3*sigma, 3*sigma]
    centroids = np.linspace(-3 * sigma, 3 * sigma, num_levels)

    for iteration in range(max_iter):
        # Step 1: Compute boundaries as midpoints between adjacent centroids
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0

        # Step 2: Update centroids as conditional expectations
        # Build full boundary list with -inf and +inf at the edges
        full_bounds = np.concatenate([[-np.inf], boundaries, [np.inf]])

        new_centroids = np.zeros_like(centroids)
        for i in range(num_levels):
            new_centroids[i] = conditional_expectation(
                full_bounds[i], full_bounds[i + 1], sigma
            )

        # Check convergence
        max_shift = np.max(np.abs(new_centroids - centroids))
        centroids = new_centroids

        if max_shift < tol:
            break

    # Final boundaries
    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return centroids, boundaries


def compute_mse(centroids: np.ndarray, boundaries: np.ndarray,
                sigma: float) -> float:
    """
    Compute the MSE of the Lloyd-Max quantizer analytically.

    MSE = E[(X - Q(X))^2] where Q maps X to its nearest centroid.
    """
    full_bounds = np.concatenate([[-np.inf], boundaries, [np.inf]])
    mse = 0.0

    for i, centroid in enumerate(centroids):
        a, b = full_bounds[i], full_bounds[i + 1]

        # E[(X - c)^2 | a < X < b] * P(a < X < b)
        # = E[X^2 | a < X < b]*P - 2*c*E[X|a<X<b]*P + c^2*P
        # = integral of (x-c)^2 * pdf(x) from a to b
        result, _ = integrate.quad(
            lambda x: (x - centroid) ** 2 * gaussian_pdf(x, sigma),
            a, b
        )
        mse += result

    return mse


def precompute_codebooks(
    head_dims: list[int] = [128],
    bit_widths: list[int] = [1, 2, 3, 4],
    output_dir: str = "codebooks"
) -> dict:
    """
    Precompute and save Lloyd-Max codebooks for all dimension/bit-width combos.

    Returns:
        Dictionary mapping (head_dim, bits) -> {centroids, boundaries, sigma, mse}
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for d in head_dims:
        sigma = 1.0 / np.sqrt(d)
        print(f"\n{'='*60}")
        print(f"Head dimension: {d}, sigma: {sigma:.6f}")
        print(f"{'='*60}")

        for bits in bit_widths:
            num_levels = 2 ** bits
            print(f"\n  {bits}-bit ({num_levels} levels):")

            centroids, boundaries = lloyd_max(num_levels, sigma)
            mse = compute_mse(centroids, boundaries, sigma)

            # Per-coordinate MSE; the paper reports per-vector MSE = d * per_coord_mse
            per_vector_mse = d * mse
            print(f"    Per-coordinate MSE: {mse:.8f}")
            print(f"    Per-vector MSE (d={d}): {per_vector_mse:.6f}")
            print(f"    Centroids: {centroids}")

            # Save
            filename = f"dim_{d}_{bits}bit.npz"
            filepath = os.path.join(output_dir, filename)
            np.savez(
                filepath,
                centroids=centroids,
                boundaries=boundaries,
                sigma=np.array([sigma]),
                mse=np.array([mse]),
                head_dim=np.array([d]),
                bits=np.array([bits])
            )
            print(f"    Saved to {filepath}")

            results[(d, bits)] = {
                "centroids": centroids,
                "boundaries": boundaries,
                "sigma": sigma,
                "mse": mse,
                "per_vector_mse": per_vector_mse,
            }

    return results


def validate_codebooks(results: dict) -> None:
    """
    Validate codebooks against known theoretical bounds from the paper.

    TurboQuant Theorem 1 upper bounds for unit vectors in R^d:
        1-bit: 0.680
        2-bit: 0.170
        3-bit: 0.043
        4-bit: 0.011
    """
    paper_bounds = {1: 0.680, 2: 0.170, 3: 0.043, 4: 0.011}

    print(f"\n{'='*60}")
    print("Validation against TurboQuant paper bounds")
    print(f"{'='*60}")

    for (d, bits), result in sorted(results.items()):
        bound = paper_bounds.get(bits)
        if bound is None:
            continue

        pv_mse = result["per_vector_mse"]
        ratio = pv_mse / bound
        status = "PASS" if pv_mse < bound else "FAIL"

        print(f"  d={d}, {bits}-bit: MSE={pv_mse:.4f}, "
              f"bound={bound:.3f}, ratio={ratio:.2f}x [{status}]")


if __name__ == "__main__":
    print("TurboQuant Codebook Precompute")
    print("Generating Lloyd-Max optimal codebooks for N(0, 1/d)\n")

    results = precompute_codebooks(
        head_dims=[128],       # Qwen3.5 attention head dimension
        bit_widths=[1, 2, 3, 4],
        output_dir="codebooks"
    )

    validate_codebooks(results)
    print("\nDone. Codebooks saved to ./codebooks/")
