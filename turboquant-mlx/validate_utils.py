"""
validate_utils.py — Similarity metrics for TurboQuant fidelity validation.
"""

import mlx.core as mx
import numpy as np


def cosine_similarity_flat(a: mx.array, b: mx.array) -> float:
    """
    Cosine similarity between two flattened score tensors.
    Used to compare entire attention score distributions.
    """
    a_f32 = a.astype(mx.float32)
    b_f32 = b.astype(mx.float32)

    a_flat = mx.reshape(a_f32, (-1,))
    b_flat = mx.reshape(b_f32, (-1,))
    dot = mx.sum(a_flat * b_flat)
    norm_a = mx.linalg.norm(a_flat)
    norm_b = mx.linalg.norm(b_flat)
    cos = dot / (norm_a * norm_b + 1e-8)
    mx.eval(cos)
    return cos.item()


def top_k_match(
    scores_compressed: mx.array, scores_original: mx.array, k: int = 1
) -> float:
    """
    For each query row, check if the true top-1 from original scores
    appears in the top-k of compressed scores.

    Args:
        scores_compressed: (num_queries, seq_len)
        scores_original:   (num_queries, seq_len)
        k: how many top positions to check

    Returns:
        fraction of queries where true top-1 is in compressed top-k
    """
    # True top-1 index per query
    true_top1 = mx.argmax(scores_original, axis=-1)  # (num_queries,)
    mx.eval(true_top1)
    true_top1_np = np.array(true_top1)

    # Compressed top-k indices per query
    # mx doesn't have top_k, so use argsort
    sorted_idx = mx.argsort(scores_compressed, axis=-1)  # ascending
    topk_compressed = sorted_idx[:, -k:]  # last k = top k
    mx.eval(topk_compressed)
    topk_np = np.array(topk_compressed)

    # Check if true_top1 is in top-k for each query
    matches = np.any(topk_np == true_top1_np[:, None], axis=-1)
    return float(np.mean(matches))
