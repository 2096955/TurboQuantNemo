"""
OpenEvolve target: TurboQuant asymmetric attention score estimator (Path A — Python/MLX).

Only the region between EVOLVE-BLOCK markers may be modified by evolution.
Do not change the function name or outer signature — the evaluator imports
`asymmetric_attention_scores` from this file.

MUST PRESERVE: mathematical equivalence of the two-term estimator (quantized key path +
QJL correction) within the fidelity gate vs the reference implementation in mlx_turboquant.py.
Fixed rotation matrices and codebooks are provided by the evaluator/fixture, not here.
"""

from __future__ import annotations

import mlx.core as mx


def asymmetric_attention_scores(
    query: mx.array,
    compressed: dict,
    rotation: mx.array,
    S: mx.array,
    qjl_scale: float,
    scale: float,
) -> mx.array:
    # EVOLVE-BLOCK-START
    x_hat_unit = mx.matmul(compressed["x_rot_quant"], rotation)
    k_hat = x_hat_unit * compressed["x_norm"]

    term1 = mx.matmul(query, mx.swapaxes(k_hat, -2, -1))

    Sq = mx.matmul(query, mx.transpose(S))
    signs_T = mx.swapaxes(compressed["residual_signs"], -2, -1)
    correction = mx.matmul(Sq, signs_T)

    r_norm = mx.swapaxes(compressed["residual_norm"], -2, -1)
    term2 = qjl_scale * r_norm * correction

    scores = term1 + term2
    return scores * scale
    # EVOLVE-BLOCK-END
