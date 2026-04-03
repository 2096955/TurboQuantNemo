"""
Benchmark and fixture helpers for TurboQuant OpenEvolve (Phase 6).

Closed-loop inputs use the same TurboQuantCompressor path as production so evolved
kernels are checked against real quantization artifacts (given codebooks on disk).

For captures from a live model (Phase 3), extend `load_fixture_from_npz` and point
`TURBOQUANT_FIXTURE_NPZ` at a file produced by your capture pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

# Repo root: turboquant-mlx/
_TQ_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ScoreFixture:
    """Tensors for one head, one forward slice — matches asymmetric_attention_scores()."""

    query: mx.array  # (1, num_queries, head_dim)
    compressed: dict[str, mx.array]
    rotation: mx.array
    S: mx.array
    qjl_scale: float
    scale: float
    head_dim: int
    bits: int


def _import_compressor():
    import sys

    root = str(_TQ_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from mlx_turboquant import TurboQuantCompressor

    return TurboQuantCompressor


def build_closed_loop_fixture(
    head_dim: int = 128,
    seq_kv: int = 256,
    num_queries: int = 64,
    bit_width: int = 4,
    codebook_dir: str | None = None,
    seed: int = 42,
) -> ScoreFixture:
    """
    Synthetic keys + queries -> compress with TurboQuantCompressor -> score inputs.
    Requires codebooks/dim_{head_dim}_{bit_width}bit.npz under codebook_dir.
    """
    TurboQuantCompressor = _import_compressor()
    cb_dir = codebook_dir or os.environ.get(
        "TURBOQUANT_CODEBOOK_DIR", str(_TQ_ROOT / "codebooks")
    )

    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(head_dim, head_dim)).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    rotation = mx.array(Q, dtype=mx.float32)

    compressor = TurboQuantCompressor(
        bit_width=bit_width,
        head_dim=head_dim,
        codebook_dir=cb_dir,
        seed=seed,
    )

    keys = mx.random.normal((seq_kv, head_dim))
    compressed = compressor.compress(keys, rotation=rotation)

    query = mx.random.normal((1, num_queries, head_dim))
    scale = 1.0

    return ScoreFixture(
        query=query,
        compressed=compressed,
        rotation=rotation,
        S=compressor.S,
        qjl_scale=compressor.qjl_scale,
        scale=scale,
        head_dim=head_dim,
        bits=bit_width,
    )


def load_fixture_from_npz(path: str | Path) -> ScoreFixture:
    """
    Load a saved fixture. Required keys (numpy -> mlx):
      query, rotation, S, qjl_scale, scale, head_dim, bits,
      x_rot_quant, x_norm, residual_signs, residual_norm

    Optional keys (from export_phase3_fixture.py) are ignored:
      layer_idx, kv_head, query_head, n_kv_heads, n_query_heads
    """
    path = Path(path)
    z = np.load(path, allow_pickle=True)

    def arr(key: str) -> mx.array:
        return mx.array(z[key])

    compressed = {
        "x_rot_quant": arr("x_rot_quant"),
        "x_norm": arr("x_norm"),
        "residual_signs": arr("residual_signs"),
        "residual_norm": arr("residual_norm"),
    }
    return ScoreFixture(
        query=arr("query"),
        compressed=compressed,
        rotation=arr("rotation"),
        S=arr("S"),
        qjl_scale=float(z["qjl_scale"]),
        scale=float(z["scale"]),
        head_dim=int(z["head_dim"]),
        bits=int(z["bits"]),
    )


def save_fixture_npz(fixture: ScoreFixture, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        query=np.array(fixture.query),
        rotation=np.array(fixture.rotation),
        S=np.array(fixture.S),
        qjl_scale=fixture.qjl_scale,
        scale=fixture.scale,
        head_dim=fixture.head_dim,
        bits=fixture.bits,
        x_rot_quant=np.array(fixture.compressed["x_rot_quant"]),
        x_norm=np.array(fixture.compressed["x_norm"]),
        residual_signs=np.array(fixture.compressed["residual_signs"]),
        residual_norm=np.array(fixture.compressed["residual_norm"]),
    )


def resolve_fixture() -> ScoreFixture:
    """Env TURBOQUANT_FIXTURE_NPZ overrides synthetic builder."""
    npz = os.environ.get("TURBOQUANT_FIXTURE_NPZ")
    if npz and Path(npz).is_file():
        return load_fixture_from_npz(npz)
    head_dim = int(os.environ.get("TURBOQUANT_HEAD_DIM", "128"))
    bits = int(os.environ.get("TURBOQUANT_BITS", "4"))
    seq_kv = int(os.environ.get("TURBOQUANT_SEQ_KV", "256"))
    num_q = int(os.environ.get("TURBOQUANT_NUM_QUERIES", "64"))
    return build_closed_loop_fixture(
        head_dim=head_dim,
        seq_kv=seq_kv,
        num_queries=num_q,
        bit_width=bits,
    )


@dataclass
class MicrobenchResult:
    eval_per_sec: float
    mean_latency_s: float
    iterations: int


def time_score_fn(
    fn: Any,
    fixture: ScoreFixture,
    warmup: int = 3,
    iterations: int = 40,
) -> MicrobenchResult:
    """Wall-time for repeated mx.eval on candidate score function."""
    import time

    q = fixture.query
    comp = fixture.compressed
    rot = fixture.rotation
    S = fixture.S
    qs = fixture.qjl_scale
    sc = fixture.scale

    for _ in range(warmup):
        out = fn(q, comp, rot, S, qs, sc)
        mx.eval(out)

    latencies: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        out = fn(q, comp, rot, S, qs, sc)
        mx.eval(out)
        latencies.append(time.perf_counter() - t0)

    mean_lat = float(sum(latencies) / max(len(latencies), 1))
    eps = 1.0 / max(mean_lat, 1e-9)
    return MicrobenchResult(
        eval_per_sec=eps, mean_latency_s=mean_lat, iterations=iterations
    )
