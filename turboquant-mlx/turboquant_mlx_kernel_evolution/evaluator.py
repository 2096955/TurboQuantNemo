"""
OpenEvolve evaluator for TurboQuant asymmetric_attention_scores (Path A).

Contract: `evaluate(program_path: str)` returns metrics (dict or EvaluationResult).
OpenEvolve 0.2.x invokes this with a path to the candidate Python file.

Fidelity: cosine similarity between flattened candidate and reference outputs must
meet TURBOQUANT_COSINE_GATE (default 0.995). On failure, throughput is reported as 0
and combined_score is strongly negative.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import traceback
from pathlib import Path

import mlx.core as mx

_EV_DIR = Path(__file__).resolve().parent
_TQ_ROOT = _EV_DIR.parent
if str(_TQ_ROOT) not in sys.path:
    sys.path.insert(0, str(_TQ_ROOT))

from openevolve.evaluation_result import EvaluationResult
from turboquant_benchmark_suite import resolve_fixture, time_score_fn
from validate_utils import cosine_similarity_flat

from mlx_turboquant import (
    asymmetric_attention_scores as reference_asymmetric_attention_scores,
)


def _load_candidate(program_path: str):
    spec = importlib.util.spec_from_file_location("tq_evolve_candidate", program_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load spec for {program_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "asymmetric_attention_scores"):
        raise RuntimeError("Candidate must define asymmetric_attention_scores(...)")
    return mod.asymmetric_attention_scores


def evaluate(program_path: str) -> dict | EvaluationResult:
    gate = float(os.environ.get("TURBOQUANT_COSINE_GATE", "0.995"))
    warmup = int(os.environ.get("TURBOQUANT_BENCH_WARMUP", "3"))
    iters = int(os.environ.get("TURBOQUANT_BENCH_ITERS", "40"))

    try:
        fixture = resolve_fixture()
    except FileNotFoundError as e:
        return EvaluationResult(
            metrics={
                "combined_score": -1_000.0,
                "throughput_eval_per_sec": 0.0,
                "cosine_similarity": 0.0,
                "error": 0.0,
            },
            artifacts={
                "stderr": str(e),
                "llm_feedback": "Codebooks missing: run codebook precompute for the chosen head_dim/bits or set TURBOQUANT_FIXTURE_NPZ.",
            },
        )
    except Exception as e:
        return EvaluationResult(
            metrics={
                "combined_score": -1_000.0,
                "throughput_eval_per_sec": 0.0,
                "cosine_similarity": 0.0,
                "error": 0.0,
            },
            artifacts={"stderr": str(e), "traceback": traceback.format_exc()},
        )

    try:
        candidate_fn = _load_candidate(program_path)
    except Exception as e:
        return EvaluationResult(
            metrics={
                "combined_score": -1_000.0,
                "throughput_eval_per_sec": 0.0,
                "cosine_similarity": 0.0,
                "error": 0.0,
            },
            artifacts={
                "stderr": str(e),
                "traceback": traceback.format_exc(),
                "llm_feedback": "Candidate failed to import; fix syntax and keep asymmetric_attention_scores signature.",
            },
        )

    q = fixture.query
    comp = fixture.compressed
    rot = fixture.rotation
    S = fixture.S
    qs = fixture.qjl_scale
    sc = fixture.scale

    try:
        ref = reference_asymmetric_attention_scores(q, comp, rot, S, qs, sc)
        cand = candidate_fn(q, comp, rot, S, qs, sc)
        mx.eval(ref, cand)
        cos = cosine_similarity_flat(cand, ref)
    except Exception as e:
        return EvaluationResult(
            metrics={
                "combined_score": -1_000.0,
                "throughput_eval_per_sec": 0.0,
                "cosine_similarity": 0.0,
                "error": 0.0,
            },
            artifacts={
                "stderr": str(e),
                "traceback": traceback.format_exc(),
                "llm_feedback": "Runtime error in score computation; check shapes/dtypes against reference.",
            },
        )

    if cos < gate:
        fb = (
            f"Fidelity fail: cosine={cos:.6f} < gate={gate:.6f}. "
            "Restore equivalence to the two-term estimator before chasing speed."
        )
        return EvaluationResult(
            metrics={
                "combined_score": -500.0 - (gate - cos) * 1_000.0,
                "throughput_eval_per_sec": 0.0,
                "cosine_similarity": float(cos),
                "error": 1.0,
            },
            artifacts={"llm_feedback": fb},
        )

    try:
        bench = time_score_fn(candidate_fn, fixture, warmup=warmup, iterations=iters)
    except Exception as e:
        return EvaluationResult(
            metrics={
                "combined_score": -100.0,
                "throughput_eval_per_sec": 0.0,
                "cosine_similarity": float(cos),
                "error": 1.0,
            },
            artifacts={
                "stderr": str(e),
                "traceback": traceback.format_exc(),
                "llm_feedback": "Correctness passed but benchmark failed.",
            },
        )

    throughput = bench.eval_per_sec
    combined = throughput * float(cos)

    return EvaluationResult(
        metrics={
            "combined_score": float(combined),
            "throughput_eval_per_sec": float(throughput),
            "cosine_similarity": float(cos),
            "error": 0.0,
        },
        artifacts={
            "llm_feedback": (
                f"Pass gate cos={cos:.5f}; ~{throughput:.1f} eval/s. "
                "Consider fusing matmuls, reusing Sq, or improving memory locality."
            )
        },
    )


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="TurboQuant OpenEvolve evaluator (standalone smoke test)"
    )
    ap.add_argument(
        "--program",
        type=str,
        default=str(_EV_DIR / "initial_program.py"),
        help="Path to candidate program (default: initial_program.py)",
    )
    ap.add_argument(
        "--json-metrics",
        action="store_true",
        help="Print only metrics as JSON (subprocess hooks)",
    )
    args = ap.parse_args()
    r = evaluate(args.program)
    if isinstance(r, EvaluationResult):
        if args.json_metrics:
            print(json.dumps(r.metrics))
        else:
            print("metrics:", r.metrics)
            if r.artifacts:
                print(
                    "artifacts:",
                    {
                        k: (v[:500] + "…") if isinstance(v, str) and len(v) > 500 else v
                        for k, v in r.artifacts.items()
                    },
                )
    else:
        print(json.dumps(r) if args.json_metrics else r)
