#!/usr/bin/env python3
"""Real-weight MLA logit parity gate for Kimi K2.5 / K2.6 IsoQuant.

Implements docs/DKV_KERNEL_SPEC.md §10 (fallback / validation gate) and
§9 test 3 (long-context drift sentinel). Closes the
``real_weight_mla`` section of the artifact emitted by
``scripts/run_kimi_npt16_parity_gate.py``: while that gate stays
``INCOMPLETE_REAL_WEIGHT_PENDING`` by design, this harness produces the
checkpoint-backed comparison that decides whether IsoQuant on MLA is
viable at all.

Decision rule from §10:

    SKIP IsoQuant on this MLA path entirely if either
      - top-1 logit match rate is below --top1-threshold (default 0.99), OR
      - PPL @ 8 K rises by more than --ppl-drift-threshold (default 0.5)
        versus the FP16 reference,
    OR §9 test 3 sees a linearly-increasing per-1 K-window PPL drift
    across context (the RoPE-smearing signature).

If the gate fails, document the negative result either way — that
outcome is publishable per §10.

Exit codes:
    0  GATE_PASS — all thresholds met across all sampled contexts.
    1  GATE_FAIL — at least one threshold violated; IsoQuant on MLA
       should be skipped (or kernel layout offsets re-checked).
    2  SETUP_ERROR — checkpoint missing, model load failure, OOM, or
       any other system-level problem before a verdict could be
       reached. Distinct from a deliberate fail so CI can route them
       differently.

Example:
    PYTHONPATH=mlx-lm python scripts/validate_real_kv.py \\
      --model /Volumes/Samsung9904tb/Kimi-K2.6 \\
      --contexts 2048,4096,8192,16384 \\
      --output artifacts/kimi_k26_profiling/real_weight_logit_parity.json

Notes:
    - Reference path: ``kv_cache_type=default`` (full FP16 KV).
    - Test path: ``kv_cache_type=isoquant`` with
      ``ISOQUANT_USE_NPT16_FUSED=1``. The harness sets the env var
      itself for the test phase; reference phase has it cleared.
    - **What this harness measures.** A short seed (``--prefill-len``,
      default 32 tokens) is single-shot-prefilled. The cache is then
      finalized — this is the point where ``KimiMLAIsoQuantCache``
      compresses its accumulated FP16 latent and starts reporting
      ``supports_fused_latent_attention=True``, which is what makes
      ``deepseek_v3.DeepseekV3Attention`` dispatch to
      ``cache.fused_latent_attention``. From there the rest of the
      sequence is teacher-forced one token at a time, capturing
      per-step logits. So the iso-path logits at decode step i reflect
      i compressed-attention dispatches — the same compression-error
      accumulation that real generation would experience, without the
      trajectory drift teacher-forcing eliminates by construction.
    - The harness does *not* enable expert offload; the parity question
      is about KV math, not residency.
    - This is teacher-forced parity, not autoregressive generation.
      Spec §9 test 3 says "Generate"; teacher-forcing is a strict
      proxy for cache-side compression error and intentionally avoids
      sampling-noise variance. If you need true generation drift
      (e.g. detecting trajectory-divergence where iso eventually
      generates different tokens than ref), that needs a separate
      harness.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Sample text used to build prompts (anything reasonably-distributed is fine
# for PPL-style comparison; the absolute PPL doesn't matter, only the
# delta between the two paths).
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "The fundamental tension in attention-based language models is that the "
    "key-value cache grows linearly with context. Multi-Head Latent "
    "Attention compresses this by projecting keys and values into a "
    "low-rank latent space, decoupling the rotary positional embedding "
    "into a separate small tensor. This has structural consequences for "
    "any post-hoc compression scheme. Lossy quantisation of the latent "
    "content dimensions is potentially safe; the same operation applied "
    "to the rotary positional dimensions smears phase information and "
    "degrades attention quality in a context-length-dependent way that "
    "short-prompt evaluations will not catch. The decoupled-key-value "
    "invariant exists to make this distinction explicit at the kernel "
    "boundary so that no future change can silently re-introduce the "
    "failure mode. "
)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class ContextResult:
    context_tokens: int
    n_compared_positions: int
    top1_match_rate: float
    ppl_ref: float
    ppl_iso: float
    ppl_drift: float
    mean_logit_cosine: float
    window_ppl_drifts: list[float] = field(default_factory=list)
    window_size: int = 1024
    long_context_slope: Optional[float] = None
    setup_seconds_ref: float = 0.0
    setup_seconds_iso: float = 0.0


@dataclass
class GateVerdict:
    passed: bool
    reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tokenizer + prompt building
# ---------------------------------------------------------------------------


def build_prompt_tokens(tokenizer, target_n_tokens: int) -> "any":
    """Build a token sequence of approximately target_n_tokens by repeating
    SAMPLE_TEXT and truncating. Returns an mlx array of shape (1, T).
    """
    import mlx.core as mx

    text = SAMPLE_TEXT
    while True:
        ids = tokenizer.encode(text)
        if len(ids) >= target_n_tokens:
            break
        text += SAMPLE_TEXT

    ids = ids[:target_n_tokens]
    return mx.array([ids])


# ---------------------------------------------------------------------------
# Forward pass with cache; captures per-position logits.
# ---------------------------------------------------------------------------


def _set_npt16(enabled: bool) -> None:
    if enabled:
        os.environ["ISOQUANT_USE_NPT16_FUSED"] = "1"
    else:
        os.environ.pop("ISOQUANT_USE_NPT16_FUSED", None)


def prefill_then_teacher_forced_decode(
    model,
    tokens,
    cache,
    prefill_len: int,
):
    """Prefill the first ``prefill_len`` tokens single-shot, finalize any
    deferred KV state on the cache list, then teacher-force the remaining
    tokens one at a time, capturing per-step logits.

    This is what actually exercises the per-step fused attention dispatch
    in the model code (e.g. ``DeepseekV3Attention`` line ~157 dispatches
    to ``cache.fused_latent_attention`` once the cache reports
    ``supports_fused_latent_attention=True`` — which only becomes true
    after ``finalize_deferred_kv_caches`` is called on a
    ``KimiMLAIsoQuantCache``).

    Returns:
      decode_logits: numpy array of shape (T - prefill_len, V) — the
                     logits emitted at each decode step. Position i in
                     this array predicts ``tokens[:, prefill_len + i + 1]``
                     (i.e. the token that follows the i-th decoded input).
    """
    import mlx.core as mx
    import numpy as np
    from mlx_lm.models.cache import finalize_deferred_kv_caches

    T = tokens.shape[1]
    if T <= prefill_len:
        raise ValueError(
            f"Need at least prefill_len + 1 = {prefill_len + 1} tokens, got {T}"
        )

    # ---- Phase 1: single-shot prefill of the seed ----
    seed = tokens[:, :prefill_len]
    _ = model(seed, cache=cache)
    # Force evaluation so the cache is committed before finalize.
    mx.eval(_)
    del _

    # ---- Phase 2: finalize deferred state ----
    # Triggers compression on KimiMLAIsoQuantCache (and any other backend
    # that defines finalize_deferred_prefill); no-op on default caches.
    finalize_deferred_kv_caches(cache)

    # ---- Phase 3: teacher-forced decode loop ----
    # At step i, feed tokens[:, prefill_len + i : prefill_len + i + 1] and
    # capture the logit for predicting tokens[:, prefill_len + i + 1].
    n_decode = T - prefill_len
    decode_logits: list[np.ndarray] = []
    for i in range(n_decode):
        in_tok = tokens[:, prefill_len + i : prefill_len + i + 1]
        logits = model(in_tok, cache=cache)
        # logits shape: (1, 1, V) — squeeze to (V,)
        mx.eval(logits)
        decode_logits.append(np.asarray(logits[0, 0], dtype=np.float32))
    return np.stack(decode_logits, axis=0)  # (n_decode, V)


def compute_metrics(
    logits_ref,  # numpy (n_decode, V) — decode-step logits
    logits_iso,  # numpy (n_decode, V) — decode-step logits
    decode_targets,  # numpy (n_decode,) — token to predict at each step
    window_size: int,
):
    """Per-step top-1 match, per-step PPL for both paths, drift,
    cosine similarity averaged over steps, and per-window PPL drift
    across the decode trajectory (the long-context drift signal).

    Inputs are aligned: ``logits_*[i]`` is the logit emitted when the
    model was fed the i-th decoded input token; the target is
    ``decode_targets[i]`` — the *next* real token in the corpus.
    """
    import numpy as np

    n = logits_ref.shape[0]
    if n <= 0 or n != logits_iso.shape[0] or n != decode_targets.shape[0]:
        return {
            "n_compared_positions": 0,
            "top1_match_rate": float("nan"),
            "ppl_ref": float("nan"),
            "ppl_iso": float("nan"),
            "ppl_drift": float("nan"),
            "mean_logit_cosine": float("nan"),
            "window_ppl_drifts": [],
        }

    ref = logits_ref
    iso = logits_iso
    targets = decode_targets

    # Top-1 match: argmax agreement on the next-token prediction.
    top1_ref = ref.argmax(axis=-1)
    top1_iso = iso.argmax(axis=-1)
    top1_match_rate = float((top1_ref == top1_iso).mean())

    # Per-position log-softmax via numerically stable trick.
    def log_softmax(x):
        m = x.max(axis=-1, keepdims=True)
        z = x - m
        return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))

    lp_ref = log_softmax(ref)
    lp_iso = log_softmax(iso)

    # NLL per position: -log p(target)
    nll_ref = -lp_ref[np.arange(n), targets]
    nll_iso = -lp_iso[np.arange(n), targets]

    ppl_ref = float(math.exp(nll_ref.mean()))
    ppl_iso = float(math.exp(nll_iso.mean()))
    ppl_drift = ppl_iso - ppl_ref

    # Cosine similarity between logit vectors (proxy for output agreement
    # before argmax).
    dot = (ref * iso).sum(axis=-1)
    nref = np.linalg.norm(ref, axis=-1)
    niso = np.linalg.norm(iso, axis=-1)
    denom = np.maximum(nref * niso, 1e-9)
    cos = dot / denom
    mean_cos = float(cos.mean())

    # Per-window PPL drift across the sequence — §9 test 3 sentinel.
    window_drifts: list[float] = []
    if window_size > 0 and n >= window_size:
        n_windows = n // window_size
        for w in range(n_windows):
            s = w * window_size
            e = s + window_size
            ppl_w_ref = math.exp(nll_ref[s:e].mean())
            ppl_w_iso = math.exp(nll_iso[s:e].mean())
            window_drifts.append(ppl_w_iso - ppl_w_ref)

    return {
        "n_compared_positions": int(n),
        "top1_match_rate": top1_match_rate,
        "ppl_ref": ppl_ref,
        "ppl_iso": ppl_iso,
        "ppl_drift": float(ppl_drift),
        "mean_logit_cosine": mean_cos,
        "window_ppl_drifts": window_drifts,
    }


def long_context_slope(window_drifts: list[float]) -> Optional[float]:
    """Least-squares slope of window-index → drift. A positive slope above
    a small tolerance is the RoPE-smearing signature called out in §9
    test 3.
    """
    import numpy as np

    if len(window_drifts) < 2:
        return None
    x = np.arange(len(window_drifts), dtype=np.float64)
    y = np.asarray(window_drifts, dtype=np.float64)
    x_mean = x.mean()
    y_mean = y.mean()
    cov = ((x - x_mean) * (y - y_mean)).sum()
    var = ((x - x_mean) ** 2).sum()
    if var == 0:
        return None
    return float(cov / var)


# ---------------------------------------------------------------------------
# Model + cache plumbing
# ---------------------------------------------------------------------------


def load_model(model_path: str):
    """Load model + tokenizer via mlx_lm.load. Raises on failure."""
    from mlx_lm import load as mlx_load

    return mlx_load(model_path)


def make_caches(model, kv_cache_type: str, bits: int):
    """Build a fresh cache list for the given kv_cache_type (default or
    isoquant). bits is only consulted on the isoquant path.
    """
    # We import inside the function so a missing checkpoint doesn't block
    # --help.
    from mlx_lm.models.cache import make_prompt_cache

    # make_prompt_cache pulls config from model.args / model.text_config;
    # passing kv_cache_type selects the IsoQuant path when it's an MLA model.
    # iso_bits is read from the env var ISOQUANT_BITS in the cache code,
    # so we set it here for clarity.
    if kv_cache_type == "isoquant":
        os.environ["ISOQUANT_BITS"] = str(bits)
    return make_prompt_cache(model, kv_cache_type=kv_cache_type)


# ---------------------------------------------------------------------------
# Per-context evaluation
# ---------------------------------------------------------------------------


def evaluate_context(
    model,
    tokenizer,
    context_tokens: int,
    bits: int,
    use_npt16: bool,
    prefill_len: int,
    window_size: int,
    log,
) -> ContextResult:
    import numpy as np

    log(f"  [{context_tokens}] building prompt of {context_tokens} tokens")
    tokens = build_prompt_tokens(tokenizer, context_tokens)
    # Decode-step i in the loop predicts tokens[prefill_len + i + 1].
    decode_targets = np.asarray(tokens[0])[prefill_len + 1 : context_tokens]

    # ---- Reference path: default cache, NPT16 disabled ----
    log(
        f"  [{context_tokens}] ref path (default cache, FP16 KV) "
        f"prefill={prefill_len} decode_steps={decode_targets.shape[0]}"
    )
    _set_npt16(False)
    t0 = time.time()
    cache_ref = make_caches(model, "default", bits=bits)
    logits_ref = prefill_then_teacher_forced_decode(
        model, tokens, cache_ref, prefill_len
    )
    # logits_ref shape: (T - prefill_len, V); we only have targets for
    # T - prefill_len - 1 of them (the last decode step has no next-token
    # target in our prompt). Trim accordingly.
    logits_ref = logits_ref[: decode_targets.shape[0]]
    setup_ref = time.time() - t0

    del cache_ref

    # ---- Test path: IsoQuant cache, NPT16 fused if requested ----
    log(f"  [{context_tokens}] iso path (IsoQuant {bits}-bit, npt16_fused={use_npt16})")
    _set_npt16(use_npt16)
    t0 = time.time()
    cache_iso = make_caches(model, "isoquant", bits=bits)
    logits_iso = prefill_then_teacher_forced_decode(
        model, tokens, cache_iso, prefill_len
    )
    logits_iso = logits_iso[: decode_targets.shape[0]]
    setup_iso = time.time() - t0

    del cache_iso

    metrics = compute_metrics(logits_ref, logits_iso, decode_targets, window_size)
    slope = long_context_slope(metrics["window_ppl_drifts"])

    return ContextResult(
        context_tokens=context_tokens,
        n_compared_positions=metrics["n_compared_positions"],
        top1_match_rate=metrics["top1_match_rate"],
        ppl_ref=metrics["ppl_ref"],
        ppl_iso=metrics["ppl_iso"],
        ppl_drift=metrics["ppl_drift"],
        mean_logit_cosine=metrics["mean_logit_cosine"],
        window_ppl_drifts=metrics["window_ppl_drifts"],
        window_size=window_size,
        long_context_slope=slope,
        setup_seconds_ref=setup_ref,
        setup_seconds_iso=setup_iso,
    )


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------


def evaluate_gate(
    results: list[ContextResult],
    top1_threshold: float,
    ppl_drift_threshold: float,
    slope_tolerance: float,
    eight_k_target: int = 8192,
) -> GateVerdict:
    reasons: list[str] = []

    for r in results:
        if r.top1_match_rate < top1_threshold:
            reasons.append(
                f"context={r.context_tokens}: top1_match_rate {r.top1_match_rate:.4f} "
                f"< threshold {top1_threshold:.4f}"
            )

    # §10: PPL drift gate is anchored at 8 K. If the user didn't sample
    # exactly 8 K, use the closest sampled context within a 2x band.
    eight_k_results = [r for r in results if r.context_tokens >= 4096]
    if eight_k_results:
        anchor = min(
            eight_k_results,
            key=lambda r: abs(r.context_tokens - eight_k_target),
        )
        if anchor.ppl_drift > ppl_drift_threshold:
            reasons.append(
                f"context={anchor.context_tokens}: ppl_drift {anchor.ppl_drift:.4f} "
                f"> §10 threshold {ppl_drift_threshold:.4f}"
            )

    # §9 test 3: per-window drift slope must be ≤ tolerance.
    for r in results:
        if r.long_context_slope is not None and r.long_context_slope > slope_tolerance:
            reasons.append(
                f"context={r.context_tokens}: window-drift slope {r.long_context_slope:.4f} "
                f"> tolerance {slope_tolerance:.4f} — possible RoPE smearing"
            )

    return GateVerdict(passed=(len(reasons) == 0), reasons=reasons)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model",
        default="/Volumes/Samsung9904tb/Kimi-K2.6",
        help="Path to Kimi K2.5/K2.6 checkpoint.",
    )
    p.add_argument(
        "--contexts",
        default="2048,4096,8192",
        help="Comma-separated context lengths to evaluate. 16384 is also "
        "permitted but the seed prefill grows linearly with context — "
        "lower --prefill-len to keep the single-shot prefill safe.",
    )
    p.add_argument(
        "--output",
        default=str(
            REPO_ROOT / "artifacts/kimi_k26_profiling/real_weight_logit_parity.json"
        ),
        help="Output JSON path.",
    )
    p.add_argument(
        "--bits",
        type=int,
        default=3,
        help="IsoQuant bit-width (sets ISOQUANT_BITS env var).",
    )
    p.add_argument(
        "--top1-threshold",
        type=float,
        default=0.99,
        help="Per-position top-1 logit agreement gate.",
    )
    p.add_argument(
        "--ppl-drift-threshold",
        type=float,
        default=0.5,
        help="§10 PPL drift gate at the 8 K anchor (iso − ref).",
    )
    p.add_argument(
        "--slope-tolerance",
        type=float,
        default=0.05,
        help="§9 test 3 long-context window-drift slope tolerance. "
        "Positive slope above this is the RoPE-smearing signature.",
    )
    p.add_argument(
        "--window-size",
        type=int,
        default=1024,
        help="Window size in tokens for §9 test 3 per-window PPL drift.",
    )
    p.add_argument(
        "--prefill-len",
        type=int,
        default=32,
        help="Number of seed tokens single-shot-prefilled before the "
        "teacher-forced decode loop begins. Must be < the smallest "
        "context being evaluated. Smaller seeds give more decode steps "
        "to integrate compression error, which is what the §10 / §9 "
        "gates are testing.",
    )
    p.add_argument(
        "--no-npt16-fused",
        action="store_true",
        help="Disable ISOQUANT_USE_NPT16_FUSED on the iso path "
        "(default: enabled, since this is the kernel under test).",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run only the shortest context for a quick smoke check; "
        "still applies §10 / §9 gates but the slope sentinel is unreliable.",
    )
    args = p.parse_args()

    contexts = sorted({int(x) for x in args.contexts.split(",") if x.strip()})
    if not contexts:
        print("ERROR: --contexts produced no integers", file=sys.stderr)
        return 2
    if args.smoke:
        contexts = [contexts[0]]

    model_path = Path(args.model)
    if not model_path.exists():
        print(
            f"ERROR: checkpoint not found at {model_path}. "
            f"Mount the volume or pass --model.",
            file=sys.stderr,
        )
        return 2

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(msg, flush=True)

    log(f"Loading model from {model_path} ...")
    t0 = time.time()
    try:
        model, tokenizer = load_model(str(model_path))
    except Exception as exc:
        print(f"ERROR: model load failed: {exc}", file=sys.stderr)
        # Write a setup-error stub so CI has an artifact to inspect.
        out_path.write_text(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                    "gate_status": "SETUP_ERROR",
                    "error": str(exc),
                },
                indent=2,
            )
        )
        return 2
    load_seconds = time.time() - t0
    log(f"Model loaded in {load_seconds:.1f} s")

    use_npt16 = not args.no_npt16_fused
    results: list[ContextResult] = []

    try:
        for ctx in contexts:
            log(f"--- context {ctx} ---")
            r = evaluate_context(
                model=model,
                tokenizer=tokenizer,
                context_tokens=ctx,
                bits=args.bits,
                use_npt16=use_npt16,
                prefill_len=args.prefill_len,
                window_size=args.window_size,
                log=log,
            )
            log(
                f"  → top1={r.top1_match_rate:.4f}  ppl_drift={r.ppl_drift:+.4f}  "
                f"cos={r.mean_logit_cosine:.4f}  slope={r.long_context_slope}"
            )
            results.append(r)
    except Exception as exc:
        print(f"ERROR: evaluation failed: {exc}", file=sys.stderr)
        out_path.write_text(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                    "gate_status": "SETUP_ERROR",
                    "error": str(exc),
                    "completed_contexts": [r.context_tokens for r in results],
                },
                indent=2,
            )
        )
        return 2

    verdict = evaluate_gate(
        results,
        top1_threshold=args.top1_threshold,
        ppl_drift_threshold=args.ppl_drift_threshold,
        slope_tolerance=args.slope_tolerance,
    )
    gate_status = "GATE_PASS" if verdict.passed else "GATE_FAIL"

    payload = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_path": str(model_path),
        "gate_status": gate_status,
        "thresholds": {
            "top1_threshold": args.top1_threshold,
            "ppl_drift_threshold": args.ppl_drift_threshold,
            "slope_tolerance": args.slope_tolerance,
            "ppl_drift_anchor_tokens": 8192,
        },
        "config": {
            "bits": args.bits,
            "use_npt16_fused": use_npt16,
            "prefill_chunk": args.prefill_chunk,
            "window_size": args.window_size,
            "smoke": args.smoke,
        },
        "verdict_reasons": verdict.reasons,
        "load_seconds": load_seconds,
        "contexts": [asdict(r) for r in results],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    log(f"Saved: {out_path}  gate_status={gate_status}")
    if verdict.reasons:
        log("Verdict reasons:")
        for r in verdict.reasons:
            log(f"  - {r}")

    return 0 if verdict.passed else 1


if __name__ == "__main__":
    sys.exit(main())
