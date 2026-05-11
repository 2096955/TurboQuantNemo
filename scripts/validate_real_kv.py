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
    - Expert offload is opt-in via ``--expert-offload`` (with
      ``--max-resident-experts`` / ``--max-cached-shards``). Required for
      Kimi K2.6 (~554 GB at 4-bit) on 128 GB hardware: without offload,
      model load OOMs before any KV math runs. The parity question is
      about KV math, not residency, but residency is a hard precondition
      for the math to run at all on this checkpoint.
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
    # Codex audit: source-specific reason when compute_metrics rejected non-
    # finite logits (e.g. "FP16 reference logits contain NaN/Inf" vs "IsoQuant
    # logits contain NaN/Inf"). None on the happy path. Lets per-context JSON
    # consumers attribute a NaN-poisoned gate failure to ref vs iso.
    non_finite_reason: Optional[str] = None
    # Codex review: prove which path the iso run actually exercised. Without
    # these counters the gate verdict is unattributable — outer
    # supports_fused_latent_attention=True does not by itself prove the
    # NPT16 metal path ran (the inner IsoQuantKVCache could still fall back
    # to reconstruct, see kimi_mla_isoquant_dkv.py:125 comment).
    iso_path_observed: Optional[str] = (
        None  # "npt16" | "3kernel" | "single_kernel" | "mlx_fallback" | "unfused_reconstruct" | "ambiguous"
    )
    iso_stats_delta: dict = field(default_factory=dict)
    ref_stats_delta: dict = field(default_factory=dict)
    # Codex review #5: margin-conditioned mismatch analysis. Populated only
    # when --dump-per-step is set. Excluded from the main artifact and
    # written to a sidecar file (kept out of asdict serialization in main).
    per_step: Optional[list] = None


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

    # ---- Phase 2b: assert IsoQuant MLA caches are now fused-capable ----
    # Codex audit HIGH 4: the harness must prove it actually exercises the
    # fused NPT16 path, not silently validate the reconstruct fallback.
    # If supports_fused_latent_attention is False here, the wrong code path
    # is being measured (e.g. wrong --bits, wrong head_dim, finalize never ran).
    try:
        from mlx_lm.models.kimi_mla_isoquant_dkv import KimiMLAIsoQuantCache
    except ImportError:
        KimiMLAIsoQuantCache = None  # type: ignore[assignment]
    if KimiMLAIsoQuantCache is not None:
        non_fused = [
            ci
            for ci, c in enumerate(cache)
            if isinstance(c, KimiMLAIsoQuantCache)
            and not getattr(c, "supports_fused_latent_attention", False)
        ]
        if non_fused:
            raise RuntimeError(
                "IsoQuant MLA caches at indices "
                f"{non_fused} report supports_fused_latent_attention=False after "
                "finalize_deferred_kv_caches(); the harness would silently "
                "validate the reconstruct fallback instead of the fused NPT16 "
                "path. Check --bits (must be 3), head_dim, and that the model "
                "actually has KimiMLAIsoQuantCache instances."
            )

    # ---- Phase 3: teacher-forced decode loop ----
    # At step i, feed tokens[:, prefill_len + i : prefill_len + i + 1] and
    # capture the logit for predicting tokens[:, prefill_len + i + 1].
    n_decode = T - prefill_len
    decode_logits: list[np.ndarray] = []
    for i in range(n_decode):
        in_tok = tokens[:, prefill_len + i : prefill_len + i + 1]
        logits = model(in_tok, cache=cache)
        # logits shape: (1, 1, V) — squeeze to (V,)
        # Cast to fp32 inside MLX before crossing into NumPy: mlx bf16 arrays
        # expose PEP 3118 format 'B' with item size 2, which numpy rejects
        # ("Item size 2 for PEP 3118 buffer format string B does not match
        # the dtype B item size 1.").
        logits_f32 = logits[0, 0].astype(mx.float32)
        mx.eval(logits_f32)
        decode_logits.append(np.asarray(logits_f32, dtype=np.float32))
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

    # Codex audit HIGH 5: reject non-finite logits before any metric is computed.
    # `nan < threshold` and `nan > threshold` both evaluate False, so a NaN-
    # poisoned run would silently report PASS while metrics are nonsense.
    if not np.isfinite(logits_ref).all():
        return {
            "n_compared_positions": int(n),
            "top1_match_rate": float("nan"),
            "ppl_ref": float("nan"),
            "ppl_iso": float("nan"),
            "ppl_drift": float("nan"),
            "mean_logit_cosine": float("nan"),
            "window_ppl_drifts": [],
            "non_finite_reason": "FP16 reference logits contain NaN/Inf",
        }
    if not np.isfinite(logits_iso).all():
        return {
            "n_compared_positions": int(n),
            "top1_match_rate": float("nan"),
            "ppl_ref": float("nan"),
            "ppl_iso": float("nan"),
            "ppl_drift": float("nan"),
            "mean_logit_cosine": float("nan"),
            "window_ppl_drifts": [],
            "non_finite_reason": "IsoQuant logits contain NaN/Inf",
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


def extract_per_step_diagnostics(
    logits_ref,
    logits_iso,
    decode_targets,
) -> list[dict]:
    """Per-position diagnostic record for margin-conditioned mismatch
    analysis.

    Codex review: with only 55 positions and ~7 argmax flips, the failing
    top-1 rate alone cannot distinguish "kernel/layout error producing
    high-margin flips" from "expected 3-bit compression error nibbling at
    tiny margins." This function emits the data needed to draw that line:

    Per position i, return:

      * ``i``                — position index in the decode loop
      * ``target``           — ground-truth next token from the prompt
      * ``ref_top1``         — argmax of the FP16 reference logits
      * ``iso_top1``         — argmax of the IsoQuant logits
      * ``match``            — bool, ref_top1 == iso_top1
      * ``ref_top1_logit``   — value of ref logit at ref_top1
      * ``ref_top2_logit``   — value of ref logit at second-highest position
      * ``ref_margin``       — ref_top1_logit - ref_top2_logit
      * ``iso_at_ref_top1``  — value of iso logit at ref_top1
      * ``iso_top1_logit``   — value of iso logit at iso_top1
      * ``iso_margin``       — iso_top1_logit - second-highest iso logit
      * ``logit_cosine``     — cosine similarity at this position

    A downstream analysis script (separate file) can then ask: of the
    mismatching positions, what is the median ``ref_margin``? If it is
    large, the kernel is producing meaningfully different logits, not
    just shifting compression noise across argmax boundaries.
    """
    import numpy as np

    n = int(logits_ref.shape[0])
    if n <= 0:
        return []

    ref = logits_ref
    iso = logits_iso

    top1_ref = ref.argmax(axis=-1)
    top1_iso = iso.argmax(axis=-1)

    # ref top-2: mask top-1 and re-argmax. Use a finite floor so this
    # works even if the original logits contain very negative values.
    ref_masked = ref.copy()
    ref_masked[np.arange(n), top1_ref] = -np.inf
    top2_ref = ref_masked.argmax(axis=-1)

    iso_masked = iso.copy()
    iso_masked[np.arange(n), top1_iso] = -np.inf
    top2_iso = iso_masked.argmax(axis=-1)

    # cosine per position
    dot = (ref * iso).sum(axis=-1)
    nref = np.linalg.norm(ref, axis=-1)
    niso = np.linalg.norm(iso, axis=-1)
    cos = dot / np.maximum(nref * niso, 1e-9)

    out: list[dict] = []
    rng = np.arange(n)
    ref_top1_vals = ref[rng, top1_ref]
    ref_top2_vals = ref[rng, top2_ref]
    iso_top1_vals = iso[rng, top1_iso]
    iso_top2_vals = iso[rng, top2_iso]
    iso_at_ref_top1 = iso[rng, top1_ref]
    for i in range(n):
        out.append(
            {
                "i": int(i),
                "target": int(decode_targets[i]),
                "ref_top1": int(top1_ref[i]),
                "iso_top1": int(top1_iso[i]),
                "match": bool(top1_ref[i] == top1_iso[i]),
                "ref_top1_logit": float(ref_top1_vals[i]),
                "ref_top2_logit": float(ref_top2_vals[i]),
                "ref_margin": float(ref_top1_vals[i] - ref_top2_vals[i]),
                "iso_at_ref_top1": float(iso_at_ref_top1[i]),
                "iso_top1_logit": float(iso_top1_vals[i]),
                "iso_margin": float(iso_top1_vals[i] - iso_top2_vals[i]),
                "logit_cosine": float(cos[i]),
            }
        )
    return out


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


def load_model(
    model_path: str,
    expert_offload: bool = False,
    max_resident_experts: Optional[int] = None,
    max_cached_shards: Optional[int] = None,
):
    """Load model + tokenizer via mlx_lm.load. Raises on failure.

    Kimi K2.6 (~554 GB at 4-bit) cannot be resident-loaded on a 128 GB box,
    so the parity harness must opt into the same expert-offload path the
    Phase 4 correctness command uses. ``expert_offload=False`` keeps the
    original behaviour for non-Kimi callers.
    """
    from mlx_lm import load as mlx_load

    if not expert_offload:
        return mlx_load(model_path)

    model_config: dict = {"expert_offload": True}
    if max_resident_experts is not None:
        model_config["max_resident_experts"] = int(max_resident_experts)
    if max_cached_shards is not None:
        model_config["max_cached_shards"] = int(max_cached_shards)
    return mlx_load(model_path, model_config=model_config)


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
# IsoQuant stats / path classification
# ---------------------------------------------------------------------------
#
# Codex review flagged that the harness's hard-fail check on
# ``KimiMLAIsoQuantCache.supports_fused_latent_attention`` only proves the
# *outer* wrapper is willing to dispatch to the fused path; it does not
# prove the *inner* IsoQuantKVCache.fused_attention then ran a metal
# kernel rather than falling back. To make the gate verdict attributable,
# snapshot the global IsoQuant counters before each iso run and diff them
# afterwards. Then classify which kernel actually ran by which timing
# counters moved.


def _snapshot_iso_stats() -> dict:
    """Return current global IsoQuant counters as a dict, or {} if module
    unavailable (e.g. mlx_lm import failed for some other reason)."""
    try:
        from mlx_lm.models.mlx_isoquant import get_stats
    except Exception:
        return {}
    s = get_stats()
    if not hasattr(s, "__slots__"):
        return {}
    return {f: int(getattr(s, f, 0) or 0) for f in s.__slots__}


def _diff_iso_stats(before: dict, after: dict) -> dict:
    if not before or not after:
        return {}
    return {k: int(after.get(k, 0) - before.get(k, 0)) for k in after}


def _classify_iso_path(delta: dict) -> str:
    """Map a stats delta to a single label naming which inner path actually
    ran during the iso phase. Returns one of:

      * ``"npt16"``                 — ``ISOQUANT_USE_NPT16_FUSED=1``
                                      kernel ran (D=512 single-pass).
      * ``"npt8_or_npt8_tiled"``    — ``ISOQUANT_USE_NPT8_FUSED=1`` ran
                                      (D=256 fused).
      * ``"single_kernel"``         — Branch ``T <= _SINGLE_KERNEL_T_THRESHOLD``
                                      kernel ran. Threshold currently 0,
                                      so this is effectively dead code.
      * ``"3kernel"``               — Default fused metal path
                                      (``_fused_attention_3kernel``).
      * ``"mlx_ops_fallback"``      — Metal path attempted then failed; a
                                      ``_fused_attention_mlx`` ran. Detected
                                      by ``fused_metal_failures > 0``.
      * ``"unfused_reconstruct"``   — Inner ``supports_fused_attention``
                                      returned False; ``reconstruct_keys``
                                      + base SDPA ran. Detected by
                                      ``unfused_fallback_calls > 0``.
      * ``"no_iso_dispatch"``       — Cache was selected but no fused
                                      attempt was made (e.g. all layers
                                      skipped, or IsoQuant cache never hit).
      * ``"ambiguous"``             — Counters disagree, e.g. mixed kernels
                                      across layers/steps; the artifact
                                      records both buckets so a human can
                                      decide.
    """
    if not delta:
        return "no_iso_dispatch"
    unfused = delta.get("unfused_fallback_calls", 0)
    metal_attempts = delta.get("fused_metal_attempts", 0)
    metal_failures = delta.get("fused_metal_failures", 0)
    if metal_attempts == 0 and unfused > 0:
        return "unfused_reconstruct"
    if metal_attempts == 0 and unfused == 0:
        return "no_iso_dispatch"
    # Metal path attempted at least once.
    if metal_failures > 0:
        return "mlx_ops_fallback"
    # Metal path succeeded. Discriminate by which fused timing counter moved.
    qk = delta.get("fused_qk_ms", 0)
    val = delta.get("fused_value_ms", 0)
    val_tiled = delta.get("fused_value_tiled_ms", 0)
    inv = delta.get("fused_inverse_ms", 0)
    sk = delta.get("fused_single_kernel_ms", 0)
    total = delta.get("fused_metal_total_ms", 0)
    # The 3-kernel path increments qk/value/inverse separately and does
    # NOT touch fused_single_kernel_ms. NPT16 / NPT8 / NPT8-tiled all run
    # under the single-kernel umbrella and bump fused_single_kernel_ms
    # (and fused_metal_total_ms), but not the three-kernel timers.
    three_kernel_signal = qk > 0 or val > 0 or val_tiled > 0 or inv > 0
    single_kernel_signal = sk > 0
    if single_kernel_signal and not three_kernel_signal:
        # Distinguish NPT16 (D=512) from NPT8 (D=256) via env var, since
        # the counters share a single timer. The only way the single-kernel
        # path runs at D=512 is if NPT16 is enabled, so the env var is a
        # reliable disambiguator at classification time.
        if os.environ.get("ISOQUANT_USE_NPT16_FUSED", "0") == "1":
            return "npt16"
        if os.environ.get("ISOQUANT_USE_NPT8_FUSED", "0") == "1":
            return "npt8_or_npt8_tiled"
        return "single_kernel"
    if three_kernel_signal and not single_kernel_signal:
        return "3kernel"
    if three_kernel_signal and single_kernel_signal:
        return "ambiguous"
    # metal_attempts > 0 but no per-kernel timer moved. Could happen if
    # mx.eval was deferred and counters lag, but harness uses mx.eval per
    # step so this should be rare.
    return "ambiguous"


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
    dump_per_step: bool = False,
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
    ref_stats_before = _snapshot_iso_stats()
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
    ref_stats_delta = _diff_iso_stats(ref_stats_before, _snapshot_iso_stats())

    del cache_ref

    # ---- Test path: IsoQuant cache, NPT16 fused if requested ----
    log(f"  [{context_tokens}] iso path (IsoQuant {bits}-bit, npt16_fused={use_npt16})")
    _set_npt16(use_npt16)
    iso_stats_before = _snapshot_iso_stats()
    t0 = time.time()
    cache_iso = make_caches(model, "isoquant", bits=bits)
    logits_iso = prefill_then_teacher_forced_decode(
        model, tokens, cache_iso, prefill_len
    )
    logits_iso = logits_iso[: decode_targets.shape[0]]
    setup_iso = time.time() - t0
    iso_stats_delta = _diff_iso_stats(iso_stats_before, _snapshot_iso_stats())
    iso_path_observed = _classify_iso_path(iso_stats_delta)
    log(f"  [{context_tokens}] iso path observed: {iso_path_observed}")

    del cache_iso

    metrics = compute_metrics(logits_ref, logits_iso, decode_targets, window_size)
    slope = long_context_slope(metrics["window_ppl_drifts"])

    per_step_records: Optional[list] = None
    if dump_per_step and metrics.get("non_finite_reason") is None:
        # Only emit per-step records when both paths produced finite logits;
        # otherwise the per-position margins are meaningless.
        per_step_records = extract_per_step_diagnostics(
            logits_ref, logits_iso, decode_targets
        )

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
        non_finite_reason=metrics.get("non_finite_reason"),
        iso_path_observed=iso_path_observed,
        iso_stats_delta=iso_stats_delta,
        ref_stats_delta=ref_stats_delta,
        per_step=per_step_records,
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

    # Codex audit HIGH 5: any non-finite metric is a hard FAIL. `nan < threshold`
    # and `nan > threshold` both evaluate False, so without an explicit guard
    # a NaN-poisoned run would silently report PASS.
    for r in results:
        if not math.isfinite(r.top1_match_rate):
            reasons.append(
                f"context={r.context_tokens}: top1_match_rate is non-finite "
                f"({r.top1_match_rate}) — likely NaN/Inf in logits"
            )
        elif r.top1_match_rate < top1_threshold:
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
        if not math.isfinite(anchor.ppl_drift):
            reasons.append(
                f"context={anchor.context_tokens}: ppl_drift is non-finite "
                f"({anchor.ppl_drift}) — likely NaN/Inf in logits"
            )
        elif anchor.ppl_drift > ppl_drift_threshold:
            reasons.append(
                f"context={anchor.context_tokens}: ppl_drift {anchor.ppl_drift:.4f} "
                f"> §10 threshold {ppl_drift_threshold:.4f}"
            )

    # §9 test 3: per-window drift slope must be ≤ tolerance.
    # Codex review: a least-squares slope on fewer than ~4 windows is
    # essentially noise and should not flip the gate. The smoke run at
    # ctx=64 / window_size=16 produced 3 windows and a +0.572 slope which
    # crossed tolerance trivially. Require enough windows for the slope
    # to carry signal; below that, record it as informational only.
    _SLOPE_MIN_WINDOWS = 4
    for r in results:
        if r.long_context_slope is None:
            continue
        if not math.isfinite(r.long_context_slope):
            reasons.append(
                f"context={r.context_tokens}: window-drift slope is non-finite "
                f"({r.long_context_slope}) — likely NaN/Inf in per-window PPL"
            )
            continue
        n_windows = len(r.window_ppl_drifts)
        if n_windows < _SLOPE_MIN_WINDOWS:
            # Informational: slope exists but sample is too small to be a
            # gate. We still log it in the artifact (long_context_slope
            # field) so a human can inspect it.
            continue
        if r.long_context_slope > slope_tolerance:
            reasons.append(
                f"context={r.context_tokens}: window-drift slope {r.long_context_slope:.4f} "
                f"> tolerance {slope_tolerance:.4f} (n_windows={n_windows}) — "
                f"possible RoPE smearing"
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
    p.add_argument(
        "--expert-offload",
        action="store_true",
        help="Enable expert offload. Required for Kimi K2.6 (~554 GB at 4-bit) "
        "on 128 GB hardware; otherwise model load OOMs before any KV math runs.",
    )
    p.add_argument(
        "--max-resident-experts",
        type=int,
        default=None,
        help="Cap on resident routed-expert instances when --expert-offload is set. "
        "Mirrors the mlx_lm.generate flag.",
    )
    p.add_argument(
        "--max-cached-shards",
        type=int,
        default=None,
        help="Cap on cached safetensor shards when --expert-offload is set. "
        "Mirrors the mlx_lm.generate flag.",
    )
    p.add_argument(
        "--dump-per-step",
        action="store_true",
        help="Write a sidecar JSON next to --output containing per-position "
        "diagnostic records (target token, ref/iso top-1, ref top-1/2 "
        "logits, iso logit at ref-top-1, per-position cosine). Enables "
        "margin-conditioned mismatch analysis. Roughly ~120 bytes per "
        "decode step per context, so cheap even at 8K.",
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

    log(
        f"Loading model from {model_path} ... "
        f"(expert_offload={args.expert_offload}, "
        f"max_resident_experts={args.max_resident_experts}, "
        f"max_cached_shards={args.max_cached_shards})"
    )
    t0 = time.time()
    try:
        model, tokenizer = load_model(
            str(model_path),
            expert_offload=args.expert_offload,
            max_resident_experts=args.max_resident_experts,
            max_cached_shards=args.max_cached_shards,
        )
    except Exception as exc:
        import traceback

        tb = traceback.format_exc()
        print(f"ERROR: model load failed: {exc}", file=sys.stderr)
        print(tb, file=sys.stderr)
        # Write a setup-error stub so CI has an artifact to inspect.
        out_path.write_text(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                    "gate_status": "SETUP_ERROR",
                    "error": str(exc),
                    "traceback": tb,
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
                dump_per_step=args.dump_per_step,
            )
            log(
                f"  → top1={r.top1_match_rate:.4f}  ppl_drift={r.ppl_drift:+.4f}  "
                f"cos={r.mean_logit_cosine:.4f}  slope={r.long_context_slope}  "
                f"path={r.iso_path_observed}"
            )
            results.append(r)
    except Exception as exc:
        import traceback

        tb = traceback.format_exc()
        print(f"ERROR: evaluation failed: {exc}", file=sys.stderr)
        print(tb, file=sys.stderr)
        out_path.write_text(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                    "gate_status": "SETUP_ERROR",
                    "error": str(exc),
                    "traceback": tb,
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

    # Strip per_step from each ContextResult before serializing to the main
    # artifact; it goes to a sidecar so the main JSON stays scannable.
    contexts_payload = []
    for r in results:
        d = asdict(r)
        d.pop("per_step", None)
        contexts_payload.append(d)

    # Snapshot the active env flags so the artifact records the kernel
    # configuration in effect at the *end* of the run. The harness toggles
    # ISOQUANT_USE_NPT16_FUSED itself; capturing it here lets a reader
    # confirm the iso phase ran with NPT16 enabled even without inspecting
    # path counters.
    env_snapshot = {
        k: os.environ.get(k)
        for k in (
            "ISOQUANT_USE_NPT16_FUSED",
            "ISOQUANT_USE_NPT8_FUSED",
            "ISOQUANT_BITS",
            "ISOQUANT_USE_METAL",
            "ISOQUANT_CACHE_MODE",
            "TURBOQUANT_SKIP_LAYERS",
        )
        if os.environ.get(k) is not None
    }

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
            "prefill_len": args.prefill_len,
            "window_size": args.window_size,
            "smoke": args.smoke,
            "expert_offload": args.expert_offload,
            "max_resident_experts": args.max_resident_experts,
            "max_cached_shards": args.max_cached_shards,
            "dump_per_step": args.dump_per_step,
        },
        "env_snapshot": env_snapshot,
        "verdict_reasons": verdict.reasons,
        "load_seconds": load_seconds,
        "contexts": contexts_payload,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    log(f"Saved: {out_path}  gate_status={gate_status}")

    # Sidecar with per-step diagnostics (for margin-conditioned analysis).
    if args.dump_per_step:
        per_step_payload = {
            "timestamp": payload["timestamp"],
            "model_path": payload["model_path"],
            "gate_status": gate_status,
            "config": payload["config"],
            "env_snapshot": env_snapshot,
            "contexts": [
                {
                    "context_tokens": r.context_tokens,
                    "iso_path_observed": r.iso_path_observed,
                    "per_step": r.per_step or [],
                }
                for r in results
            ],
        }
        sidecar_path = out_path.with_suffix(".per_step.json")
        sidecar_path.write_text(json.dumps(per_step_payload, indent=2))
        log(f"Saved per-step diagnostics: {sidecar_path}")
    if verdict.reasons:
        log("Verdict reasons:")
        for r in verdict.reasons:
            log(f"  - {r}")

    return 0 if verdict.passed else 1


if __name__ == "__main__":
    sys.exit(main())
