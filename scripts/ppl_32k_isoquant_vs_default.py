#!/usr/bin/env python3
"""Phase 4: long-context PPL regression at 32K. Flag divergence > 5%.

Compares IsoQuant vs default KV cache perplexity on wikitext-103-raw-test
at 32K context length. Uses nvfp4 model weights for consistency with the
benchmark matrix.
"""

import json
import os
import sys

os.environ.setdefault("ISOQUANT_BITS", "3")
os.environ.setdefault("ISOQUANT_USE_NPT8_FUSED", "1")

import mlx.core as mx

from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODEL = os.environ.get("PPL_MODEL", "/Users/anthonylui/Models/Qwen3.6-35B-A3B-nvfp4")
CTX = int(os.environ.get("PPL_CTX", "32768"))
DATA_PATH = os.environ.get("PPL_DATA", "data/wikitext-103-raw-test.txt")
DIVERGENCE_THRESHOLD = 5.0


def compute_ppl(model, tokenizer, kv_cache_type: str, text: str) -> float:
    cache = make_prompt_cache(model, kv_cache_type=kv_cache_type)
    ids = tokenizer.encode(text)[:CTX]
    if len(ids) < CTX:
        print(
            f"  Warning: text only has {len(ids)} tokens, requested {CTX}",
            file=sys.stderr,
        )
    arr = mx.array(ids)
    logits = model(arr[None, :], cache=cache)
    mx.eval(logits)
    log_probs = logits[0, :-1, :] - mx.logsumexp(
        logits[0, :-1, :], axis=-1, keepdims=True
    )
    targets = arr[1:]
    nll = -mx.take_along_axis(log_probs, targets[:, None], axis=-1).mean()
    mx.eval(nll)
    return float(mx.exp(nll))


def main():
    if not os.path.exists(DATA_PATH):
        print(f"Data not found: {DATA_PATH}")
        print("Download with: see Phase 4 plan Task 5")
        sys.exit(1)

    text = open(DATA_PATH).read()
    print(f"Loading model {MODEL}...")
    model, tokenizer = load(MODEL)

    print(f"Computing PPL with default KV at {CTX} context...")
    ppl_default = compute_ppl(model, tokenizer, "default", text)
    print(f"  PPL (default): {ppl_default:.4f}")

    print(f"Computing PPL with IsoQuant KV at {CTX} context...")
    ppl_iso = compute_ppl(model, tokenizer, "isoquant", text)
    print(f"  PPL (isoquant): {ppl_iso:.4f}")

    divergence_pct = abs(ppl_iso - ppl_default) / ppl_default * 100.0
    passed = divergence_pct < DIVERGENCE_THRESHOLD

    result = {
        "model": MODEL,
        "ctx": CTX,
        "ppl_default": ppl_default,
        "ppl_isoquant": ppl_iso,
        "divergence_pct": round(divergence_pct, 3),
        "threshold_pct": DIVERGENCE_THRESHOLD,
        "pass": passed,
    }

    out_dir = "artifacts/phase4_scaling"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ppl_32k.json")
    json.dump(result, open(out_path, "w"), indent=2)

    print(f"\nDivergence: {divergence_pct:.2f}% (threshold: {DIVERGENCE_THRESHOLD}%)")
    print(f"Gate: {'PASS' if passed else 'FAIL'}")
    print(f"Wrote {out_path}")
    print(json.dumps(result, indent=2))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
