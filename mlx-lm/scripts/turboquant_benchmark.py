#!/usr/bin/env python3
"""
Benchmark KV cache memory footprint (TurboQuant vs full-precision).

Requires a local MLX model directory. Does not run full generation by default;
use --generate to run a short mlx_lm.generate smoke test.

Example:
  python scripts/turboquant_benchmark.py --model /path/to/model \\
      --prompt-tokens 256 --kv-cache-type default
  python scripts/turboquant_benchmark.py --model /path/to/model \\
      --prompt-tokens 256 --kv-cache-type turboquant
"""

from __future__ import annotations

import argparse
import gc
import time

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.models import cache as cache_mod


def nbytes_list(caches) -> int:
    return sum(getattr(c, "nbytes", 0) for c in caches)


def main():
    p = argparse.ArgumentParser(description="TurboQuant KV cache benchmark")
    p.add_argument("--model", required=True, help="MLX model path or HF id")
    p.add_argument("--prompt-tokens", type=int, default=256)
    p.add_argument(
        "--kv-cache-type",
        choices=["default", "turboquant"],
        default="default",
    )
    p.add_argument(
        "--generate",
        action="store_true",
        help="Run a short generate() after cache fill (adds decode overhead)",
    )
    p.add_argument("--max-tokens", type=int, default=8)
    args = p.parse_args()

    print(f"Loading {args.model} ...")
    model, tokenizer = load(args.model)
    mx.clear_cache()
    gc.collect()

    base = "Hello. " * max(1, args.prompt_tokens // 2)
    prompt_ids = tokenizer.encode(base)[: args.prompt_tokens]
    prompt = tokenizer.decode(prompt_ids)
    print(f"Prompt length: {len(prompt_ids)} tokens")

    caches = cache_mod.make_prompt_cache(model, kv_cache_type=args.kv_cache_type)
    t0 = time.perf_counter()
    toks = mx.array(prompt_ids)[None]
    model(toks, cache=caches)
    mx.eval([x for x in toks])
    from mlx_lm.generate import _eval_prompt_cache_state

    _eval_prompt_cache_state(caches)
    t1 = time.perf_counter()

    nb = nbytes_list(caches)
    print(f"KV cache type: {args.kv_cache_type}")
    print(f"Prefill wall time: {t1 - t0:.3f}s")
    print(f"Sum(cache.nbytes): {nb / 1e6:.2f} MB")

    if args.generate:
        text = generate(
            model,
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            verbose=False,
            prompt_cache=caches,
            kv_cache_type=args.kv_cache_type,
        )
        print(f"Generated ({args.max_tokens} max): {text[:200]!r}...")


if __name__ == "__main__":
    main()
