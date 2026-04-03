#!/usr/bin/env python3
"""
Export a single-head OpenEvolve fixture from real model KV (Phase 3 style capture).

Uses validate_real_kv.capture_kv_from_attention, builds TurboQuantKVCache for the
chosen layer (same layer_idx/seed as production), then writes tensors for evaluator.py.

Run:
  PYTHONPATH=/path/to/turboquant-mlx python export_phase3_fixture.py [options]

Requires mlx-lm, model weights, and codebooks (see --codebook-dir).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_EV = Path(__file__).resolve().parent
if str(_EV) not in sys.path:
    sys.path.insert(0, str(_EV))

from mlx_lm import load
from mlx_lm.models.mlx_turboquant import TurboQuantKVCache
from validate_real_kv import capture_kv_from_attention


def main() -> None:
    ap = argparse.ArgumentParser(description="Export .npz fixture from captured KV")
    ap.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    ap.add_argument("--prompt", default=None, help="Prompt text (default: long filler)")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--kv-head", type=int, default=0)
    ap.add_argument(
        "--query-head-offset",
        type=int,
        default=0,
        help="Query head within GQA group (0 .. n_q//n_kv - 1)",
    )
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--codebook-dir", type=str, default=None)
    ap.add_argument(
        "--output",
        type=Path,
        default=_EV / "fixtures" / "phase3_layer0_h0.npz",
    )
    args = ap.parse_args()

    codebook_dir = args.codebook_dir or str(_ROOT / "codebooks")
    prompt = args.prompt or ("The quick brown fox jumps over the lazy dog. " * 200)

    print(f"Loading {args.model}...")
    model, tokenizer = load(args.model, lazy=True)

    config = getattr(model, "args", getattr(model, "config", None))
    n_q = getattr(config, "num_attention_heads", 32)
    n_kv = getattr(config, "num_key_value_heads", n_q)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None and config is not None:
        head_dim = getattr(config, "hidden_size", 4096) // n_q

    if args.kv_head < 0 or args.kv_head >= n_kv:
        raise SystemExit(f"--kv-head must be in [0, {n_kv - 1}]")
    heads_per_kv = n_q // n_kv
    if args.query_head_offset < 0 or args.query_head_offset >= heads_per_kv:
        raise SystemExit(f"--query-head-offset must be in [0, {heads_per_kv - 1}]")

    print("Capturing KV...")
    captured, _ = capture_kv_from_attention(
        model, tokenizer, prompt, max_tokens=args.prompt_tokens
    )
    if args.layer not in captured:
        raise SystemExit(
            f"Layer {args.layer} not in capture (keys: {list(captured.keys())})"
        )

    queries, keys, values = captured[args.layer][0]

    tq = TurboQuantKVCache(
        num_heads=n_kv,
        head_dim=head_dim,
        bit_width=args.bits,
        layer_idx=args.layer,
        codebook_dir=codebook_dir,
        seed=args.seed,
    )
    tq.update_and_fetch(keys, values)

    qh = args.kv_head * heads_per_kv + args.query_head_offset
    q_slice = queries[:, qh : qh + 1, :, :]
    comp = tq.compressed_keys[args.kv_head]
    rot = tq.rotation_matrices[args.kv_head]
    S = tq.compressor.S
    qjl_scale = tq.compressor.qjl_scale
    scale = 1.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        query=np.array(q_slice),
        rotation=np.array(rot),
        S=np.array(S),
        qjl_scale=qjl_scale,
        scale=scale,
        head_dim=head_dim,
        bits=args.bits,
        x_rot_quant=np.array(comp["x_rot_quant"]),
        x_norm=np.array(comp["x_norm"]),
        residual_signs=np.array(comp["residual_signs"]),
        residual_norm=np.array(comp["residual_norm"]),
        layer_idx=args.layer,
        kv_head=args.kv_head,
        query_head=qh,
        n_kv_heads=n_kv,
        n_query_heads=n_q,
    )
    print(
        f"Wrote {args.output.resolve()} (layer={args.layer}, kv_head={args.kv_head}, query_head={qh})"
    )


if __name__ == "__main__":
    main()
