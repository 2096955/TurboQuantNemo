#!/usr/bin/env python3
"""Stage 2 quantize Kimi K2.6 -> 2-bit routed experts, chunked-eval edition.

Why this exists: the standard `mlx_lm convert --quantize --mixed-expert-bits 2`
path crashes with kIOGPUCommandBufferCallbackErrorTimeout during save_safetensors
on Kimi K2.6. Root cause: lazy-eval graph for the 2-bit quantization stays
unmaterialized until save time, then save_safetensors triggers giant Metal
command buffers that exceed macOS GPU watchdog (~5-second limit).

Fix: same quantize_model + same predicate, but before save we walk every leaf
parameter and call mx.eval + mx.synchronize per tensor. This forces small,
bounded Metal command buffers and avoids the watchdog.

Two source modes:

(1) bf16 source (the original two-stage path):
    PYTHONPATH=mlx-lm python3 scripts/convert_kimi_2bit_chunked.py \
        --src /Volumes/Samsung9904tb/Kimi-K2.6-bf16 \
        --dst /Volumes/Samsung9904tb/Kimi-K2.6-2bit-experts

(2) 4-bit pack-quantized source, in-memory dequantize-then-requantize
    (saves ~1.9 TB of disk by skipping the bf16 intermediate):
    PYTHONPATH=mlx-lm python3 scripts/convert_kimi_2bit_chunked.py \
        --src /Volumes/Samsung9904tb/Kimi-K2.6 \
        --dst /Volumes/Samsung9904tb/Kimi-K2.6-2bit-experts \
        --source-quantized

In mode (2) the dequantize step swaps QuantizedSwitchLinear -> SwitchLinear in
the module tree (so the quantize predicate's `to_quantized` check fires) and
mx.dequantize is lazy: the bf16 graph stays symbolic through the dequantize
phase (verified: RSS delta 0.00 GB during dequantize_model on Kimi K2.6).
The per-tensor mx.eval + mx.synchronize in the save loop bounds materialised
memory to one quantized projection's subgraph at a time — i.e. one
{weight, scales, biases} triple per dispatch, not literally one tensor —
since mx.quantize emits all three outputs from a single op (Codex review
2026-05-07 MEDIUM #4). That bound is tight enough to keep RSS well under
the 32 GB budget; the whole-graph eval that the standard mlx_lm.convert
triggers is what the chunked loop avoids.

Disk peak: source + 2-bit final (~856 GB). The two-stage path peaks at
source + bf16 intermediate + 2-bit final (~2.76 TB).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MLX_LM_ROOT = REPO_ROOT / "mlx-lm"
if MLX_LM_ROOT.exists() and str(MLX_LM_ROOT) not in sys.path:
    sys.path.insert(0, str(MLX_LM_ROOT))

KIMI_REQUIRED_VOLUME = Path("/Volumes/Samsung9904tb")


def assert_kimi_volume(*paths: Path) -> None:
    """Refuse to run unless every path lives on the expected Samsung 4 TB SSD.

    Bypass with KIMI_VOLUME_GUARD=0 (not recommended). The diskutil identity
    check guards against phantom mounts where /Volumes/Samsung9904tb routes
    to a different physical disk after a reconnect.
    """
    if os.environ.get("KIMI_VOLUME_GUARD", "1") == "0":
        return
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        try:
            resolved.relative_to(KIMI_REQUIRED_VOLUME)
        except ValueError:
            raise SystemExit(
                f"Refusing to use {resolved}: Kimi conversions must target "
                f"{KIMI_REQUIRED_VOLUME} only. Set KIMI_VOLUME_GUARD=0 to bypass."
            )
    if not KIMI_REQUIRED_VOLUME.is_dir():
        raise SystemExit(f"{KIMI_REQUIRED_VOLUME} is not mounted.")
    try:
        info = subprocess.run(
            ["diskutil", "info", str(KIMI_REQUIRED_VOLUME)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"diskutil info failed for {KIMI_REQUIRED_VOLUME}: {exc}")
    if "Volume Name:" not in info or "Samsung9904tb" not in info:
        raise SystemExit(
            f"{KIMI_REQUIRED_VOLUME} does not report Volume Name 'Samsung9904tb'; "
            "possible phantom mount."
        )
    if "APFS" not in info:
        raise SystemExit(f"{KIMI_REQUIRED_VOLUME} is not APFS as expected.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src",
        required=True,
        help="Source path. bf16 dequantized checkpoint by default; pass "
        "--source-quantized to point at the 4-bit pack-quantized source.",
    )
    p.add_argument("--dst", required=True, help="2-bit final output path")
    p.add_argument("--mixed-expert-bits", type=int, default=2)
    p.add_argument("--shared-expert-bits", type=int, default=4)
    p.add_argument("--q-bits", type=int, default=4)
    p.add_argument("--q-group-size", type=int, default=64)
    p.add_argument("--q-mode", default="affine")
    p.add_argument(
        "--source-quantized",
        action="store_true",
        help="Source is already pack-quantized (e.g. the upstream Kimi K2.6 "
        "checkpoint). dequantize_model() is run in-memory before "
        "quantize_model() so the bf16 intermediate never hits disk.",
    )
    args = p.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    src = Path(args.src)
    dst = Path(args.dst)
    assert_kimi_volume(src, dst)
    if not src.is_dir():
        raise SystemExit(f"src not found: {src}")
    if dst.exists():
        raise SystemExit(f"dst already exists: {dst} (rm -rf if you want to re-run)")

    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.convert import _build_mixed_expert_quant_predicate
    from mlx_lm.utils import dequantize_model, load, quantize_model, save_config

    print(f"[load] {src}", flush=True)
    t0 = time.time()
    model, tokenizer, config = load(
        str(src),
        return_config=True,
        tokenizer_config={"trust_remote_code": True},
        lazy=True,
    )
    print(f"[load] done in {time.time() - t0:.1f}s", flush=True)

    if args.source_quantized:
        print("[dequantize] in-memory dequantize before requantize", flush=True)
        t0 = time.time()
        model = dequantize_model(model)
        # Drop the source quantization stanza so quantize_model can write the
        # new one cleanly. mx.dequantize is lazy — no eval triggered here.
        config.pop("quantization", None)
        config.pop("quantization_config", None)
        print(
            f"[dequantize] graph rebuilt in {time.time() - t0:.1f}s (lazy)",
            flush=True,
        )

    predicate = _build_mixed_expert_quant_predicate(
        mixed_expert_bits=args.mixed_expert_bits,
        shared_expert_bits=args.shared_expert_bits,
        default_bits=args.q_bits,
        default_group_size=args.q_group_size,
        mode=args.q_mode,
    )

    print(
        f"[quantize] mixed_expert_bits={args.mixed_expert_bits} "
        f"shared_expert_bits={args.shared_expert_bits} "
        f"q_bits={args.q_bits} q_group_size={args.q_group_size} "
        f"q_mode={args.q_mode}",
        flush=True,
    )
    t0 = time.time()
    model, config = quantize_model(
        model,
        config,
        args.q_group_size,
        args.q_bits,
        mode=args.q_mode,
        quant_predicate=predicate,
    )
    print(f"[quantize] graph built in {time.time() - t0:.1f}s (lazy)", flush=True)

    # Per-shard, per-tensor materialize-then-save loop. This bounds memory
    # to one shard's worth of materialized tensors at any time AND keeps
    # Metal command buffers small enough to dodge the GPU watchdog.
    #
    # The previous attempts failed two ways:
    #   1. Default mlx_lm.convert: save_safetensors evals a whole shard
    #      worth of lazy quantization graph -> kIOGPUCommandBufferCallback-
    #      ErrorTimeout (5s GPU watchdog).
    #   2. Materialize-everything-then-save: held all materialized weights
    #      resident -> kernel SIGKILL on memory pressure.
    # This third path: stream shard-by-shard, eval per-tensor inside the
    # shard, write the shard, drop all references, mx.clear_cache().
    import gc
    import json

    from mlx_lm.utils import get_total_parameters, make_shards

    print(f"[save] writing to {dst}", flush=True)
    t0 = time.time()
    dst.mkdir(parents=True, exist_ok=True)

    weights = dict(tree_flatten(model.parameters()))
    shards = make_shards(weights)
    shards_count = len(shards)
    shard_format = (
        "model-{:05d}-of-{:05d}.safetensors"
        if shards_count > 1
        else "model.safetensors"
    )
    total_size = sum(v.nbytes for v in weights.values())
    index_data = {
        "metadata": {
            "total_size": total_size,
            "total_parameters": get_total_parameters(model),
        },
        "weight_map": {},
    }

    # Drop the now-redundant flattened dict so refcount drops correctly
    # when shards are processed and released.
    weights.clear()
    del weights

    # CRITICAL for memory: donate the model. The model object holds
    # references to every parameter through its module hierarchy, which
    # would prevent gc from freeing tensors after we save+del each shard.
    # Replacing every parameter with an empty mx.array breaks those refs
    # so the only remaining references are in `shards` — which we
    # explicitly free per iteration.
    from mlx.utils import tree_map

    model.update(tree_map(lambda _: mx.array([]), model.parameters()))
    gc.collect()

    print(
        f"[save] {shards_count} shards, {total_size / 1024**3:.1f} GB total",
        flush=True,
    )
    last_print = 0.0
    for i in range(shards_count):
        shard = shards[i]
        shards[i] = None  # break the list's ref so shard refcount can drop
        shard_name = shard_format.format(i + 1, shards_count)
        shard_path = dst / shard_name

        # Materialize this shard's tensors one at a time (bounded command
        # buffer per eval).
        for _key, arr in shard.items():
            mx.eval(arr)
            mx.synchronize()

        # All tensors materialized; save_safetensors should not need to
        # evaluate anything further.
        mx.save_safetensors(str(shard_path), shard, metadata={"format": "mlx"})

        for k in shard:
            index_data["weight_map"][k] = shard_name

        # Aggressively free this shard before moving on.
        del shard
        gc.collect()
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

        now = time.time()
        if now - last_print > 5.0 or i == shards_count - 1:
            elapsed = now - t0
            done = i + 1
            rate = done / max(elapsed, 0.001)
            eta = (shards_count - done) / rate
            print(
                f"  shard {done}/{shards_count} written "
                f"({elapsed:.0f}s elapsed, {rate:.2f} shard/s, ETA {eta:.0f}s)",
                flush=True,
            )
            last_print = now

    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }
    with open(dst / "model.safetensors.index.json", "w") as f:
        json.dump(index_data, f, indent=4)
    print(f"[save] all shards written in {time.time() - t0:.1f}s", flush=True)

    # Weights already written above. Just save config + tokenizer.
    print("[save] writing config + tokenizer", flush=True)
    save_config(config, config_path=dst / "config.json")
    if tokenizer is not None:
        tokenizer.save_pretrained(dst)
    print(f"[save] all done in {time.time() - t0:.1f}s", flush=True)

    print(f"\nDONE: {dst}", flush=True)


if __name__ == "__main__":
    main()
