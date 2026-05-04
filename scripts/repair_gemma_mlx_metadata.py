#!/usr/bin/env python3
"""Repair missing MLX metadata for local Gemma 4 checkpoints.

This script reconstructs a usable MLX model directory by:
1. Copying tokenizer / generation assets from a cached HF snapshot.
2. Rebuilding config.json with inferred quantization overrides.
3. Rebuilding model.safetensors.index.json so expert weights come from the
   repacked shards while non-expert weights stay on the original model shards.

It is intentionally specific to the local Gemma 4 checkpoints in this workspace.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from safetensors import safe_open


ASSET_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "chat_template.jinja",
    "processor_config.json",
)
COMMON_GROUP_SIZES = (64, 32, 128, 16, 256)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair missing MLX metadata for local Gemma 4 checkpoints."
    )
    parser.add_argument(
        "--snapshot",
        required=True,
        type=Path,
        help="Cached HF snapshot root for google/gemma-4-26B-A4B-it.",
    )
    parser.add_argument(
        "models",
        nargs="+",
        type=Path,
        help="One or more local MLX Gemma model directories to repair.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def copy_assets(snapshot: Path, model_dir: Path) -> None:
    for name in ASSET_FILES:
        src = snapshot / name
        if src.is_file():
            shutil.copy2(src, model_dir / name)


def infer_bits_and_group_size(weight_shape: tuple[int, ...], scale_shape: tuple[int, ...]) -> tuple[int, int]:
    packed_cols = int(weight_shape[-1])
    scale_cols = int(scale_shape[-1])
    for group_size in COMMON_GROUP_SIZES:
        denom = scale_cols * group_size
        if denom <= 0:
            continue
        numerator = packed_cols * 32
        if numerator % denom != 0:
            continue
        bits = numerator // denom
        if 1 <= bits <= 8:
            return bits, group_size
    raise ValueError(
        f"Could not infer quantization for weight_shape={weight_shape}, "
        f"scale_shape={scale_shape}"
    )


def infer_quantization_map(model_dir: Path) -> dict[str, dict[str, Any]]:
    quant_map: dict[str, dict[str, Any]] = {}
    for shard in sorted(model_dir.glob("model-*.safetensors")):
        with safe_open(str(shard), framework="pt") as handle:
            keys = set(handle.keys())
            for key in keys:
                if not key.endswith(".weight"):
                    continue
                base = key[:-7]
                scales_key = f"{base}.scales"
                if scales_key not in keys:
                    continue

                weight = handle.get_tensor(key)
                scales = handle.get_tensor(scales_key)
                if len(weight.shape) < 2 or len(scales.shape) < 2:
                    continue

                bits, group_size = infer_bits_and_group_size(weight.shape, scales.shape)
                quant_map[base] = {
                    "bits": bits,
                    "group_size": group_size,
                    "mode": "affine",
                }

    if not quant_map:
        raise ValueError(f"No quantized tensors found under {model_dir}")

    return quant_map


def build_quantization_config(
    inferred_quant: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    counts = Counter(
        (cfg["bits"], cfg["group_size"], cfg["mode"]) for cfg in inferred_quant.values()
    )
    (default_bits, default_group_size, default_mode), _ = counts.most_common(1)[0]

    quantization: dict[str, Any] = {
        "bits": default_bits,
        "group_size": default_group_size,
        "mode": default_mode,
    }
    for path in sorted(inferred_quant):
        cfg = inferred_quant[path]
        if (
            cfg["bits"] != default_bits
            or cfg["group_size"] != default_group_size
            or cfg["mode"] != default_mode
        ):
            quantization[path] = cfg
    return quantization


def build_config(snapshot_config: dict[str, Any], model_dir: Path) -> dict[str, Any]:
    config = copy.deepcopy(snapshot_config)
    config.pop("_name_or_path", None)
    config.pop("vision_config", None)

    quantization = build_quantization_config(infer_quantization_map(model_dir))
    config["quantization"] = quantization
    config["quantization_config"] = copy.deepcopy(quantization)
    return config


def build_index(model_dir: Path) -> dict[str, Any]:
    weight_map: dict[str, str] = {}

    for shard in sorted(model_dir.glob("model-*.safetensors")):
        with safe_open(str(shard), framework="pt") as handle:
            for key in handle.keys():
                if ".experts.switch_glu." in key:
                    continue
                weight_map[key] = shard.name

    repacked_shards = sorted(model_dir.glob("repacked-model-*.safetensors"))
    for shard in repacked_shards:
        with safe_open(str(shard), framework="pt") as handle:
            for key in handle.keys():
                weight_map[key] = shard.name

    if not weight_map:
        raise ValueError(f"No weights found under {model_dir}")
    if not repacked_shards:
        raise ValueError(f"No repacked expert shards found under {model_dir}")

    return {
        "metadata": {
            "repaired": True,
            "source": "scripts/repair_gemma_mlx_metadata.py",
        },
        "weight_map": weight_map,
    }


def repair_model(snapshot: Path, model_dir: Path) -> None:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    snapshot_config = load_json(snapshot / "config.json")
    copy_assets(snapshot, model_dir)
    dump_json(model_dir / "config.json", build_config(snapshot_config, model_dir))
    dump_json(model_dir / "model.safetensors.index.json", build_index(model_dir))


def main() -> None:
    args = parse_args()
    snapshot = args.snapshot.expanduser().resolve()
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Snapshot directory not found: {snapshot}")

    for model_dir in args.models:
        model_dir = model_dir.expanduser().resolve()
        repair_model(snapshot, model_dir)
        print(f"repaired {model_dir}")


if __name__ == "__main__":
    main()
