# Copyright © 2026 Apple Inc.

"""Selective safetensors loading for MoE expert offload (per-key reads, no full-shard eager load)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import mlx.core as mx

from .expert_offload import is_expert_weight_key


def load_weight_map(model_path: Path) -> dict[str, str] | None:
    """Return weight_map from model.safetensors.index.json, or None if missing."""
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        return None
    with open(index_path) as f:
        data = json.load(f)
    return data.get("weight_map", data)


def build_weight_map_single_shard(
    model_path: Path, shard_name: str = "model.safetensors"
) -> dict[str, str]:
    """Map every tensor key to a single shard file (no index.json)."""
    path = model_path / shard_name
    if not path.is_file():
        raise FileNotFoundError(f"No {shard_name} in {model_path}")
    shard_data = mx.load(str(path))
    return {k: shard_name for k in shard_data.keys()}


def build_weight_map_multi_glob(model_path: Path) -> dict[str, str]:
    """Scan all model*.safetensors shards and assign keys (slow; avoids index.json)."""
    weight_map: dict[str, str] = {}
    for wf in sorted(model_path.glob("model*.safetensors")):
        shard = wf.name
        shard_data = mx.load(str(wf))
        for k in shard_data.keys():
            weight_map[k] = shard
    if not weight_map:
        raise FileNotFoundError(
            f"No model*.safetensors tensors found under {model_path} (empty glob or unreadable shards)."
        )
    return weight_map


def load_non_expert_weights(
    model_path: Path, weight_map: dict[str, str]
) -> dict[str, mx.array]:
    """Load only tensors whose keys are not routed expert weights (Nemotron-H or Gemma 4)."""
    # Group keys by shard for batched open
    shard_to_keys: dict[str, list] = defaultdict(list)
    for key, shard in weight_map.items():
        is_expert = (
            is_expert_weight_key(key, model_type="nemotron_h")
            or is_expert_weight_key(key, model_type="gemma4_text")
            or is_expert_weight_key(key, model_type="qwen3_moe")
            or is_expert_weight_key(key, model_type="qwen3_5_moe")
        )
        if not is_expert and not key.startswith("mtp."):
            shard_to_keys[shard].append(key)

    if not weight_map:
        raise ValueError(
            "load_non_expert_weights: empty weight_map (misconfigured checkpoint path)."
        )

    weights: dict[str, mx.array] = {}
    base = Path(model_path)
    for shard, keys in shard_to_keys.items():
        path = base / shard
        if not path.is_file():
            raise FileNotFoundError(f"Shard missing: {path}")
        shard_tensors = mx.load(str(path))
        for k in keys:
            if k not in shard_tensors:
                raise KeyError(f"Key {k!r} missing from {path}")
            weights[k] = shard_tensors[k]
    return weights


def validate_weight_map_shards_exist(
    model_path: Path, weight_map: dict[str, str]
) -> None:
    """Ensure every shard file referenced by the map exists on disk (fast index sanity check)."""
    base = Path(model_path)
    for shard in sorted(set(weight_map.values())):
        p = base / shard
        if not p.is_file():
            raise FileNotFoundError(
                f"Checkpoint index references missing shard {shard!r} at {p}"
            )


def resolve_weight_map(model_path: Path) -> dict[str, str]:
    """Prefer index.json; else single-shard or multi-shard scan."""
    wm = load_weight_map(model_path)
    if wm is not None:
        if not wm:
            raise ValueError(
                f"Empty weight_map in {model_path / 'model.safetensors.index.json'} (corrupt index or wrong path)."
            )
        validate_weight_map_shards_exist(model_path, wm)
        return wm
    single = model_path / "model.safetensors"
    if single.is_file():
        return build_weight_map_single_shard(model_path)
    return build_weight_map_multi_glob(model_path)
