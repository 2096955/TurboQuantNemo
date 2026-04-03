import argparse
import json
from pathlib import Path

import mlx.core as mx

from mlx.utils import tree_flatten


def _build_repacked_weights_for_shard(
    *,
    shard: str,
    keys: list[str],
    weights: dict[str, mx.array],
    n_routed_experts: int,
) -> dict[str, mx.array]:
    new_weights: dict[str, mx.array] = {}
    for k in keys:
        if ".switch_mlp." not in k:
            new_weights[k] = weights.pop(k)
            continue

        # e.g. backbone.layers.1.mixer.switch_mlp.fc1.weight
        parts = k.split(".")
        layer_idx = parts[2]
        proj = parts[5]  # fc1 or fc2
        suffix = parts[6]  # weight, scales, biases
        orig_proj = "up_proj" if proj == "fc1" else "down_proj"

        stacked_tensor = weights.pop(k)
        num_experts = stacked_tensor.shape[0]
        if num_experts != n_routed_experts:
            raise ValueError(
                f"Tensor {k} has {num_experts} experts, expected {n_routed_experts} from config."
            )

        for e in range(num_experts):
            expert_key = (
                f"backbone.layers.{layer_idx}.mixer.experts.{e}.{orig_proj}.{suffix}"
            )
            new_weights[expert_key] = stacked_tensor[e]
    return new_weights


def repack_checkpoint(model_path: Path) -> None:
    model_path = Path(model_path)
    if not model_path.is_dir():
        raise ValueError(f"Model path {model_path} is not a directory.")

    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    if config.get("model_type") != "nemotron_h":
        raise ValueError(
            f"Expected model_type 'nemotron_h', got '{config.get('model_type')}'."
        )

    n_routed_experts = config.get("n_routed_experts")
    if n_routed_experts is None:
        raise ValueError("Missing 'n_routed_experts' in config.json")

    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index file: {index_path}")

    with open(index_path, "r") as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})

    # Group keys by the shard they belong to
    shards_to_keys: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        shards_to_keys.setdefault(shard, []).append(key)

    new_weight_map: dict[str, str] = {}
    new_shards: set[str] = set()

    for shard, keys in shards_to_keys.items():
        has_switch = any(".switch_mlp." in k for k in keys)
        shard_file = model_path / shard

        if not has_switch:
            # Keep shard mapping unchanged
            for k in keys:
                new_weight_map[k] = shard
            continue

        print(f"Repacking {shard}...")
        weights = dict(mx.load(str(shard_file)).items())
        repacked = _build_repacked_weights_for_shard(
            shard=shard,
            keys=keys,
            weights=weights,
            n_routed_experts=n_routed_experts,
        )

        # Write to a new shard file name so index swap is an all-or-nothing checkpoint flip.
        repacked_shard = f"repacked-{shard}"
        repacked_path = model_path / repacked_shard
        mx.save_safetensors(str(repacked_path), repacked)
        print(f"Prepared {repacked_path}")

        for name, _ in tree_flatten(repacked):
            new_weight_map[name] = repacked_shard
                
        new_shards.add(repacked_shard)

    # Atomic commit: replace only the index last.
    index["weight_map"] = new_weight_map
    tmp_index_path = index_path.with_suffix(".json.tmp")
    with open(tmp_index_path, "w") as f:
        json.dump(index, f, indent=2)
    tmp_index_path.replace(index_path)
    print("Updated model.safetensors.index.json")

    if new_shards:
        print(f"Repacked shards: {len(new_shards)}")


def main():
    parser = argparse.ArgumentParser(
        description="Repack stacked switch_mlp tensors into per-expert tensors for offloading."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the converted MLX model."
    )
    args = parser.parse_args()
    repack_checkpoint(Path(args.model))


if __name__ == "__main__":
    main()
