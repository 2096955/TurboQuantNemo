#!/usr/bin/env python3
"""Fast-fail checks for MLX model directories: index.json, shard presence, config sanity.

Uses the same weight-map resolution as expert offload (`resolve_weight_map`).
Exit: 0 = OK, 1 = validation error, 2 = bad arguments.

Optional checks:
  --expect-repack     index must reference at least one repacked-*.safetensors shard
  --expect-expert-keys  weight map must contain Nemotron-style routed expert keys
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _expert_like_keys(weight_map: dict[str, str]) -> list[str]:
    return [k for k in weight_map if ".experts." in k and "shared_experts" not in k]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate model.safetensors index, shards on disk, and optional repack/config rules."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to an MLX model directory (contains config.json and safetensors).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON summary on stdout.",
    )
    parser.add_argument(
        "--require-config",
        action="store_true",
        help="Fail if config.json is missing.",
    )
    parser.add_argument(
        "--expect-repack",
        action="store_true",
        help="Require at least one shard file name starting with repacked- in the index.",
    )
    parser.add_argument(
        "--expect-expert-keys",
        action="store_true",
        help="Require weight_map keys that look like Nemotron routed experts (.experts. in key).",
    )
    args = parser.parse_args()

    mp = Path(args.model).expanduser().resolve()
    if not mp.is_dir():
        print(f"ERROR: not a directory: {mp}", file=sys.stderr)
        sys.exit(2)

    cfg = mp / "config.json"
    config_obj: dict | None = None
    if cfg.is_file():
        try:
            with open(cfg) as f:
                config_obj = json.load(f)
        except json.JSONDecodeError as e:
            print(f"ERROR: invalid JSON in config.json: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.require_config:
        print(f"ERROR: missing config.json under {mp}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"WARNING: config.json not found under {mp}", file=sys.stderr)

    try:
        from mlx_lm.expert_weight_loader import resolve_weight_map

        wm = resolve_weight_map(mp)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    shards = sorted(set(wm.values()))
    repacked_shards = [s for s in shards if s.startswith("repacked-")]
    non_repacked_safetensors = [
        s
        for s in shards
        if s.endswith(".safetensors") and not s.startswith("repacked-")
    ]

    expert_keys = _expert_like_keys(wm)

    errors: list[str] = []
    if args.expect_repack and not repacked_shards:
        errors.append(
            "expect-repack: no repacked- shards in index (run mlx_lm.repack_experts?)"
        )
    if args.expect_expert_keys and not expert_keys:
        errors.append(
            "expect-expert-keys: no .experts. keys found (wrong model or not Nemotron MoE?)"
        )

    # Partial repack detection: index lists repacked-X but file missing (resolve_weight_map already errors)
    # Mixed state: some expert keys point to repacked shards and some to old names — warn only
    repack_key_targets = {k for k, v in wm.items() if v.startswith("repacked-")}
    old_key_targets = {
        k for k, v in wm.items() if ".experts." in k and not v.startswith("repacked-")
    }
    mixed_repack = bool(repack_key_targets and old_key_targets)

    if config_obj is not None:
        arch = config_obj.get("model_type") or config_obj.get("architectures")
        if arch is None:
            print(
                "WARNING: config.json has no model_type or architectures",
                file=sys.stderr,
            )

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    summary = {
        "ok": True,
        "model_path": str(mp),
        "n_weight_keys": len(wm),
        "n_shard_files": len(shards),
        "shard_files": shards,
        "has_config_json": cfg.is_file(),
        "repacked_shard_count": len(repacked_shards),
        "non_repacked_shard_count": len(non_repacked_safetensors),
        "nemotron_style_expert_key_count": len(expert_keys),
        "mixed_repack_warning": mixed_repack,
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"OK: {len(wm)} weight keys across {len(shards)} shard file(s)")
        for s in shards:
            print(f"  {s}")
        if mixed_repack:
            print(
                "WARNING: some expert keys use repacked shards and some use non-repacked names.",
                file=sys.stderr,
            )

    sys.exit(0)


if __name__ == "__main__":
    main()
