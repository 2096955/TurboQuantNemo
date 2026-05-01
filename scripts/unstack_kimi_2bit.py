#!/usr/bin/env python3
"""Unstack Kimi K2.6 2-bit checkpoint from SwitchLinear-stacked layout to
per-expert layout that the ExpertOffloadManager can load on demand.

Input:  /Volumes/Samsung9904tb/Kimi-K2.6-2bit-experts (stacked)
Output: /Volumes/Samsung9904tb/Kimi-K2.6-2bit-experts-offload (per-expert)

Stacked key:
    language_model.model.layers.<L>.mlp.switch_mlp.<proj>.<kind>
        shape (384, *, *) for weight/scales

Per-expert key (what _KIMI_EXPERT_KEY_RE expects):
    language_model.model.layers.<L>.mlp.experts.<E>.<proj>.<kind>
        shape (*, *) per expert, E in [0, 384)

Non-routed-expert tensors pass through unchanged.

config.json quantization_config entries are also rewritten: each
"...switch_mlp.<proj>" entry is replaced by 384 "...experts.<E>.<proj>"
entries with the same bits/group_size.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MLX_LM_ROOT = REPO_ROOT / "mlx-lm"
if MLX_LM_ROOT.exists() and str(MLX_LM_ROOT) not in sys.path:
    sys.path.insert(0, str(MLX_LM_ROOT))


_STACKED_RE = re.compile(
    r"^(?P<prefix>(?:language_model\.)?model\.layers\.\d+\.mlp)"
    r"\.switch_mlp\.(?P<proj>gate_proj|up_proj|down_proj)"
    r"\.(?P<kind>weight|scales|biases)$"
)
_STACKED_CONFIG_RE = re.compile(
    r"^(?P<prefix>(?:language_model\.)?model\.layers\.\d+\.mlp)"
    r"\.switch_mlp\.(?P<proj>gate_proj|up_proj|down_proj)$"
)


def unstack_tensor(name, tensor):
    """If name is a stacked routed-expert tensor, return [(new_name, slice), ...].
    Else return [(name, tensor)]."""
    m = _STACKED_RE.match(name)
    if m is None:
        return [(name, tensor)]
    prefix = m.group("prefix")
    proj = m.group("proj")
    kind = m.group("kind")
    n_experts = tensor.shape[0]
    out = []
    for i in range(n_experts):
        new_name = f"{prefix}.experts.{i}.{proj}.{kind}"
        out.append((new_name, tensor[i]))
    return out


def rewrite_quant_config(quant_config: dict, n_experts: int) -> dict:
    """Replace each switch_mlp.<proj> entry with n_experts experts.<E>.<proj> entries."""
    new_q = {}
    for path, params in quant_config.items():
        if not isinstance(params, dict):
            new_q[path] = params
            continue
        m = _STACKED_CONFIG_RE.match(path)
        if m is None:
            new_q[path] = params
            continue
        prefix = m.group("prefix")
        proj = m.group("proj")
        for i in range(n_experts):
            new_q[f"{prefix}.experts.{i}.{proj}"] = dict(params)
    return new_q


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="stacked 2-bit checkpoint")
    p.add_argument("--dst", required=True, help="per-expert output dir")
    p.add_argument(
        "--shard-bytes",
        type=int,
        default=5 * 1024**3,
        help="Target bytes per output shard (default 5 GiB)",
    )
    args = p.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    src = Path(args.src)
    dst = Path(args.dst)
    if not src.is_dir():
        raise SystemExit(f"src not found: {src}")
    if dst.exists():
        raise SystemExit(f"dst exists: {dst} (rm -rf if you want to re-run)")
    dst.mkdir(parents=True)

    import mlx.core as mx

    # Read source index
    idx_path = src / "model.safetensors.index.json"
    if not idx_path.is_file():
        raise SystemExit(f"missing source index: {idx_path}")
    src_idx = json.load(open(idx_path))
    src_weight_map = src_idx["weight_map"]

    # Group keys by source shard
    shards_in_src = sorted(set(src_weight_map.values()))
    print(
        f"[plan] {len(src_weight_map)} src tensors across {len(shards_in_src)} shards",
        flush=True,
    )

    # Streaming write to .tmp shards; renumber at end when total count is known
    out_buffer: dict = {}
    out_buffer_bytes = 0
    tmp_shards: list[Path] = []  # ordered list of written tmp paths
    out_weight_map: dict[str, str] = {}  # name -> tmp filename (renamed later)
    n_experts_seen = 0
    t0 = time.time()
    last_print = 0.0

    def flush_buffer():
        nonlocal out_buffer, out_buffer_bytes
        if not out_buffer:
            return
        idx = len(tmp_shards) + 1
        tmp_name = f"model-{idx:05d}-of-XXXXX.safetensors.tmp"
        tmp_path = dst / tmp_name
        mx.save_safetensors(str(tmp_path), out_buffer, metadata={"format": "mlx"})
        for k in out_buffer:
            out_weight_map[k] = tmp_name
        tmp_shards.append(tmp_path)
        out_buffer.clear()
        out_buffer_bytes = 0
        gc.collect()
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

    for src_shard_idx, shard_name in enumerate(shards_in_src):
        shard_path = src / shard_name
        # mx.load returns dict[str, mx.array] (lazy mmap'd)
        loaded = mx.load(str(shard_path))
        for key in list(loaded.keys()):
            tensor = loaded[key]
            for new_name, slice_arr in unstack_tensor(key, tensor):
                # Materialize the slice (single-tensor eval, bounded buffer)
                mx.eval(slice_arr)
                mx.synchronize()
                out_buffer[new_name] = slice_arr
                out_buffer_bytes += slice_arr.nbytes
                if new_name.endswith(".gate_proj.weight") and ".experts.0." in new_name:
                    # Capture n_experts from the first stacked tensor we expand
                    if n_experts_seen == 0:
                        n_experts_seen = tensor.shape[0]
                if out_buffer_bytes >= args.shard_bytes:
                    flush_buffer()
            # Drop the source tensor
            del loaded[key]
            del tensor
        del loaded
        gc.collect()
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

        now = time.time()
        if now - last_print > 5.0 or src_shard_idx == len(shards_in_src) - 1:
            elapsed = now - t0
            done = src_shard_idx + 1
            rate = done / max(elapsed, 0.001)
            eta = (len(shards_in_src) - done) / rate
            buf_gb = out_buffer_bytes / 1024**3
            print(
                f"  src shard {done}/{len(shards_in_src)} processed "
                f"({elapsed:.0f}s, {rate:.2f}/s, ETA {eta:.0f}s, "
                f"out_buffer={buf_gb:.1f} GB, written_shards={len(tmp_shards)})",
                flush=True,
            )
            last_print = now

    flush_buffer()
    n_out_shards = len(tmp_shards)
    print(
        f"[unstack] done in {time.time() - t0:.1f}s. "
        f"{n_out_shards} output shards. n_experts={n_experts_seen}",
        flush=True,
    )

    # Rename tmp shards with correct N-of-M and update weight_map
    print("[rename] correcting shard numbering", flush=True)
    rename_map = {}
    for i, tmp_path in enumerate(tmp_shards):
        new_name = f"model-{i + 1:05d}-of-{n_out_shards:05d}.safetensors"
        new_path = dst / new_name
        tmp_path.rename(new_path)
        old_tmp_name = tmp_path.name
        rename_map[old_tmp_name] = new_name
    final_weight_map = {k: rename_map[old] for k, old in out_weight_map.items()}

    # Write the index
    print(
        f"[index] writing model.safetensors.index.json ({len(final_weight_map)} keys)",
        flush=True,
    )
    index_data = {
        "metadata": {
            "total_size": sum(
                # we no longer have the tensors but src_idx had total_size
                src_idx.get("metadata", {}).get("total_size", 0),
            )
            if isinstance(src_idx.get("metadata", {}).get("total_size"), int)
            else 0,
        },
        "weight_map": {k: final_weight_map[k] for k in sorted(final_weight_map)},
    }
    if "metadata" in src_idx and "total_parameters" in src_idx["metadata"]:
        index_data["metadata"]["total_parameters"] = src_idx["metadata"][
            "total_parameters"
        ]
    with open(dst / "model.safetensors.index.json", "w") as f:
        json.dump(index_data, f, indent=4)

    # Update config.json: rewrite per-layer quantization entries
    src_cfg_path = src / "config.json"
    dst_cfg_path = dst / "config.json"
    if src_cfg_path.is_file():
        cfg = json.load(open(src_cfg_path))
        for q_key in ("quantization", "quantization_config"):
            if q_key in cfg and isinstance(cfg[q_key], dict):
                cfg[q_key] = rewrite_quant_config(cfg[q_key], n_experts_seen)
        with open(dst_cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[config] wrote {dst_cfg_path}", flush=True)

    # Copy passthrough files (tokenizer, chat template, etc.)
    import shutil

    PASSTHROUGH = (
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "tiktoken.model",
        "special_tokens_map.json",
        "chat_template.jinja",
        "generation_config.json",
        "configuration_deepseek.py",
        "configuration_kimi_k25.py",
        "modeling_deepseek.py",
        "modeling_kimi_k25.py",
        "kimi_k25_processor.py",
        "kimi_k25_vision_processing.py",
        "media_utils.py",
    )
    for fname in PASSTHROUGH:
        sf = src / fname
        if sf.is_file():
            shutil.copy2(sf, dst / fname)

    print(f"\nDONE: {dst}", flush=True)


if __name__ == "__main__":
    main()
