#!/usr/bin/env python3
"""Recover from a partial unstack: rename tmp shards to proper names,
rebuild model.safetensors.index.json, rewrite config.json's per-layer
quantization entries to per-expert form. Idempotent.

Codex review 2026-05-07 surfaced two issues with the previous version:

  HIGH #1   Recovery had no completeness check — a mixed state of
            already-renamed final shards plus leftover tmp shards (e.g.
            from a future crash *during* the rename loop) would be
            silently treated as "n = number of tmp shards", losing
            half the data without warning.

  MEDIUM #2 Recovery recomputed total_parameters by summing safetensor
            element counts, which inflates the count by treating
            packed-weight + scales + biases as three separate
            "parameters". The fixed unstack copies total_parameters
            from the source index. This script now does the same.

This version:
  - Detects mixed final/tmp state and refuses to proceed without
    explicit operator confirmation.
  - Validates contiguous coverage 1..N after rename.
  - Cross-checks the reconstructed weight_map against the source
    stacked-checkpoint key count (each .switch_mlp.<proj>.<kind> key
    expands to 384 .experts.<E>.<proj>.<kind> keys).
  - Carries source total_parameters when available; falls back to
    summed element counts only when the source is silent.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path("/Users/anthonylui/QwenCoderLocal")
sys.path.insert(0, str(REPO_ROOT / "mlx-lm"))

DST = Path("/Volumes/Samsung9904tb/Kimi-K2.6-2bit-experts-offload")
SRC = Path("/Volumes/Samsung9904tb/Kimi-K2.6-2bit-experts")  # stacked source

N_EXPERTS = 384

# Step 1: list tmp shards. Accept both legacy and current tmp filename
# patterns:
#   - legacy (pre-fix): model-NNNNN-of-XXXXX.safetensors.tmp.safetensors
#   - current:          model-NNNNN-of-XXXXX.tmp.safetensors
TMP_RE = re.compile(r"^model-(\d{5})-of-XXXXX\.(?:safetensors\.)?tmp\.safetensors$")
FINAL_RE = re.compile(r"^model-(\d{5})-of-(\d{5})\.safetensors$")


def _scan_dst() -> tuple[list[tuple[int, Path]], list[tuple[int, int, Path]]]:
    tmp_files: list[tuple[int, Path]] = []
    final_files: list[tuple[int, int, Path]] = []
    for p in sorted(DST.iterdir()):
        m = TMP_RE.match(p.name)
        if m:
            tmp_files.append((int(m.group(1)), p))
            continue
        m = FINAL_RE.match(p.name)
        if m:
            final_files.append((int(m.group(1)), int(m.group(2)), p))
    tmp_files.sort(key=lambda t: t[0])
    final_files.sort(key=lambda t: t[0])
    return tmp_files, final_files


def main() -> int:
    if not DST.is_dir():
        sys.exit(f"DST does not exist: {DST}")

    tmp_files, final_files = _scan_dst()
    n_tmp = len(tmp_files)
    n_final = len(final_files)
    print(f"[scan] {n_tmp} tmp shards, {n_final} final shards", flush=True)

    if n_tmp == 0 and n_final == 0:
        sys.exit("nothing to recover (no tmp or final shards in DST)")

    # Mixed state: a previous run partially renamed before crashing. We
    # cannot infer N reliably from the tmp count alone, so require either
    # all-final or all-tmp.
    if n_tmp > 0 and n_final > 0:
        finals_n = {n for _, n, _ in final_files}
        if len(finals_n) == 1:
            n = next(iter(finals_n))
            print(
                f"[scan] MIXED state. Final shards report N={n}; tmp shards "
                f"would report N={n_tmp}.",
                flush=True,
            )
            if n_tmp + n_final != n:
                sys.exit(
                    f"FAIL: mixed state with inconsistent count "
                    f"(tmp={n_tmp} + final={n_final} != advertised N={n}). "
                    f"Manual investigation required."
                )
            print(
                f"[scan] mixed state is consistent; will rename "
                f"{n_tmp} remaining tmp shards using N={n}.",
                flush=True,
            )
            mixed_n = n
        else:
            sys.exit(
                f"FAIL: final shards advertise inconsistent N values: "
                f"{sorted(finals_n)}. Manual investigation required."
            )
    else:
        mixed_n = None

    # Step 2: rename tmp -> model-NNNNN-of-NNNNN.safetensors
    if n_tmp > 0:
        n = mixed_n if mixed_n is not None else n_tmp
        print(f"[rename] -> model-NNNNN-of-{n:05d}.safetensors", flush=True)
        for idx, src_path in tmp_files:
            if idx < 1 or idx > n:
                sys.exit(
                    f"FAIL: tmp shard index {idx} is outside [1, {n}]; "
                    f"refusing to rename."
                )
            new_name = f"model-{idx:05d}-of-{n:05d}.safetensors"
            new_path = DST / new_name
            if new_path.exists():
                sys.exit(
                    f"FAIL: rename target {new_path} already exists; "
                    f"manual investigation required."
                )
            src_path.rename(new_path)
    else:
        # All-final state — recovery just rewrites index/config from
        # whatever is on disk. n is the advertised total.
        n = final_files[0][1]

    # After rename: enforce contiguous 1..n coverage. This is the check
    # the prior version was missing.
    expected_names = {f"model-{i:05d}-of-{n:05d}.safetensors" for i in range(1, n + 1)}
    actual_names = {p.name for p in DST.iterdir() if FINAL_RE.match(p.name)}
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing:
        sys.exit(
            f"FAIL: shard coverage gap after rename. Missing {len(missing)} "
            f"shard(s), e.g. {missing[:5]}. Recovery would produce an "
            f"incomplete checkpoint."
        )
    if extra:
        sys.exit(
            f"FAIL: unexpected final shards on disk: {extra[:5]}. "
            f"Manual investigation required."
        )

    # Step 3: build weight_map by reading each shard
    print("[index] scanning shards to rebuild weight_map", flush=True)
    from safetensors import safe_open

    _DTYPE_BYTES = {
        "F64": 8,
        "F32": 4,
        "F16": 2,
        "BF16": 2,
        "I64": 8,
        "I32": 4,
        "I16": 2,
        "I8": 1,
        "U64": 8,
        "U32": 4,
        "U16": 2,
        "U8": 1,
        "BOOL": 1,
        "F8_E5M2": 1,
        "F8_E4M3": 1,
    }
    weight_map: dict[str, str] = {}
    total_size = 0
    summed_elem_count = 0
    for idx in range(1, n + 1):
        shard_name = f"model-{idx:05d}-of-{n:05d}.safetensors"
        with safe_open(str(DST / shard_name), framework="numpy") as f:
            for key in f.keys():
                weight_map[key] = shard_name
                sl = f.get_slice(key)
                shape = sl.get_shape()
                dt = sl.get_dtype()
                nelem = 1
                for s in shape:
                    nelem *= s
                summed_elem_count += nelem
                total_size += nelem * _DTYPE_BYTES.get(dt, 0)
    print(f"  {len(weight_map)} keys, {total_size / 1024**3:.2f} GB", flush=True)

    # Cross-check key count against the stacked source. Each stacked
    # .switch_mlp.<proj>.<kind> key in SRC expands to N_EXPERTS
    # .experts.<E>.<proj>.<kind> keys here; non-stacked keys pass through 1:1.
    src_idx_path = SRC / "model.safetensors.index.json"
    if src_idx_path.is_file():
        with open(src_idx_path) as f:
            src_idx = json.load(f)
        src_keys = list(src_idx.get("weight_map", {}).keys())
        stacked_re = re.compile(
            r"^(?:language_model\.)?model\.layers\.\d+\.mlp"
            r"\.switch_mlp\.(?:gate_proj|up_proj|down_proj)\.(?:weight|scales|biases)$"
        )
        n_stacked = sum(1 for k in src_keys if stacked_re.match(k))
        n_passthrough = len(src_keys) - n_stacked
        expected_unstacked = n_passthrough + (n_stacked * N_EXPERTS)
        print(
            f"[verify] src has {len(src_keys)} keys "
            f"({n_stacked} stacked × {N_EXPERTS} = {n_stacked * N_EXPERTS} "
            f"+ {n_passthrough} passthrough = {expected_unstacked} expected)",
            flush=True,
        )
        if len(weight_map) != expected_unstacked:
            sys.exit(
                f"FAIL: reconstructed weight_map has {len(weight_map)} keys; "
                f"source expansion predicts {expected_unstacked}. Mismatch "
                f"of {expected_unstacked - len(weight_map)} keys. Aborting."
            )
        # Carry source total_parameters when present (MEDIUM #2 from
        # 2026-05-07 Codex review). Falls back to summed element count
        # only if source is silent — that fallback double-counts packed
        # weights + scales + biases, so it is strictly worse.
        src_total_params = src_idx.get("metadata", {}).get("total_parameters")
    else:
        print(
            f"[verify] WARNING: source index {src_idx_path} not found; "
            f"skipping cross-check and falling back to summed element count "
            f"for total_parameters (will overcount packed quantized weights).",
            flush=True,
        )
        src_total_params = None

    metadata = {"total_size": total_size}
    if isinstance(src_total_params, int) and src_total_params > 0:
        metadata["total_parameters"] = src_total_params
    else:
        metadata["total_parameters"] = summed_elem_count

    index_data = {
        "metadata": metadata,
        "weight_map": {k: weight_map[k] for k in sorted(weight_map)},
    }
    with open(DST / "model.safetensors.index.json", "w") as f:
        json.dump(index_data, f, indent=4)
    print(
        f"[index] wrote model.safetensors.index.json "
        f"(total_parameters={metadata['total_parameters']:,})",
        flush=True,
    )

    # Step 4: rewrite config.json's per-layer quantization entries.
    _STACKED_CFG_RE = re.compile(
        r"^(?P<prefix>(?:language_model\.)?model\.layers\.\d+\.mlp)"
        r"\.switch_mlp\.(?P<proj>gate_proj|up_proj|down_proj)$"
    )
    src_cfg_path = SRC / "config.json"
    if not src_cfg_path.is_file():
        sys.exit(f"FAIL: source config.json not found at {src_cfg_path}")
    with open(src_cfg_path) as f:
        src_cfg = json.load(f)

    def rewrite_quant(q: dict) -> dict:
        out: dict = {}
        for path, params in q.items():
            if not isinstance(params, dict):
                out[path] = params
                continue
            m = _STACKED_CFG_RE.match(path)
            if m is None:
                out[path] = params
                continue
            prefix = m.group("prefix")
            proj = m.group("proj")
            for i in range(N_EXPERTS):
                out[f"{prefix}.experts.{i}.{proj}"] = dict(params)
        return out

    for k in ("quantization", "quantization_config"):
        if k in src_cfg and isinstance(src_cfg[k], dict):
            src_cfg[k] = rewrite_quant(src_cfg[k])
            print(
                f"[config] rewrote {k}: {len(src_cfg[k])} entries after expansion",
                flush=True,
            )

    with open(DST / "config.json", "w") as f:
        json.dump(src_cfg, f, indent=2)
    print("[config] wrote config.json", flush=True)

    # Step 5: copy passthrough files
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
        sf = SRC / fname
        if sf.is_file() and not (DST / fname).exists():
            shutil.copy2(sf, DST / fname)
            print(f"[passthrough] {fname}", flush=True)

    print(f"\n[done] recovered: {DST}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
