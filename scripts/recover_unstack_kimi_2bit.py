#!/usr/bin/env python3
"""Recover from the unstack crash: rename tmp shards to proper names,
rebuild model.safetensors.index.json, rewrite config.json's per-layer
quantization entries to per-expert form. Idempotent."""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path("/Users/anthonylui/QwenCoderLocal")
sys.path.insert(0, str(REPO_ROOT / "mlx-lm"))

DST = Path("/Volumes/Samsung9904tb/Kimi-K2.6-2bit-experts-offload")
SRC = Path(
    "/Volumes/Samsung9904tb/Kimi-K2.6-2bit-experts"
)  # for config.json + tokenizer pass-through

# Step 1: list tmp shards in numeric order
tmp_re = re.compile(r"^model-(\d{5})-of-XXXXX\.safetensors\.tmp\.safetensors$")
tmp_files = []
for p in sorted(DST.iterdir()):
    m = tmp_re.match(p.name)
    if m:
        tmp_files.append((int(m.group(1)), p))
tmp_files.sort(key=lambda t: t[0])
n = len(tmp_files)
print(f"[recover] found {n} tmp shards", flush=True)
if n == 0:
    sys.exit("no tmp shards to recover")

# Step 2: rename tmp -> model-NNNNN-of-NNNNN.safetensors
print("[rename] -> model-NNNNN-of-{n:05d}.safetensors".format(n=n), flush=True)
rename_map = {}
for idx, src_path in tmp_files:
    new_name = f"model-{idx:05d}-of-{n:05d}.safetensors"
    new_path = DST / new_name
    src_path.rename(new_path)
    rename_map[src_path.name] = new_name

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
weight_map = {}
total_size = 0
total_params = 0
for idx in range(1, n + 1):
    new_name = f"model-{idx:05d}-of-{n:05d}.safetensors"
    p = DST / new_name
    with safe_open(str(p), framework="numpy") as f:
        for key in f.keys():
            weight_map[key] = new_name
            sl = f.get_slice(key)
            shape = sl.get_shape()
            dt = sl.get_dtype()
            nelem = 1
            for s in shape:
                nelem *= s
            total_params += nelem
            total_size += nelem * _DTYPE_BYTES.get(dt, 0)

print(f"  {len(weight_map)} keys, {total_size / 1024**3:.2f} GB", flush=True)

index_data = {
    "metadata": {
        "total_size": total_size,
        "total_parameters": total_params,
    },
    "weight_map": {k: weight_map[k] for k in sorted(weight_map)},
}
with open(DST / "model.safetensors.index.json", "w") as f:
    json.dump(index_data, f, indent=4)
print("[index] wrote model.safetensors.index.json", flush=True)

# Step 4: rewrite config.json's per-layer quantization entries
_STACKED_CFG_RE = re.compile(
    r"^(?P<prefix>(?:language_model\.)?model\.layers\.\d+\.mlp)"
    r"\.switch_mlp\.(?P<proj>gate_proj|up_proj|down_proj)$"
)
N_EXPERTS = 384
src_cfg = json.load(open(SRC / "config.json"))


def rewrite_quant(q):
    out = {}
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
    sf = SRC / fname
    if sf.is_file() and not (DST / fname).exists():
        shutil.copy2(sf, DST / fname)
        print(f"[passthrough] {fname}", flush=True)

print("\n[done] recovered:", DST, flush=True)
