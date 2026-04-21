#!/usr/bin/env python3
"""Run Mojo kernel benchmarks and augment their JSON output with provenance.

Captures machine + runtime metadata once per invocation, executes the requested
pixi tasks, then patches every JSON written to results/ during the run with the
same provenance block (so a single invocation is a self-consistent group).

Usage:
    python scripts/bench_with_provenance.py --kernel matmul
    python scripts/bench_with_provenance.py --kernel matmul softmax rope
    python scripts/bench_with_provenance.py --all
    python scripts/bench_with_provenance.py --kernel smoke
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import socket
import subprocess
import sys
import uuid
from pathlib import Path

KERNELS = (
    "matmul",
    "softmax",
    "rope",
    "isoquant",
    "kv-compress",
    "fused-attn",
    "smoke",
)

# Resolve project root assuming this file lives in <root>/scripts/.
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"
PIXI_LOCK = ROOT / "pixi.lock"


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """subprocess.run wrapper that fails loudly on non-zero exit."""
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kwargs)


def gather_provenance() -> dict:
    """Collect runtime + machine metadata. Raises on any failure."""
    if not PIXI_LOCK.is_file():
        raise FileNotFoundError(f"pixi.lock missing at {PIXI_LOCK}")

    mojo_proc = _run(["pixi", "run", "mojo", "--version"], cwd=ROOT)
    mojo_version = (
        (mojo_proc.stdout + mojo_proc.stderr).strip().splitlines()[-1].strip()
    )
    if not mojo_version:
        raise RuntimeError("Empty mojo --version output")

    chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).stdout.strip()
    macos_version = _run(["sw_vers", "-productVersion"]).stdout.strip()

    lock_sha = hashlib.sha256(PIXI_LOCK.read_bytes()).hexdigest()
    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_uuid = uuid.uuid4().hex

    prov = {
        "mojo_version": mojo_version,
        "hostname": socket.gethostname(),
        "chip": chip,
        "macos_version": macos_version,
        "pixi_lock_sha256": lock_sha,
        "run_timestamp_utc": timestamp,
        "run_uuid": run_uuid,
    }
    for k, v in prov.items():
        if not v:
            raise RuntimeError(f"Empty provenance field: {k}")
    return prov


def snapshot_results_mtimes() -> dict[Path, float]:
    """Snapshot mtime of every JSON in results/ before a kernel runs."""
    if not RESULTS_DIR.is_dir():
        return {}
    return {p: p.stat().st_mtime_ns for p in RESULTS_DIR.glob("*.json")}


def changed_jsons_since(snapshot: dict[Path, float]) -> list[Path]:
    """Return JSONs created or modified since the snapshot."""
    if not RESULTS_DIR.is_dir():
        return []
    changed: list[Path] = []
    for p in RESULTS_DIR.glob("*.json"):
        prev = snapshot.get(p)
        cur = p.stat().st_mtime_ns
        if prev is None or cur > prev:
            changed.append(p)
    return sorted(changed)


def run_kernel(name: str) -> subprocess.CompletedProcess:
    """Run a single pixi bench task; capture stdout+stderr to logs/."""
    task = f"bench-{name}"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{task}.log"
    print(f"[run] pixi run {task}  (log: {log_path.relative_to(ROOT)})", flush=True)
    try:
        proc = _run(["pixi", "run", task], cwd=ROOT)
    except subprocess.CalledProcessError as exc:
        log_path.write_text((exc.stdout or "") + (exc.stderr or ""))
        sys.stderr.write(f"[fail] {task} exited {exc.returncode}; see {log_path}\n")
        raise
    log_path.write_text((proc.stdout or "") + (proc.stderr or ""))
    return proc


def augment_json(path: Path, prov: dict) -> None:
    """Add/overwrite top-level 'provenance' on a JSON file. Atomic write."""
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level JSON is not an object")
    data["provenance"] = prov
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def _resolve_kernels(args: argparse.Namespace) -> list[str]:
    if args.all:
        return [k for k in KERNELS if k != "smoke"]
    requested = list(args.kernel or [])
    invalid = [k for k in requested if k not in KERNELS]
    if invalid:
        raise SystemExit(f"Unknown kernel(s): {invalid}. Valid: {list(KERNELS)}")
    if not requested:
        raise SystemExit("Must specify --kernel <name> [<name>...] or --all")
    return requested


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--kernel", nargs="+", choices=KERNELS, help="Kernel(s) to run")
    group.add_argument(
        "--all", action="store_true", help="Run all real kernels (excludes smoke)"
    )
    args = parser.parse_args()

    kernels = _resolve_kernels(args)

    print("[provenance] gathering...", flush=True)
    prov = gather_provenance()
    prov_digest = hashlib.sha256(
        json.dumps(prov, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    print(f"[provenance] run_uuid={prov['run_uuid']} digest={prov_digest}", flush=True)

    all_changed: list[Path] = []
    for name in kernels:
        snap = snapshot_results_mtimes()
        run_kernel(name)
        changed = changed_jsons_since(snap)
        if not changed:
            print(
                f"[warn] {name}: no JSONs modified in results/ (smoke kernels may not write)"
            )
        for path in changed:
            augment_json(path, prov)
            print(f"[augment] {path.relative_to(ROOT)}")
            all_changed.append(path)

    print()
    print("=== summary ===")
    print(f"kernels run     : {kernels}")
    print(f"jsons augmented : {len(all_changed)}")
    print(f"run_uuid        : {prov['run_uuid']}")
    print(f"prov_digest     : {prov_digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
