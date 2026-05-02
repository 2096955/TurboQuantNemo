#!/usr/bin/env python3
"""Record NPT16 parity gate status for Kimi MLA (D=512) profiling lane.

Writes JSON suitable for archiving as:
  artifacts/kimi_k26_profiling/npt16_real_weight_logit_parity.json

Stages:
  1. Synthetic: run `mlx-lm/tests/test_fused_npt16.py` (Metal / reference agreement).
  2. Real-weight MLA: not automated here — requires comparing `KimiMLAIsoQuantCache`
     fused latent attention against an FP16 latent reference on checkpoint-backed
     packed state. Extend this script when that hook exists.

Exit codes:
  0  --synthetic-only and synthetic pytest passed.
  1  --synthetic-only and synthetic pytest failed.
  2  Default mode: real-weight MLA stage is not yet implemented, so the gate is
     incomplete regardless of the synthetic result. Also returned on pytest
     timeout, missing pytest file, or other system-level failures.

Example (synthetic-only smoke):
  PYTHONPATH=mlx-lm python scripts/run_kimi_npt16_parity_gate.py --synthetic-only \\
    --output artifacts/kimi_k26_profiling/npt16_real_weight_logit_parity.json

Assumptions:
  Synthetic stage exercises ``test_fused_npt16.py`` only; ``real_weight_mla`` stays
  ``executed: false`` until a checkpoint-backed fused-vs-FP16 latent comparison exists.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        default=str(
            REPO_ROOT
            / "artifacts/kimi_k26_profiling/npt16_real_weight_logit_parity.json"
        ),
    )
    p.add_argument(
        "--pytest-path",
        default=str(REPO_ROOT / "mlx-lm/tests/test_fused_npt16.py"),
        help="Synthetic parity test file.",
    )
    p.add_argument(
        "--synthetic-only",
        action="store_true",
        help=(
            "Acknowledge that the real-weight MLA stage is unimplemented and gate "
            "exit only on the synthetic pytest. Without this flag the script exits 2 "
            "even when synthetic passes, since the named gate cannot be certified."
        ),
    )
    p.add_argument(
        "--pytest-timeout",
        type=int,
        default=600,
        help="Seconds before the synthetic pytest is killed (default: 600).",
    )
    args = p.parse_args()

    pytest_file = Path(args.pytest_path)
    if not pytest_file.is_file():
        print(f"ERROR: missing {pytest_file}", flush=True)
        sys.exit(2)

    env = dict(**os.environ)
    mlx = REPO_ROOT / "mlx-lm"
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(mlx) if not pp else f"{mlx}{os.pathsep}{pp}"

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(pytest_file),
        "-q",
        "--tb=no",
    ]

    timed_out = False
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=args.pytest_timeout,
        )
        synth_ok = proc.returncode == 0
        out_t = proc.stdout or ""
        err_t = proc.stderr or ""
        exit_code = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        synth_ok = False
        out_t = exc.stdout.decode("utf-8", "replace") if exc.stdout else ""
        err_t = exc.stderr.decode("utf-8", "replace") if exc.stderr else ""
        exit_code = None
        print(
            f"ERROR: pytest timed out after {args.pytest_timeout}s",
            flush=True,
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "synthetic_only" if args.synthetic_only else "full"
    if timed_out:
        gate_status = "TIMEOUT"
    elif args.synthetic_only:
        gate_status = "SYNTHETIC_PASS" if synth_ok else "SYNTHETIC_FAIL"
    else:
        # Real-weight MLA stage is unimplemented; the named gate cannot pass.
        gate_status = "INCOMPLETE_REAL_WEIGHT_PENDING"

    payload = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": mode,
        "gate_status": gate_status,
        "pytest_command": cmd,
        "pytest_timeout_s": args.pytest_timeout,
        "pytest_timed_out": timed_out,
        "synthetic_pytest_pass": synth_ok,
        "pytest_exit_code": exit_code,
        # Pytest emits failures on stdout (-q keeps it short); keep both tails for triage.
        "pytest_stdout_tail": out_t[-4000:],
        "pytest_stderr_tail": err_t[-4000:],
        "real_weight_mla": {
            "executed": False,
            "reason": (
                "Pending: compare fused_latent_attention (ISOQUANT_USE_NPT16_FUSED=1) "
                "vs FP16 latent reference using checkpoint-derived packed cache."
            ),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(
        f"Saved: {out_path}  mode={mode}  gate_status={gate_status}  "
        f"synthetic_pytest_pass={synth_ok}",
        flush=True,
    )
    if timed_out:
        sys.exit(2)
    if args.synthetic_only:
        sys.exit(0 if synth_ok else 1)
    sys.exit(2)


if __name__ == "__main__":
    main()
