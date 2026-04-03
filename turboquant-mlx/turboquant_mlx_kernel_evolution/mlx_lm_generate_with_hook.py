#!/usr/bin/env python3
"""
Subprocess helpers for TurboQuant Phase 6.

1) **eval** (default): run evaluator.py in a fresh process (fidelity + micro-throughput).
2) **generate**: run `python -m mlx_lm.generate` with TurboQuant cache and optional
   `TURBOQUANT_ASYMMETRIC_SCORE_MODULE` so the evolved `asymmetric_attention_scores`
   is used inside the real attention path (mlx-lm fork with env hook).

`make_prompt_cache(..., turboquant)` resolves `codebooks/` relative to the process
**current working directory** — use `--workdir` pointing at turboquant-mlx (or export
`TURBOQUANT_CODEBOOK_DIR` if you extend cache construction to honor it).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run_eval_subprocess(program: Path, repo_root: Path) -> int:
    evaluator = Path(__file__).resolve().parent / "evaluator.py"
    env = {
        **os.environ,
        "PYTHONPATH": f"{repo_root}:{os.environ.get('PYTHONPATH', '')}",
    }
    proc = subprocess.run(
        [sys.executable, str(evaluator), "--program", str(program), "--json-metrics"],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or "")
        return proc.returncode
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "{}"
    try:
        metrics = json.loads(line)
    except json.JSONDecodeError:
        print(proc.stdout)
        return 1
    print(json.dumps(metrics, indent=2))
    return 0


def run_generate_subprocess(
    program: Path | None,
    repo_root: Path,
    model: str,
    prompt: str,
    max_tokens: int,
    extra_args: list[str],
) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}"
    if program is not None:
        env["TURBOQUANT_ASYMMETRIC_SCORE_MODULE"] = str(program.resolve())

    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.generate",
        "--model",
        model,
        "--prompt",
        prompt,
        "--max-tokens",
        str(max_tokens),
        "--kv-cache-type",
        "turboquant",
    ]
    cmd.extend(extra_args)

    print("Running:", " ".join(cmd), file=sys.stderr)
    if program:
        print(f"TURBOQUANT_ASYMMETRIC_SCORE_MODULE={program}", file=sys.stderr)

    return subprocess.call(cmd, cwd=str(repo_root), env=env)


def main() -> int:
    root_default = Path(__file__).resolve().parent.parent
    # Back-compat: `... mlx_lm_generate_with_hook.py --program foo.py` => eval
    if len(sys.argv) > 1 and sys.argv[1] not in (
        "eval",
        "generate",
        "-h",
        "--help",
    ):
        sys.argv.insert(1, "eval")

    p = argparse.ArgumentParser(description="TurboQuant evolution subprocess hooks")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser(
        "eval", help="Run evaluator in a child process (default workflow)"
    )
    pe.add_argument("--program", type=Path, required=True)
    pe.add_argument("--repo-root", type=Path, default=root_default)

    pg = sub.add_parser(
        "generate",
        help="mlx_lm.generate with turboquant + optional evolved score module",
    )
    pg.add_argument(
        "--program",
        type=Path,
        default=None,
        help="Candidate .py defining asymmetric_attention_scores (omit for baseline)",
    )
    pg.add_argument("--repo-root", type=Path, default=root_default)
    pg.add_argument("--model", type=str, required=True)
    pg.add_argument("--prompt", type=str, default="Hello")
    pg.add_argument("--max-tokens", type=int, default=32)
    pg.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Additional args after -- for mlx_lm.generate",
    )

    args = p.parse_args()
    if args.cmd == "eval":
        return run_eval_subprocess(args.program, args.repo_root)
    if args.cmd == "generate":
        extra = args.extra
        if extra and extra[0] == "--":
            extra = extra[1:]
        return run_generate_subprocess(
            args.program,
            args.repo_root,
            args.model,
            args.prompt,
            args.max_tokens,
            extra,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
