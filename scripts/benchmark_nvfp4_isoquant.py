#!/usr/bin/env python3
"""2x2 benchmark: NVFP4 weights x IsoQuant KV cache.

Tests whether weight-quant (NVFP4) and KV-quant (IsoQuant) gains are additive,
which is the open question raised by Ollama 0.19's NVFP4 release. Both pipelines
already exist in this fork independently; this harness is the first measurement
of them together.

Matrix (4 runs, model loaded fresh each time):

                 default KV     IsoQuant KV
  baseline (Q4)  baseline       IsoQuant-only gain
  NVFP4 weights  NVFP4-only     combined (the question)

Reuses run_benchmark() from benchmark_moe_offload.py so the timing/memory
methodology matches existing comparison artifacts.

Example:
  python scripts/benchmark_nvfp4_isoquant.py \\
    --baseline-model ~/Models/Qwen3.6-35B-A3B-4bit \\
    --nvfp4-model ~/Models/Qwen3.6-35B-A3B-nvfp4 \\
    --output artifacts/nvfp4_isoquant/qwen3_5_moe_2x2.json

If --nvfp4-model does not exist, the harness prints the exact mlx_lm.convert
command needed to produce it and runs only the rows it has weights for.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Reuse the same run_benchmark used by compare_pathway_kv_modes.py.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from benchmark_moe_offload import run_benchmark  # noqa: E402


def _model_exists(path: str | None) -> bool:
    if not path:
        return False
    p = Path(path).expanduser()
    return p.is_dir() and (p / "config.json").exists()


def _read_quant_mode(model_path: str) -> str | None:
    """Best-effort read of the quantization mode from config.json."""
    cfg_path = Path(model_path).expanduser() / "config.json"
    if not cfg_path.exists():
        return None
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    quant = cfg.get("quantization") or {}
    mode = quant.get("mode") if isinstance(quant, dict) else None
    bits = quant.get("bits") if isinstance(quant, dict) else None
    if mode and bits:
        return f"{mode}-{bits}bit"
    return mode or None


# Model types whose stacked SwitchGLU tensors must be repacked before
# expert_offload can resolve per-expert weight keys. Mirrors the runtime
# check in mlx_lm/utils.py:345.
_NEEDS_REPACK_FOR_OFFLOAD = {"gemma4_text", "gemma4", "qwen3_moe", "qwen3_5_moe"}


def _read_model_type(model_path: str) -> str | None:
    cfg_path = Path(model_path).expanduser() / "config.json"
    if not cfg_path.exists():
        return None
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return cfg.get("model_type")


def _checkpoint_has_repacked_experts(model_path: str) -> bool:
    """Heuristic: repacked checkpoints expose `*.experts.<idx>.*` weight keys.

    Cheap to evaluate from the safetensors index; avoids importing mlx_lm just
    for a preflight."""
    idx_path = Path(model_path).expanduser() / "model.safetensors.index.json"
    if not idx_path.exists():
        return False
    try:
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    keys = (idx.get("weight_map") or {}).keys()
    for k in keys:
        if ".experts." in k and (
            ".gate_proj" in k or ".up_proj" in k or ".down_proj" in k
        ):
            return True
    return False


def _preflight_expert_offload(models: dict[str, str]) -> list[str]:
    """Returns repack commands needed before --expert-offload can run, or []."""
    need: list[str] = []
    for label, path in models.items():
        if not _model_exists(path):
            continue
        mt = _read_model_type(path) or ""
        if mt in _NEEDS_REPACK_FOR_OFFLOAD and not _checkpoint_has_repacked_experts(
            path
        ):
            need.append(f"python -m mlx_lm.repack_experts --model {path}")
    return need


def _print_convert_hint(bf16_source: str | None, target_path: str) -> None:
    src = bf16_source or "<path/to/bf16-source>"
    print(
        "\n[hint] To produce the NVFP4 model run:",
        f"  mlx_lm.convert --hf-path {src} \\",
        f"    --mlx-path {target_path} \\",
        "    --quantize --q-mode nvfp4",
        "(The bf16 source must be a Hugging Face directory or repo id.)",
        sep="\n",
    )


def _summary_table(rows: list[dict[str, Any]]) -> str:
    cols = (
        "label",
        "kv",
        "prefill_ttft_s",
        "decode_tps",
        "peak_mb",
    )
    header = "  ".join(f"{c:<22}" for c in cols)
    lines = [header, "-" * len(header)]
    for row in rows:
        cells = [
            f"{row.get('label', '?'):<22}",
            f"{row.get('kv', '?'):<22}",
            f"{(row.get('prefill_ttft_s') or float('nan')):<22.3f}",
            f"{(row.get('decode_tps') or float('nan')):<22.2f}",
            f"{(row.get('peak_mb') or float('nan')):<22.0f}",
        ]
        lines.append("  ".join(cells))
    return "\n".join(lines)


def _delta_pct(new: float | None, base: float | None) -> float | None:
    if new is None or base is None or base == 0:
        return None
    return round((new - base) / base * 100.0, 2)


def main() -> None:
    p = argparse.ArgumentParser(
        description="2x2 NVFP4 weights x IsoQuant KV combination benchmark"
    )
    p.add_argument(
        "--baseline-model",
        type=str,
        required=True,
        help="Path to existing affine Q4 (or other baseline) MLX model directory.",
    )
    p.add_argument(
        "--nvfp4-model",
        type=str,
        required=True,
        help="Path to NVFP4-quantized MLX model directory. "
        "Build with: mlx_lm.convert --quantize --q-mode nvfp4 ...",
    )
    p.add_argument(
        "--bf16-source",
        type=str,
        default=None,
        help="Optional bf16 source path/repo (used only to print convert hint "
        "if --nvfp4-model is missing).",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="JSON output file (parent dirs created if missing).",
    )
    p.add_argument(
        "--profile",
        type=str,
        choices=["A", "B"],
        default="A",
        help="A: 512 prefill / 128 decode; B: 1024 prefill / 256 decode. "
        "Ignored if --prefill-tokens / --decode-tokens are supplied.",
    )
    p.add_argument(
        "--prefill-tokens",
        type=int,
        default=None,
        help="Override profile prefill token count (e.g. 4096, 8192). "
        "Requires --decode-tokens.",
    )
    p.add_argument(
        "--decode-tokens",
        type=int,
        default=None,
        help="Override profile decode token count (e.g. 512, 1024). "
        "Requires --prefill-tokens.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--isoquant-bits",
        type=int,
        default=3,
        help="ISOQUANT_BITS for the isoquant rows (default 3).",
    )
    p.add_argument(
        "--prefill-step-size",
        type=int,
        default=64,
    )
    p.add_argument(
        "--expert-offload",
        action="store_true",
        help="Enable expert offload. NOTE: gemma4/gemma4_text/qwen3_moe/qwen3_5_moe "
        "checkpoints must be pre-processed with `python -m mlx_lm.repack_experts` "
        "first; the harness preflights this and aborts with the exact command if "
        "any model still has stacked SwitchGLU tensors.",
    )
    p.add_argument(
        "--max-resident-experts",
        type=int,
        default=None,
    )
    p.add_argument(
        "--skip",
        type=str,
        default="",
        help="Comma-separated cells to skip, e.g. 'baseline_default' to skip the baseline default-KV row.",
    )
    args = p.parse_args()

    os.environ["ISOQUANT_BITS"] = str(args.isoquant_bits)

    baseline_exists = _model_exists(args.baseline_model)
    nvfp4_exists = _model_exists(args.nvfp4_model)

    if not baseline_exists:
        print(
            f"[error] baseline model not found at {args.baseline_model}",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.expert_offload:
        repack_cmds = _preflight_expert_offload(
            {"baseline": args.baseline_model, "nvfp4": args.nvfp4_model}
        )
        if repack_cmds:
            print(
                "[error] --expert-offload requires repacked per-expert weights for "
                "this model_type. Run the following before retrying:",
                file=sys.stderr,
            )
            for cmd in repack_cmds:
                print(f"  {cmd}", file=sys.stderr)
            sys.exit(2)

    if not nvfp4_exists:
        print(
            f"[warn] NVFP4 model not found at {args.nvfp4_model} — skipping NVFP4 rows",
            file=sys.stderr,
        )
        _print_convert_hint(args.bf16_source, args.nvfp4_model)

    if (args.prefill_tokens is None) != (args.decode_tokens is None):
        print(
            "[error] --prefill-tokens and --decode-tokens must be provided together",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.prefill_tokens is not None:
        prefill_tokens, decode_tokens = args.prefill_tokens, args.decode_tokens
        profile_label = f"custom-{prefill_tokens}p{decode_tokens}d"
    else:
        prefill_tokens, decode_tokens = (
            (512, 128) if args.profile == "A" else (1024, 256)
        )
        profile_label = args.profile

    skip_cells = {s.strip() for s in args.skip.split(",") if s.strip()}

    matrix: list[tuple[str, str, str, bool]] = [
        # (cell_id, label, model_path, kv_cache_type, model_exists)
        ("baseline_default", "baseline+defaultKV", args.baseline_model, "default"),
        ("baseline_isoquant", "baseline+isoquant", args.baseline_model, "isoquant"),
        ("nvfp4_default", "nvfp4+defaultKV", args.nvfp4_model, "default"),
        ("nvfp4_isoquant", "nvfp4+isoquant", args.nvfp4_model, "isoquant"),
    ]

    runs: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []

    for cell_id, label, model_path, kv in matrix:
        if cell_id in skip_cells:
            print(f"\n[skip] {label} ({cell_id})")
            runs[cell_id] = {"status": "skipped", "reason": "user --skip"}
            continue
        if not _model_exists(model_path):
            print(f"\n[skip] {label}: model missing at {model_path}")
            runs[cell_id] = {"status": "skipped", "reason": "model missing"}
            continue

        print(f"\n======== {label} ({cell_id}) ========")
        print(f"  model: {model_path}")
        print(f"  kv:    {kv}")
        try:
            metrics, gate_failed = run_benchmark(
                model_path,
                profile=profile_label,
                prefill_tokens=prefill_tokens,
                prefill_step_size=args.prefill_step_size,
                decode_tokens=decode_tokens,
                seed=args.seed,
                kv_cache_type=kv,
                expert_offload=args.expert_offload,
                max_resident_experts=args.max_resident_experts,
                expert_offload_dir=None,
                memory_mode=None,
                strict_gates=False,
                split_decode_timing=False,
                warm_second_pass=False,
                use_predictor=False,
                use_dedekimi_observer=False,
                task_expert_cliques=None,
                target_envelope_mb=None,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {label} failed: {type(exc).__name__}: {exc}")
            runs[cell_id] = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            continue

        metrics["status"] = "ok"
        metrics["label"] = label
        metrics["model_path"] = model_path
        metrics["model_quant"] = _read_quant_mode(model_path)
        if gate_failed:
            metrics["_soft_gate_failed"] = True
        runs[cell_id] = metrics

        summary_rows.append(
            {
                "label": label,
                "kv": kv,
                "prefill_ttft_s": metrics.get("prefill_plus_first_token_s"),
                "decode_tps": metrics.get("decode_tok_per_s"),
                "peak_mb": metrics.get("peak_memory_mb"),
            }
        )

    # Compute pairwise deltas (only when both halves exist).
    def _ok(cell_id: str) -> dict[str, Any] | None:
        r = runs.get(cell_id)
        return r if r and r.get("status") == "ok" else None

    base = _ok("baseline_default")
    iso_only = _ok("baseline_isoquant")
    nvfp4_only = _ok("nvfp4_default")
    combined = _ok("nvfp4_isoquant")

    deltas: dict[str, Any] = {}
    if base is not None:
        if iso_only is not None:
            deltas["isoquant_only_vs_baseline"] = {
                "decode_tps_delta_pct": _delta_pct(
                    iso_only.get("decode_tok_per_s"), base.get("decode_tok_per_s")
                ),
                "prefill_ttft_delta_pct": _delta_pct(
                    iso_only.get("prefill_plus_first_token_s"),
                    base.get("prefill_plus_first_token_s"),
                ),
                "peak_mem_delta_pct": _delta_pct(
                    iso_only.get("peak_memory_mb"), base.get("peak_memory_mb")
                ),
            }
        if nvfp4_only is not None:
            deltas["nvfp4_only_vs_baseline"] = {
                "decode_tps_delta_pct": _delta_pct(
                    nvfp4_only.get("decode_tok_per_s"), base.get("decode_tok_per_s")
                ),
                "prefill_ttft_delta_pct": _delta_pct(
                    nvfp4_only.get("prefill_plus_first_token_s"),
                    base.get("prefill_plus_first_token_s"),
                ),
                "peak_mem_delta_pct": _delta_pct(
                    nvfp4_only.get("peak_memory_mb"), base.get("peak_memory_mb")
                ),
            }
        if combined is not None:
            deltas["combined_vs_baseline"] = {
                "decode_tps_delta_pct": _delta_pct(
                    combined.get("decode_tok_per_s"), base.get("decode_tok_per_s")
                ),
                "prefill_ttft_delta_pct": _delta_pct(
                    combined.get("prefill_plus_first_token_s"),
                    base.get("prefill_plus_first_token_s"),
                ),
                "peak_mem_delta_pct": _delta_pct(
                    combined.get("peak_memory_mb"), base.get("peak_memory_mb")
                ),
            }

    # Independence check: compare actual combined throughput to the
    # multiplicative-composition prediction. Throughput factors compose
    # multiplicatively when two changes act on independent time components
    # (e.g. weight FLOPs vs KV bandwidth), so multiplicative is the correct
    # null model — not addition of percentage deltas.
    if (
        base is not None
        and iso_only is not None
        and nvfp4_only is not None
        and combined is not None
    ):
        base_tps = base.get("decode_tok_per_s")
        iso_tps = iso_only.get("decode_tok_per_s")
        nvfp4_tps = nvfp4_only.get("decode_tok_per_s")
        combined_tps = combined.get("decode_tok_per_s")
        if (
            base_tps
            and iso_tps is not None
            and nvfp4_tps is not None
            and combined_tps is not None
        ):
            iso_factor = iso_tps / base_tps
            nvfp4_factor = nvfp4_tps / base_tps
            combined_factor = combined_tps / base_tps
            predicted_factor = iso_factor * nvfp4_factor
            predicted_tps = predicted_factor * base_tps
            ratio = combined_factor / predicted_factor if predicted_factor else None
            deltas["independence_check"] = {
                "iso_factor_vs_baseline": round(iso_factor, 4),
                "nvfp4_factor_vs_baseline": round(nvfp4_factor, 4),
                "combined_factor_vs_baseline": round(combined_factor, 4),
                "predicted_combined_factor_multiplicative": round(predicted_factor, 4),
                "predicted_combined_tok_per_s": round(predicted_tps, 2),
                "actual_combined_tok_per_s": round(combined_tps, 2),
                "actual_over_predicted_ratio": round(ratio, 4) if ratio else None,
                "interpretation": (
                    "actual exceeds independent-effects prediction "
                    "(constructive interaction — overheads overlap or share a bottleneck)"
                    if ratio is not None and ratio > 1.05
                    else "actual underperforms independent-effects prediction "
                    "(destructive interaction — extra overhead when combined)"
                    if ratio is not None and ratio < 0.95
                    else "approximately matches independent-effects prediction"
                ),
                "note": (
                    "Two changes are independent if their effects compose multiplicatively. "
                    "Adding percentage deltas overstates the combined effect for two "
                    "throughput-reducing changes and is mathematically incorrect."
                ),
            }

    out: dict[str, Any] = {
        "harness": "benchmark_nvfp4_isoquant",
        "profile": profile_label,
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
        "seed": args.seed,
        "isoquant_bits": args.isoquant_bits,
        "expert_offload": args.expert_offload,
        "models": {
            "baseline": {
                "path": args.baseline_model,
                "quant": _read_quant_mode(args.baseline_model),
            },
            "nvfp4": {
                "path": args.nvfp4_model,
                "quant": _read_quant_mode(args.nvfp4_model)
                if nvfp4_exists
                else "missing",
            },
        },
        "runs": runs,
        "deltas": deltas,
    }

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== 2x2 Matrix Summary ===")
    if summary_rows:
        print(_summary_table(summary_rows))
    else:
        print("(no successful runs)")

    if deltas:
        print("\n=== Deltas vs baseline (decode tok/s) ===")
        for k, v in deltas.items():
            if k == "independence_check":
                continue
            d = v.get("decode_tps_delta_pct") if isinstance(v, dict) else None
            print(f"  {k}: {d:+.2f}%" if d is not None else f"  {k}: n/a")
        if "independence_check" in deltas:
            ic = deltas["independence_check"]
            print(
                f"\n  Independence: actual {ic['actual_combined_tok_per_s']:.2f} tok/s "
                f"vs multiplicative-prediction {ic['predicted_combined_tok_per_s']:.2f} tok/s "
                f"(ratio {ic['actual_over_predicted_ratio']})"
            )
            print(f"  -> {ic['interpretation']}")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
