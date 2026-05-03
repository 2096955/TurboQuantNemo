"""Summarize IsoQuant submission bundle artifacts into paper-ready tables.

Reads JSON artifacts from results/isoquant_submission/ (or a custom dir)
and produces a markdown comparison table collapsing per-model, per-KV
results into one view.

Usage:
  python scripts/summarize_submission_results.py
  python scripts/summarize_submission_results.py --input-dir results/isoquant_submission
  python scripts/summarize_submission_results.py --output results/submission_summary.md
"""

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fmt(val: Any, precision: int = 2) -> str:
    if val is None:
        return "—"
    if isinstance(val, bool):
        return "PASS" if val else "FAIL"
    if isinstance(val, float):
        return f"{val:.{precision}f}"
    return str(val)


def extract_quality(data: dict) -> dict[str, Any]:
    """Extract quality gate summary from eval_quality_gate output."""
    results = data.get("results", data.get("prompts", []))
    if isinstance(results, list):
        n_pass = sum(1 for r in results if r.get("passed", False))
        n_total = len(results)
    else:
        n_pass = 0
        n_total = 0
    return {"score": f"{n_pass}/{n_total}", "n_pass": n_pass, "n_total": n_total}


def extract_benchmark(data: dict) -> dict[str, Any]:
    """Extract benchmark summary from benchmark_moe_offload output."""
    tok_s = data.get("decode_tok_per_s")
    if tok_s is None:
        decode = data.get("decode", data.get("decode_pass", {}))
        tok_s = decode.get("tok_per_sec", decode.get("tokens_per_sec"))
    peak_mb = data.get("peak_memory_mb", data.get("peak_memory_after_load_mb"))
    return {
        "tok_s": tok_s,
        "peak_mb": peak_mb,
    }


def extract_fidelity(data: dict) -> dict[str, Any]:
    """Extract fidelity summary."""
    backends = data.get("backends", {})
    default = backends.get("default", {})
    iso = backends.get("isoquant", {})
    turbo = backends.get("turboquant", {})
    default_ppl = default.get("mean_ppl")
    if not backends or default_ppl is None:
        return {}
    out = {}
    if iso:
        iso_ppl = iso.get("mean_ppl")
        if iso_ppl is not None:
            out["iso_ppl_delta"] = round(iso_ppl - default_ppl, 2)
        out["iso_cosine"] = iso.get(
            "cosine_sim_vs_default", iso.get("logit_cosine_sim_vs_default")
        )
    if turbo:
        turbo_ppl = turbo.get("mean_ppl")
        if turbo_ppl is not None:
            out["turbo_ppl_delta"] = round(turbo_ppl - default_ppl, 2)
        out["turbo_cosine"] = turbo.get(
            "cosine_sim_vs_default", turbo.get("logit_cosine_sim_vs_default")
        )
    return out


def extract_stress(data: dict) -> dict[str, Any]:
    """Extract stress test summary."""
    tests = data.get("tests", {})
    out = {}

    cold = tests.get("cold_prompt", {})
    prompts = cold.get("prompts", [])
    if prompts:
        avg_decode_hit = sum(
            p.get("cold", {}).get("decode_hit_rate", 0) for p in prompts
        ) / len(prompts)
        out["cold_decode_hit"] = round(avg_decode_hit, 3)

    slot = tests.get("slot_competition", {})
    summary = slot.get("summary", {})
    if summary:
        out["churn_single"] = summary.get("single_domain_avg_churn")
        out["churn_multi"] = summary.get("multi_domain_avg_churn")

    poison = tests.get("kv_poison", {})
    psummary = poison.get("summary", {})
    if psummary:
        out["poison_recovery"] = psummary.get("recovery_rate")

    return out


def extract_longctx(data: dict) -> dict[str, Any]:
    """Extract long-context eval summary."""
    tests = data.get("tests", {})
    out = {}

    retention = tests.get("state_retention", {})
    rsummary = retention.get("summary", {})
    if rsummary:
        out["retention_pass_rate"] = rsummary.get("pass_rate")

    ppl = tests.get("ppl_at_depth", {})
    psummary = ppl.get("summary", {})
    if psummary:
        out["avg_delta_ppl_vs_default"] = psummary.get("avg_delta_ppl_vs_default")
        out["avg_logit_cosine_vs_default"] = psummary.get(
            "avg_logit_cosine_vs_default"
        )
        out["avg_logit_top5_agreement_vs_default"] = psummary.get(
            "avg_logit_top5_agreement_vs_default"
        )

    consistency = tests.get("consistency_replay", {})
    csummary = consistency.get("summary", {})
    if csummary:
        out["identical_rate"] = csummary.get("identical_rate")
        out["avg_jaccard"] = csummary.get("avg_jaccard")

    return out


MODELS = ["qwen3", "gemma4", "nemotron120b"]
KV_MODES = ["default", "turboquant", "isoquant"]


def build_summary(input_dir: Path) -> str:
    lines: list[str] = []
    lines.append("# IsoQuant Submission Results Summary\n")

    # ── Quality gate table ──
    lines.append("## Quality Gate (5-prompt, 500 tokens)\n")
    lines.append("| Model | default | TurboQuant | IsoQuant |")
    lines.append("| --- | --- | --- | --- |")
    for model in MODELS:
        cells = [model]
        for kv in KV_MODES:
            path = input_dir / f"{model}_{kv}_500tok_quality.json"
            data = load_json(path)
            if data:
                q = extract_quality(data)
                cells.append(q["score"])
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # ── Benchmark table ──
    lines.append("## Throughput & Memory (benchmark triad)\n")
    lines.append("| Model | KV mode | tok/s | Peak MB |")
    lines.append("| --- | --- | --- | --- |")
    for model in MODELS:
        for kv in KV_MODES:
            path = input_dir / f"{model}_{kv}_profileA_bench.json"
            if not path.exists():
                path = input_dir / f"{model}_{kv}_profileB_bench.json"
            data = load_json(path)
            if data:
                b = extract_benchmark(data)
                lines.append(
                    f"| {model} | {kv} | {fmt(b['tok_s'])} | {fmt(b['peak_mb'], 0)} |"
                )
            else:
                lines.append(f"| {model} | {kv} | — | — |")
    lines.append("")

    # ── Fidelity table ──
    lines.append("## KV Fidelity\n")
    lines.append(
        "| Model | IsoQuant ΔPPL | IsoQuant cosine | TurboQuant ΔPPL | TurboQuant cosine |"
    )
    lines.append("| --- | --- | --- | --- | --- |")
    for model in MODELS:
        path = input_dir / f"{model}_kv_fidelity.json"
        data = load_json(path)
        if data:
            f = extract_fidelity(data)
            lines.append(
                f"| {model} | {fmt(f.get('iso_ppl_delta'))} | "
                f"{fmt(f.get('iso_cosine'), 4)} | "
                f"{fmt(f.get('turbo_ppl_delta'))} | "
                f"{fmt(f.get('turbo_cosine'), 4)} |"
            )
        else:
            lines.append(f"| {model} | — | — | — | — |")
    lines.append("")

    # ── Stress test table ──
    lines.append("## Stress Tests\n")
    lines.append(
        "| Model | KV mode | Cold decode hit | "
        "Churn (single) | Churn (multi) | Poison recovery |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for model in MODELS:
        for kv in KV_MODES:
            path = input_dir / f"{model}_stress_{kv}.json"
            data = load_json(path)
            if data:
                s = extract_stress(data)
                lines.append(
                    f"| {model} | {kv} | "
                    f"{fmt(s.get('cold_decode_hit'), 3)} | "
                    f"{fmt(s.get('churn_single'), 3)} | "
                    f"{fmt(s.get('churn_multi'), 3)} | "
                    f"{fmt(s.get('poison_recovery'), 2)} |"
                )
            else:
                lines.append(f"| {model} | {kv} | — | — | — | — |")
    lines.append("")

    # ── Long-context table ──
    lines.append("## Long-Context KV Evaluation\n")
    lines.append(
        "| Model | KV mode | Retention pass | Avg ΔPPL | Avg cosine | Identical rate |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for model in MODELS:
        for kv in ["default", "isoquant"]:
            path = input_dir / f"{model}_longctx_{kv}.json"
            data = load_json(path)
            if data:
                lc = extract_longctx(data)
                lines.append(
                    f"| {model} | {kv} | "
                    f"{fmt(lc.get('retention_pass_rate'), 2)} | "
                    f"{fmt(lc.get('avg_delta_ppl_vs_default'), 4)} | "
                    f"{fmt(lc.get('avg_logit_cosine_vs_default'), 4)} | "
                    f"{fmt(lc.get('identical_rate'), 2)} |"
                )
            else:
                lines.append(f"| {model} | {kv} | — | — | — | — |")
    lines.append("")

    # ── Missing artifacts ──
    all_expected = []
    for model in MODELS:
        for kv in KV_MODES:
            all_expected.append(f"{model}_{kv}_500tok_quality.json")
            all_expected.append(f"{model}_stress_{kv}.json")
        all_expected.append(f"{model}_kv_fidelity.json")
        for kv in ["default", "isoquant"]:
            all_expected.append(f"{model}_longctx_{kv}.json")

    missing = [f for f in all_expected if not (input_dir / f).exists()]
    if missing:
        lines.append("## Missing Artifacts\n")
        for f in missing:
            lines.append(f"- `{f}`")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize IsoQuant submission results into paper-ready tables"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="results/isoquant_submission",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write summary to file (default: stdout)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Warning: input directory does not exist: {input_dir}")
        print("Creating empty summary with all artifacts marked missing.")
        input_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(input_dir)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            f.write(summary)
        print(f"Summary written to: {out}")
    else:
        print(summary)


if __name__ == "__main__":
    main()
