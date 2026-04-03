#!/usr/bin/env python3
"""Benchmark harness for MoE expert offload + TurboQuant (Profile A / B from implementation plan).

Single-process, single-run measurements (not a statistical or parity-vs-baseline gate unless
you add that externally). Peak memory after load is reported separately from the final peak
so load-time spikes are visible for large-checkpoint runs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from typing import Any

_RUNTIME = None


def _runtime():
    global _RUNTIME
    if _RUNTIME is None:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.generate import generate_step

        _RUNTIME = (mx, load, generate_step)
    return _RUNTIME


def _build_prompt_tokens(tokenizer, target_len: int) -> list[int]:
    """Approximate target token count by growing a repeated segment."""
    unit = tokenizer.encode("hello ", add_special_tokens=False)
    if not unit:
        unit = [tokenizer.eos_token_id or 0]
    out: list[int] = []
    while len(out) < target_len:
        out.extend(unit)
    return out[:target_len]


def run_benchmark(
    model_path: str,
    *,
    profile: str,
    prefill_tokens: int,
    prefill_step_size: int | None,
    decode_tokens: int,
    seed: int,
    kv_cache_type: str,
    expert_offload: bool,
    max_resident_experts: int,
    expert_offload_dir: str | None,
    memory_mode: str | None,
    strict_gates: bool,
    split_decode_timing: bool,
    warm_second_pass: bool,
) -> tuple[dict[str, Any], bool]:
    mx, load, generate_step = _runtime()
    mx.random.seed(seed)

    expert_offload_flag = expert_offload
    kv = kv_cache_type
    if memory_mode == "120b-32gb":
        expert_offload_flag = True
        kv = "turboquant"

    model_config = {"quantize_activations": False}
    if expert_offload_flag:
        model_config["expert_offload"] = True
        model_config["max_resident_experts"] = max_resident_experts
        model_config["expert_offload_dir"] = expert_offload_dir

    print(f"Loading model: {model_path}...")
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    model, tokenizer = load(model_path, model_config=model_config)
    load_time = time.perf_counter() - t0
    peak_after_load_mb = mx.get_peak_memory() / (1024 * 1024)
    print(
        f"Load time: {load_time:.2f}s (peak memory after load: {peak_after_load_mb:.2f} MB)"
    )

    prompt_ids = _build_prompt_tokens(tokenizer, prefill_tokens)
    prompt = mx.array(prompt_ids)
    effective_prefill_step_size = (
        min(64, max(1, prefill_tokens))
        if prefill_step_size is None
        else max(1, min(prefill_step_size, prefill_tokens))
    )

    t_prefill = time.perf_counter()
    first_tok = None
    decode_tps = None
    peak_mb = None
    t_decode_start = None
    t_mid = None
    t_decode_end = None
    split_point = None
    if split_decode_timing and decode_tokens >= 4:
        split_point = max(1, decode_tokens // 2)

    for n, (token, logprobs) in enumerate(
        generate_step(
            prompt,
            model,
            max_tokens=decode_tokens,
            kv_cache_type=kv,
            prefill_step_size=effective_prefill_step_size,
        )
    ):
        mx.eval(token, logprobs)
        now = time.perf_counter()
        if n == 0:
            first_tok = now
            prefill_s = first_tok - t_prefill
            print(
                f"Prefill+first-token latency: {prefill_s:.3f}s for {prefill_tokens} prompt tokens"
            )
            t_decode_start = now
        if (
            split_point is not None
            and t_decode_start is not None
            and n + 1 == split_point
        ):
            t_mid = now
            first_s = max(t_mid - t_decode_start, 1e-9)
            print(
                f"Decode (first half): {split_point / first_s:.2f} tok/s "
                f"over first {split_point} tokens"
            )
        if n + 1 == decode_tokens and t_decode_start is not None:
            t_decode_end = now
            decode_s = max(t_decode_end - t_decode_start, 1e-9)
            decode_tps = decode_tokens / decode_s
            print(f"Decode: {decode_tps:.2f} tok/s over {decode_tokens} tokens")
            if split_point is not None and t_mid is not None:
                second = decode_tokens - split_point
                second_s = max(t_decode_end - t_mid, 1e-9)
                print(
                    f"Decode (second half): {second / second_s:.2f} tok/s "
                    f"over last {second} tokens (warm-cache proxy)"
                )
        peak_mb = mx.get_peak_memory() / (1024 * 1024)

    mgr = getattr(model, "expert_offload_manager", None)
    cache_stats = mgr.stats_summary() if mgr is not None else {}

    warm_decode_tps = None
    warm_peak_mb = None
    warm_cache_stats: dict[str, Any] = {}
    if warm_second_pass and t_decode_start is not None:
        print("\n--- Warm second pass (same loaded model, fresh generate_step) ---")
        t_ds = None
        for n, (token, logprobs) in enumerate(
            generate_step(
                prompt,
                model,
                max_tokens=decode_tokens,
                kv_cache_type=kv,
                prefill_step_size=effective_prefill_step_size,
            )
        ):
            mx.eval(token, logprobs)
            now = time.perf_counter()
            if n == 0:
                t_ds = now
            if n + 1 == decode_tokens and t_ds is not None:
                warm_decode_tps = decode_tokens / max(now - t_ds, 1e-9)
                print(
                    f"Warm pass decode: {warm_decode_tps:.2f} tok/s over {decode_tokens} tokens"
                )
            warm_peak_mb = mx.get_peak_memory() / (1024 * 1024)
        mgr_w = getattr(model, "expert_offload_manager", None)
        warm_cache_stats = mgr_w.stats_summary() if mgr_w is not None else {}
        if warm_peak_mb is not None and peak_mb is not None:
            peak_mb = max(peak_mb, warm_peak_mb)

    first_half_tps = None
    second_half_tps = None
    if (
        split_point is not None
        and t_decode_start is not None
        and t_mid is not None
        and t_decode_end is not None
    ):
        first_half_tps = split_point / max(t_mid - t_decode_start, 1e-9)
        second = decode_tokens - split_point
        second_half_tps = second / max(t_decode_end - t_mid, 1e-9)

    metrics: dict[str, Any] = {
        "profile": profile,
        "load_time_s": load_time,
        "peak_memory_after_load_mb": peak_after_load_mb,
        "prefill_plus_first_token_s": (first_tok - t_prefill) if first_tok else None,
        "prefill_step_size": effective_prefill_step_size,
        "decode_tok_per_s": decode_tps,
        "decode_split_point_tokens": split_point,
        "decode_first_half_tok_per_s": first_half_tps,
        "decode_second_half_tok_per_s": second_half_tps,
        "peak_memory_mb": peak_mb,
        "kv_cache_type": kv,
        "expert_offload": expert_offload_flag,
        "expert_cache": cache_stats,
        "warm_second_pass": warm_second_pass,
        "warm_pass_decode_tok_per_s": warm_decode_tps,
        "warm_pass_peak_memory_mb": warm_peak_mb,
        "warm_pass_expert_cache": warm_cache_stats,
    }

    print(
        f"Peak memory (session, includes load + inference): {peak_mb:.2f} MB "
        f"(after load only: {peak_after_load_mb:.2f} MB)"
    )
    print("\n--- JSON Metrics ---")
    print(json.dumps(metrics, indent=2))

    gate_failed = False

    # Merge gates (plan defaults; tune after baselines on real hardware).
    # Use session peak so load + inference both count toward envelope claims.
    peak_gate = metrics["peak_memory_mb"]
    if profile == "A":
        if peak_gate and peak_gate > 24000:
            msg = "Peak memory exceeded soft gate for Profile A (>24 GB)"
            print(f"WARNING: {msg}")
            if strict_gates:
                gate_failed = True
    elif profile == "B":
        if peak_gate and peak_gate > 30000:
            msg = "Peak memory exceeded soft gate for Profile B (>30 GB)"
            print(f"WARNING: {msg}")
            if strict_gates:
                gate_failed = True
        decode_hit_rate = cache_stats.get(
            "decode_hit_rate", cache_stats.get("hit_rate", 0)
        )
        if expert_offload_flag and decode_hit_rate < 0.5:
            msg = (
                "Expert cache decode hit rate below 0.5 after run "
                "(prefill is excluded from this gate)."
            )
            print(f"WARNING: {msg}")
            if strict_gates:
                gate_failed = True

    return metrics, gate_failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MoE offload + TurboQuant benchmark harness"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Local MLX model directory"
    )
    parser.add_argument("--profile", type=str, choices=["A", "B"], default="A")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--kv-cache-type",
        type=str,
        default="default",
        choices=["default", "turboquant"],
    )
    parser.add_argument("--expert-offload", action="store_true")
    parser.add_argument("--max-resident-experts", type=int, default=16)
    parser.add_argument("--expert-offload-dir", type=str, default=None)
    parser.add_argument(
        "--memory-mode",
        type=str,
        default=None,
        help='Use "120b-32gb" to force expert offload + turboquant KV.',
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=64,
        help="Chunk size for prefill. Smaller sizes prevent OOM during prefill burst (default: 64).",
    )
    parser.add_argument(
        "--strict-gates",
        action="store_true",
        help="Exit with status 1 if any soft memory/cache gate triggers (for CI).",
    )
    parser.add_argument(
        "--split-decode-timing",
        action="store_true",
        help="Report first-half vs second-half decode tok/s (warm-cache proxy).",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Write metrics JSON to this file (for CI artifacts).",
    )
    parser.add_argument(
        "--warm-second-pass",
        action="store_true",
        help="After the first decode, run a second full decode (expert cache warm).",
    )
    parser.add_argument(
        "--repeat-runs",
        type=int,
        default=1,
        help="Run the full benchmark N times (reloads model each time); summary in JSON.",
    )
    args = parser.parse_args()

    if args.profile == "A":
        prefill_tokens, decode_tokens = 512, 128
    else:
        prefill_tokens, decode_tokens = 1024, 256

    try:
        runs_out: list[dict[str, Any]] = []
        gate_failed = False
        for r in range(max(1, args.repeat_runs)):
            if args.repeat_runs > 1:
                print(f"\n======== Run {r + 1}/{args.repeat_runs} ========\n")
            metrics, gf = run_benchmark(
                args.model,
                profile=args.profile,
                prefill_tokens=prefill_tokens,
                prefill_step_size=args.prefill_step_size,
                decode_tokens=decode_tokens,
                seed=args.seed + r,
                kv_cache_type=args.kv_cache_type,
                expert_offload=args.expert_offload,
                max_resident_experts=args.max_resident_experts,
                expert_offload_dir=args.expert_offload_dir,
                memory_mode=args.memory_mode,
                strict_gates=args.strict_gates,
                split_decode_timing=args.split_decode_timing,
                warm_second_pass=args.warm_second_pass,
            )
            gate_failed = gate_failed or gf
            runs_out.append(metrics)

        metrics = runs_out[0]
        if len(runs_out) > 1:
            dts = [
                m.get("decode_tok_per_s") for m in runs_out if m.get("decode_tok_per_s")
            ]
            peaks = [
                m.get("peak_memory_mb") for m in runs_out if m.get("peak_memory_mb")
            ]
            summary: dict[str, Any] = {
                "repeat_runs": len(runs_out),
                "seed_start": args.seed,
            }
            if dts:
                summary["decode_tok_per_s_mean"] = statistics.mean(dts)
                summary["decode_tok_per_s_stdev"] = (
                    statistics.pstdev(dts) if len(dts) > 1 else 0.0
                )
            if peaks:
                summary["peak_memory_mb_max"] = max(peaks)
                summary["peak_memory_mb_mean"] = statistics.mean(peaks)
            metrics = {
                "multi_run_summary": summary,
                "runs": runs_out,
            }

        if args.json_output:
            with open(args.json_output, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"Wrote metrics JSON: {args.json_output}")
        if gate_failed:
            print(
                "FAIL: benchmark exceeded one or more gates (--strict-gates).",
                file=sys.stderr,
            )
            sys.exit(1)
    except Exception as e:
        print(f"Benchmark failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
