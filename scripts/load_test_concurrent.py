"""Concurrent load test for mlx-lm server.

Sends N simultaneous requests to the /v1/completions endpoint and measures
per-request latency, aggregate throughput, and whether expert offloading
serializes under concurrent load.

Usage:
  # Start server first:
  python -m mlx_lm.server --model ./gemma4-layer-aware \\
      --expert-offload --kv-cache-type isoquant --port 8080

  # Then run load test:
  python scripts/load_test_concurrent.py --port 8080 --concurrency 4
  python scripts/load_test_concurrent.py --port 8080 --concurrency 1,2,4,8
"""

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen


PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a Python function to compute the nth Fibonacci number.",
    "What are the three laws of thermodynamics?",
    "Describe the process of photosynthesis step by step.",
    "Write a SQL query to find duplicate rows in a table.",
    "What is the difference between TCP and UDP?",
    "Explain how a hash map works internally.",
    "What causes ocean tides?",
]


def send_request(
    base_url: str,
    prompt: str,
    max_tokens: int,
    seed: int,
    request_id: int,
) -> dict:
    """Send a single completion request and measure latency."""
    payload = json.dumps(
        {
            "model": "default",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": seed,
        }
    ).encode()

    req = Request(
        f"{base_url}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.monotonic()
    try:
        with urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read())
        elapsed = time.monotonic() - t0

        tokens = body.get("usage", {}).get("completion_tokens", 0)
        return {
            "request_id": request_id,
            "status": "ok",
            "latency_s": round(elapsed, 3),
            "tokens": tokens,
            "tok_per_s": round(tokens / elapsed, 2) if elapsed > 0 else 0,
            "prompt_preview": prompt[:50],
        }
    except Exception as e:
        elapsed = time.monotonic() - t0
        return {
            "request_id": request_id,
            "status": "error",
            "latency_s": round(elapsed, 3),
            "error": str(e),
            "prompt_preview": prompt[:50],
        }


def run_batch(
    base_url: str,
    concurrency: int,
    max_tokens: int,
    seed: int,
) -> dict:
    """Run a batch of concurrent requests and collect metrics."""
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(concurrency)]

    t0 = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(send_request, base_url, prompt, max_tokens, seed + i, i): i
            for i, prompt in enumerate(prompts)
        }
        for future in as_completed(futures):
            results.append(future.result())
    wall_time = time.monotonic() - t0

    results.sort(key=lambda r: r["request_id"])

    ok_results = [r for r in results if r["status"] == "ok"]
    failed = len(results) - len(ok_results)

    latencies = [r["latency_s"] for r in ok_results]
    tok_rates = [r["tok_per_s"] for r in ok_results]
    total_tokens = sum(r["tokens"] for r in ok_results)

    summary = {
        "concurrency": concurrency,
        "wall_time_s": round(wall_time, 3),
        "requests_ok": len(ok_results),
        "requests_failed": failed,
        "total_tokens": total_tokens,
        "aggregate_tok_per_s": round(total_tokens / wall_time, 2)
        if wall_time > 0
        else 0,
    }

    if latencies:
        summary["latency_mean_s"] = round(statistics.mean(latencies), 3)
        summary["latency_p50_s"] = round(sorted(latencies)[len(latencies) // 2], 3)
        summary["latency_max_s"] = round(max(latencies), 3)
        if len(latencies) > 1:
            summary["latency_stdev_s"] = round(statistics.stdev(latencies), 3)

    if tok_rates:
        summary["per_request_tok_per_s_mean"] = round(statistics.mean(tok_rates), 2)

    summary["results"] = results
    return summary


def check_server(base_url: str) -> bool:
    """Check if server is reachable."""
    try:
        req = Request(f"{base_url}/v1/models", method="GET")
        with urlopen(req, timeout=5):
            return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Concurrent load test for mlx-lm server"
    )
    parser.add_argument(
        "--host", default="localhost", help="Server host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Server port (default: 8080)"
    )
    parser.add_argument(
        "--concurrency",
        default="1,2,4",
        help="Comma-separated concurrency levels to test (default: 1,2,4)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Max tokens per request (default: 100)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument("--output-json", help="Write results to JSON file")
    parser.add_argument(
        "--warmup", action="store_true", help="Send a single warmup request first"
    )
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    concurrency_levels = [int(c.strip()) for c in args.concurrency.split(",")]

    print(f"Load test: {base_url}")
    print(f"Concurrency levels: {concurrency_levels}")
    print(f"Max tokens per request: {args.max_tokens}")
    print()

    if not check_server(base_url):
        print(f"ERROR: Server not reachable at {base_url}", file=sys.stderr)
        print("Start the server first:", file=sys.stderr)
        print(
            f"  python -m mlx_lm.server --model <path> --port {args.port}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.warmup:
        print("Warmup request...")
        send_request(base_url, "Hello", args.max_tokens, args.seed, -1)
        print()

    all_results = []
    print(
        f"{'Concurrency':<14} {'Wall (s)':<10} {'Agg tok/s':<12} {'Per-req tok/s':<16} {'Lat mean':<10} {'Lat max':<10} {'Failed':<8}"
    )
    print(
        f"{'-' * 14:<14} {'-' * 10:<10} {'-' * 12:<12} {'-' * 16:<16} {'-' * 10:<10} {'-' * 10:<10} {'-' * 8:<8}"
    )

    for level in concurrency_levels:
        batch = run_batch(base_url, level, args.max_tokens, args.seed)
        all_results.append(batch)

        print(
            f"{batch['concurrency']:<14} "
            f"{batch['wall_time_s']:<10} "
            f"{batch['aggregate_tok_per_s']:<12} "
            f"{batch.get('per_request_tok_per_s_mean', 'N/A'):<16} "
            f"{batch.get('latency_mean_s', 'N/A'):<10} "
            f"{batch.get('latency_max_s', 'N/A'):<10} "
            f"{batch['requests_failed']:<8}"
        )

    # Scaling analysis
    if len(all_results) >= 2:
        base = all_results[0]
        print()
        print("Scaling analysis (vs single request):")
        base_agg = base["aggregate_tok_per_s"]
        if base_agg > 0:
            for batch in all_results[1:]:
                speedup = batch["aggregate_tok_per_s"] / base_agg
                efficiency = speedup / batch["concurrency"] * 100
                print(
                    f"  {batch['concurrency']}x concurrency: "
                    f"{speedup:.2f}x aggregate throughput, "
                    f"{efficiency:.0f}% scaling efficiency"
                )

    if args.output_json:
        output = {
            "server": base_url,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "batches": [
                {k: v for k, v in b.items() if k != "results"} for b in all_results
            ],
            "detailed_results": all_results,
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {args.output_json}")


if __name__ == "__main__":
    main()
