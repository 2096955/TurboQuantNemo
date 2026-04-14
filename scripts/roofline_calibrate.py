#!/usr/bin/env python3
"""Measure M4 Max hardware ceilings for roofline analysis.

Measures:
  - Peak FP16 TFLOPS (sustained FMA throughput)
  - Peak FP32 TFLOPS (sustained FMA throughput)
  - Peak memory bandwidth (stream-triad pattern)
  - MLX dispatch overhead (no-op mx.eval)
  - Power baseline (avg package + GPU watts via powermetrics)

Usage:
    python scripts/roofline_calibrate.py --output results/roofline_m4max.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time

import mlx.core as mx


def measure_peak_flops(dtype: mx.Dtype, warmup: int = 10, iters: int = 50) -> float:
    """Sustained FMA throughput via large square matmul.

    Uses 4096x4096 matrices — large enough to saturate GPU compute units.
    Returns TFLOPS.
    """
    N = 4096
    a = mx.random.normal((N, N)).astype(dtype)
    b = mx.random.normal((N, N)).astype(dtype)
    mx.eval(a, b)

    # Warmup
    for _ in range(warmup):
        c = mx.matmul(a, b)
        mx.eval(c)

    # Measure
    t0 = time.perf_counter()
    for _ in range(iters):
        c = mx.matmul(a, b)
        mx.eval(c)
    elapsed = (time.perf_counter() - t0) / iters

    flops = 2 * N * N * N  # 2*N^3 for matmul
    tflops = flops / elapsed / 1e12
    return tflops


def measure_peak_bandwidth(warmup: int = 10, iters: int = 50) -> float:
    """Stream-triad pattern: a = b + scalar * c.

    Measures sustained memory bandwidth in GB/s.
    Uses large arrays to avoid cache effects.
    """
    N = 64 * 1024 * 1024  # 64M elements = 256 MB per array (float32)
    a = mx.zeros((N,), dtype=mx.float32)
    b = mx.random.normal((N,), dtype=mx.float32)
    c = mx.random.normal((N,), dtype=mx.float32)
    scalar = mx.array(3.0, dtype=mx.float32)
    mx.eval(a, b, c, scalar)

    for _ in range(warmup):
        a = b + scalar * c
        mx.eval(a)

    t0 = time.perf_counter()
    for _ in range(iters):
        a = b + scalar * c
        mx.eval(a)
    elapsed = (time.perf_counter() - t0) / iters

    # 2 reads (b, c) + 1 write (a) = 3 * N * 4 bytes
    bytes_moved = 3 * N * 4
    gbs = bytes_moved / elapsed / 1e9
    return gbs


def measure_dispatch_overhead(warmup: int = 100, iters: int = 500) -> float:
    """Time mx.eval() on a pre-compiled no-op to measure dispatch latency.

    Returns overhead in microseconds.
    """
    x = mx.array(0.0)
    mx.eval(x)

    for _ in range(warmup):
        y = x + 0.0
        mx.eval(y)

    t0 = time.perf_counter()
    for _ in range(iters):
        y = x + 0.0
        mx.eval(y)
    elapsed = (time.perf_counter() - t0) / iters

    return elapsed * 1e6  # microseconds


def read_powermetrics(duration_ms: int = 2000) -> dict:
    """Read average power from powermetrics (requires sudo).

    Returns dict with avg_package_watts and avg_gpu_watts, or None values
    if powermetrics is unavailable or not run with sudo.
    """
    try:
        result = subprocess.run(
            [
                "sudo",
                "powermetrics",
                "--samplers",
                "gpu_power,cpu_power",
                "-i",
                str(duration_ms),
                "-n",
                "1",
                "--format",
                "plist",
            ],
            capture_output=True,
            text=True,
            timeout=duration_ms / 1000 + 5,
        )
        if result.returncode != 0:
            return {"avg_package_watts": None, "avg_gpu_watts": None}

        # Parse plist for power values
        import plistlib

        data = plistlib.loads(result.stdout.encode())
        pkg_power = data.get("processor", {}).get("package_watts", None)
        gpu_power = data.get("gpu", {}).get("gpu_power", None)
        return {"avg_package_watts": pkg_power, "avg_gpu_watts": gpu_power}
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return {"avg_package_watts": None, "avg_gpu_watts": None}


def get_hardware_info() -> dict:
    """Collect hardware metadata with sanity checks."""
    import platform

    gpu_family = "unknown"
    gpu_cores = 0
    memory_gb = 0
    device_name = ""

    try:
        info = mx.device_info()
        gpu_family = info.get("architecture", "unknown")
        device_name = info.get("device_name", "")
        memory_gb = round(info.get("memory_size", 0) / (1024**3))
        # Extract GPU core count from device name (e.g. "Apple M4 Max" -> 40 cores)
        core_map = {
            "M4 Max": 40,
            "M4 Pro": 20,
            "M4": 10,
            "M3 Max": 40,
            "M3 Pro": 18,
            "M3": 10,
            "M2 Max": 38,
            "M2 Ultra": 76,
            "M1 Max": 32,
            "M1 Ultra": 64,
        }
        for chip, cores in core_map.items():
            if chip in device_name:
                gpu_cores = cores
                break
    except Exception:
        pass

    # Sanity checks
    if gpu_cores == 0:
        print("WARNING: gpu_cores=0 detected - hardware metadata may be incomplete")
    if gpu_family == "unknown":
        print("WARNING: metal_gpu_family=unknown - hardware detection failed")

    return {
        "chip": device_name or platform.processor() or "Apple Silicon",
        "memory_gb": memory_gb,
        "gpu_cores": gpu_cores,
        "macos_version": platform.mac_ver()[0],
        "metal_gpu_family": gpu_family,
    }


def main():
    parser = argparse.ArgumentParser(description="Roofline hardware calibration")
    parser.add_argument("--output", type=str, default="results/roofline_m4max.json")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--skip-power", action="store_true", help="Skip powermetrics (requires sudo)"
    )
    args = parser.parse_args()

    print("=== Roofline Calibration ===")

    print("Measuring peak FP16 TFLOPS...")
    fp16_tflops = measure_peak_flops(mx.float16, args.warmup, args.iters)
    print(f"  Peak FP16: {fp16_tflops:.2f} TFLOPS")

    print("Measuring peak FP32 TFLOPS...")
    fp32_tflops = measure_peak_flops(mx.float32, args.warmup, args.iters)
    print(f"  Peak FP32: {fp32_tflops:.2f} TFLOPS")

    print("Measuring peak memory bandwidth...")
    bw_gbs = measure_peak_bandwidth(args.warmup, args.iters)
    theoretical_bw = 546.0  # M4 Max 128GB: 16-channel LPDDR5X-8533
    bw_ratio = bw_gbs / theoretical_bw
    print(f"  Peak bandwidth: {bw_gbs:.1f} GB/s (measured)")
    print(
        f"  Theoretical: {theoretical_bw:.0f} GB/s, ratio: {bw_ratio:.2f} (expected 0.73-0.85)"
    )

    print("Measuring dispatch overhead...")
    dispatch_us = measure_dispatch_overhead()
    print(f"  Dispatch overhead: {dispatch_us:.2f} us")

    power = {"avg_package_watts": None, "avg_gpu_watts": None}
    if not args.skip_power:
        print("Measuring baseline power (2s sample)...")
        power = read_powermetrics()
        if power["avg_package_watts"]:
            print(
                f"  Package: {power['avg_package_watts']:.1f} W, GPU: {power['avg_gpu_watts']:.1f} W"
            )
        else:
            print("  Power metrics unavailable (run with sudo for power data)")

    # Validation checks
    hw_info = get_hardware_info()
    valid = True
    warnings = []

    if hw_info["gpu_cores"] == 0 or hw_info["metal_gpu_family"] == "unknown":
        valid = False
        warnings.append(
            "Hardware metadata incomplete (gpu_cores=0 or gpu_family=unknown)"
        )

    if bw_ratio < 0.5:
        valid = False
        warnings.append(
            f"Bandwidth efficiency {bw_ratio:.2f} < 0.5 - calibration may be unreliable (too few iters={args.iters})"
        )

    if not valid:
        print("\n*** VALIDATION FAILED ***")
        for w in warnings:
            print(f"  - {w}")
        print("Run with more --iters (>=100) and check hardware detection")

    result = {
        "hardware": hw_info,
        "calibration": {
            "peak_fp16_tflops": round(fp16_tflops, 2),
            "peak_fp32_tflops": round(fp32_tflops, 2),
            "peak_memory_bandwidth_gbs": round(bw_gbs, 1),
            "theoretical_memory_bandwidth_gbs": 546.0,
            "bandwidth_efficiency_ratio": round(bw_gbs / 546.0, 3),
            "noop_dispatch_us": round(dispatch_us, 2),
        },
        "power_baseline": power,
        "valid": valid,
        "warnings": warnings,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    import os

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
