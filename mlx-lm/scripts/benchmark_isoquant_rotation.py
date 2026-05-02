import argparse
import json
import time

import mlx.core as mx
import numpy as np

from mlx_lm.models.mlx_isoquant import (
    build_isoquant_rotation_components,
    structured_rotate_forward,
    structured_rotate_inverse,
)
from mlx_lm.models.isoquant_metal_kernels import (
    metal_rotate_forward,
    metal_rotate_inverse,
)


def _bench(fn, x, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        y = fn(x)
        mx.eval(y)
    start = time.perf_counter()
    for _ in range(iters):
        y = fn(x)
        mx.eval(y)
    return (time.perf_counter() - start) / iters


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark structured IsoQuant rotation against dense matmul."
    )
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dims", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[1, 32, 256])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--include-metal", action="store_true")
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    results = []
    rng = np.random.default_rng(args.seed)

    for head_dim in args.head_dims:
        components = build_isoquant_rotation_components(
            args.num_heads, head_dim, seed=args.seed, layer_idx=args.layer_idx
        )
        rotation = components["rotation_matrices"]
        block_matrices = components["block_matrices"]
        block_matrices_t = components["block_matrices_t"]
        use_hadamard = bool(components["use_hadamard"])

        dense_forward = lambda x: mx.matmul(x, mx.swapaxes(rotation, -2, -1))
        dense_inverse = lambda x: mx.matmul(x, rotation)
        struct_forward = lambda x: structured_rotate_forward(
            x, block_matrices_t, use_hadamard
        )
        struct_inverse = lambda x: structured_rotate_inverse(
            x, block_matrices, use_hadamard
        )
        metal_forward = lambda x: metal_rotate_forward(
            x, block_matrices, use_hadamard
        )
        metal_inverse = lambda x: metal_rotate_inverse(
            x, block_matrices_t, use_hadamard
        )

        for seq_len in args.seq_lens:
            x = mx.array(
                rng.normal(size=(args.num_heads, seq_len, head_dim)).astype(np.float32)
            )
            dense_rot = dense_forward(x)
            struct_rot = struct_forward(x)
            dense_back = dense_inverse(dense_rot)
            struct_back = struct_inverse(struct_rot)
            eval_tensors = [dense_rot, struct_rot, dense_back, struct_back]

            result = {
                "num_heads": args.num_heads,
                "head_dim": head_dim,
                "seq_len": seq_len,
                "use_hadamard": use_hadamard,
                "forward_max_abs_diff": float(
                    np.max(np.abs(np.array(dense_rot) - np.array(struct_rot)))
                ),
                "inverse_max_abs_diff": float(
                    np.max(np.abs(np.array(dense_back) - np.array(struct_back)))
                ),
            }

            if args.include_metal:
                metal_rot = metal_forward(x)
                metal_back = metal_inverse(metal_rot)
                eval_tensors.extend([metal_rot, metal_back])
                result["forward_metal_max_abs_diff"] = float(
                    np.max(np.abs(np.array(dense_rot) - np.array(metal_rot)))
                )
                result["inverse_metal_max_abs_diff"] = float(
                    np.max(np.abs(np.array(dense_back) - np.array(metal_back)))
                )
                result["metal_roundtrip_max_abs_diff"] = float(
                    np.max(np.abs(np.array(x) - np.array(metal_back)))
                )

            mx.eval(*eval_tensors)

            result["forward_dense_ms"] = _bench(
                dense_forward, x, args.warmup, args.iters
            ) * 1000.0
            result["forward_struct_ms"] = _bench(
                struct_forward, x, args.warmup, args.iters
            ) * 1000.0
            result["inverse_dense_ms"] = _bench(
                dense_inverse, dense_rot, args.warmup, args.iters
            ) * 1000.0
            result["inverse_struct_ms"] = _bench(
                struct_inverse, struct_rot, args.warmup, args.iters
            ) * 1000.0
            result["forward_speedup"] = (
                result["forward_dense_ms"] / result["forward_struct_ms"]
                if result["forward_struct_ms"] > 0
                else None
            )
            result["inverse_speedup"] = (
                result["inverse_dense_ms"] / result["inverse_struct_ms"]
                if result["inverse_struct_ms"] > 0
                else None
            )
            if args.include_metal:
                result["forward_metal_ms"] = _bench(
                    metal_forward, x, args.warmup, args.iters
                ) * 1000.0
                result["inverse_metal_ms"] = _bench(
                    metal_inverse, metal_rot, args.warmup, args.iters
                ) * 1000.0
                result["forward_metal_speedup"] = (
                    result["forward_dense_ms"] / result["forward_metal_ms"]
                    if result["forward_metal_ms"] > 0
                    else None
                )
                result["inverse_metal_speedup"] = (
                    result["inverse_dense_ms"] / result["inverse_metal_ms"]
                    if result["inverse_metal_ms"] > 0
                    else None
                )
            results.append(result)

    payload = {
        "benchmark": "isoquant_rotation_dense_vs_structured"
        + ("_vs_metal" if args.include_metal else ""),
        "seed": args.seed,
        "warmup": args.warmup,
        "iters": args.iters,
        "results": results,
    }

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
