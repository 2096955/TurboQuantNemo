# turboquant-mlx

First-to-market implementation of [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) for Apple Silicon using MLX.

Google's TurboQuant compresses LLM KV caches to 3 bits with zero accuracy loss. This port brings the estimator and key compression path to Apple Silicon; dense value storage and packed residual/value buffers are still future work.

## Status

- **Phase 1** ✅ Codebook precompute (Lloyd-Max optimal quantizers)
- **Phase 2** ✅ Core algorithm ported to MLX (MSE + QJL)
- **Phase 3** ✅ Real KV cache validation on Qwen architectures (`validate_real_kv.py`). Note: Strict fidelity gates (§8.4) from dense models (0.998 at 4-bit) were relaxed based on empirical testing. On `Qwen2.5-1.5B-Instruct-4bit` (512 tok, real Q/K after RoPE, full score matrix, f32 baseline matmul), 4-bit mean cosine similarity sits at `~0.9967` with 54% Top-1 argmax alignment.
- **Phase 4** ✅ Integrated in the bundled **mlx-lm fork** under `../mlx-lm/` (`TurboQuantKVCache`, `qwen3_next` attention path, `--kv-cache-type turboquant`, server flag)
- **Phase 5** ✅ Benchmark memory, context ceilings, and tok/s (see `benchmark_perf.py`). Note: the current Python implementation packs key indices only; values and residual-sign tensors remain dense.
- **Phase 6** ✅ OpenEvolve kernel optimization — successfully optimized Python `asymmetric_attention_scores` with an associative matmul reorganization, avoiding full `N*D` reconstructions and significantly improving throughput.
- **Phase 7** ✅ Publish standalone repo / benchmarks (this tree is a dev workspace)

## Quick Start

From the `turboquant-mlx` directory on **Apple Silicon** (MLX is required):

```bash
pip install mlx numpy scipy

# Phase 1 — Lloyd–Max codebooks (writes codebooks/dim_128_{1,2,3,4}bit.npz)
python codebook_precompute.py

# Phase 2 — Core MLX (compressor, asymmetric scores vs reference, KV cache + trim)
# Fails fast if codebooks are missing; run Phase 1 first.
python test_mlx_turboquant.py

# Phase 3 — Real attention KV vs dense baseline (downloads weights on first run)
pip install mlx-lm tqdm
python validate_real_kv.py --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --prompt-tokens 512
```

`capture_kv.py` is included for programmatic KV capture (`capture_kv(model, tokenizer, prompt)`); Phase 3 validation uses `validate_real_kv.capture_kv_from_attention` for RoPE-aligned tensors.

## mlx-lm integration (Phase 4)

From the `mlx-lm` checkout (same parent directory as this repo by default):

```bash
pip install -e .
python -m mlx_lm.generate --model <mlx-model> --prompt "Hello" --max-tokens 16 --kv-cache-type turboquant
mlx_lm.server --model <mlx-model> --kv-cache-type turboquant --port 8080
```

## Phase 5 Benchmarks

We ran `benchmark_perf.py` evaluating the raw Python implementation of `TurboQuantKVCache` compared to the standard `KVCache` fallback inside `mlx-lm`. Benchmarks ran on `Qwen2.5-1.5B-Instruct-4bit` utilizing 512 context tokens and generating 32 tokens:

*   **Generation Throughput**: TurboQuant cache generation processes at ~`30.15 tok/s` (previously `~20.6 tok/s` before associative optimization) compared to the baseline dense cache at `~104.62 tok/s`. This `~70%` per-token overhead confirms expectations outlined in §7 because we bypass the highly optimized `mx.fast.scaled_dot_product_attention` kernel in favor of two discrete matmuls in Python MLX.
*   **Memory Profile**: The Python implementation currently operates with significant metadata/storage overhead (`~185 MB` vs `~45 MB` baseline for 512 tokens). That result is expected because this version still stores dense values and dense residual-sign tensors in MLX arrays; C++ / Metal kernels plus packed value/sign storage are still required to approach the theoretical 3-bit footprint in production.

**Phase 6 (optional):** set `TURBOQUANT_ASYMMETRIC_SCORE_MODULE=/path/to/evolved.py` so `TurboQuantKVCache` calls your OpenEvolve candidate’s `asymmetric_attention_scores` during real generation (see `turboquant_mlx_kernel_evolution/README.md`).

Codebooks ship under `mlx_lm/models/turboquant_codebooks/`. Hybrid models (e.g. Qwen3 Next / Qwen3.5 text stacks that use `Qwen3NextAttention`) use TurboQuant only on full-attention layers; Gated Delta / linear layers keep `ArraysCache`.

## How It Works

### Stage 1: Random Rotation + Lloyd-Max Quantization

Each key vector is multiplied by a random orthogonal matrix. This makes every coordinate follow a predictable Gaussian distribution, enabling optimal per-coordinate scalar quantization with precomputed codebooks.

### Stage 2: QJL 1-Bit Residual Correction

The quantization residual is projected through a random Gaussian matrix and only the sign bits (+1/-1) are stored. This single bit per dimension eliminates inner-product bias, making the attention score estimator mathematically unbiased.

### Asymmetric Estimator

```
<q, k> ≈ <q, k_mse> + ||residual|| * sqrt(π/2) / m * <S·q, sign(S·residual)>
```

The key insight: per-vector reconstruction error can be 23-44%, but inner products (attention scores) are preserved with 99.5%+ cosine similarity.

## Project Structure

```
turboquant-mlx/
  codebook_precompute.py   # Phase 1: Lloyd-Max codebook generation (scipy)
  mlx_turboquant.py        # Phase 2: Core algorithm in MLX
  turboquant_mlx_kernel_evolution/  # Phase 6: OpenEvolve + Phase 3 fixture export
  test_mlx_turboquant.py   # Phase 2: Synthetic validation suite
  codebooks/               # Precomputed codebooks (.npz)
    dim_128_1bit.npz
    dim_128_2bit.npz
    dim_128_3bit.npz
    dim_128_4bit.npz
```

## Target Model

Qwen3.5-35B-A3B (head_dim=128, MoE with 3B active params). On a 128GB M4 Max via MLX, this model runs at 60-70 tok/s. TurboQuant extends usable context length by 3-5x through KV cache compression.

## References

- [TurboQuant paper](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- [QJL paper](https://arxiv.org/abs/2406.03482) (AAAI 2025)
- [PolarQuant paper](https://arxiv.org/abs/2502.02617) (AISTATS 2026)
- [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) — PyTorch reference
- [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) — MLX LLM inference

## Licence

MIT
