# IsoQuant3 vs TurboQuant3 — Comparative Profile

**Model:** Qwen2.5-1.5B-Instruct Q6_K (n_head=12, n_head_kv=2, n_gqa=6, head_dim=128)
**Hardware:** Apple M4 Max, Metal backend, flash_attn=1
**Context:** n_prompt=512, n_gen=128, r=3

## Same-session benchmark (pinned)

Source: `results/llama_cpp_turbo3_vs_isoquant3_final.json`

| Configuration | Prompt (t/s) | Generation (t/s) |
|---|---|---|
| turbo3 | 4101 | 99.4 |
| isoquant3 fused | 4078 (-0.6%) | 95.8 (-3.6%) |

Source: `results/llama_cpp_isoquant3_vs_turbo3.json` (derived comparison)

## Historical composed baseline

Source: `results/llama_cpp_isoquant3_bench.json` (pre-fusion, 2026-04-13)

| Configuration | Prompt (t/s) | Generation (t/s) |
|---|---|---|
| isoquant3 composed (280 extra launches) | 2306 (-44%) | 81.9 (-18%) |

## Analysis: Is the SO(4) machinery worthwhile?

**Yes, with the fused kernel. No, with graph-level composition.**

The graph-composed path added 280 extra Metal kernel launches per token (5 ops × 2 applications × 28 layers), producing a -44% prompt and -18% generation penalty. This was entirely dispatch overhead, not compute — the actual SO(4) block rotation is only 512 FMAs per application.

The fused `kernel_turbo_wht_so4` eliminates all 280 extra launches by performing the SO(4) block matvec inside the existing WHT butterfly kernel. The result:

- **Prompt eval**: recovered from -44% to -0.6% (within measurement noise)
- **Generation**: recovered from -18% to -3.6% (irreducible SO(4) compute cost)

The residual -3.6% generation gap is the actual arithmetic cost: 32 blocks × 16 FMAs = 512 extra FMAs per WHT dispatch, on top of the 896 FMAs for the WHT butterfly. This is ~57% more compute per kernel, but the kernel is a small fraction of total decode time, so the end-to-end impact is ~3.6%.

## Numerical equivalence

The fused kernel operates in half4 precision for the SO(4) matvec, while the composed path uses F32 via `ggml_mul_mat`. This produces:

- Token-identical output over 30 decode steps (argmax agreement)
- Logit-level divergence: max_abs_diff=4.31, rmse=0.95, top-10 overlap 7/10
- Same argmax for the final prompt logits

The divergence is consistent with half-precision accumulation through 28 layers of attention. It does not affect top-1 token selection on the tested prompt.

Source: `results/llama_cpp_isoquant3_logit_diff.json`, `results/llama_cpp_isoquant3_correctness.md`

## Conclusion

The SO(4) block rotation machinery is worthwhile **only with a fused kernel**. Graph-level composition destroys the performance advantage entirely. The fused kernel recovers turbo3 parity at -3.6% generation cost, which is the irreducible price of the extra 512 FMAs. For models where the theoretical quality benefit of SO(4) decorrelation over WHT-only rotation is proven, this is an acceptable tradeoff.

## Limitations

- Tested on a single small model (1.5B parameters). Larger models may show different proportional overhead.
- The fused kernel is currently restricted to group_size=128 (64-wide sign tables not yet generated).
- Metal-only; CPU/CUDA backends assert on SO(4) rotation.
- InnerQ per-channel scaling is not applied on Metal (pre-existing gap, not introduced by fusion).
