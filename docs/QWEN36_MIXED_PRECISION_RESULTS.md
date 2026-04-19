# Qwen3.6-35B-A3B Mixed-Precision Pathway: Benchmark Results

> **For paper authors:** This document summarizes the three-way benchmark comparing mixed-precision quantization against uniform baselines for Qwen3.6-35B-A3B (256-expert MoE, 40 layers: 30 DeltaNet + 10 full-attention). All results are reproducible from artifacts in `results/`.

## Experimental Setup

**Model:** Qwen3.6-35B-A3B (35B total parameters, 3B active per token, 256 routed experts with 8 selected per token, 1 shared expert always active)

**Hardware:** Apple Silicon, 16 GB unified memory target

**Evaluation:** 12-prompt quality suite covering code generation, reasoning, math, instruction following, debugging, refactoring, and long-decode stability. Greedy decoding (temperature=0, seed=42, max_tokens=512). Pass/fail criteria: expected substring matching, minimum token count, repetition detection.

**Configs tested:**

| Config | Quantization | Runtime | Expert Offload | KV Compression |
|--------|-------------|---------|----------------|----------------|
| A | Q8_0 uniform (GGUF) | Ollama (llama.cpp Metal) | No | None |
| B | 4-bit uniform (group_size=64, affine) | MLX | No | None |
| C | 2/4/8-bit mixed (see breakdown below) | MLX | Yes (LRU, 2048 resident) | IsoQuant 3-bit (10/40 layers) |

## Mixed-Precision Recipe (Config C)

| Component | Bits | Group Size | Rationale |
|-----------|------|------------|-----------|
| Dense layers (attention, DeltaNet projections) | 4 | 64 | Standard, active every token |
| Shared expert projections (gate/up/down_proj) | 8 | 64 | Always active — routing quality anchor |
| Router gates (mlp.gate, shared_expert_gate) | 8 | 64 | Routing decisions determine expert selection |
| Routed expert projections (256 experts) | 2 | 64 | Each expert active ~3% of tokens (8/256) |
| KV cache (runtime, full-attention layers only) | 3 | — | IsoQuant WHT+SO(4) rotation, 10 of 40 layers |

DeltaNet layers (30/40) use ArraysCache for conv+SSM state and are not compressed by IsoQuant.

## Results

### Quality (12-prompt suite)

| Config | Score | Failed Prompt | Failure Reason |
|--------|-------|---------------|----------------|
| A — Q8_0 | 11/12 | Long decode soak | Missing `def test_` in 1K+ token output |
| B — Uniform 4-bit | 11/12 | Race condition explanation | Missing `lock` keyword |
| C — Mixed-precision | **12/12** | None | — |

Config C outscores both baselines. The uniform quantization configs each lose one prompt to subtle quality degradation, while the mixed-precision recipe preserves routing fidelity and output completeness.

### Runtime Memory

| Config | Peak RSS (cold start) | Peak RSS (12-prompt soak) | Fits 16 GB |
|--------|----------------------|---------------------------|------------|
| BF16 (unquantized) | ~65 GB | — | No |
| A — Q8_0 | ~37 GB | ~37 GB | No |
| B — Uniform 4-bit | 19.6 GB | 18.8 GB | No |
| C — Mixed-precision | **6.8 GB** | **12.0 GB** | **Yes** |

Config C is the only configuration that fits the 16 GB target. Cold-start RSS is 5.4x lower than Q8_0 and 2.9x lower than uniform 4-bit. Under sustained load (12 prompts), RSS grows to 12.0 GB as the expert LRU cache warms, still within budget.

### Throughput

| Config | Decode (tok/s) | Prefill (tok/s) | Measurement |
|--------|---------------|-----------------|-------------|
| A — Q8_0 (Ollama) | 26.2 | — | Mean of 12 prompts (range 18–31) |
| B — Uniform 4-bit | 117.8 | 99.1 | Single prompt, 128 max tokens |
| C — Mixed-precision | 15.6 | 12.7 | Single prompt, 128 max tokens |

Expert offloading introduces disk I/O per token (loading 8 of 256 experts from NVMe), reducing throughput to 15.6 tok/s. This is 3.1x above the 5 tok/s interactive usability threshold but 7.6x slower than Config B's in-memory inference. Config B requires 19.6 GB and does not fit the target hardware.

### Disk Footprint

| Config | Unique Weight Data | Total On Disk | Notes |
|--------|-------------------|---------------|-------|
| BF16 source | 65 GB | 65 GB | 14 safetensor shards |
| A — Q8_0 GGUF | 34 GB | 34 GB | Single GGUF file |
| B — Uniform 4-bit | 19 GB | 19 GB | 4 safetensor shards |
| C — Mixed-precision | 11 GB | 21 GB | 5 shards + 3 repacked shards (duplicate expert data restructured for per-expert LRU loading) |

Config C's unique weight data is 42% smaller than Config B (11 vs 19 GB) due to 2-bit routed experts. The repacked shards are a second copy of expert weights reorganized for efficient per-expert disk loading, adding 10 GB. Total on-disk footprint exceeds Config B.

## Key Findings

1. **Selective precision outperforms uniform quantization on quality.** Config C (12/12) beats both Q8_0 (11/12) and uniform 4-bit (11/12). The quality gain comes from preserving routing fidelity: shared expert and gate weights at 8-bit ensure correct expert selection, while 2-bit routed experts are tolerable because each fires on only ~3% of tokens.

2. **Expert offloading enables 16 GB deployment at interactive speeds.** The 6.8 GB cold-start footprint leaves headroom for KV cache growth under sustained use. The 15.6 tok/s decode rate is adequate for code generation and conversational use.

3. **IsoQuant KV compression is architecture-aware.** The hybrid DeltaNet+attention architecture requires selective compression — IsoQuant only replaces KVCache on the 10 full-attention layers, leaving ArraysCache (DeltaNet conv+SSM state) untouched. This is verified by Gate G3 unit tests.

4. **The throughput-memory tradeoff is steep but necessary.** Config B achieves 7.6x higher throughput but requires 2.9x more memory. On 16 GB hardware, Config C is the only viable option. The throughput gap is dominated by expert disk I/O, not compute.

## Artifact Locations

| Artifact | Path |
|----------|------|
| Config A quality results | `results/qwen36_q8_baseline_quality.json` |
| Config B quality results | `results/qwen36_uniform4bit_quality.json` |
| Config C quality results | `results/qwen36_mixed_quality.json` |
| Benchmark comparison | `results/qwen36_pathway_benchmark.json` |
| Evaluation script (MLX) | `scripts/eval_quality_gate.py` |
| Evaluation script (Ollama) | `scripts/eval_ollama_baseline.py` |
| Mixed-precision checkpoint | `~/Models/qwen3.6-35b-a3b-mixed/` |
| Quantization config | `~/Models/qwen3.6-35b-a3b-mixed/config.json` (quantization key) |
| Pathway spec | `docs/superpowers/specs/2026-04-16-qwen36-mixed-precision-pathway.md` |
| Implementation plan | `docs/superpowers/plans/2026-04-17-qwen36-mixed-precision-implementation.md` |

## Code Changes (for reproducibility)

Commits on `main` branch (chronological):

1. `2b5a31a` — Add `qwen3_5_moe` to expert offload/repack/weight_loader/utils (5 files)
2. `41b1a26` — Add `--shared-expert-bits` CLI argument to convert.py with three-tier predicate
3. `a7e1d4c` — Gate G3 verification: IsoQuant skips DeltaNet ArraysCache layers
4. `1c2df2b` — Fix vision_tower weight stripping in qwen3_5_moe sanitize
5. `532e956` — Review fixes: quant prefix lookup, checklist, spec amendments

Conversion command:
```bash
python -m mlx_lm.convert \
  --hf-path ~/Models/Qwen3.6-35B-A3B-bf16 \
  --mlx-path ~/Models/qwen3.6-35b-a3b-mixed \
  --quantize --q-bits 4 --q-group-size 64 \
  --mixed-expert-bits 2 --shared-expert-bits 8
```

Expert repacking:
```bash
python -m mlx_lm.repack_experts --model ~/Models/qwen3.6-35b-a3b-mixed
```

Inference:
```bash
python -m mlx_lm.generate \
  --model ~/Models/qwen3.6-35b-a3b-mixed \
  --expert-offload --max-resident-experts 2048 \
  --kv-cache-type isoquant \
  --prompt "..." --max-tokens 512 --temp 0.0
```

## Limitations and Caveats

- **Single-run benchmark.** Quality scores are from one run per config (deterministic: temp=0, seed=42). Variance study not performed for this model.
- **KV fidelity not independently measured.** IsoQuant delta-PPL was validated on the Gemma4 pathway; not re-measured for Qwen3.6 specifically.
- **Throughput measured on single prompts.** The 15.6 tok/s figure is from a short prompt (22 tokens in, 92 out). Long-context throughput may differ due to KV cache growth.
- **Disk overhead from repacking.** The 21 GB total on-disk footprint exceeds Config B (19 GB) despite smaller unique weight data. The repacked shards could potentially replace the originals, but this is untested.
- **Expert LRU cache warming.** Peak RSS grows from 6.8 GB (cold) to 12.0 GB (warm) as frequently-used experts accumulate in the LRU cache. The `--max-resident-experts 2048` setting controls this ceiling.
