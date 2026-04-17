# Qwen3.6-35B-A3B Mixed-Precision Pathway — 16 GB Target

**Date:** 2026-04-16
**Status:** Design approved, pending implementation
**Approach:** A — Direct MLX convert from BF16

## Objective

Convert Qwen3.6-35B-A3B to a mixed-precision MLX checkpoint that fits within a 16 GB memory budget using expert offloading + IsoQuant KV compression. Benchmark against two baselines (Q8_0 GGUF on Ollama, mlx-community uniform 4-bit) to demonstrate quality preservation and throughput at a fraction of the memory.

## Source Model

| Property | Value |
|----------|-------|
| HF path | `mlx-community/Qwen3.6-35B-A3B-bf16` |
| Total params | 35B |
| Active params | 3B per token |
| Layers | 40 (30 DeltaNet + 10 full attention) |
| Experts | 256 routed + 1 shared per MoE layer |
| TopK | 8 |
| Head dim | 256 (attention), 128 (DeltaNet) |
| KV heads | 2 (GQA) |
| Context | 262K native |
| model_type | `qwen3_5_moe` |

Architecture: hybrid DeltaNet (linear attention) + full attention. Pattern is `3 x DeltaNet -> 1 x full attention`, repeated 10 times. Only the 10 full-attention layers produce KV caches; IsoQuant compresses these 10 layers (25% of total, similar to Gemma4's 5/30 sliding-window exposure).

## Mixed-Precision Recipe

| Component | Bits | Group Size | Rationale |
|-----------|------|------------|-----------|
| Dense/attention weights | 4 | 64 | Standard quality, proven on all pathways |
| DeltaNet projections (in_proj, out_proj, conv1d) | 4 | 64 | Active every token, quality-sensitive |
| Routed expert weights (switch_mlp.gate/up/down_proj) | 2 | 32 | 97% idle per token, offloaded to disk |
| Shared expert weights | 8 | 64 | Always active, high kurtosis expected |
| Embeddings / lm_head | 4 | 64 | Default |
| KV cache (runtime) | IsoQuant 3-bit | — | WHT + SO(4), only 10 full-attention layers |

### Conversion Commands

```bash
# Step 1: Convert BF16 -> mixed-precision MLX
python -m mlx_lm.convert \
  --hf-path mlx-community/Qwen3.6-35B-A3B-bf16 \
  --mlx-path ~/Models/qwen3.6-35b-a3b-mixed \
  --quantize --q-bits 4 --q-group-size 64 \
  --mixed-expert-bits 2

# Step 2: Repack experts for LRU offload
python -m mlx_lm.repack_experts --model ~/Models/qwen3.6-35b-a3b-mixed
```

### Runtime

```bash
python -m mlx_lm.generate \
  --model ~/Models/qwen3.6-35b-a3b-mixed \
  --expert-offload --max-resident-experts 2048 \
  --kv-cache-type isoquant \
  --prompt "Hello"
```

## Three-Way Benchmark Comparison

The core deliverable is a head-to-head comparison proving our mixed-precision stack preserves quality while fitting in 16 GB.

### Configuration A: Q8_0 GGUF on Ollama (baseline)

- **Model:** `~/Models/Qwen3.6-35B-A3B-Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf`
- **Serving:** `ollama run qwen3.6-35b-a3b:q8_0`
- **Memory:** ~37 GB (does NOT fit 16 GB)
- **Purpose:** Gold-standard quality reference. Maximum fidelity.

### Configuration B: mlx-community uniform 4-bit

- **Model:** `mlx-community/Qwen3.6-35B-A3B-4bit` (download from HF)
- **Serving:** `python -m mlx_lm.generate --model mlx-community/Qwen3.6-35B-A3B-4bit`
- **Memory:** ~6-8 GB estimated (fits 16 GB without offload)
- **Purpose:** Off-the-shelf baseline. Uniform 4-bit, no expert offload, no IsoQuant.

### Configuration C: Our mixed-precision (the proof)

- **Model:** `~/Models/qwen3.6-35b-a3b-mixed`
- **Serving:** `python -m mlx_lm.generate --model ~/Models/qwen3.6-35b-a3b-mixed --expert-offload --max-resident-experts 2048 --kv-cache-type isoquant`
- **Memory:** Target < 16 GB with memory cap
- **Purpose:** Demonstrates quality parity with Q8_0, throughput parity with uniform 4-bit, at 16 GB budget.

### Benchmark Matrix

| Metric | Config A (Q8_0) | Config B (4-bit) | Config C (ours) | Notes |
|--------|----------------|-----------------|-----------------|-------|
| Smoke test (12 prompts) | Baseline | ? | Target: match A | Same rubric, same evaluator |
| Peak RSS | ~37 GB | ~6-8 GB | < 16 GB | Measured via `/usr/bin/time -l` |
| Decode tok/s | ~41 tok/s | ? | Target: > 5 tok/s | A vs B/C not comparable (different runtimes) |
| C vs B throughput ratio | — | 1.0x (ref) | Target: > 0.5x of B | Apples-to-apples MLX comparison |
| KV PPL delta @2048 | 0 (no compression) | 0 (no compression) | Target: < 0.01 | |
| Fits 16 GB? | No | Yes | Yes | |
| Expert offload? | N/A (GGUF) | No | Yes (LRU) | |
| IsoQuant KV? | No | No | Yes | |

### Benchmark Procedure

1. Run `eval_quality_gate.py --suite all` on configs B and C (MLX). For config A (Ollama/GGUF), run the same 12 prompts via `ollama run` or the Ollama `/api/chat` endpoint and manually score against the same pass/fail criteria
2. Run `measure_kv_fidelity.py --depths 512,2048` on configs B and C (A is GGUF, different runtime)
3. Run `benchmark_moe_offload.py` on config C for memory and throughput profiling
4. Compare quality scores head-to-head in a summary table

## Acceptance Criteria (16 GB pathway)

- **Smoke test:** 10+/12 on Config C (same rubric as A/B, thinking tags stripped)
- **Quality parity:** Config C score ≥ Config A score - 1, AND no A-pass/C-fail attributable to quantization
- **Peak RSS:** Config C < 16,384 MB measured via `/usr/bin/time -l`
- **Decode throughput:** Config C > 5 tok/s AND Config C ≥ 0.5x Config B tok/s (MLX-to-MLX)
- **KV fidelity:** IsoQuant delta PPL < 0.01 @ 2048
- **Preflight gates:** All four gates (G1-G4) pass before conversion begins

## Output Artifacts

```
~/Models/qwen3.6-35b-a3b-mixed/
  config.json, model.safetensors.index.json, model-*.safetensors,
  repacked-*.safetensors, tokenizer.json, tokenizer_config.json

results/
  qwen36_q8_baseline_quality.json      # Config A
  qwen36_uniform4bit_quality.json      # Config B
  qwen36_mixed_quality.json            # Config C
  qwen36_kv_ppl_depth.json             # IsoQuant fidelity
  qwen36_pathway_benchmark.json        # Memory + throughput
```

## Pathway Checklist Entry

New row in `docs/PATHWAY_PROVEN_CHECKLIST.md`:

| Model | RAM class | Full stack script | Quality JSON | Benchmark JSON | Notes |
|-------|-----------|-------------------|--------------|----------------|-------|
| Qwen3.6-35B-A3B | 16 GB | this spec | `results/qwen36_mixed_quality.json` | `results/qwen36_pathway_benchmark.json` | Three-way comparison: Q8_0 vs uniform-4bit vs mixed-precision |

## Preflight Gates (hard stop/go)

All four gates must pass before conversion begins. If any gate fails, stop and fix before proceeding.

| Gate | Check | Pass condition | How to verify |
|------|-------|----------------|---------------|
| G1: repack_experts arch support | `repack_experts.py` accepts `qwen3_5_moe` | Script completes on a toy model or the model_type guard is widened | `python -m mlx_lm.repack_experts --model <path> --dry-run` (add `--dry-run` flag if needed, or inspect code) |
| G2: Shared expert Q8_0 override | `_build_mixed_expert_quant_predicate()` applies 8-bit to shared expert, 2-bit to routed experts | Inspect quantization config in output `config.json`: shared expert entries show `bits=8`, routed show `bits=2` | Post-conversion: `python -c "import json; c=json.load(open('config.json')); [print(k,v) for k,v in c.get('quantization_config',{}).items()]"` |
| G3: IsoQuant skips DeltaNet layers | `_replace_attention_caches()` in `cache.py` only replaces `KVCache`, not `ArraysCache` | All 30 DeltaNet layers keep `ArraysCache`; only 10 full-attention layers get IsoQuant | Unit test: load model, create cache, assert `isinstance(cache[0], ArraysCache)` for DeltaNet layers |
| G4: Disk budget | BF16 download (~70 GB) + mixed output (~8 GB) + repacked shards fit available disk | `df -h ~` shows ≥ 90 GB free | Check before `huggingface-cli download` |

**Rollback acceptance:** If G2 (shared expert Q8_0) cannot be implemented before conversion, accept all-expert 2-bit as a fallback. Document the quality delta in results. The benchmark still proceeds — the fallback just means shared experts are 2-bit instead of 8-bit, which may show a small quality drop vs the ideal recipe.

## Benchmark Comparability Locks

All benchmark runs MUST use identical generation parameters to ensure fair comparison:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Temperature | 0.0 (greedy) | Eliminates sampling variance |
| Max tokens | 512 | Consistent across all configs |
| Seed | 42 | Reproducibility (where supported) |
| Prompt set | Exactly the 12 prompts from `eval_quality_gate.py --suite all` | Same inputs across all configs |

### Scoring protocol for Config A (Ollama/GGUF)

Config A runs outside MLX, so scoring must be manually aligned:

1. Run each of the 12 prompts via `ollama run qwen3.6-35b-a3b:q8_0` (or the `/api/generate` endpoint with `temperature: 0, seed: 42, num_predict: 512`)
2. Apply the **same rubric** used by `eval_quality_gate.py` — the pass/fail criteria are defined in the script, not ad-hoc
3. Record results in `results/qwen36_q8_baseline_quality.json` using the same schema as the MLX quality JSONs
4. One evaluator scores all three configs in a single session to avoid drift

**"Quality parity"** is defined as: Config C score ≥ Config A score - 1, AND no prompt that passes on A fails on C unless the failure is attributable to quantization (not runtime differences).

## Memory-Cap Protocol

"Fits 16 GB" means peak RSS stays under 16,384 MB during the full benchmark run.

### Measurement method

```bash
# Start generation with memory monitoring
/usr/bin/time -l python -m mlx_lm.generate \
  --model ~/Models/qwen3.6-35b-a3b-mixed \
  --expert-offload --max-resident-experts 2048 \
  --kv-cache-type isoquant \
  --prompt "<benchmark prompt>" \
  --max-tokens 512 2>&1 | grep "maximum resident set size"
```

If peak RSS exceeds 16 GB, reduce `--max-resident-experts` (try 1024, 512) until it fits. Record the final value used.

For continuous monitoring during the full 12-prompt run:

```bash
# In a separate terminal
while true; do ps -o rss= -p $(pgrep -f mlx_lm.generate) 2>/dev/null | awk '{printf "%.0f MB\n", $1/1024}'; sleep 1; done
```

### Throughput comparison caveat

Config A (Ollama/GGUF) and Configs B/C (MLX) use different runtimes with different Metal backends. Throughput numbers (tok/s) are **not directly comparable** across runtimes. The benchmark table reports them side-by-side for reference, but the primary throughput claim is:

- **Config C vs Config B** (both MLX): apples-to-apples comparison showing expert offload + IsoQuant overhead
- **Config A tok/s**: reported for completeness only; not used in pass/fail criteria

## Risks

These risks are tracked separately from the preflight gates. Gates are hard blockers; risks are known issues that may require mitigation during implementation.

1. **repack_experts.py currently validates `model_type=="nemotron_h"`** — needs to be extended for `qwen3_5_moe` or the check removed. The actual repacking logic is architecture-agnostic (it operates on `switch_mlp` keys). **→ Covered by Gate G1.**
2. **Shared expert Q8_0** — `--mixed-expert-bits 2` sets ALL expert weights to 2-bit. The shared expert override may need a per-layer config entry in `config.json` post-conversion, or a code change to `_build_mixed_expert_quant_predicate()`. **→ Covered by Gate G2. Rollback: accept all-expert 2-bit.**
3. **DeltaNet layers have no KV cache** — IsoQuant must skip these 30 layers. The existing `_replace_attention_caches()` in `cache.py` should handle this (it only replaces `KVCache` instances, not `ArraysCache`), but needs verification. **→ Covered by Gate G3.**
4. **BF16 download is ~70 GB** — one-time cost, can be cleaned up after conversion. **→ Covered by Gate G4.**
5. **Qwen3.6 is a thinking model** — `<think>` tags in output. For benchmark scoring, strip `<think>...</think>` blocks before applying the quality rubric. Both Ollama and MLX outputs will contain thinking tokens.
