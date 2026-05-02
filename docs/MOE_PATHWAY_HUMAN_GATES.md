# MoE pathway — tasks that **cannot** be closed by code alone

This complements `apex_positioning_and_roadmap` (plan file — do not edit). Use it to track **human / hardware / weights** gates.

| Plan todo | Blocker | What to do |
| --------- | ------- | ---------- |
| `prereq-disk-kimi` | Human confirms ~2.1 TB free + staging on the volume used for HF cache + convert scratch | `bash scripts/check_kimi_disk_prereq.sh` (soft check) + operator sign-off |
| `pathway-proven` | Measured full stack on **target RAM class** for Qwen + Gemma + Nemotron | Fill `docs/PATHWAY_PROVEN_CHECKLIST.md` with pinned JSON paths |
| `nemotron-32gb-validation` | Physical 32 GB-class host (or capped VM) | `docs/NEMOTRON_PATHWAY_32GB.md` + `scripts/run_nemotron_pathway_full_stack.sh` |
| `kimi-light-quant`, `kimi-mla-isoquant-validation`, `kimi-stability-soak` | Kimi K2.5 weights + 128 GB-class machine | After `pathway-proven` + disk gate |
| `phase7-ssd-codegen-lora` | Training run (optional) | See `docs/PHASE7_SSD_OUTLINE.md` |
| `qes-routing-optimisation` | Stable stack + DedeKimi logs + offline batch | Spec only until Phase 5a gates pass |

## Runbooks (commands + scripts)

- **Qwen3 full stack:** `docs/QWEN3_FULL_STACK.md` — `scripts/run_qwen3_full_stack.sh`
- **Nemotron 32 GB pathway:** `docs/NEMOTRON_PATHWAY_32GB.md` — `scripts/run_nemotron_pathway_full_stack.sh`

## Automatable measurement (local)

- **IsoQuant vs TurboQuant (3-bit default):** `python scripts/compare_pathway_kv_modes.py --model <mlx_dir> --output results/<name>.json --expert-offload`
- **Single mode:** `python scripts/benchmark_moe_offload.py --model <mlx_dir> --kv-cache-type isoquant --json-output results/foo.json --expert-offload`
- **Bits:** pass `--turboquant-bits 3` (sets `TURBOQUANT_BITS`).

## Qwen3 KV attention

TurboQuant/IsoQuant require full KV reconstruction in attention (`qwen3_moe.py`). See `results/QWEN3_KV_RECONSTRUCTION_FIX.md` if quality collapses.
