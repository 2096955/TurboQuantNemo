# Plan todos that stay **open** until external gates clear

Per `AGENTS.md`, **measured** artifacts or **human** confirmation are required — documentation alone does not close these.

| Todo theme | Blocker | Repo support |
|------------|---------|--------------|
| `prereq-disk-kimi` | Operator confirms ~2.1 TB + staging policy | `scripts/check_kimi_disk_prereq.sh` (soft `df` check) |
| `pathway-proven` | Pinned JSON on 16 GB / 32 GB-class hosts for Qwen + Gemma + Nemotron | `docs/PATHWAY_PROVEN_CHECKLIST.md` |
| `nemotron-32gb-validation` | Physical 32 GB-class machine | `docs/NEMOTRON_PATHWAY_32GB.md`, `scripts/run_nemotron_pathway_full_stack.sh` |
| `phase1-16gb-validation` | 16 GB-class repeat for Gemma/Qwen | Same runbooks; pin `results/*.json` |
| `kimi-light-quant`, `kimi-mla-isoquant-validation`, `kimi-stability-soak`, `kimi-attnres-forward`, `kimi-128gb-turboquant-ablation` | Kimi weights + 128 GB-class hardware | `mlx_lm/models/kimi_mla_isoquant_dkv.py` (DKV placeholder), plan Appendix H |
| `phase7-ssd-codegen-lora` | Training / SFT run | `docs/PHASE7_SSD_OUTLINE.md` |
| `qes-routing-optimisation` | Phase 5a offline batch + stable stack | Spec / external QES reference only |

Closing any row above requires **artifacts or human sign-off**, not additional markdown claims.
