# Self-Speculative Kimi K2.6 — Pause Handover

**Paused:** 2026-04-29 (session disconnect for office)
**Resume:** later same day (tonight)
**Plan:** `docs/superpowers/plans/2026-04-29-self-speculative-kimi.md`
**Branch:** `main` (per user direction; no worktree)
**Background process at pause:** PID 90552 — Kimi 2-bit quantization

---

## Current state at pause

### Commits landed this session (most recent first)
```
0edbd69 feat(kimi-mla): add trim() to KimiMLAIsoQuantCache         <- Phase 1 / Task 1.2
c596fa1 test(kimi-mla): failing tests for KimiMLAIsoQuantCache.trim()  <- Phase 1 / Task 1.1
16a6b0c evidence(kimi): post-fix A/B + first-principles attribution    <- Pre-plan (prior turn)
40501b2 checkpoint(kimi-mla-isoquant): NPT=16 dispatch + Phase 4-6 evidence  <- Pre-plan (prior turn)
```

### Phase progress
| Phase | Status | Notes |
|---|---|---|
| 1: trim() on KimiMLAIsoQuantCache | ✅ DONE | 16/16 tests pass; spec ✅ + code-quality ✅ |
| 2: Quantize to 2-bit experts | 🟡 IN PROGRESS | Background PID 90552 launched ~start of pause; ETA 30-90 min |
| 3: Quality validation | ⏸️ Pending | Blocked on Phase 2 |
| 4: Speculative smoke + correctness | ⏸️ Pending | Blocked on Phases 2+3 |
| 5: Benchmark sweeps | ⏸️ Pending | Blocked on Phase 4 |
| 6: 1-bit experts (optional) | ⏸️ Pending | Conditional on Phase 5 outcome |
| 7: Document + finalize | ⏸️ Pending | Last |

---

## ON RESUME — first-things-first checklist

Run these in order. Do NOT skip the verification step.

### 1. Verify the workspace state matches this handover
```bash
cd /Users/anthonylui/QwenCoderLocal
pwd  # expect /Users/anthonylui/QwenCoderLocal
git branch --show-current  # expect main
git log --oneline -4  # expect: 0edbd69, c596fa1, 16a6b0c, 40501b2
```

If any of these don't match → STOP. Something changed during the pause. Don't proceed until you understand why.

### 2. Check Phase 2 quantization status

```bash
# Is it still running?
ps -p 90552 -o pid,etime,comm 2>&1 || echo "process exited"

# What does the log say?
tail -30 /tmp/kimi_2bit_convert.log

# Is the output checkpoint there?
ls -la /Volumes/Samsung9904tb/Kimi-K2.6-2bit-experts/ 2>&1 | head -10
du -sh /Volumes/Samsung9904tb/Kimi-K2.6-2bit-experts 2>&1
```

**Three possible outcomes:**
- **Process still running** → wait. Tail the log for hints on progress (shard count, MB/s).
- **Process exited successfully + ~285 GB checkpoint exists** → proceed to Phase 2 / Task 2.5 (smoke load).
- **Process exited unsuccessfully** → read full log: `cat /tmp/kimi_2bit_convert.log | tail -100`. Common failures:
  - OOM during conversion (will say so explicitly)
  - Path predicate didn't match Kimi's MoE paths (output much smaller than expected, ~50 GB or less)
  - Disk filled up (df -h /Volumes/Samsung9904tb to confirm)

### 3. SSD discipline reminder (post-302GB-incident)

Before any operation against `/Volumes/Samsung9904tb`:

```bash
diskutil info /Volumes/Samsung9904tb | grep -E "Volume Name|File System|Disk Size|Mount Point"
```

Confirm `Volume Name: Samsung9904tb` and `File System Personality: APFS` and `Disk Size: 4.0 TB`. If anything mismatches → STOP. Phantom mount risk.

### 4. If Phase 2 succeeded, resume from Task 2.5

Per `docs/superpowers/plans/2026-04-29-self-speculative-kimi.md`:

```bash
cd /Users/anthonylui/QwenCoderLocal
PYTHONPATH=mlx-lm python3 -c "
from mlx_lm import load, generate
model, tokenizer = load(
    '/Volumes/Samsung9904tb/Kimi-K2.6-2bit-experts',
    model_config={'expert_offload': True, 'max_resident_experts': 2000},
)
print('Model loaded.')
out = generate(model, tokenizer, prompt='What is 2+2?', max_tokens=8, verbose=False)
print(f'Output: {out!r}')
"
```

Then commit Phase 2 (Task 2.6), and proceed to Phase 3.

---

## Outstanding infrastructure issues (flagged by Phase 1 implementer)

Surface these to the user when resuming if any block subsequent phases:

1. **`mcp__qwen-coder__code_generation` is down** — local llama.cpp at :8080 is not running. The plan's primary delegation channel for code-gen is bypassed. Fallback used in Phase 1: ollama-via-script then direct write. Same fallback applies to Phase 3 (eval script) and Phase 4 (speculative profile script).

2. **`delegate-to-council` Stage 0 contract validation broken** — Gemini drafter emits invalid JSON for `acceptance_criteria.json` and `feature_list.json`. The council pattern (planned for Task 1.2 review) was unavailable; fell back to test-suite-only correctness gate. Same risk for any future phase that planned council use.

3. **`delegate-to-ollama` and `delegate-to-council` wrappers crash on long task strings** — task string becomes a filename and hits "filename too long". Workaround: pass long tasks via `--context` file argument.

None of these block correctness; they affect orchestration channels. The `~/.claude/CLAUDE.md` and project memory note these scripts as primary delegation tools, so they should be fixed eventually but not before Phase 7.

---

## Plan stop conditions (rehearse before resuming)

From the plan, stop the entire effort and report up if:

1. **Phase 2 quantize fails or produces a clearly wrong-sized output** — investigate before re-running. Don't blindly retry.
2. **Phase 3 quality eval fails the gate** (variant has high-rep outputs OR <20% prefix match against 4-bit reference) — 2-bit may be too aggressive; surface to user before trying 3-bit or different q-mode.
3. **Phase 4 correctness gate fails** (speculative output diverges from greedy character-for-character) — the trim() implementation has a subtle bug despite passing unit tests; surface to user.
4. **Phase 5 best speedup is < 1.2×** — speculative is not winning materially; surface to user before pursuing 1-bit.

---

## Pending TodoWrite tasks (excerpt)

- #11 (in_progress): Self-speculative decode + aggressive expert quantization — parent
- #18 ✅ COMPLETE: Phase 1 trim()
- #12 (in_progress): Phase 2 quantize 2-bit
- #13–#17: Phases 3–7 pending

---

## Quick-reference paths

- Plan: `docs/superpowers/plans/2026-04-29-self-speculative-kimi.md`
- This handover: `docs/superpowers/handovers/2026-04-29-self-speculative-kimi-pause.md`
- Quantize script: `scripts/quantize_kimi_2bit.sh`
- Quantize log: `/tmp/kimi_2bit_convert.log`
- 4-bit source checkpoint: `/Volumes/Samsung9904tb/Kimi-K2.6/` (554 GB, untouched)
- 2-bit destination: `/Volumes/Samsung9904tb/Kimi-K2.6-2bit-experts/` (in flight)
- Test file: `mlx-lm/tests/test_kimi_mla_isoquant_dkv.py`
- Cache file: `mlx-lm/mlx_lm/models/kimi_mla_isoquant_dkv.py`

---

## What NOT to do on resume

- Do NOT `rm -rf /Volumes/Samsung9904tb/Kimi-K2.6-2bit-experts/` if quantization completed but seems "wrong" — inspect first, ask user before deleting
- Do NOT kill PID 90552 unless you've confirmed it's actually stuck (no log activity for >15 min)
- Do NOT switch off `main` to a worktree mid-stream — user's choice was main
- Do NOT skip the spec/code-quality review for any task; the discipline matters
