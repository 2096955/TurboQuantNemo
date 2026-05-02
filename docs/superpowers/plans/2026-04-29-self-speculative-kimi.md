# Self-Speculative Kimi K2.6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Materially improve Kimi K2.6 decode throughput on M4 Max 128 GB beyond the current 0.42 tok/s ceiling by combining (a) sub-bit quantization of routed experts to enable a viable in-RAM draft model and (b) self-speculative decoding via the existing `speculative_generate_step` infrastructure.

**Architecture:** Quantize Kimi K2.6 routed experts to 2-bit (and optionally 1-bit) using the existing `mlx_lm.convert --mixed-expert-bits` tooling. Add a missing `trim()` method to `KimiMLAIsoQuantCache` (the only blocker for speculative decoding's cache rewind path). Wire the existing `speculative_generate_step` with the 4-bit Kimi as target and the 2-bit Kimi as draft. Measure end-to-end throughput and acceptance rate. Push to 1-bit if 2-bit acceptance is too low.

**Tech Stack:** Python 3.11, MLX, mlx-lm fork (this repo), pytest. Target hardware: Apple M4 Max 128 GB. Source checkpoint: `/Volumes/Samsung9904tb/Kimi-K2.6` (554 GB, already 4-bit pack-quantized for routed experts).

**Pre-existing infrastructure (verified):**
- `mlx_lm.convert` supports `--mixed-expert-bits 1..8` via `_build_mixed_expert_quant_predicate` (mlx-lm/mlx_lm/convert.py:102).
- `speculative_generate_step` is implemented at mlx-lm/mlx_lm/generate.py:530 and uses `cache.trim_prompt_cache()` for the rewind path (line 642-644).
- `IsoQuantKVCache.trim(n)` is implemented at mlx-lm/mlx_lm/models/mlx_isoquant.py:1702 and handles packed-cache invalidation correctly.
- `cache.trim_prompt_cache()` walks every cache and calls `.trim(n)` (cache.py:279).

**The single missing blocker:** `KimiMLAIsoQuantCache` (mlx-lm/mlx_lm/models/kimi_mla_isoquant_dkv.py) has no `trim()` method, so any speculative decode that rejects a draft token would leave the MLA cache state out of sync with the accepted token sequence.

**File structure (what this plan creates / modifies):**
- Modify: `mlx-lm/mlx_lm/models/kimi_mla_isoquant_dkv.py` — add `trim()` method
- Modify: `mlx-lm/tests/test_kimi_mla_isoquant_dkv.py` — add trim tests
- Create: `scripts/quantize_kimi_2bit.sh` — wrapper around `mlx_lm.convert` with the right flags for Kimi
- Create: `scripts/eval_kimi_quality.py` — quality eval harness reusing the Phase 4 prompts (currently inline in the prior agent session)
- Create: `scripts/profile_kimi_speculative.py` — speculative-decode A/B harness (target-only vs target+draft)
- Create: `artifacts/kimi_k26_speculative/` — output dir for speculative artifacts
- Modify: `docs/KIMI_K26_FULL_STACK.md` — document new draft model + speculative path

**Stop conditions:**
- After Phase 2 (cache trim + tests): if you cannot get the trim tests to pass, STOP and surface the blocker. Speculative decode cannot work without trim.
- After Phase 4 (speculative smoke test): if the speculative output diverges from greedy output at temperature=0, STOP. Do not proceed to benchmarking until correctness is established.
- After Phase 5 (initial benchmark): if speedup is negative or the acceptance rate is < 30%, STOP and report. Do not push to 1-bit (Phase 7) until you understand why 2-bit isn't working.

**What this plan does NOT do:**
- Does not build new fused kernels (NPT=16 tiled is deferred to a separate plan; first-principles math says it's not the bottleneck at T<8K).
- Does not wire the `AttnResExpertPredictor` (separate plan).
- Does not optimize `prepare_gather_triple_quantized` Python overhead (separate plan; speculative amortizes it across K tokens, which may be enough).

---

## Execution & Delegation Strategy

This plan assumes execution by Claude main as the orchestrator (CEO role per CLAUDE.md), with explicit delegation to local models for the work where they're cheaper or better. The orchestrator owns: TDD discipline (test design, gate decisions), commits, scope decisions, and cross-task synthesis. Everything else can and should be delegated.

**Per-task delegation matrix:**

| Task | Primary executor | Why | Fallback |
|------|------------------|-----|----------|
| 1.1 (write trim tests) | Claude subagent | Test design needs TDD discipline + understanding of MLA cache semantics | — |
| 1.2 (implement trim) | `mcp__qwen-coder__code_generation` with the failing tests as spec | Pure code-gen against well-specified tests; Qwen3-Coder-Next is the local specialist | `delegate-to-ollama --model qwen2.5-coder:32b-instruct-q4_K_M` |
| 1.2 (review the implementation, before commit) | `delegate-to-council "<task>" --max-rounds 2` | trim() has subtle off-by-one risks (PE buffer, packed cache) — council pattern (Gemini drafts → Ollama critiques → Claude ARB) catches them | Claude main solo review |
| 1.3 (run test suite) | Claude subagent | Needs Bash + result interpretation | — |
| 1.4 (commit) | Claude main | Scope + message decisions | — |
| 2.1, 2.3 (preflight + dry-run) | Claude subagent | Quick verification with Bash | — |
| 2.2 (write quantize wrapper) | `mcp__qwen-coder__code_generation` | Bash boilerplate; well-specified | `delegate-to-ollama` |
| 2.4, 2.5 (run quantize, smoke load) | Claude subagent (background) | Multi-hour Bash; subagent launches and reports back | — |
| 2.6 (commit) | Claude main | — | — |
| 3.1 (write quality eval script) | `mcp__qwen-coder__code_generation` | Python with clear spec (PHASE_4_PROMPTS, repetition_ratio, prefix_match_chars functions) | `delegate-to-ollama` |
| 3.2 (run eval) | Claude subagent (background) | Multi-hour wait; subagent reports completion | — |
| 3.3 (commit) | Claude main | — | — |
| 4.1 (speculative profile script) | `mcp__qwen-coder__code_generation` with the existing speculative_generate_step API as spec | Python with clear interface | `delegate-to-ollama` |
| 4.2, 4.3 (smoke + correctness) | Claude subagent (background) | Multi-hour; subagent runs + reports | — |
| 4.3 (correctness gate analysis if it FAILS) | `delegate-to-council "<failure debugging>"` | Multi-model debug for cache-state divergence is high-stakes | Claude main + `mcp__qwen-coder__debug_assistance` |
| 4.4 (commit) | Claude main | — | — |
| 5.1, 5.2 (sweeps) | Claude subagent (background, sequential) | Multi-hour bash loops | — |
| 5.3 (decision point) | Claude main | Cross-artifact synthesis decision | — |
| 5.4 (commit) | Claude main | — | — |
| 6.1 (1-bit quantize) | Claude subagent (background) | Same shape as 2.4 | — |
| 6.2, 6.3 (eval + speculative) | Claude subagent (background) | Multi-hour | — |
| 6.4 (commit) | Claude main | — | — |
| 7.1 (doc update) | Claude main | Cross-artifact synthesis | — |
| 7.2 (final commit) | Claude main | — | — |

**Session memory tracking (every task):**

Every coding task should bracket its work with session-memory MCP calls:

```
mcp__session-memory__start_task(task_type="kimi-spec-phase-N", task_description="...")
# ... do the work ...
mcp__session-memory__log_step_fast(session_id, tokens=N, latency_ms=M)
mcp__session-memory__finish_task(session_id, outcome="success"|"failure"|"partial")
```

This populates the local SQLite memory and feeds future strategy lookups via `mcp__session-memory__quick_strategy_check(task_type="kimi-spec-*")`.

**Token efficiency:**
- The orchestrator should run all coding work via the qwen-coder MCP path so the actual code synthesis tokens cost zero on the Claude API.
- Use `compress_handover` (handover-compression-mcp) to compress any large benchmark output before quoting back to the user.
- RTK hook is active; raw bash output is auto-compacted by the hook layer.

**Council-pattern usage (strict):**
- Reserve `delegate-to-council` for: Task 1.2 review, Task 4.3 debug-on-failure, Phase 5.3 final speedup interpretation. The council adds latency (multi-round) so don't use it for code-gen — use single-shot Qwen for that.
- Default council settings: `--drafter gemini --critic qwen2.5-coder:32b --max-rounds 2`.

**Subagent dispatch (REQUIRED SUB-SKILL):**

When the orchestrator dispatches a Claude subagent for a task, the subagent prompt MUST:
1. Quote the relevant task header from this plan (so the subagent has the spec)
2. Include explicit delegation instruction (e.g., "use mcp__qwen-coder__code_generation for the implementation; use Bash to run tests")
3. Demand a specific reportable outcome (e.g., "commit hash + test pass count + new artifact path")
4. Set a context budget (e.g., "report in under 300 words")

This pattern is documented in superpowers:subagent-driven-development.

**Failure / fallback policy:**
- If `mcp__qwen-coder__code_generation` returns garbage or unrelated code: fall back to `delegate-to-ollama --model qwen2.5-coder:32b-instruct-q4_K_M --context <relevant-files>`. If that also fails: Claude main does it directly and notes the local-model failure in the task's commit message.
- If a Claude subagent gets stuck (no progress in 2 turns): orchestrator intervenes with the `codex:codex-rescue` or `gemini:gemini-rescue` skill.
- If the qwen-coder MCP server is down (the local llama.cpp at :8080 is not running): all qwen-coder calls fall through to delegate-to-ollama transparently.

---

## Phase 1: Add `trim()` to `KimiMLAIsoQuantCache`

### Task 1.1: Add the failing trim test

**Files:**
- Modify: `mlx-lm/tests/test_kimi_mla_isoquant_dkv.py`

- [ ] **Step 1: Add the test class at the end of the file**

Read the existing file first to find the right insertion point and helper imports:

```bash
grep -n "^class\|^def\|^import\|^from" /Users/anthonylui/QwenCoderLocal/mlx-lm/tests/test_kimi_mla_isoquant_dkv.py | head -30
```

Append this new test class (use the same imports the file already has — check the file's top before adding):

```python
class TestTrim(unittest.TestCase):
    """Trim must restore the cache to the state it was in N tokens ago.

    Required for speculative_generate_step's cache rewind path. When the
    target rejects K-1 of K draft tokens, the cache must be trimmed back
    to the accepted prefix.
    """

    def _build_with_n_tokens(self, n: int) -> KimiMLAIsoQuantCache:
        cb = get_default_codebook_dir()
        cache = KimiMLAIsoQuantCache(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            bit_width=3,
            layer_idx=0,
            codebook_dir=cb,
        )
        rng = np.random.default_rng(42)
        for _ in range(n):
            kv_latent = mx.array(rng.normal(size=(1, 1, 1, 512)).astype(np.float32))
            k_pe = mx.array(rng.normal(size=(1, 1, 1, 64)).astype(np.float32))
            cache.update_and_fetch(kv_latent, k_pe)
        cache.finalize_deferred_prefill()
        return cache

    def test_trim_reduces_offset(self):
        cache = self._build_with_n_tokens(8)
        self.assertEqual(cache.offset, 8)
        cache.trim(3)
        self.assertEqual(cache.offset, 5)

    def test_trim_to_zero_clears_state(self):
        cache = self._build_with_n_tokens(4)
        cache.trim(4)
        self.assertEqual(cache.offset, 0)

    def test_trim_then_extend_matches_direct_build(self):
        """Build to T=8, trim to T=5, append 3 more, compare against direct build to T=8."""
        ref = self._build_with_n_tokens(8)
        ref_lat, ref_pe = ref.update_and_fetch(
            mx.zeros((1, 1, 0, 512)), mx.zeros((1, 1, 0, 64))
        )

        rng = np.random.default_rng(42)
        cache = KimiMLAIsoQuantCache(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            bit_width=3,
            layer_idx=0,
            codebook_dir=get_default_codebook_dir(),
        )
        # Push 8 tokens
        for _ in range(8):
            kv_latent = mx.array(rng.normal(size=(1, 1, 1, 512)).astype(np.float32))
            k_pe = mx.array(rng.normal(size=(1, 1, 1, 64)).astype(np.float32))
            cache.update_and_fetch(kv_latent, k_pe)
        cache.finalize_deferred_prefill()

        # Trim back to T=5 then re-append the same last 3 tokens
        cache.trim(3)
        self.assertEqual(cache.offset, 5)

        # Re-append by recreating the rng to reproduce the same last 3 tokens
        rng2 = np.random.default_rng(42)
        # Skip the first 5 tokens
        for _ in range(5):
            _ = mx.array(rng2.normal(size=(1, 1, 1, 512)).astype(np.float32))
            _ = mx.array(rng2.normal(size=(1, 1, 1, 64)).astype(np.float32))
        # Now append tokens 6, 7, 8
        for _ in range(3):
            kv_latent = mx.array(rng2.normal(size=(1, 1, 1, 512)).astype(np.float32))
            k_pe = mx.array(rng2.normal(size=(1, 1, 1, 64)).astype(np.float32))
            cache.update_and_fetch(kv_latent, k_pe)

        self.assertEqual(cache.offset, 8)

        # Compare full-context PE buffer shapes (sanity)
        new_lat, new_pe = cache.update_and_fetch(
            mx.zeros((1, 1, 0, 512)), mx.zeros((1, 1, 0, 64))
        )
        self.assertEqual(new_lat.shape, ref_lat.shape)
        self.assertEqual(new_pe.shape, ref_pe.shape)

    def test_trim_invalidates_packed_cache(self):
        cache = self._build_with_n_tokens(8)
        # Force a fused call to populate _packed_latent_cache
        q = mx.array(np.random.randn(1, 64, 1, 512).astype(np.float32))
        pe = mx.array(np.random.randn(1, 64, 1, 8).astype(np.float32))
        scale = 1.0 / (512 ** 0.5)
        _ = cache.fused_latent_attention(q, pe, scale)
        self.assertIsNotNone(cache._packed_latent_cache)
        packed_T_before = cache._packed_latent_cache.shape[1]
        cache.trim(3)
        # After trim, packed cache must be either invalidated (None) or
        # have shape consistent with new offset
        if cache._packed_latent_cache is not None:
            self.assertEqual(cache._packed_latent_cache.shape[1], cache.offset)
        # Either way, offset must reflect the trim
        self.assertEqual(cache.offset, 5)

    def test_trim_zero_is_noop(self):
        cache = self._build_with_n_tokens(4)
        cache.trim(0)
        self.assertEqual(cache.offset, 4)

    def test_trim_more_than_offset_raises(self):
        cache = self._build_with_n_tokens(4)
        with self.assertRaises((ValueError, AssertionError)):
            cache.trim(5)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /Users/anthonylui/QwenCoderLocal
PYTHONPATH=mlx-lm python3 -m pytest mlx-lm/tests/test_kimi_mla_isoquant_dkv.py::TestTrim -v
```

Expected: 6 FAIL with `AttributeError: 'KimiMLAIsoQuantCache' object has no attribute 'trim'`

If tests pass: stop and re-read the file. The trim method may already exist and the plan's premise is wrong.

### Task 1.2: Implement `trim()` on `KimiMLAIsoQuantCache`

**Files:**
- Modify: `mlx-lm/mlx_lm/models/kimi_mla_isoquant_dkv.py`

- [ ] **Step 1: Read the file to find where to insert**

```bash
grep -n "def \|self.offset\|self._compressed_latent\|self._pe_buffer\|self._packed_latent_cache" /Users/anthonylui/QwenCoderLocal/mlx-lm/mlx_lm/models/kimi_mla_isoquant_dkv.py
```

Identify all state that depends on the seq dim:
- `self.offset` — number of cached tokens
- `self._compressed_latent` — dict[str, mx.array], each shape `(1, T, ...)` along axis=1
- `self._pe_buffer` — mx.array shape `(1, T, 64)` along axis=1
- `self._packed_latent_cache` — mx.array shape `(1, T, 192)` along axis=1 (or None)

- [ ] **Step 2: Add the trim method**

Insert after the `finalize_deferred_prefill` method (around line 165 — confirm by reading):

```python
    def trim(self, n: int) -> int:
        """Trim the last n tokens from the cache.

        Required for speculative_generate_step. When the target model
        rejects K-1 of K draft tokens, the cache must be rewound to
        the accepted prefix length.

        Args:
            n: Number of tokens to remove from the end. Must be 0 <= n <= offset.

        Returns:
            The new offset after trimming.
        """
        if n == 0:
            return self.offset
        if n < 0:
            raise ValueError(f"trim(n) requires n >= 0, got {n}")
        if n > self.offset:
            raise ValueError(
                f"trim({n}) exceeds current offset {self.offset}"
            )

        new_T = self.offset - n

        # Trim _compressed_latent (dict of arrays, each shape (1, T, ...) on axis 1)
        if self._compressed_latent is not None:
            self._compressed_latent = {
                k: v[:, :new_T, :] for k, v in self._compressed_latent.items()
            }

        # Trim _pe_buffer (shape (1, T, 64))
        if self._pe_buffer is not None:
            self._pe_buffer = self._pe_buffer[:, :new_T, :]

        # Trim _packed_latent_cache (shape (1, T, 192)) — invalidate is also valid
        # but trimming preserves the work done so far.
        if self._packed_latent_cache is not None:
            self._packed_latent_cache = self._packed_latent_cache[:, :new_T, :]

        self.offset = new_T
        return self.offset
```

- [ ] **Step 3: Run the trim tests to verify they pass**

```bash
cd /Users/anthonylui/QwenCoderLocal
PYTHONPATH=mlx-lm python3 -m pytest mlx-lm/tests/test_kimi_mla_isoquant_dkv.py::TestTrim -v
```

Expected: 6 PASS.

If `test_trim_then_extend_matches_direct_build` fails on shape mismatch: the issue is likely that `_packed_latent_cache` got out of sync. Check with:

```bash
PYTHONPATH=mlx-lm python3 -c "
from mlx_lm.models.mlx_turboquant import get_default_codebook_dir
from mlx_lm.models.kimi_mla_isoquant_dkv import KimiMLAIsoQuantCache
import mlx.core as mx, numpy as np
c = KimiMLAIsoQuantCache(kv_lora_rank=512, qk_rope_head_dim=64, bit_width=3, layer_idx=0, codebook_dir=get_default_codebook_dir())
rng = np.random.default_rng(42)
for _ in range(8):
    c.update_and_fetch(mx.array(rng.normal(size=(1,1,1,512)).astype(np.float32)), mx.array(rng.normal(size=(1,1,1,64)).astype(np.float32)))
c.finalize_deferred_prefill()
print('before trim: offset=', c.offset, 'packed=', c._packed_latent_cache.shape if c._packed_latent_cache is not None else None)
c.trim(3)
print('after trim: offset=', c.offset, 'packed=', c._packed_latent_cache.shape if c._packed_latent_cache is not None else None, 'pe=', c._pe_buffer.shape)
print('compressed_latent shapes:', {k: v.shape for k, v in c._compressed_latent.items()})
"
```

Expected output: `offset=5, packed=(1,5,192), pe=(1,5,64), compressed_latent values all have second dim = 5`.

### Task 1.3: Run the full Kimi MLA test suite to verify no regression

- [ ] **Step 1: Run full test file**

```bash
cd /Users/anthonylui/QwenCoderLocal
PYTHONPATH=mlx-lm python3 -m pytest mlx-lm/tests/test_kimi_mla_isoquant_dkv.py -v --timeout=60
```

Expected: 16 PASS (10 prior + 6 new).

If any prior test fails: revert Task 1.2 and investigate. The trim method should have zero impact on tests that don't call trim.

### Task 1.4: Commit Phase 1

- [ ] **Step 1: Stage and commit**

```bash
cd /Users/anthonylui/QwenCoderLocal
git add mlx-lm/mlx_lm/models/kimi_mla_isoquant_dkv.py mlx-lm/tests/test_kimi_mla_isoquant_dkv.py
git commit -m "feat(kimi-mla): add trim() to KimiMLAIsoQuantCache

Required for speculative_generate_step's cache rewind path.
When the target rejects K-1 of K draft tokens, the cache must
be trimmed back to the accepted prefix length.

Trims _compressed_latent (dict of arrays), _pe_buffer, and
_packed_latent_cache along the seq dimension (axis=1). Updates
offset accordingly. Raises ValueError on out-of-range n.

6 new tests cover: offset reduction, full clear, trim-then-extend
shape consistency, packed cache invalidation, no-op zero, and
out-of-range error handling. All 16 KimiMLAIsoQuantCache tests pass."
```

---

## Phase 2: Quantize Kimi K2.6 routed experts to 2-bit

### Task 2.1: Pre-flight — confirm storage and existing tooling

- [ ] **Step 1: Verify SSD has space for the 2-bit checkpoint**

Estimated size: 4-bit Kimi is 554 GB; 2-bit experts will be roughly 285 GB. Need ~300 GB free.

```bash
df -h /Volumes/Samsung9904tb
```

Expected: Available > 350 GB. If less, STOP — choose a different output volume.

- [ ] **Step 2: Confirm `--mixed-expert-bits` predicate routes Kimi paths correctly**

Read the predicate source:

```bash
sed -n '120,140p' /Users/anthonylui/QwenCoderLocal/mlx-lm/mlx_lm/convert.py
```

Verify the routed-expert predicate matches `.switch_mlp.gate_proj`, `.switch_mlp.up_proj`, `.switch_mlp.down_proj`. These are the Kimi MoE layer weight paths.

If Kimi uses a different path naming (check via reading `mlx-lm/mlx_lm/models/deepseek_v3.py:264` `self.switch_mlp = SwitchGLU(...)`), the predicate may not match. Test with a tiny dry-run convert in Step 4.

- [ ] **Step 3: Check whether the source checkpoint is in HF format that `mlx_lm.convert` can read**

```bash
ls /Volumes/Samsung9904tb/Kimi-K2.6/ | head -10
cat /Volumes/Samsung9904tb/Kimi-K2.6/config.json | python3 -m json.tool | head -20
```

Expected: `config.json` shows `architectures: ["KimiK25ForConditionalGeneration"]` or similar with `text_config.model_type: "kimi_k25"`. Safetensors shards present.

If the checkpoint is already MLX-converted (not HF), convert may need a different invocation. Inspect the safetensors index for any MLX-specific markers.

### Task 2.2: Write the quantize wrapper script

**Files:**
- Create: `scripts/quantize_kimi_2bit.sh`

- [ ] **Step 1: Create the script**

```bash
cat > /Users/anthonylui/QwenCoderLocal/scripts/quantize_kimi_2bit.sh <<'EOF'
#!/usr/bin/env bash
# Quantize Kimi K2.6 with 2-bit routed experts (4-bit shared/dense/attn).
# Output goes to /Volumes/Samsung9904tb/Kimi-K2.6-2bit-experts/
#
# Per CLAUDE.md: "routed experts at 4-bit INT4 group-32; attention/shared/
# dense are unquantized BF16" is the current default. This script overrides
# routed experts to 2-bit while keeping the rest at the source precision.
#
# Estimated wall time: 30-90 minutes depending on disk speed.
# Estimated output size: ~285 GB.

set -euo pipefail

SRC="${KIMI_SRC:-/Volumes/Samsung9904tb/Kimi-K2.6}"
DST="${KIMI_DST:-/Volumes/Samsung9904tb/Kimi-K2.6-2bit-experts}"

if [ ! -d "$SRC" ]; then
    echo "ERROR: source not found: $SRC" >&2
    exit 1
fi

if [ -d "$DST" ]; then
    echo "ERROR: destination already exists: $DST"
    echo "  Remove it first if you want to re-quantize:"
    echo "    rm -rf '$DST'"
    exit 1
fi

# Disk space check — need ~300 GB
AVAIL_KB=$(df -k "$(dirname "$DST")" | awk 'NR==2 {print $4}')
AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
if [ "$AVAIL_GB" -lt 350 ]; then
    echo "ERROR: only ${AVAIL_GB} GB free at $(dirname "$DST"); need >= 350 GB" >&2
    exit 1
fi

cd "$(dirname "$0")/.."

PYTHONPATH=mlx-lm python3 -m mlx_lm convert \
    --hf-path "$SRC" \
    --mlx-path "$DST" \
    --quantize \
    --q-bits 4 \
    --q-group-size 64 \
    --q-mode affine \
    --mixed-expert-bits 2 \
    --shared-expert-bits 4 \
    --trust-remote-code

echo
echo "Quantization complete: $DST"
du -sh "$DST"
EOF

chmod +x /Users/anthonylui/QwenCoderLocal/scripts/quantize_kimi_2bit.sh
```

- [ ] **Step 2: Sanity check the script syntax**

```bash
bash -n /Users/anthonylui/QwenCoderLocal/scripts/quantize_kimi_2bit.sh && echo OK
```

Expected: `OK`.

### Task 2.3: Dry-run convert with `--help` to confirm flags work

- [ ] **Step 1: Verify `--mixed-expert-bits` is recognized**

```bash
cd /Users/anthonylui/QwenCoderLocal
PYTHONPATH=mlx-lm python3 -m mlx_lm convert --help 2>&1 | grep -E "mixed-expert-bits|shared-expert-bits|q-bits"
```

Expected: all three flags appear in the help output.

### Task 2.4: Run the quantization

- [ ] **Step 1: Kick off quantization in the background**

```bash
cd /Users/anthonylui/QwenCoderLocal
nohup bash scripts/quantize_kimi_2bit.sh > /tmp/kimi_2bit_convert.log 2>&1 &
echo "PID: $!"
```

This will take 30-90 minutes. Do not wait synchronously; check periodically with:

```bash
tail -20 /tmp/kimi_2bit_convert.log
df -h /Volumes/Samsung9904tb
```

- [ ] **Step 2: Verify completion and output size**

When the process exits:

```bash
ls /Volumes/Samsung9904tb/Kimi-K2.6-2bit-experts/ | head -5
du -sh /Volumes/Samsung9904tb/Kimi-K2.6-2bit-experts/
cat /Volumes/Samsung9904tb/Kimi-K2.6-2bit-experts/config.json | python3 -m json.tool | grep -E "num_bits|group_size|format" | head -10
```

Expected:
- Directory exists with safetensors shards
- Total size ~280-300 GB (down from 554 GB)
- `quantization_config` shows `num_bits: 2` for routed experts and `num_bits: 4` for shared

If output is larger than ~320 GB or smaller than ~250 GB: investigate. The mixed predicate may not have matched routed-expert paths correctly.

If the convert errors out: capture the full log and STOP. Common failure modes:
- OOM during conversion (try in chunks; check `mlx_lm.convert` for streaming)
- Path predicate doesn't match (read the actual paths via `python3 -c "from safetensors import safe_open; print(list(safe_open('$SRC/model-00001-of-000064.safetensors').keys())[:30])"`)

### Task 2.5: Smoke test that the 2-bit checkpoint loads

- [ ] **Step 1: Load and run a 1-token generation**

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

Expected: model loads without error, generates 8 tokens. Output should at minimum be valid UTF-8 text. Quality at this point is informational only; Phase 3 measures it properly.

If load fails on missing keys or shape mismatch: the convert may not have produced the right index. Inspect `model.safetensors.index.json` and compare to the source.

### Task 2.6: Commit the quantize script + checkpoint reference

- [ ] **Step 1: Commit the script (the checkpoint itself stays on the SSD, not in git)**

```bash
cd /Users/anthonylui/QwenCoderLocal
git add scripts/quantize_kimi_2bit.sh
git commit -m "feat(scripts): quantize Kimi K2.6 routed experts to 2-bit

Wrapper around mlx_lm.convert with --mixed-expert-bits 2
--shared-expert-bits 4. Produces a 2-bit-experts variant of
Kimi K2.6 at /Volumes/Samsung9904tb/Kimi-K2.6-2bit-experts/
(~285 GB, down from 554 GB at 4-bit).

Includes pre-flight disk-space check and refuses to overwrite
an existing destination, per the workspace's no-bullshit-after-
the-302GB-incident rule.

Intended use: draft model for self-speculative decoding
against the 4-bit Kimi target."
```

---

## Phase 3: Quality validation harness

### Task 3.1: Write the quality-eval script

**Files:**
- Create: `scripts/eval_kimi_quality.py`

- [ ] **Step 1: Create the script**

This reuses the Phase 4 prompt set the prior session used inline. The script must:
- Load both 4-bit and 2-bit Kimi (one at a time to fit in memory)
- Generate the same N tokens at temperature=0 (greedy) for each prompt
- Compare outputs at three levels: exact match, prefix-match length, and a coarse "factually correct + no repetition" gate

```python
#!/usr/bin/env python3
"""Quality eval for Kimi quantization variants.

Uses the Phase 4 prompts (3 simple factual prompts) plus a longer-form
generation, both at temperature=0 (greedy) for deterministic comparison.

Outputs a JSON artifact with per-prompt outputs, prefix-match lengths,
and a coarse pass/fail gate.

Usage:
    python3 scripts/eval_kimi_quality.py \\
        --reference /Volumes/Samsung9904tb/Kimi-K2.6 \\
        --variant /Volumes/Samsung9904tb/Kimi-K2.6-2bit-experts \\
        --max-tokens 64 \\
        --output artifacts/kimi_k26_speculative/quality_2bit_vs_4bit.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MLX_LM_ROOT = REPO_ROOT / "mlx-lm"
if MLX_LM_ROOT.exists() and str(MLX_LM_ROOT) not in sys.path:
    sys.path.insert(0, str(MLX_LM_ROOT))


PHASE_4_PROMPTS = [
    "What is 2+2?",
    "The capital of France is",
    "Explain the concept of attention in transformers in one sentence.",
]


def repetition_ratio(text: str) -> float:
    """Bigram repetition ratio. >0.3 indicates collapse."""
    words = text.lower().split()
    if len(words) < 10:
        return 0.0
    bigrams = [f"{words[j]} {words[j+1]}" for j in range(len(words) - 1)]
    return 1.0 - len(set(bigrams)) / max(len(bigrams), 1)


def run_one(model_path: str, max_tokens: int, max_resident_experts: int) -> list[dict]:
    """Load model, run all prompts, return per-prompt outputs."""
    from mlx_lm import load, generate

    print(f"  Loading {model_path}...", flush=True)
    t0 = time.time()
    model, tokenizer = load(
        model_path,
        model_config={
            "expert_offload": True,
            "max_resident_experts": max_resident_experts,
        },
    )
    load_s = time.time() - t0
    print(f"  Loaded in {load_s:.1f}s", flush=True)

    results = []
    for prompt in PHASE_4_PROMPTS:
        t0 = time.time()
        # Greedy generation (temperature=0 implied by default sampler)
        out = generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False
        )
        elapsed = time.time() - t0
        results.append(
            {
                "prompt": prompt,
                "output": out,
                "elapsed_s": round(elapsed, 2),
                "char_count": len(out),
                "repetition_ratio": round(repetition_ratio(out), 4),
            }
        )
        print(
            f"  [{prompt[:40]}...] {elapsed:.1f}s, "
            f"{len(out)} chars, rep={results[-1]['repetition_ratio']}",
            flush=True,
        )

    # Free model before loading the next one
    del model, tokenizer
    gc.collect()
    return results


def prefix_match_chars(a: str, b: str) -> int:
    """Length of the longest matching prefix in characters."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True, help="4-bit reference Kimi path")
    p.add_argument("--variant", required=True, help="Quantized variant Kimi path")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--max-resident-experts", type=int, default=2000)
    p.add_argument(
        "--output",
        default="artifacts/kimi_k26_speculative/quality_eval.json",
    )
    args = p.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Reference: {args.reference} ===")
    ref_results = run_one(args.reference, args.max_tokens, args.max_resident_experts)

    print(f"\n=== Variant: {args.variant} ===")
    var_results = run_one(args.variant, args.max_tokens, args.max_resident_experts)

    # Compare
    comparisons = []
    for ref, var in zip(ref_results, var_results):
        prefix = prefix_match_chars(ref["output"], var["output"])
        comparisons.append(
            {
                "prompt": ref["prompt"],
                "ref_chars": ref["char_count"],
                "var_chars": var["char_count"],
                "prefix_match_chars": prefix,
                "exact_match": ref["output"] == var["output"],
                "ref_repetition": ref["repetition_ratio"],
                "var_repetition": var["repetition_ratio"],
                "ref_output": ref["output"],
                "var_output": var["output"],
            }
        )

    # Summary
    n = len(comparisons)
    exact_matches = sum(1 for c in comparisons if c["exact_match"])
    avg_prefix = sum(c["prefix_match_chars"] for c in comparisons) / n
    avg_ref_chars = sum(c["ref_chars"] for c in comparisons) / n
    avg_prefix_ratio = avg_prefix / max(avg_ref_chars, 1)
    high_rep = sum(1 for c in comparisons if c["var_repetition"] > 0.3)

    payload = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reference": args.reference,
        "variant": args.variant,
        "max_tokens": args.max_tokens,
        "comparisons": comparisons,
        "summary": {
            "n_prompts": n,
            "exact_matches": exact_matches,
            "avg_prefix_match_chars": round(avg_prefix, 1),
            "avg_prefix_match_ratio": round(avg_prefix_ratio, 3),
            "variant_high_repetition_count": high_rep,
            "passes_quality_gate": (
                high_rep == 0 and avg_prefix_ratio >= 0.20
            ),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Exact matches: {exact_matches}/{n}")
    print(f"  Avg prefix match: {avg_prefix:.1f} chars ({avg_prefix_ratio:.1%} of ref)")
    print(f"  High-repetition variant outputs: {high_rep}/{n}")
    print(
        f"  Quality gate: {'PASS' if payload['summary']['passes_quality_gate'] else 'FAIL'}"
    )
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax check**

```bash
python3 -c "import ast; ast.parse(open('/Users/anthonylui/QwenCoderLocal/scripts/eval_kimi_quality.py').read()); print('OK')"
```

Expected: `OK`.

### Task 3.2: Run quality eval — 4-bit reference vs 2-bit variant

- [ ] **Step 1: Run the eval (background; takes ~30-60 minutes total for both models)**

```bash
cd /Users/anthonylui/QwenCoderLocal
mkdir -p artifacts/kimi_k26_speculative
nohup python3 scripts/eval_kimi_quality.py \
    --reference /Volumes/Samsung9904tb/Kimi-K2.6 \
    --variant /Volumes/Samsung9904tb/Kimi-K2.6-2bit-experts \
    --max-tokens 64 \
    --output artifacts/kimi_k26_speculative/quality_2bit_vs_4bit.json \
    > /tmp/kimi_quality.log 2>&1 &
echo "PID: $!"
```

Watch with `tail -20 /tmp/kimi_quality.log`. Wait for completion.

- [ ] **Step 2: Read the summary**

```bash
python3 -c "
import json
d = json.load(open('artifacts/kimi_k26_speculative/quality_2bit_vs_4bit.json'))
print(json.dumps(d['summary'], indent=2))
print()
for c in d['comparisons']:
    print(f\"[{c['prompt'][:50]}]\")
    print(f\"  prefix={c['prefix_match_chars']} ref_chars={c['ref_chars']} var_chars={c['var_chars']}\")
    print(f\"  ref_rep={c['ref_repetition']} var_rep={c['var_repetition']}\")
    print()
"
```

**Decision point:**
- `passes_quality_gate: true` → proceed to Phase 4.
- `passes_quality_gate: false` → STOP. Document the failure mode. Options to consider before continuing:
  - 2-bit may be too aggressive for routed experts → try 3-bit (`--mixed-expert-bits 3`)
  - May need a different `q-mode` (mxfp4 or nvfp4) — re-quantize and re-eval
  - May need to keep the first/last layers at higher precision

For speculative decoding, exact-match is NOT required (the target verifies). What matters is that the draft output **distribution overlaps the target** — high-repetition or pure garbage means the draft will get rejected on every token and speculative is worse than no-speculative.

The 20% prefix-match threshold + zero-repetition gate is a coarse first-line check, not the final acceptance-rate measurement (Phase 5 measures that directly).

### Task 3.3: Commit the quality eval script + first artifact

- [ ] **Step 1: Commit**

```bash
cd /Users/anthonylui/QwenCoderLocal
git add scripts/eval_kimi_quality.py artifacts/kimi_k26_speculative/quality_2bit_vs_4bit.json
git commit -m "feat(eval): Kimi quantization-variant quality harness + 2-bit baseline

Loads reference + variant Kimi (one at a time), runs Phase 4 prompts
at temperature=0 (greedy), compares outputs by exact-match,
prefix-match length, and repetition ratio.

Coarse quality gate: variant must have zero high-repetition outputs
and at least 20% average prefix match against the reference. This
gate is not sufficient for production but is the minimum bar before
investing in speculative decoding (low-quality draft = high reject
rate = no speedup).

First artifact: 2-bit-experts variant against 4-bit reference."
```

---

## Phase 4: Speculative decode smoke test

### Task 4.1: Write the speculative decode smoke test script

**Files:**
- Create: `scripts/profile_kimi_speculative.py`

- [ ] **Step 1: Create the script**

Note: `speculative_generate_step` from `mlx_lm.generate` is a generator. We need to convert it to a finite token list. Read the function signature first:

```bash
sed -n '530,560p' /Users/anthonylui/QwenCoderLocal/mlx-lm/mlx_lm/generate.py
```

Then write the script:

```python
#!/usr/bin/env python3
"""Self-speculative decode smoke test on Kimi K2.6.

Loads the 4-bit Kimi as target and the 2-bit-experts Kimi as draft.
Runs speculative_generate_step from mlx_lm.generate. Compares output
to a baseline greedy run with the target alone. Records throughput
and acceptance rate.

Memory note: both models in RAM simultaneously is the major risk.
Each uses expert_offload, so resident expert memory is bounded by
max_resident_experts. Set draft's max_resident lower than target's
to avoid evicting target's experts.

Usage:
    python3 scripts/profile_kimi_speculative.py \\
        --target /Volumes/Samsung9904tb/Kimi-K2.6 \\
        --draft /Volumes/Samsung9904tb/Kimi-K2.6-2bit-experts \\
        --num-draft-tokens 4 \\
        --max-tokens 64 \\
        --output artifacts/kimi_k26_speculative/speculative_smoke.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MLX_LM_ROOT = REPO_ROOT / "mlx-lm"
if MLX_LM_ROOT.exists() and str(MLX_LM_ROOT) not in sys.path:
    sys.path.insert(0, str(MLX_LM_ROOT))


SMOKE_PROMPTS = [
    "The capital of France is",
    "Explain the concept of attention in transformers in one sentence.",
]


def run_baseline(target_path: str, max_tokens: int, max_resident: int):
    """Greedy generation with the target model alone."""
    from mlx_lm import load, generate

    model, tokenizer = load(
        target_path,
        model_config={
            "expert_offload": True,
            "max_resident_experts": max_resident,
        },
    )
    results = []
    for prompt in SMOKE_PROMPTS:
        t0 = time.time()
        out = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        elapsed = time.time() - t0
        tok_count = len(tokenizer.encode(out)) - len(tokenizer.encode(prompt))
        results.append(
            {
                "prompt": prompt,
                "output": out,
                "elapsed_s": round(elapsed, 2),
                "approx_tok_count": tok_count,
                "tok_per_s": round(max(tok_count, 1) / elapsed, 3),
            }
        )
        print(
            f"  baseline [{prompt[:40]}] {elapsed:.1f}s, ~{tok_count} tok, "
            f"{results[-1]['tok_per_s']} tok/s",
            flush=True,
        )
    del model, tokenizer
    gc.collect()
    return results


def run_speculative(target_path: str, draft_path: str, num_draft: int,
                    max_tokens: int, max_resident_target: int,
                    max_resident_draft: int):
    """Speculative decode with target + draft."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import speculative_generate_step

    print(f"  Loading target: {target_path}", flush=True)
    target, tokenizer = load(
        target_path,
        model_config={
            "expert_offload": True,
            "max_resident_experts": max_resident_target,
        },
    )
    print(f"  Loading draft: {draft_path}", flush=True)
    draft, _ = load(
        draft_path,
        model_config={
            "expert_offload": True,
            "max_resident_experts": max_resident_draft,
        },
    )

    results = []
    for prompt in SMOKE_PROMPTS:
        prompt_ids = mx.array(tokenizer.encode(prompt))
        accepted_count = 0
        n_calls = 0
        out_tokens = []
        t0 = time.time()
        for token, _logprobs, accepted_flag in speculative_generate_step(
            prompt_ids, target, draft,
            num_draft_tokens=num_draft,
            max_tokens=max_tokens,
        ):
            out_tokens.append(int(token))
            n_calls += 1
            if accepted_flag:
                accepted_count += 1
            if len(out_tokens) >= max_tokens:
                break
        elapsed = time.time() - t0
        out_text = tokenizer.decode(out_tokens)
        tok_count = len(out_tokens)
        results.append(
            {
                "prompt": prompt,
                "output": out_text,
                "elapsed_s": round(elapsed, 2),
                "tok_count": tok_count,
                "tok_per_s": round(tok_count / elapsed, 3),
                "n_accepted": accepted_count,
                "n_total": n_calls,
                "acceptance_rate": round(accepted_count / max(n_calls, 1), 3),
            }
        )
        print(
            f"  spec [{prompt[:40]}] {elapsed:.1f}s, {tok_count} tok, "
            f"{results[-1]['tok_per_s']} tok/s, accept={results[-1]['acceptance_rate']}",
            flush=True,
        )

    del target, draft, tokenizer
    gc.collect()
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--num-draft-tokens", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--max-resident-target", type=int, default=1500)
    p.add_argument("--max-resident-draft", type=int, default=500)
    p.add_argument(
        "--output",
        default="artifacts/kimi_k26_speculative/speculative_smoke.json",
    )
    p.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline run (assume baseline numbers from prior runs)",
    )
    args = p.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target": args.target,
        "draft": args.draft,
        "num_draft_tokens": args.num_draft_tokens,
        "max_tokens": args.max_tokens,
        "max_resident_target": args.max_resident_target,
        "max_resident_draft": args.max_resident_draft,
    }

    if not args.skip_baseline:
        print("\n=== Baseline (target only) ===")
        payload["baseline"] = run_baseline(
            args.target, args.max_tokens, args.max_resident_target
        )

    print("\n=== Speculative (target + draft) ===")
    payload["speculative"] = run_speculative(
        args.target, args.draft, args.num_draft_tokens,
        args.max_tokens, args.max_resident_target, args.max_resident_draft,
    )

    if "baseline" in payload:
        bl_med = float(np.median([r["tok_per_s"] for r in payload["baseline"]]))
        sp_med = float(np.median([r["tok_per_s"] for r in payload["speculative"]]))
        accept_med = float(np.median([r["acceptance_rate"] for r in payload["speculative"]]))
        payload["summary"] = {
            "baseline_tok_per_s_median": round(bl_med, 3),
            "speculative_tok_per_s_median": round(sp_med, 3),
            "speedup": round(sp_med / max(bl_med, 1e-6), 2),
            "acceptance_rate_median": round(accept_med, 3),
        }
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for k, v in payload["summary"].items():
            print(f"  {k}: {v}")

    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax check**

```bash
python3 -c "import ast; ast.parse(open('/Users/anthonylui/QwenCoderLocal/scripts/profile_kimi_speculative.py').read()); print('OK')"
```

### Task 4.2: Smoke test — does the speculative path even run end-to-end?

- [ ] **Step 1: Tiny smoke run with --skip-baseline (avoid 2x model load)**

```bash
cd /Users/anthonylui/QwenCoderLocal
nohup python3 scripts/profile_kimi_speculative.py \
    --target /Volumes/Samsung9904tb/Kimi-K2.6 \
    --draft /Volumes/Samsung9904tb/Kimi-K2.6-2bit-experts \
    --num-draft-tokens 2 \
    --max-tokens 16 \
    --max-resident-target 1200 \
    --max-resident-draft 300 \
    --skip-baseline \
    --output artifacts/kimi_k26_speculative/speculative_smoke_micro.json \
    > /tmp/kimi_spec_smoke.log 2>&1 &
echo "PID: $!"
```

Watch with `tail -20 /tmp/kimi_spec_smoke.log`. Likely failure modes to watch for:
- OOM with both models loaded → reduce `--max-resident-target` and `--max-resident-draft` (e.g., 800 + 200)
- Tokenizer mismatch error → ABORT and re-investigate; both models came from same source so this should not happen
- `KimiMLAIsoQuantCache` trim error → Phase 1's trim is broken; revisit Task 1.2

- [ ] **Step 2: Verify output is non-empty and looks valid**

```bash
cat /tmp/kimi_spec_smoke.log
python3 -c "
import json
d = json.load(open('artifacts/kimi_k26_speculative/speculative_smoke_micro.json'))
for r in d['speculative']:
    print(f\"[{r['prompt'][:30]}] tok={r['tok_count']}, accept={r['acceptance_rate']}\")
    print(f'  out: {r[\"output\"][:80]!r}')
"
```

Expected: each prompt produced some output with a valid acceptance rate (likely 0.3-0.7 if 2-bit draft is reasonably aligned).

If `acceptance_rate == 0`: the draft is producing tokens the target never agrees with. Likely causes:
- Sampling mismatch (default sampler may not be greedy for both)
- Tokenizer / vocab mismatch
- The 2-bit quantization broke the model

If `acceptance_rate == 1`: suspicious — verify both models are actually different by checking the draft output looks like the 2-bit baseline from Phase 3.

### Task 4.3: Correctness gate — speculative output must match target greedy

- [ ] **Step 1: Run baseline + speculative on the same prompts, compare prefix-by-prefix**

```bash
cd /Users/anthonylui/QwenCoderLocal
nohup python3 scripts/profile_kimi_speculative.py \
    --target /Volumes/Samsung9904tb/Kimi-K2.6 \
    --draft /Volumes/Samsung9904tb/Kimi-K2.6-2bit-experts \
    --num-draft-tokens 4 \
    --max-tokens 32 \
    --max-resident-target 1200 \
    --max-resident-draft 300 \
    --output artifacts/kimi_k26_speculative/speculative_correctness.json \
    > /tmp/kimi_spec_correct.log 2>&1 &
echo "PID: $!"
```

- [ ] **Step 2: Verify baseline and speculative outputs match exactly (correctness gate)**

```bash
python3 -c "
import json
d = json.load(open('artifacts/kimi_k26_speculative/speculative_correctness.json'))
all_match = True
for bl, sp in zip(d['baseline'], d['speculative']):
    match = bl['output'] == sp['output']
    print(f\"[{bl['prompt'][:30]}] match={match}\")
    if not match:
        # First divergence point
        for i, (a, b) in enumerate(zip(bl['output'], sp['output'])):
            if a != b:
                print(f'  diverge at char {i}: {a!r} vs {b!r}')
                print(f'  baseline tail:    {bl[\"output\"][i:i+50]!r}')
                print(f'  speculative tail: {sp[\"output\"][i:i+50]!r}')
                break
        all_match = False
print(f'\\nALL MATCH: {all_match}')
"
```

**Decision point:**
- All match → proceed to Phase 5 (benchmarking).
- Some don't match → STOP. The cache trim is likely incorrect. Common bugs:
  - Trim is not a true reversal: an extra token's residual state lingers
  - PE buffer trim is off-by-one
  - Packed cache trim is incorrect

To debug: add a target-only single-step path to the script that uses the SAME model.trim path and verify trim is bit-correct on its own.

### Task 4.4: Commit Phase 4

- [ ] **Step 1: Commit the script + smoke artifacts**

```bash
cd /Users/anthonylui/QwenCoderLocal
git add scripts/profile_kimi_speculative.py \
    artifacts/kimi_k26_speculative/speculative_smoke_micro.json \
    artifacts/kimi_k26_speculative/speculative_correctness.json
git commit -m "feat(speculative): smoke test + correctness gate for Kimi self-speculative

Loads 4-bit Kimi (target) + 2-bit Kimi (draft), runs speculative_generate_step
with greedy sampling. Correctness gate compares output to target-only baseline:
must match character-for-character, otherwise the cache trim is wrong.

Smoke micro artifact: tiny --max-tokens=16 run to verify the path doesn't
crash or OOM with both models in RAM (target=1200 resident, draft=300).
Correctness artifact: --max-tokens=32 with character-level diff against
baseline. Recorded acceptance rate per prompt."
```

---

## Phase 5: Benchmark + tune

### Task 5.1: Sweep `num_draft_tokens` (acceptance rate vs throughput tradeoff)

- [ ] **Step 1: Run the same speculative script at K=2, 4, 8, 16**

For each K, run separately so results are isolated:

```bash
cd /Users/anthonylui/QwenCoderLocal
for K in 2 4 8 16; do
    echo "=== K=$K ==="
    python3 scripts/profile_kimi_speculative.py \
        --target /Volumes/Samsung9904tb/Kimi-K2.6 \
        --draft /Volumes/Samsung9904tb/Kimi-K2.6-2bit-experts \
        --num-draft-tokens $K \
        --max-tokens 64 \
        --max-resident-target 1200 \
        --max-resident-draft 300 \
        --output artifacts/kimi_k26_speculative/sweep_K${K}.json \
        2>&1 | tee /tmp/kimi_spec_K${K}.log
done
```

This is a multi-hour run (4 sweeps × 2 model loads each × 2 prompts).

- [ ] **Step 2: Aggregate the sweep**

```bash
python3 -c "
import json
print(f'{\"K\":>4} {\"baseline tok/s\":>15} {\"spec tok/s\":>12} {\"speedup\":>8} {\"accept\":>8}')
for K in [2, 4, 8, 16]:
    d = json.load(open(f'artifacts/kimi_k26_speculative/sweep_K{K}.json'))
    s = d['summary']
    print(f'{K:>4} {s[\"baseline_tok_per_s_median\"]:>15} {s[\"speculative_tok_per_s_median\"]:>12} {s[\"speedup\"]:>8} {s[\"acceptance_rate_median\"]:>8}')
"
```

Pick the K with the highest speedup. Typical pattern: speedup peaks around K=4-8 then declines as wasted draft work (rejected tokens) outweighs the batched-verification gain.

### Task 5.2: Resident-experts sweep at the chosen K

- [ ] **Step 1: With the chosen K, sweep target+draft resident allocations**

```bash
K=<chosen-K-from-task-5.1>
cd /Users/anthonylui/QwenCoderLocal
for ALLOC in "1500 200" "1200 300" "900 500"; do
    TGT=$(echo $ALLOC | cut -d' ' -f1)
    DFT=$(echo $ALLOC | cut -d' ' -f2)
    echo "=== target=$TGT draft=$DFT ==="
    python3 scripts/profile_kimi_speculative.py \
        --target /Volumes/Samsung9904tb/Kimi-K2.6 \
        --draft /Volumes/Samsung9904tb/Kimi-K2.6-2bit-experts \
        --num-draft-tokens $K \
        --max-tokens 64 \
        --max-resident-target $TGT \
        --max-resident-draft $DFT \
        --output artifacts/kimi_k26_speculative/alloc_t${TGT}_d${DFT}.json \
        2>&1 | tee /tmp/kimi_alloc_t${TGT}_d${DFT}.log
done
```

- [ ] **Step 2: Pick the best allocation and document**

The right allocation maximizes target's hit rate while keeping the draft fast enough that draft generation doesn't dominate the step. There is no a priori best — it depends on draft acceptance rate.

### Task 5.3: Decision point — proceed to Phase 6 (1-bit) or stop?

- [ ] **Step 1: Compute speedup vs the 0.42 tok/s baseline**

If best speedup is:
- **< 1.2x**: speculative is not winning materially. STOP. Either 2-bit draft is too poor, or the OOM-avoidance reduced target hit rate too much. Phase 6 (1-bit) is unlikely to help since 1-bit will have lower acceptance rate.
- **1.2x - 2x**: marginal win. Phase 6 (1-bit) worth trying since smaller draft = more target residency + possibly higher K.
- **> 2x**: clear win. Document, commit, and consider Phase 6 only if pursuing maximum throughput.

### Task 5.4: Commit Phase 5

- [ ] **Step 1: Commit the sweep artifacts**

```bash
cd /Users/anthonylui/QwenCoderLocal
git add artifacts/kimi_k26_speculative/sweep_K*.json artifacts/kimi_k26_speculative/alloc_*.json
git commit -m "evidence(speculative): K-token sweep + resident-allocation sweep

K sweep (num_draft_tokens=2,4,8,16) at fixed allocation, finds the
acceptance/throughput sweet spot.

Allocation sweep at chosen K, finds the target/draft resident-experts
split that maximizes end-to-end tok/s.

Honest reporting: speedup vs 0.42 tok/s baseline is included in each
sweep summary. Numbers are noisy (single run per config); consider
re-running with --max-tokens 256 for tighter variance before quoting."
```

---

## Phase 6 (optional): 1-bit experts

Only execute if Phase 5 task 5.3 indicates Phase 6 is worth pursuing.

### Task 6.1: Quantize Kimi to 1-bit experts

- [ ] **Step 1: Adapt the quantize script**

```bash
cp /Users/anthonylui/QwenCoderLocal/scripts/quantize_kimi_2bit.sh /Users/anthonylui/QwenCoderLocal/scripts/quantize_kimi_1bit.sh
sed -i '' 's|Kimi-K2.6-2bit-experts|Kimi-K2.6-1bit-experts|g; s|--mixed-expert-bits 2|--mixed-expert-bits 1|g; s|2-bit|1-bit|g' /Users/anthonylui/QwenCoderLocal/scripts/quantize_kimi_1bit.sh
chmod +x /Users/anthonylui/QwenCoderLocal/scripts/quantize_kimi_1bit.sh
```

Verify the substitution worked:

```bash
grep -E "mixed-expert-bits|KIMI_DST" /Users/anthonylui/QwenCoderLocal/scripts/quantize_kimi_1bit.sh
```

- [ ] **Step 2: Run the quantization (background, ~30-90 min)**

```bash
cd /Users/anthonylui/QwenCoderLocal
nohup bash scripts/quantize_kimi_1bit.sh > /tmp/kimi_1bit_convert.log 2>&1 &
echo "PID: $!"
```

Wait for completion. Verify size: should be ~143 GB (down from 285 GB at 2-bit).

If `mlx_lm.convert` errors on `--mixed-expert-bits 1`: investigate whether MLX supports 1-bit affine quantization. The path predicate validates `[1, 8]` but the mlx primitive `mx.quantize(..., bits=1)` may not be supported. If unsupported, this phase ends here — sub-bit below 2 requires non-trivial custom kernel work.

### Task 6.2: Quality eval for 1-bit

- [ ] **Step 1: Run the eval script**

```bash
cd /Users/anthonylui/QwenCoderLocal
python3 scripts/eval_kimi_quality.py \
    --reference /Volumes/Samsung9904tb/Kimi-K2.6 \
    --variant /Volumes/Samsung9904tb/Kimi-K2.6-1bit-experts \
    --max-tokens 64 \
    --output artifacts/kimi_k26_speculative/quality_1bit_vs_4bit.json
```

If `passes_quality_gate: false`: 1-bit is unusable as a draft. STOP this phase. The acceptance rate would be near zero.

### Task 6.3: Speculative decode with 1-bit draft

- [ ] **Step 1: Run with the K and allocations chosen in Task 5.2**

```bash
K=<chosen-K>
TGT=<chosen-target-resident>
DFT_1BIT=<draft-resident-doubled-since-1bit-experts-are-half-size>

cd /Users/anthonylui/QwenCoderLocal
python3 scripts/profile_kimi_speculative.py \
    --target /Volumes/Samsung9904tb/Kimi-K2.6 \
    --draft /Volumes/Samsung9904tb/Kimi-K2.6-1bit-experts \
    --num-draft-tokens $K \
    --max-tokens 64 \
    --max-resident-target $TGT \
    --max-resident-draft $DFT_1BIT \
    --output artifacts/kimi_k26_speculative/speculative_1bit.json \
    2>&1 | tee /tmp/kimi_spec_1bit.log
```

- [ ] **Step 2: Compare 1-bit vs 2-bit speedup**

```bash
python3 -c "
import json
d2 = json.load(open('artifacts/kimi_k26_speculative/sweep_K<K>.json'))
d1 = json.load(open('artifacts/kimi_k26_speculative/speculative_1bit.json'))
print('2-bit draft:', d2['summary'])
print('1-bit draft:', d1['summary'])
"
```

If 1-bit is faster than 2-bit: switch to 1-bit as the production draft.
If 1-bit is slower (likely due to lower acceptance rate): 2-bit remains the choice.

### Task 6.4: Commit Phase 6

- [ ] **Step 1: Commit**

```bash
cd /Users/anthonylui/QwenCoderLocal
git add scripts/quantize_kimi_1bit.sh \
    artifacts/kimi_k26_speculative/quality_1bit_vs_4bit.json \
    artifacts/kimi_k26_speculative/speculative_1bit.json
git commit -m "evidence(speculative): 1-bit draft variant

1-bit Kimi quantization (mixed-expert-bits 1, ~143 GB on disk).
Quality eval + speculative decode A/B against the 2-bit baseline.

Honest comparison: lower-bit draft has lower acceptance rate per
token but more residency in RAM. Whichever wins on end-to-end
tok/s for this hardware is the production choice."
```

---

## Phase 7: Document and finalize

### Task 7.1: Update KIMI_K26_FULL_STACK.md

**Files:**
- Modify: `docs/KIMI_K26_FULL_STACK.md`

- [ ] **Step 1: Read the current state of the doc**

```bash
cat /Users/anthonylui/QwenCoderLocal/docs/KIMI_K26_FULL_STACK.md
```

- [ ] **Step 2: Add a "Self-Speculative Decode" section after the Phase 5/6 results**

Append (or insert at the appropriate location based on the existing structure):

```markdown
## Self-Speculative Decode

Following the first-principles analysis in commit 16a6b0c, single-stream
decode at 0.42 tok/s on this hardware is bounded by per-token expert
dispatch overhead. Speculative decoding amortizes that overhead across
K accepted tokens.

**Setup (final):**
- Target: `/Volumes/Samsung9904tb/Kimi-K2.6` (4-bit routed experts)
- Draft: `/Volumes/Samsung9904tb/Kimi-K2.6-<2|1>bit-experts`
  (`mlx_lm.convert --mixed-expert-bits <2|1> --shared-expert-bits 4`)
- Cache: `KimiMLAIsoQuantCache` with `trim()` (added in commit <hash>)
- Decode: `mlx_lm.generate.speculative_generate_step` (existing infra)

**Best results from sweep:**
- K (num_draft_tokens): <chosen>
- Target resident experts: <chosen>
- Draft resident experts: <chosen>
- Median tok/s: <number>
- Median acceptance rate: <number>
- Speedup vs baseline 0.42 tok/s: <number>x

**Artifacts:**
- `artifacts/kimi_k26_speculative/quality_*_vs_4bit.json` — quantization quality eval
- `artifacts/kimi_k26_speculative/sweep_K*.json` — K-token sweep
- `artifacts/kimi_k26_speculative/alloc_*.json` — resident-experts allocation sweep
- `artifacts/kimi_k26_speculative/speculative_*.json` — final + 1-bit comparison
```

Replace the angle-bracketed placeholders with the real numbers from the artifacts.

### Task 7.2: Final commit

- [ ] **Step 1: Commit doc update**

```bash
cd /Users/anthonylui/QwenCoderLocal
git add docs/KIMI_K26_FULL_STACK.md
git commit -m "docs(kimi): self-speculative decode results

Final speedup vs the 0.42 tok/s baseline, the chosen draft variant
(2-bit or 1-bit), the chosen K, and the allocation split. Pointers
to all artifacts."
```

---

## Self-review (executed before saving)

**Spec coverage:**
- Sub-bit quantization tooling → Phases 2 (2-bit) and 6 (1-bit)
- Quality validation on Phase 4 prompts → Phase 3 + Task 6.2
- K-token batched verification → reuses existing `speculative_generate_step` (verified at generate.py:530); no new code required
- Self-speculative loop → Phase 4 + 5
- Cache trim prerequisite → Phase 1
- All five spec items have explicit tasks.

**Placeholder scan:**
- "TBD" / "TODO" / "implement later": none present
- Steps describing what without showing how: none — every code step has a complete code block
- The `<chosen-K>` and `<chosen-target-resident>` markers in Phase 6 are intentionally bound by Phase 5 outputs, with explicit `bash` substitution syntax shown.

**Type consistency:**
- `KimiMLAIsoQuantCache.trim(n: int) -> int` — defined in Task 1.2, called via `cache.trim_prompt_cache()` (the existing API at cache.py:279). The `trim_prompt_cache` walks each cache and calls `.trim(n)`, so the signature lines up.
- `speculative_generate_step` signature matches the verified one in generate.py:530.
- `_compressed_latent` is a `dict[str, mx.array]` per kimi_mla_isoquant_dkv.py:99-103 — Task 1.2's trim handles it as a dict.
- `_pe_buffer` and `_packed_latent_cache` are `mx.array | None` — Task 1.2's trim handles None correctly.
