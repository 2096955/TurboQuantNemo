# Qwen3.6-35B-A3B Mixed-Precision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert Qwen3.6-35B-A3B to a mixed-precision MLX checkpoint (4-bit dense, 2-bit routed experts, 8-bit shared expert) that fits 16 GB, with three-way benchmark proving quality preservation.

**Architecture:** The model uses `model_type=qwen3_5_moe` but our expert offload/repack pipeline only recognizes `qwen3_moe`. We add `qwen3_5_moe` to all type-check sets, add `--shared-expert-bits` to the conversion CLI for shared expert Q8_0 override, then run conversion → repack → three-way benchmark.

**Tech Stack:** Python, MLX, mlx-lm, safetensors, Ollama (GGUF baseline)

**Spec:** `docs/superpowers/specs/2026-04-16-qwen36-mixed-precision-pathway.md`

---

### Task 1: Add `qwen3_5_moe` to repack_experts.py

**Files:**
- Modify: `mlx-lm/mlx_lm/repack_experts.py:133,141`
- Test: `mlx-lm/tests/test_models.py` (existing repack tests)

- [ ] **Step 1: Write the failing test**

Create a minimal test that verifies `qwen3_5_moe` is accepted by the repack type check:

```python
# Add to mlx-lm/tests/test_qwen35_moe_pathway.py (new file)
import unittest
import json
import tempfile
import os
from pathlib import Path

class TestQwen35MoeRepacking(unittest.TestCase):

    def test_repack_accepts_qwen3_5_moe_model_type(self):
        """Gate G1: repack_experts.py must accept qwen3_5_moe."""
        from mlx_lm.repack_experts import _repack_supported_types
        self.assertIn("qwen3_5_moe", _repack_supported_types)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mlx-lm && python -m pytest tests/test_qwen35_moe_pathway.py::TestQwen35MoeRepacking::test_repack_accepts_qwen3_5_moe_model_type -v`
Expected: FAIL — `"qwen3_5_moe"` not in set

- [ ] **Step 3: Add `qwen3_5_moe` to `_repack_supported_types` and `is_qwen3` check**

In `mlx-lm/mlx_lm/repack_experts.py`:

Line 133 — change:
```python
_repack_supported_types = {"nemotron_h", "gemma4_text", "gemma4", "qwen3_moe"}
```
to:
```python
_repack_supported_types = {"nemotron_h", "gemma4_text", "gemma4", "qwen3_moe", "qwen3_5_moe"}
```

Line 141 — change:
```python
is_qwen3 = model_type == "qwen3_moe"
```
to:
```python
is_qwen3 = model_type in ("qwen3_moe", "qwen3_5_moe")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mlx-lm && python -m pytest tests/test_qwen35_moe_pathway.py::TestQwen35MoeRepacking::test_repack_accepts_qwen3_5_moe_model_type -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mlx-lm/mlx_lm/repack_experts.py mlx-lm/tests/test_qwen35_moe_pathway.py
git commit -m "feat: add qwen3_5_moe to repack_experts supported types"
```

---

### Task 2: Add `qwen3_5_moe` to expert_offload.py

**Files:**
- Modify: `mlx-lm/mlx_lm/expert_offload.py:67,71,134,1025`
- Test: `mlx-lm/tests/test_qwen35_moe_pathway.py`

- [ ] **Step 1: Write the failing tests**

Add to `mlx-lm/tests/test_qwen35_moe_pathway.py`:

```python
class TestQwen35MoeOffload(unittest.TestCase):

    def test_offload_type_sets_include_qwen3_5_moe(self):
        """qwen3_5_moe must be in both EXPERT_OFFLOAD_MODEL_TYPES and MOE_MODEL_TYPES."""
        from mlx_lm.expert_offload import EXPERT_OFFLOAD_MODEL_TYPES, MOE_MODEL_TYPES
        self.assertIn("qwen3_5_moe", EXPERT_OFFLOAD_MODEL_TYPES)
        self.assertIn("qwen3_5_moe", MOE_MODEL_TYPES)

    def test_parse_expert_key_qwen3_5_moe(self):
        """parse_expert_key must handle qwen3_5_moe like qwen3_moe."""
        from mlx_lm.expert_offload import parse_expert_key
        result = parse_expert_key(
            "language_model.model.layers.5.mlp.experts.3.gate_proj.weight",
            model_type="qwen3_5_moe",
        )
        self.assertIsNotNone(result)
        layer_idx, expert_id, proj, suffix = result
        self.assertEqual(layer_idx, 5)
        self.assertEqual(expert_id, 3)
        self.assertEqual(proj, "gate_proj")
        self.assertEqual(suffix, "weight")

    def test_is_expert_weight_key_qwen3_5_moe(self):
        """is_expert_weight_key must recognize qwen3_5_moe expert keys."""
        from mlx_lm.expert_offload import is_expert_weight_key
        self.assertTrue(
            is_expert_weight_key(
                "model.layers.1.mlp.experts.0.gate_proj.weight",
                model_type="qwen3_5_moe",
            )
        )
        self.assertFalse(
            is_expert_weight_key(
                "model.layers.1.mlp.shared_expert.gate_proj.weight",
                model_type="qwen3_5_moe",
            )
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd mlx-lm && python -m pytest tests/test_qwen35_moe_pathway.py::TestQwen35MoeOffload -v`
Expected: FAIL — `"qwen3_5_moe"` not recognized

- [ ] **Step 3: Add `qwen3_5_moe` to expert_offload.py**

In `mlx-lm/mlx_lm/expert_offload.py`:

Line 67 — change:
```python
    {"nemotron_h", "gemma4_text", "gemma4", "qwen3_moe"}
```
to:
```python
    {"nemotron_h", "gemma4_text", "gemma4", "qwen3_moe", "qwen3_5_moe"}
```

Line 71 — change:
```python
MOE_MODEL_TYPES = frozenset({"nemotron_h", "gemma4_text", "gemma4", "qwen3_moe"})
```
to:
```python
MOE_MODEL_TYPES = frozenset({"nemotron_h", "gemma4_text", "gemma4", "qwen3_moe", "qwen3_5_moe"})
```

Line 134 — change:
```python
    elif model_type == "qwen3_moe":
```
to:
```python
    elif model_type in ("qwen3_moe", "qwen3_5_moe"):
```

Line 1025 — change:
```python
    elif model_type == "qwen3_moe":
```
to:
```python
    elif model_type in ("qwen3_moe", "qwen3_5_moe"):
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd mlx-lm && python -m pytest tests/test_qwen35_moe_pathway.py::TestQwen35MoeOffload -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mlx-lm/mlx_lm/expert_offload.py mlx-lm/tests/test_qwen35_moe_pathway.py
git commit -m "feat: add qwen3_5_moe to expert offload supported types"
```

---

### Task 3: Add `qwen3_5_moe` to expert_weight_loader.py and utils.py

**Files:**
- Modify: `mlx-lm/mlx_lm/expert_weight_loader.py:62`
- Modify: `mlx-lm/mlx_lm/utils.py:305,339,486,516,528`
- Test: `mlx-lm/tests/test_qwen35_moe_pathway.py`

- [ ] **Step 1: Write the failing test**

Add to `mlx-lm/tests/test_qwen35_moe_pathway.py`:

```python
class TestQwen35MoeUtils(unittest.TestCase):

    def test_offload_supported_types_in_utils(self):
        """utils.py _offload_supported_types must include qwen3_5_moe."""
        # This is checked indirectly — utils.load_model raises NotImplementedError
        # for unsupported types. We verify the set directly.
        import mlx_lm.utils as utils
        # Access the module-level set used in load_model
        src = Path(utils.__file__).read_text()
        self.assertIn('"qwen3_5_moe"', src, "qwen3_5_moe must be in _offload_supported_types")

    def test_expert_weight_loader_recognizes_qwen3_5_moe(self):
        """expert_weight_loader.load_non_expert_weights must filter qwen3_5_moe expert keys."""
        from mlx_lm.expert_offload import is_expert_weight_key
        # Shared expert keys are NOT expert keys (should be loaded normally)
        self.assertFalse(
            is_expert_weight_key(
                "language_model.model.layers.5.mlp.shared_expert.gate_proj.weight",
                model_type="qwen3_5_moe",
            )
        )
        # Routed expert keys ARE expert keys (should be offloaded)
        self.assertTrue(
            is_expert_weight_key(
                "language_model.model.layers.5.mlp.experts.10.down_proj.scales",
                model_type="qwen3_5_moe",
            )
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mlx-lm && python -m pytest tests/test_qwen35_moe_pathway.py::TestQwen35MoeUtils -v`
Expected: FAIL

- [ ] **Step 3: Add `qwen3_5_moe` to expert_weight_loader.py**

In `mlx-lm/mlx_lm/expert_weight_loader.py`, line 62 — add after the `qwen3_moe` check:

```python
            or is_expert_weight_key(key, model_type="qwen3_moe")
            or is_expert_weight_key(key, model_type="qwen3_5_moe")
```

- [ ] **Step 4: Add `qwen3_5_moe` to utils.py**

In `mlx-lm/mlx_lm/utils.py`:

Line 305 — change:
```python
    _offload_supported_types = {"nemotron_h", "gemma4_text", "gemma4", "qwen3_moe"}
```
to:
```python
    _offload_supported_types = {"nemotron_h", "gemma4_text", "gemma4", "qwen3_moe", "qwen3_5_moe"}
```

Line 339 — change:
```python
        if _mt in ("gemma4_text", "gemma4", "qwen3_moe"):
```
to:
```python
        if _mt in ("gemma4_text", "gemma4", "qwen3_moe", "qwen3_5_moe"):
```

Line 486 — change:
```python
        if _mt == "qwen3_moe":
```
to:
```python
        if _mt in ("qwen3_moe", "qwen3_5_moe"):
```

Line 516 — change:
```python
        elif _model_type == "qwen3_moe":
```
to:
```python
        elif _model_type in ("qwen3_moe", "qwen3_5_moe"):
```

Line 528 — change:
```python
        if _model_type == "qwen3_moe":
```
to:
```python
        if _model_type in ("qwen3_moe", "qwen3_5_moe"):
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd mlx-lm && python -m pytest tests/test_qwen35_moe_pathway.py::TestQwen35MoeUtils -v`
Expected: PASS

- [ ] **Step 6: Run existing qwen3_moe tests to confirm no regressions**

Run: `cd mlx-lm && python -m pytest tests/test_models.py::TestModels::test_qwen3_moe tests/test_qwen3_offload_module_selection.py tests/test_qwen3_wiring.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add mlx-lm/mlx_lm/expert_weight_loader.py mlx-lm/mlx_lm/utils.py mlx-lm/tests/test_qwen35_moe_pathway.py
git commit -m "feat: add qwen3_5_moe to expert weight loader and utils"
```

---

### Task 4: Add `--shared-expert-bits` to conversion pipeline

**Files:**
- Modify: `mlx-lm/mlx_lm/convert.py:102-135,363-374,484-488`
- Test: `mlx-lm/tests/test_qwen35_moe_pathway.py`

- [ ] **Step 1: Write the failing test**

Add to `mlx-lm/tests/test_qwen35_moe_pathway.py`:

```python
import mlx.nn as nn

class TestSharedExpertQuantPredicate(unittest.TestCase):

    def test_shared_expert_gets_higher_bits(self):
        """Gate G2: shared expert must get shared_expert_bits, not mixed_expert_bits."""
        from mlx_lm.convert import _build_mixed_expert_quant_predicate
        from mlx_lm.models.switch_layers import SwitchLinear

        predicate = _build_mixed_expert_quant_predicate(
            mixed_expert_bits=2,
            default_bits=4,
            default_group_size=64,
            mode="affine",
            shared_expert_bits=8,
        )

        # Shared expert (nn.Linear) should get 8-bit
        linear = nn.Linear(128, 256)
        result = predicate("language_model.model.layers.5.mlp.shared_expert.gate_proj", linear)
        self.assertEqual(result["bits"], 8)

        result = predicate("language_model.model.layers.5.mlp.shared_expert.up_proj", linear)
        self.assertEqual(result["bits"], 8)

        result = predicate("language_model.model.layers.5.mlp.shared_expert.down_proj", linear)
        self.assertEqual(result["bits"], 8)

        # Router gate should get 8-bit
        gate = nn.Linear(128, 256, bias=False)
        result = predicate("language_model.model.layers.5.mlp.gate", gate)
        self.assertEqual(result["bits"], 8)

        # Shared expert gate should get 8-bit
        result = predicate("language_model.model.layers.5.mlp.shared_expert_gate", gate)
        self.assertEqual(result["bits"], 8)

        # Dense layers should get default (4-bit)
        dense = nn.Linear(128, 256)
        result = predicate("language_model.model.layers.5.self_attn.q_proj", dense)
        self.assertEqual(result["bits"], 4)

    def test_shared_expert_bits_defaults_to_default_bits(self):
        """When shared_expert_bits is None, shared expert gets default_bits."""
        from mlx_lm.convert import _build_mixed_expert_quant_predicate

        predicate = _build_mixed_expert_quant_predicate(
            mixed_expert_bits=2,
            default_bits=4,
            default_group_size=64,
            mode="affine",
            shared_expert_bits=None,
        )

        linear = nn.Linear(128, 256)
        result = predicate("language_model.model.layers.5.mlp.shared_expert.gate_proj", linear)
        self.assertEqual(result["bits"], 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mlx-lm && python -m pytest tests/test_qwen35_moe_pathway.py::TestSharedExpertQuantPredicate -v`
Expected: FAIL — `_build_mixed_expert_quant_predicate` doesn't accept `shared_expert_bits`

- [ ] **Step 3: Modify `_build_mixed_expert_quant_predicate()` in convert.py**

In `mlx-lm/mlx_lm/convert.py`, update the function signature and body (lines 102-135):

```python
def _build_mixed_expert_quant_predicate(
    *,
    mixed_expert_bits: int,
    default_bits: int,
    default_group_size: int,
    mode: str,
    shared_expert_bits: Optional[int] = None,
) -> Callable[[str, nn.Module], Union[bool, dict]]:
    if mixed_expert_bits < 1 or mixed_expert_bits > 8:
        raise ValueError("--mixed-expert-bits must be an integer in [1, 8].")

    resolved_shared_bits = shared_expert_bits if shared_expert_bits is not None else default_bits

    def _predicate(path: str, module: nn.Module) -> Union[bool, dict]:
        # Shared expert projections: always-active, higher quality
        if ".shared_expert." in path and isinstance(module, nn.Linear):
            return {
                "bits": resolved_shared_bits,
                "group_size": default_group_size,
                "mode": mode,
            }

        # Router gate and shared expert gate: always 8-bit
        if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
            return {"group_size": default_group_size, "bits": 8, "mode": mode}

        # Route lower-bit quantization only to routed expert projections.
        # SwitchLinear is the leaf module at paths like .switch_mlp.fc1,
        # .switch_glu.gate_proj, .switch_mlp.gate_proj etc.
        is_routed_switch = isinstance(module, SwitchLinear) and (
            ".mixer.switch_mlp.fc1" in path
            or ".mixer.switch_mlp.fc2" in path
            or ".switch_glu.gate_proj" in path
            or ".switch_glu.up_proj" in path
            or ".switch_glu.down_proj" in path
            or ".switch_mlp.gate_proj" in path
            or ".switch_mlp.up_proj" in path
            or ".switch_mlp.down_proj" in path
        )
        if is_routed_switch:
            return {
                "bits": mixed_expert_bits,
                "group_size": default_group_size,
                "mode": mode,
            }

        return {"bits": default_bits, "group_size": default_group_size, "mode": mode}

    return _predicate
```

- [ ] **Step 4: Add `--shared-expert-bits` CLI argument**

In `mlx-lm/mlx_lm/convert.py`, near the `--mixed-expert-bits` argument definition (around line 484-488), add:

```python
parser.add_argument(
    "--shared-expert-bits",
    type=int,
    default=None,
    help="Bit-width for shared expert projections (default: same as --q-bits). "
    "Use 8 for maximum quality on always-active shared experts.",
)
```

- [ ] **Step 5: Wire `--shared-expert-bits` into the predicate builder call**

In `mlx-lm/mlx_lm/convert.py`, where `_build_mixed_expert_quant_predicate` is called (around line 369):

```python
    quant_predicate = _build_mixed_expert_quant_predicate(
        mixed_expert_bits=mixed_expert_bits,
        default_bits=resolved_q_bits,
        default_group_size=resolved_q_group_size,
        mode=q_mode,
        shared_expert_bits=shared_expert_bits,
    )
```

And ensure the `convert()` function signature accepts `shared_expert_bits`:

Find the `convert()` function signature and add `shared_expert_bits: Optional[int] = None` as a parameter.

Wire it from the CLI `args.shared_expert_bits` to the `convert()` call.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd mlx-lm && python -m pytest tests/test_qwen35_moe_pathway.py::TestSharedExpertQuantPredicate -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add mlx-lm/mlx_lm/convert.py mlx-lm/tests/test_qwen35_moe_pathway.py
git commit -m "feat: add --shared-expert-bits for per-component quantization"
```

---

### Task 5: Verify IsoQuant skips DeltaNet layers (Gate G3)

**Files:**
- Test: `mlx-lm/tests/test_qwen35_moe_pathway.py`
- Reference: `mlx-lm/mlx_lm/models/cache.py:140-173`

- [ ] **Step 1: Write the verification test**

Add to `mlx-lm/tests/test_qwen35_moe_pathway.py`:

```python
class TestIsoQuantDeltaNetSkip(unittest.TestCase):

    def test_deltanet_layers_keep_arrays_cache(self):
        """Gate G3: IsoQuant must NOT replace ArraysCache (DeltaNet layers)."""
        from mlx_lm.models.cache import KVCache, ArraysCache, _replace_attention_caches

        # Simulate a mixed cache list: DeltaNet (ArraysCache) + Attention (KVCache)
        # Pattern: 3 DeltaNet + 1 Attention, repeated
        caches = []
        for i in range(8):
            if (i + 1) % 4 == 0:  # full attention layer
                caches.append(KVCache())
            else:  # DeltaNet layer
                caches.append(ArraysCache(size=2))

        def make_quant_cache(layer_idx):
            return f"quant_cache_{layer_idx}"  # sentinel value

        replaced = []
        for c in caches:
            result = _replace_attention_caches(
                c,
                make_quant_cache=make_quant_cache,
                skip_layers=0,
                layer_idx=None,
                layer_idx_ref=[0],
            )
            replaced.append(result)
            # Reset layer_idx_ref for next iteration
            # Actually _replace_attention_caches increments layer_idx_ref internally
            # only for KVCache instances, so we need to track properly

        # Re-do with proper layer_idx tracking
        layer_idx_ref = [0]
        replaced = []
        for c in caches:
            result = _replace_attention_caches(
                c,
                make_quant_cache=make_quant_cache,
                skip_layers=0,
                layer_idx=None,
                layer_idx_ref=layer_idx_ref,
            )
            replaced.append(result)

        # Verify: ArraysCache instances untouched, KVCache instances replaced
        for i, (original, result) in enumerate(zip(caches, replaced)):
            if isinstance(original, ArraysCache):
                self.assertIs(result, original, f"Layer {i} (DeltaNet) should keep ArraysCache")
            else:
                self.assertIsInstance(result, str, f"Layer {i} (attention) should be replaced")
                self.assertTrue(result.startswith("quant_cache_"))
```

- [ ] **Step 2: Run the test**

Run: `cd mlx-lm && python -m pytest tests/test_qwen35_moe_pathway.py::TestIsoQuantDeltaNetSkip -v`
Expected: PASS (this is a verification of existing behavior, not a code change)

- [ ] **Step 3: Commit test**

```bash
git add mlx-lm/tests/test_qwen35_moe_pathway.py
git commit -m "test: verify IsoQuant skips DeltaNet ArraysCache layers (G3)"
```

---

### Task 6: Preflight disk budget check (Gate G4) and BF16 download

**Files:**
- No code changes — operational task

- [ ] **Step 1: Check disk budget**

Run: `df -h ~`
Expected: ≥ 90 GB free (70 GB BF16 + 8 GB mixed output + headroom)

If insufficient, clean up old model checkpoints first:
```bash
du -sh ~/Models/*/
```

- [ ] **Step 2: Download BF16 source model**

```bash
huggingface-cli download mlx-community/Qwen3.6-35B-A3B-bf16 \
  --local-dir ~/Models/Qwen3.6-35B-A3B-bf16 \
  --local-dir-use-symlinks False
```

Expected: ~70 GB download, creates `~/Models/Qwen3.6-35B-A3B-bf16/` with config.json, model safetensors, tokenizer files.

- [ ] **Step 3: Verify download integrity**

```bash
ls -la ~/Models/Qwen3.6-35B-A3B-bf16/config.json
python -c "import json; c=json.load(open('$HOME/Models/Qwen3.6-35B-A3B-bf16/config.json')); print(f'model_type={c[\"model_type\"]}, num_experts={c.get(\"num_experts\", c.get(\"text_config\",{}).get(\"num_experts\",\"?\"))}')"
```

Expected: `model_type=qwen3_5_moe` (or similar), `num_experts=256`

---

### Task 7: Run mixed-precision conversion

**Files:**
- No code changes — operational task
- Input: `~/Models/Qwen3.6-35B-A3B-bf16/`
- Output: `~/Models/qwen3.6-35b-a3b-mixed/`

- [ ] **Step 1: Run conversion with our mixed-precision recipe**

```bash
cd ~/QwenCoderLocal && python -m mlx_lm.convert \
  --hf-path ~/Models/Qwen3.6-35B-A3B-bf16 \
  --mlx-path ~/Models/qwen3.6-35b-a3b-mixed \
  --quantize --q-bits 4 --q-group-size 64 \
  --mixed-expert-bits 2 \
  --shared-expert-bits 8
```

Expected: Creates `~/Models/qwen3.6-35b-a3b-mixed/` with quantized safetensors.

- [ ] **Step 2: Verify Gate G2 — shared expert quantization**

```bash
python -c "
import json
c = json.load(open('$HOME/Models/qwen3.6-35b-a3b-mixed/config.json'))
qc = c.get('quantization_config', c.get('quantization', {}))
print('Quantization config:')
for k, v in sorted(qc.items()):
    print(f'  {k}: {v}')
"
```

Expected: Quantization config shows mixed bits (shared=8, routed=2, default=4).

- [ ] **Step 3: Verify output size**

```bash
du -sh ~/Models/qwen3.6-35b-a3b-mixed/
```

Expected: ~6-10 GB (much smaller than 70 GB BF16)

---

### Task 8: Repack experts for LRU offload

**Files:**
- No code changes — operational task (uses code from Tasks 1-3)
- Input: `~/Models/qwen3.6-35b-a3b-mixed/`

- [ ] **Step 1: Run expert repacking**

```bash
cd ~/QwenCoderLocal && python -m mlx_lm.repack_experts \
  --model ~/Models/qwen3.6-35b-a3b-mixed
```

Expected: Creates `repacked-*.safetensors` files, updates `model.safetensors.index.json`.

- [ ] **Step 2: Verify repacked shards exist**

```bash
ls ~/Models/qwen3.6-35b-a3b-mixed/repacked-*.safetensors | head -5
python -c "
import json
idx = json.load(open('$HOME/Models/qwen3.6-35b-a3b-mixed/model.safetensors.index.json'))
repacked = [v for v in set(idx['weight_map'].values()) if 'repacked' in v]
print(f'{len(repacked)} repacked shards')
expert_keys = [k for k in idx['weight_map'] if '.experts.' in k]
print(f'{len(expert_keys)} expert weight entries in index')
"
```

Expected: Multiple repacked shard files, expert keys pointing to repacked shards.

---

### Task 9: Smoke test Config C (our mixed-precision)

**Files:**
- No code changes — operational verification

- [ ] **Step 1: Quick generation test without offload**

```bash
cd ~/QwenCoderLocal && python -m mlx_lm.generate \
  --model ~/Models/qwen3.6-35b-a3b-mixed \
  --prompt "What is 2+2?" \
  --max-tokens 100
```

Expected: Coherent response (may include `<think>` tags). Verifies model loads correctly.

- [ ] **Step 2: Generation test with expert offload + IsoQuant**

```bash
cd ~/QwenCoderLocal && python -m mlx_lm.generate \
  --model ~/Models/qwen3.6-35b-a3b-mixed \
  --expert-offload --max-resident-experts 2048 \
  --kv-cache-type isoquant \
  --prompt "What is 2+2?" \
  --max-tokens 100
```

Expected: Coherent response. Expert offload and IsoQuant activate without errors.

- [ ] **Step 3: Memory measurement**

```bash
/usr/bin/time -l python -m mlx_lm.generate \
  --model ~/Models/qwen3.6-35b-a3b-mixed \
  --expert-offload --max-resident-experts 2048 \
  --kv-cache-type isoquant \
  --prompt "Explain the theory of relativity to a 5 year old." \
  --max-tokens 512 2>&1 | grep "maximum resident set size"
```

Expected: Peak RSS < 16,384 MB (16 GB). If over, reduce `--max-resident-experts` to 1024 or 512.

---

### Task 10: Download Config B (uniform 4-bit) model

**Files:**
- No code changes — operational task

- [ ] **Step 1: Download mlx-community uniform 4-bit**

```bash
huggingface-cli download mlx-community/Qwen3.6-35B-A3B-4bit \
  --local-dir ~/Models/Qwen3.6-35B-A3B-4bit \
  --local-dir-use-symlinks False
```

Expected: ~6-8 GB download.

- [ ] **Step 2: Quick smoke test**

```bash
cd ~/QwenCoderLocal && python -m mlx_lm.generate \
  --model ~/Models/Qwen3.6-35B-A3B-4bit \
  --prompt "What is 2+2?" \
  --max-tokens 100
```

Expected: Coherent response.

---

### Task 11: Run three-way benchmark

**Files:**
- Output: `results/qwen36_q8_baseline_quality.json` (Config A)
- Output: `results/qwen36_uniform4bit_quality.json` (Config B)
- Output: `results/qwen36_mixed_quality.json` (Config C)

**Benchmark comparability locks:** All runs use `--temp 0.0 --seed 42 --max-tokens 512`.

- [ ] **Step 1: Run Config B benchmark (uniform 4-bit on MLX)**

```bash
cd ~/QwenCoderLocal && python scripts/eval_quality_gate.py \
  --model ~/Models/Qwen3.6-35B-A3B-4bit \
  --suite all \
  --temp 0.0 --seed 42 --max-tokens 512 \
  --output results/qwen36_uniform4bit_quality.json
```

Expected: JSON output with pass/fail for 12 prompts.

- [ ] **Step 2: Run Config C benchmark (our mixed-precision on MLX)**

```bash
cd ~/QwenCoderLocal && python scripts/eval_quality_gate.py \
  --model ~/Models/qwen3.6-35b-a3b-mixed \
  --expert-offload --max-resident-experts 2048 \
  --kv-cache-type isoquant \
  --suite all \
  --temp 0.0 --seed 42 --max-tokens 512 \
  --output results/qwen36_mixed_quality.json
```

Expected: 10+/12 pass. JSON output saved.

- [ ] **Step 3: Run Config A benchmark (Q8_0 on Ollama)**

For each of the 12 prompts from `eval_quality_gate.py --suite all`, run via Ollama API:

```bash
# Ensure Ollama is running with the Q8_0 model
ollama run qwen3.6-35b-a3b:q8_0 "Write a Python function to compute the nth Fibonacci number. Output only the function, no explanation." --format plain
```

Score each response against the same rubric (expected substrings, min tokens, repetition check). Record results in `results/qwen36_q8_baseline_quality.json` using the same JSON schema.

Note: This is manual scoring. Use the Ollama `/api/generate` endpoint for reproducibility:

```bash
curl -s http://localhost:11434/api/generate -d '{
  "model": "qwen3.6-35b-a3b:q8_0",
  "prompt": "<prompt text>",
  "options": {"temperature": 0, "seed": 42, "num_predict": 512},
  "stream": false
}' | python -c "import sys,json; print(json.load(sys.stdin)['response'])"
```

- [ ] **Step 4: Run memory + throughput profiling on Config C**

```bash
cd ~/QwenCoderLocal && python scripts/benchmark_moe_offload.py \
  --model ~/Models/qwen3.6-35b-a3b-mixed \
  --expert-offload --max-resident-experts 2048 \
  --kv-cache-type isoquant \
  --output results/qwen36_pathway_benchmark.json
```

Expected: JSON with decode tok/s, peak memory, latency.

- [ ] **Step 5: Run KV fidelity test (if measure_kv_fidelity.py exists)**

```bash
cd ~/QwenCoderLocal && python scripts/measure_kv_fidelity.py \
  --model ~/Models/qwen3.6-35b-a3b-mixed \
  --depths 512,2048 \
  --output results/qwen36_kv_ppl_depth.json
```

Expected: IsoQuant delta PPL < 0.01 @ 2048 tokens.

---

### Task 12: Compile results and update pathway checklist

**Files:**
- Modify: `docs/PATHWAY_PROVEN_CHECKLIST.md`

- [ ] **Step 1: Load and compare results**

```bash
python -c "
import json

configs = {
    'A (Q8_0)': 'results/qwen36_q8_baseline_quality.json',
    'B (4-bit)': 'results/qwen36_uniform4bit_quality.json',
    'C (mixed)': 'results/qwen36_mixed_quality.json',
}

for name, path in configs.items():
    try:
        data = json.load(open(path))
        print(f'{name}: {data[\"n_pass\"]}/{data[\"n_total\"]} pass, peak {data.get(\"memory\",{}).get(\"peak_mb\",\"?\"):.0f} MB')
    except FileNotFoundError:
        print(f'{name}: FILE NOT FOUND')
"
```

- [ ] **Step 2: Verify acceptance criteria**

Check each criterion from the spec:
- Smoke test: Config C ≥ 10/12
- Quality parity: Config C score ≥ Config A score - 1
- Peak RSS: Config C < 16,384 MB
- Decode throughput: Config C > 5 tok/s AND ≥ 0.5x Config B tok/s
- KV fidelity: IsoQuant delta PPL < 0.01 @ 2048

- [ ] **Step 3: Add row to pathway checklist**

Add to `docs/PATHWAY_PROVEN_CHECKLIST.md`:

```markdown
| Qwen3.6-35B-A3B | 16 GB | `docs/superpowers/specs/2026-04-16-qwen36-mixed-precision-pathway.md` | `results/qwen36_mixed_quality.json` | `results/qwen36_pathway_benchmark.json` | Three-way: Q8_0 vs uniform-4bit vs mixed-precision |
```

- [ ] **Step 4: Commit results and checklist update**

```bash
git add results/qwen36_*.json docs/PATHWAY_PROVEN_CHECKLIST.md
git commit -m "feat: Qwen3.6-35B-A3B 16GB pathway — three-way benchmark complete"
```

---

## Rollback Path

If Gate G2 fails (shared expert Q8_0 not implementable):
1. Skip `--shared-expert-bits 8` in Task 7
2. Run conversion with `--mixed-expert-bits 2` only (all experts 2-bit)
3. Document the delta in results
4. Proceed with remaining tasks — benchmark is still valid

## Cleanup

After successful benchmark:
```bash
# Remove BF16 source (~70 GB) — only needed for conversion
rm -rf ~/Models/Qwen3.6-35B-A3B-bf16
```
