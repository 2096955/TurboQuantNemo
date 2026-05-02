# Qwen3 Plumbing, Deferred Prefill, and DedeKimi Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Implement Qwen3 MoE offload wiring, Deferred Prefill (including Gemma sliding window logic), and the DedeKimi Observer with task-aware pinning.

**Architecture:** 
1. **Qwen3 Plumbing:** We will wire `repack_experts.py`, `expert_offload.py`, and `convert.py` to handle the Qwen3 pattern (`switch_mlp.{gate,up,down}_proj`), safely ignore the lack of shared experts, and integrate the `AttnResExpertPredictor` into `qwen3_moe.py`.
2. **Deferred Prefill:** We will modify `RotorQuantKVCache` to keep the cache in FP16 during prefill, and execute bulk compression before the first decode step. For Gemma 4, sliding-window layers will skip RotorQuant compression.
3. **DedeKimi Observer:** We will introduce a pure observer in `expert_offload.py` that computes EMA expert activation frequencies and entropy. We will also add a `pre_populate_task_clique` method to `ExpertOffloadManager` to support task-aware pinning.

**Tech Stack:** Python, MLX, PyTest.

---

## Chunk 1: Qwen3 MoE Offload Plumbing

**Files:**
- Create: `mlx-lm/tests/test_qwen3_wiring.py`
- Modify: `mlx-lm/mlx_lm/repack_experts.py`
- Modify: `mlx-lm/mlx_lm/expert_offload.py`
- Modify: `mlx-lm/mlx_lm/convert.py`
- Modify: `mlx-lm/mlx_lm/models/qwen3_moe.py`

- [x] **Step 1: Write the failing test for Qwen3 offload wiring**

```python
import pytest
from mlx_lm.expert_offload import ExpertOffloadManager
from mlx_lm.repack_experts import extract_expert_pattern

def test_qwen3_repack_pattern():
    key = "model.layers.4.mlp.experts.127.up_proj.weight"
    match = extract_expert_pattern(key)
    assert match is not None
    assert match["layer"] == 4
    assert match["expert"] == 127
    assert match["proj"] == "up_proj"

def test_qwen3_offload_manager_regex():
    manager = ExpertOffloadManager("test_qwen3_model", max_resident_experts=32)
    assert manager._is_expert_key("model.layers.1.mlp.experts.0.gate_proj.weight")
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest mlx-lm/tests/test_qwen3_wiring.py -v`
Expected: FAIL (Key patterns and extraction logic not yet implemented for Qwen3).

- [x] **Step 3: Write minimal implementation in `repack_experts.py`**

Add Qwen3 regex support for `switch_mlp.{gate,up,down}_proj` in `extract_expert_pattern`.

- [x] **Step 4: Write minimal implementation in `expert_offload.py`**

Update `_is_expert_key` and key mapping logic in `ExpertOffloadManager` to correctly identify and process Qwen3 keys.

- [x] **Step 5: Write minimal implementation in `convert.py`**

Add Qwen3 projection markers (`switch_mlp.{gate,up,down}_proj`) to the target predicates for layer-aware quantization, ensuring it gracefully skips missing shared experts.

- [x] **Step 6: Write minimal implementation in `qwen3_moe.py`**

Wire `AttnResExpertPredictor` into the MoE layer forward pass in `qwen3_moe.py`, passing the layer proxy signal to `predict_experts` and `record_activation`. Note: `max_resident_experts` default should account for `decoder_sparse_step`.

- [x] **Step 7: Run test to verify it passes**

Run: `pytest mlx-lm/tests/test_qwen3_wiring.py -v`
Expected: PASS

- [x] **Step 8: Commit**

```bash
cd mlx-lm
git add tests/test_qwen3_wiring.py mlx_lm/repack_experts.py mlx_lm/expert_offload.py mlx_lm/convert.py mlx_lm/models/qwen3_moe.py
git commit -m "feat: implement Qwen3 offload plumbing and wiring"
```

---

## Chunk 2: Deferred Prefill & Gemma Sliding Window

**Files:**
- Create: `mlx-lm/tests/test_deferred_prefill.py`
- Modify: `mlx-lm/mlx_lm/models/mlx_turboquant.py`
- Modify: `mlx-lm/mlx_lm/models/gemma4_text.py`

- [x] **Step 1: Write the failing test for deferred prefill**

```python
import pytest
import mlx.core as mx
from mlx_lm.models.mlx_turboquant import RotorQuantKVCache

def test_deferred_prefill_compression():
    cache = RotorQuantKVCache(head_dim=128, bit_width=3)
    keys = mx.random.normal((1, 4, 2048, 128))
    values = mx.random.normal((1, 4, 2048, 128))
    
    # Simulate prefill without compression
    cache.update_and_fetch(keys, values, prefill_mode=True)
    assert cache.is_deferred == True
    
    # Simulate transition to decode
    k_hat, v_hat = cache.update_and_fetch(mx.random.normal((1, 4, 1, 128)), mx.random.normal((1, 4, 1, 128)), prefill_mode=False)
    assert cache.is_deferred == False
    assert cache.keys is not None # Compression happened
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest mlx-lm/tests/test_deferred_prefill.py -v`
Expected: FAIL (`prefill_mode` argument not expected, `is_deferred` not found).

- [x] **Step 3: Write minimal implementation in `mlx_turboquant.py`**

Modify `RotorQuantKVCache.update_and_fetch` to accept a `prefill_mode` flag (or infer from sequence length). Keep incoming tokens in FP16 buffer up to a limit (~8K). Bulk-compress when switching from prefill to decode.

- [x] **Step 4: Write minimal implementation in `gemma4_text.py`**

Pass `is_global_attention` boolean to the `KVCache` factory. Skip RotorQuant compression for sliding-window layers (evicted <1024 tokens) entirely, returning FP16/native cache instead.

- [x] **Step 5: Run test to verify it passes**

Run: `pytest mlx-lm/tests/test_deferred_prefill.py -v`
Expected: PASS

- [x] **Step 6: Commit**

```bash
cd mlx-lm
git add tests/test_deferred_prefill.py mlx_lm/models/mlx_turboquant.py mlx_lm/models/gemma4_text.py
git commit -m "feat: implement deferred prefill and Gemma sliding window skip"
```

---

## Chunk 3: DedeKimi Observer & Task Pinning

**Files:**
- Create: `mlx-lm/tests/test_dedekimi_observer.py`
- Modify: `mlx-lm/mlx_lm/expert_offload.py`

- [x] **Step 1: Write the failing test for DedeKimi observer**

```python
import pytest
from mlx_lm.expert_offload import ExpertOffloadManager, DedeKimiObserver

def test_dedekimi_observer_ema():
    observer = DedeKimiObserver(num_layers=10, num_experts=128)
    observer.record_activation(layer=0, expert_ids=[1, 2, 3])
    
    entropy = observer.get_layer_entropy(layer=0)
    assert entropy > 0

def test_task_aware_pinning():
    manager = ExpertOffloadManager("test_model", max_resident_experts=32)
    # Mocking task clique
    manager.pre_populate_task_clique("Rust_Refactor", layer_expert_map={0: [1, 2, 3]})
    assert manager._pinned_tasks["Rust_Refactor"] is not None
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest mlx-lm/tests/test_dedekimi_observer.py -v`
Expected: FAIL (`DedeKimiObserver` undefined, `pre_populate_task_clique` missing).

- [x] **Step 3: Write minimal implementation in `expert_offload.py`**

Implement `DedeKimiObserver` to track EMA frequencies and compute entropy. Ensure control mode is gated by a `--dede-mode` flag (default `observer`).
Implement `pre_populate_task_clique` in `ExpertOffloadManager` to asynchronously issue `madvise(WILLNEED)` or load specific experts when a task type is flagged.

- [x] **Step 4: Run test to verify it passes**

Run: `pytest mlx-lm/tests/test_dedekimi_observer.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
cd mlx-lm
git add tests/test_dedekimi_observer.py mlx_lm/expert_offload.py
git commit -m "feat: implement DedeKimi observer and task-aware pinning API"
```
