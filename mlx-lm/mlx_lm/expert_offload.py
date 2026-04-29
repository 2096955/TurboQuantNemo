# Copyright © 2026 Apple Inc.

"""Expert weight offloading for large MoE models (LRU + per-key safetensors loads).

Architecture Support
--------------------
Supports **Nemotron-H** and **Gemma 4** expert offloading:

**Nemotron-H** (nemotron_h):
  - Experts stored as individual per-expert tensors after repacking
    (``backbone.layers.{L}.mixer.experts.{E}.{up_proj|down_proj}.{weight|scales|biases}``)
  - Two projections per expert: fc1 (up_proj) and fc2 (down_proj)
  - Activation: ReLU²

**Gemma 4** (gemma4_text):
  - Experts stored as individual per-expert tensors after repacking via repack_experts.py
    (``model.layers.{L}.experts.{E}.{gate_proj|up_proj|down_proj}.{weight|scales|biases}``)
  - Three projections per expert: gate, up, down (GeGLU activation)
  - Original checkpoint uses stacked SwitchGLU format; repack_experts.py splits to individual

Both architectures use the same ExpertOffloadManager with LRU caching.
The ``projections`` constructor parameter controls whether 2 or 3 projections are loaded.
"""

from __future__ import annotations

import re
import threading
import time
import concurrent.futures
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import mlx.core as mx

# ---------------------------------------------------------------------------
# Expert key patterns — one per supported architecture
# ---------------------------------------------------------------------------

# Nemotron-H: individual per-expert tensors (after repack_experts.py)
_NEMOTRON_EXPERT_KEY_RE = re.compile(
    r"^backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(up_proj|down_proj)\.(weight|scales|biases)$"
)

# Gemma 4: stacked SwitchGLU tensors (pre-repack format)
# Handles both bare (gemma4_text) and wrapped (gemma4 multimodal) key prefixes:
#   model.layers.{L}.experts.switch_glu.{proj}.{suffix}
#   language_model.model.layers.{L}.experts.switch_glu.{proj}.{suffix}
_GEMMA4_STACKED_KEY_RE = re.compile(
    r"^(?:language_model\.)?model\.layers\.(\d+)\.experts\.switch_glu\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$"
)

# Gemma 4: individual per-expert tensors (after repack_experts.py)
_GEMMA4_EXPERT_KEY_RE = re.compile(
    r"^(?:language_model\.)?model\.layers\.(\d+)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$"
)

# Qwen3: individual per-expert tensors (after repack_experts.py)
_QWEN3_EXPERT_KEY_RE = re.compile(
    r"^(?:language_model\.)?model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$"
)

# Kimi K2/K2.5: individual per-expert compressed-tensors keys in the raw
# checkpoint. The multimodal wrapper adds language_model.; text-only Kimi K2
# uses the bare model.layers prefix.
_KIMI_EXPERT_KEY_RE = re.compile(
    r"^(?:language_model\.)?model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases|weight_packed|weight_scale|weight_shape)$"
)

# Supported model types for expert offloading
EXPERT_OFFLOAD_MODEL_TYPES = frozenset(
    {
        "nemotron_h",
        "gemma4_text",
        "gemma4",
        "qwen3_moe",
        "qwen3_5_moe",
        "kimi_k25",
        "kimi_k2",
    }
)

# Model types with known MoE architectures (for future extensibility)
MOE_MODEL_TYPES = frozenset(
    {
        "nemotron_h",
        "gemma4_text",
        "gemma4",
        "qwen3_moe",
        "qwen3_5_moe",
        "kimi_k25",
        "kimi_k2",
    }
)

# Legacy alias — kept for backward compatibility
_EXPERT_KEY_RE = _NEMOTRON_EXPERT_KEY_RE


class ExpertLoadError(RuntimeError):
    """Raised when a routed expert tensor cannot be read from disk."""

    def __init__(
        self,
        message: str,
        *,
        shard_path: str = "",
        tensor_key: str = "",
        layer_idx: int = -1,
    ):
        super().__init__(message)
        self.shard_path = shard_path
        self.tensor_key = tensor_key
        self.layer_idx = layer_idx


def http_status_for_generation_failure(exc: BaseException) -> int:
    """HTTP status for chat/completions when generation fails before streaming."""
    return 503 if isinstance(exc, (ExpertLoadError, MemoryError)) else 500


def parse_nemotron_expert_key(key: str) -> tuple[int, int, str, str] | None:
    """Return (layer_idx, expert_id, 'up_proj'|'down_proj', suffix) or None if not a routed expert weight.

    Only matches Nemotron-H individual per-expert tensor keys (post-repack format).
    """
    m = _NEMOTRON_EXPERT_KEY_RE.match(key)
    if m is None:
        return None
    proj = m.group(3)
    suffix = m.group(4)
    return int(m.group(1)), int(m.group(2)), proj, suffix


def parse_expert_key(
    key: str, model_type: str = "nemotron_h"
) -> tuple[int, int, str, str] | None:
    """Architecture-aware expert key parser.

    Returns (layer_idx, expert_id, proj_name, tensor_type) or None.

    For Nemotron-H (individual per-expert tensors):
        - expert_id is the explicit expert index from the key
        - proj_name is 'up_proj' or 'down_proj'

    For Gemma 4 (individual per-expert tensors after repack):
        - expert_id is the explicit expert index from the key
        - proj_name is 'gate_proj', 'up_proj', or 'down_proj'
    """
    if model_type == "nemotron_h":
        return parse_nemotron_expert_key(key)
    elif model_type in ("gemma4_text", "gemma4"):
        m = _GEMMA4_EXPERT_KEY_RE.match(key)
        if m is None:
            return None
        return int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
    elif model_type in ("qwen3_moe", "qwen3_5_moe"):
        m = _QWEN3_EXPERT_KEY_RE.match(key)
        if m is None:
            return None
        return int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
    elif model_type in ("kimi_k25", "kimi_k2"):
        m = _KIMI_EXPERT_KEY_RE.match(key)
        if m is None:
            return None
        return int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
    return None


def is_nemotron_routed_expert_weight_key(key: str) -> bool:
    return parse_nemotron_expert_key(key) is not None


def is_expert_weight_key(key: str, model_type: str = "nemotron_h") -> bool:
    """Architecture-aware check for expert weight keys."""
    return parse_expert_key(key, model_type=model_type) is not None


def build_nemotron_expert_key_table(
    weight_map: dict[str, str],
) -> dict[tuple[int, int], dict[str, str]]:
    """Map (layer_idx, expert_id) -> {'fc1_weight': key, 'fc1_scales': key, etc.} using checkpoint names."""
    table: dict[tuple[int, int], dict[str, str]] = {}
    for key in weight_map.keys():
        parsed = parse_nemotron_expert_key(key)
        if parsed is None:
            continue
        layer_idx, expert_id, proj, suffix = parsed
        slot = table.setdefault((layer_idx, expert_id), {})
        if proj == "up_proj":
            slot[f"fc1_{suffix}"] = key
        else:
            slot[f"fc2_{suffix}"] = key
    return table


class AttnResExpertPredictor:
    """Predicts required experts from AttnRes block attention weights.

    Online affinity matrix (num_layers, num_blocks, num_experts) accumulated
    from observed alpha x expert activations. Phase 2c deliverable — see plan
    Appendix F for the unified framework.
    """

    def __init__(self, num_blocks: int, num_experts: int, num_layers: int):
        self.block_expert_affinity = mx.zeros((num_layers, num_blocks, num_experts))
        self.observation_count = mx.zeros((num_layers, num_blocks))

    def record_activation(
        self,
        layer_idx: int,
        block_attention_weights: mx.array,
        expert_ids: mx.array,
    ) -> None:
        alpha = block_attention_weights.mean(axis=(1, 2))
        mx.eval(expert_ids)
        flat_eids = mx.array(list(set(expert_ids.flatten().tolist())), dtype=mx.int32)
        affinity = self.block_expert_affinity[layer_idx]
        affinity[:, flat_eids] = affinity[:, flat_eids] + alpha[:, None]
        self.block_expert_affinity[layer_idx] = affinity
        self.observation_count[layer_idx] = self.observation_count[layer_idx] + 1

    def predict_experts(
        self,
        layer_idx: int,
        block_attention_weights: mx.array,
        top_k: int = 16,
    ) -> list[int]:
        alpha = block_attention_weights.mean(axis=(1, 2))
        affinity = self.block_expert_affinity[layer_idx]
        scores = mx.matmul(alpha, affinity)
        total_counts = mx.matmul(alpha, self.observation_count[layer_idx])
        scores = mx.where(total_counts > 0, scores / total_counts, scores)
        predicted = mx.argsort(scores)[-top_k:]
        mx.eval(predicted)
        return predicted.tolist()


class SimulatedAttnResPredictor(AttnResExpertPredictor):
    """Generates a proxy block-attention signal for models without native AttnRes."""

    def __init__(
        self, num_blocks: int, num_experts: int, num_layers: int, hidden_dim: int
    ):
        super().__init__(num_blocks, num_experts, num_layers)
        self.hidden_dim = hidden_dim
        self.block_size = max(1, num_layers // num_blocks)
        self.block_representations = mx.zeros((num_blocks, hidden_dim))

    def compute_proxy_alpha(self, layer_idx: int, hidden_state: mx.array) -> mx.array:
        """Compute cosine similarity-based proxy alpha and update block representations."""
        block_idx = layer_idx // self.block_size
        if block_idx >= self.observation_count.shape[1]:
            block_idx = self.observation_count.shape[1] - 1

        # x is typically [batch*seq_len, hidden_dim] or [batch, seq_len, hidden_dim]
        # We want to average over all dimensions except the last one (hidden_dim)
        axes = tuple(range(hidden_state.ndim - 1))
        mean_h = hidden_state.mean(axis=axes)

        # update block representation online
        self.block_representations[block_idx] = (
            0.9 * self.block_representations[block_idx] + 0.1 * mean_h
        )

        norm_h = mean_h / (mx.linalg.norm(mean_h) + 1e-6)
        norm_blocks = self.block_representations / (
            mx.linalg.norm(self.block_representations, axis=-1, keepdims=True) + 1e-6
        )

        logits = mx.matmul(norm_blocks, norm_h)
        alpha = mx.softmax(logits * 5.0, axis=0)  # Temp to sharpen

        # Match shape expected by base class: [num_blocks, 1, 1]
        return alpha[:, None, None]


def build_expert_importance_from_router(
    layer_idx: int,
    top_k_indices: mx.array,
    top_k_weights: mx.array,
) -> dict[tuple[int, int], float]:
    """Aggregate mean router weight per expert id for eviction hints (dynamic offload).

    Shapes: top_k_indices and top_k_weights (..., K) e.g. (B, S, K).
    """
    mx.eval(top_k_indices, top_k_weights)
    flat_i = top_k_indices.reshape(-1)
    flat_w = top_k_weights.reshape(-1)
    idx_list = flat_i.tolist()
    w_list = flat_w.tolist()
    acc: dict[tuple[int, int], float] = {}
    cnt: dict[tuple[int, int], int] = {}
    for eid, wt in zip(idx_list, w_list):
        key = (layer_idx, int(eid))
        acc[key] = acc.get(key, 0.0) + float(wt)
        cnt[key] = cnt.get(key, 0) + 1
    return {k: acc[k] / cnt[k] for k in acc}


def build_gemma4_expert_key_table(
    weight_map: dict[str, str],
) -> dict[tuple[int, int], dict[str, str]]:
    """Map (layer_idx, expert_id) -> {'gate_weight': key, 'up_weight': ..., 'down_weight': ...}.

    Expects repacked individual per-expert keys (post repack_experts.py):
      model.layers.{L}.experts.{E}.{gate_proj|up_proj|down_proj}.{weight|scales|biases}
    """
    table: dict[tuple[int, int], dict[str, str]] = {}
    for key in weight_map.keys():
        m = _GEMMA4_EXPERT_KEY_RE.match(key)
        if m is None:
            continue
        layer_idx, expert_id = int(m.group(1)), int(m.group(2))
        proj = m.group(3)
        suffix = m.group(4)
        short = proj.replace("_proj", "")
        slot = table.setdefault((layer_idx, expert_id), {})
        slot[f"{short}_{suffix}"] = key
    return table


def build_qwen3_expert_key_table(
    weight_map: dict[str, str],
) -> dict[tuple[int, int], dict[str, str]]:
    """Map (layer_idx, expert_id) -> {'gate_weight': key, 'up_weight': ..., 'down_weight': ...}.

    Expects repacked individual per-expert keys (post repack_experts.py):
      model.layers.{L}.mlp.experts.{E}.{gate_proj|up_proj|down_proj}.{weight|scales|biases}
    """
    table: dict[tuple[int, int], dict[str, str]] = {}
    for key in weight_map.keys():
        m = _QWEN3_EXPERT_KEY_RE.match(key)
        if m is None:
            continue
        layer_idx, expert_id = int(m.group(1)), int(m.group(2))
        proj = m.group(3)
        suffix = m.group(4)
        short = proj.replace("_proj", "")
        slot = table.setdefault((layer_idx, expert_id), {})
        slot[f"{short}_{suffix}"] = key
    return table


def build_kimi_expert_key_table(
    weight_map: dict[str, str],
) -> dict[tuple[int, int], dict[str, str]]:
    """Map Kimi compressed-tensors expert keys to SwitchGLU offload specs.

    Kimi checkpoints store routed experts as per-expert tensors with
    ``weight_packed`` / ``weight_scale`` / ``weight_shape`` suffixes. The
    manager exposes those to the quantized SwitchGLU path as
    ``{gate,up,down}_{weight,scales}``; ``_load_expert_pair_tensors`` performs
    the final dtype aliasing and synthesized-bias handling.
    """
    table: dict[tuple[int, int], dict[str, str]] = {}
    for key in weight_map.keys():
        parsed = parse_expert_key(key, model_type="kimi_k25")
        if parsed is None:
            continue
        layer_idx, expert_id, proj, suffix = parsed
        short = proj.replace("_proj", "")
        slot = table.setdefault((layer_idx, expert_id), {})
        if suffix == "weight_packed":
            slot[f"{short}_weight"] = key
        elif suffix == "weight_scale":
            slot[f"{short}_scales"] = key
        elif suffix == "weight_shape":
            slot[f"{short}_shape"] = key
        else:
            slot[f"{short}_{suffix}"] = key
    return table


class DedeKimiObserver:
    """Pure observer for expert activation logging + entropy tracking (Phase 1.6).

    No control hooks — logs EMA activations and per-layer entropy for offline
    clique building and collapse detection. Does **not** prefetch or evict.
    """

    def __init__(self, num_layers: int, num_experts: int, decay: float = 0.99):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.decay = decay
        self.activation_ema = np.zeros((num_layers, num_experts))
        self.lock = threading.Lock()

    def record_activation(self, layer: int, expert_ids: list[int]):
        with self.lock:
            indicator = np.zeros(self.num_experts)
            if hasattr(expert_ids, "tolist"):
                expert_ids = expert_ids.tolist()

            for eid in expert_ids:
                indicator[eid] = 1.0

            self.activation_ema[layer] = (
                self.decay * self.activation_ema[layer] + (1 - self.decay) * indicator
            )

    def get_layer_entropy(self, layer: int) -> float:
        with self.lock:
            probs = self.activation_ema[layer]
            probs = probs / (probs.sum() + 1e-9)
            entropy = -np.sum(probs * np.log(probs + 1e-9))
            return float(entropy)

    def layers_below_entropy(self, min_entropy: float) -> list[tuple[int, float]]:
        """Layers whose routing distribution entropy is below ``min_entropy`` (collapse risk)."""
        out: list[tuple[int, float]] = []
        for layer in range(self.num_layers):
            h = self.get_layer_entropy(layer)
            if h < min_entropy:
                out.append((layer, h))
        return out

    def expert_collapse_risk(
        self, min_entropy: float = 0.5, min_active_mass: float = 1e-3
    ) -> bool:
        """Heuristic: True if any layer looks collapsed (low entropy or one-hot-ish EMA)."""
        for layer in range(self.num_layers):
            if self.get_layer_entropy(layer) < min_entropy:
                return True
            with self.lock:
                row = self.activation_ema[layer]
            if float(np.max(row)) > 1.0 - min_active_mass and float(np.sum(row)) > 0:
                return True
        return False

    def health_summary(self, min_entropy: float = 0.5) -> dict[str, Any]:
        """Structured snapshot for JSONL logging (no I/O here)."""
        risky = self.layers_below_entropy(min_entropy)
        return {
            "min_entropy_threshold": min_entropy,
            "layers_below_entropy": [{"layer": a, "entropy": b} for a, b in risky],
            "collapse_risk": self.expert_collapse_risk(min_entropy=min_entropy),
            "mean_entropy": float(
                np.mean([self.get_layer_entropy(i) for i in range(self.num_layers)])
            ),
        }


class ExpertOffloadManager:
    """LRU-backed resident set for routed expert weights; thread-safe bookkeeping."""

    def __init__(
        self,
        base_path: Path,
        weight_map: dict[str, str],
        expert_key_table: dict[tuple[int, int], dict[str, str]],
        max_resident_experts: int = 16,
        max_cached_shards: int = 4,
        projections: tuple[str, ...] = ("fc1", "fc2"),
    ):
        self.base_path = Path(base_path)
        self.weight_map = weight_map
        self.expert_key_table = expert_key_table
        self.max_resident_experts = max(1, int(max_resident_experts))
        self.max_cached_shards = max(1, int(max_cached_shards))
        self._projections = projections

        self._prefetch_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._lru: OrderedDict[tuple[int, int], None] = OrderedDict()
        self._cache: dict[tuple[int, int], dict[str, mx.array]] = {}
        self._loading: set[tuple[int, int]] = set()
        self._expert_importance: dict[tuple[int, int], float] = {}

        self.predictor = None
        self.prefetch_top_k = (
            2  # conservative default; was 16 which crashed at low max_resident
        )
        self._pinned_tasks = {}
        self.dedekimi_observer = None

        # Persistent shard file handles — avoid repeated open/close
        self._shard_handles: OrderedDict[str, Any] = OrderedDict()

        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.prefill_evals = 0
        self.decode_evals = 0
        self.prefill_hits = 0
        self.prefill_misses = 0
        self.decode_hits = 0
        self.decode_misses = 0
        self.load_time_ms_total = 0.0
        self.load_count = 0
        self.prefetch_requests = 0
        self.prefetch_already_cached = 0

        self._phase: str = "decode"

    def pre_populate_task_clique(
        self, task_name: str, layer_expert_map: dict[int, list[int]]
    ):
        """Asynchronously pre-populate LRU with likely experts for a specific task."""
        with self._lock:
            self._pinned_tasks[task_name] = layer_expert_map

        for layer, experts in layer_expert_map.items():
            self.prefetch(layer, experts)

    def set_phase(self, phase: str) -> None:
        """Hint for prefetch / metrics: prefill vs decode (autoregressive) phases."""
        self._phase = phase if phase in ("prefill", "decode") else "decode"

    def stats_summary(self) -> dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total) if total else 0.0
            prefill_total = self.prefill_hits + self.prefill_misses
            prefill_hit_rate = (
                (self.prefill_hits / prefill_total) if prefill_total else 0.0
            )
            decode_total = self.decode_hits + self.decode_misses
            decode_hit_rate = (self.decode_hits / decode_total) if decode_total else 0.0
            avg_load_ms = (
                (self.load_time_ms_total / self.load_count) if self.load_count else 0.0
            )
            return {
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": hit_rate,
                "prefill_hits": self.prefill_hits,
                "prefill_misses": self.prefill_misses,
                "prefill_hit_rate": prefill_hit_rate,
                "decode_hits": self.decode_hits,
                "decode_misses": self.decode_misses,
                "decode_hit_rate": decode_hit_rate,
                "avg_load_ms": avg_load_ms,
                "load_count": self.load_count,
                "load_time_ms_total": self.load_time_ms_total,
                "resident_slots": len(self._lru),
                "prefill_gather_calls": self.prefill_evals,
                "decode_gather_calls": self.decode_evals,
                "prefetch_requests": self.prefetch_requests,
                "prefetch_already_cached": self.prefetch_already_cached,
            }

    def _record_load_metrics(self, dt_ms: float) -> None:
        """Update load timing counters."""
        with self._lock:
            self.load_time_ms_total += dt_ms
            self.load_count += 1

    def _get_shard_handle(self, shard_name: str):
        """Return cached shard tensor dict, loading on first use via mx.load (handles uint32 + bfloat16)."""
        with self._lock:
            data = self._shard_handles.get(shard_name)
            if data is not None:
                self._shard_handles.move_to_end(shard_name)
                return data
        shard_path = self.base_path / shard_name
        if not shard_path.is_file():
            raise ExpertLoadError(
                f"Missing safetensors shard: {shard_path}",
                shard_path=str(shard_path),
            )
        data = mx.load(str(shard_path))
        with self._lock:
            existing = self._shard_handles.get(shard_name)
            if existing is not None:
                self._shard_handles.move_to_end(shard_name)
                return existing
            self._shard_handles[shard_name] = data
            self._shard_handles.move_to_end(shard_name)
            while len(self._shard_handles) > self.max_cached_shards:
                self._shard_handles.popitem(last=False)
            return data

    def _load_expert_pair_tensors(self, spec: dict[str, str]) -> dict[str, mx.array]:
        """Load projection tensors for one expert using persistent shard handles.

        Uses self._projections to determine which projections to load:
          - Nemotron-H: ("fc1", "fc2") — 2 projections
          - Gemma 4: ("gate", "up", "down") — 3 projections
        """
        result = {}
        t0 = time.perf_counter()
        try:
            for proj in self._projections:
                base_key = spec[f"{proj}_weight"]
                scales_key = spec.get(f"{proj}_scales")
                biases_key = spec.get(f"{proj}_biases")

                # Check if quantized keys are present in the spec
                if scales_key:
                    keys_to_load = [("weight", base_key), ("scales", scales_key)]
                    if biases_key:
                        keys_to_load.append(("biases", biases_key))

                    for suffix, key in keys_to_load:
                        shard = self.weight_map.get(key)
                        if not shard:
                            raise ExpertLoadError(
                                f"No shard mapping for expert key {key!r}",
                                tensor_key=key,
                            )
                        handle = self._get_shard_handle(shard)
                        result[f"{proj}_{suffix}"] = handle[key]

                    if base_key.endswith(".weight_packed"):
                        result[f"{proj}_weight"] = result[f"{proj}_weight"].view(
                            mx.uint32
                        )
                        if not biases_key:
                            result[f"{proj}_biases"] = -8.0 * result[f"{proj}_scales"]
                else:
                    # Dense expert
                    shard = self.weight_map.get(base_key)
                    if not shard:
                        raise ExpertLoadError(
                            f"No shard mapping for expert key {base_key!r}",
                            tensor_key=base_key,
                        )
                    handle = self._get_shard_handle(shard)
                    result[f"{proj}_weight"] = handle[base_key]
        except ExpertLoadError:
            raise
        except Exception as e:
            raise ExpertLoadError(
                f"Failed loading expert tensors: {e}", tensor_key=base_key
            ) from e

        # H3 fix: validate loaded tensor dtypes to catch corrupt checkpoints early.
        for tname, tensor in result.items():
            if tname.endswith("_weight") and tensor.dtype not in (
                mx.uint32,
                mx.float16,
                mx.bfloat16,
                mx.float32,
            ):
                raise ExpertLoadError(
                    f"Unexpected dtype {tensor.dtype} for {tname} "
                    f"(expected uint32/float16/bfloat16/float32)",
                    tensor_key=tname,
                )
            if tname.endswith("_scales") and tensor.dtype not in (
                mx.float16,
                mx.bfloat16,
                mx.float32,
            ):
                raise ExpertLoadError(
                    f"Unexpected dtype {tensor.dtype} for {tname} "
                    f"(expected float16/bfloat16/float32)",
                    tensor_key=tname,
                )

        dt = (time.perf_counter() - t0) * 1000.0
        self._record_load_metrics(dt)
        return result

    def _touch(self, key: tuple[int, int]) -> None:
        if key in self._lru:
            self._lru.move_to_end(key)
        else:
            self._lru[key] = None

    def _record_gather_call(self) -> None:
        if self._phase == "prefill":
            self.prefill_evals += 1
        else:
            self.decode_evals += 1

    def _record_cache_access(self, *, hit: bool) -> None:
        if hit:
            self.hits += 1
            if self._phase == "prefill":
                self.prefill_hits += 1
            else:
                self.decode_hits += 1
        else:
            self.misses += 1
            if self._phase == "prefill":
                self.prefill_misses += 1
            else:
                self.decode_misses += 1

    def update_expert_importance(
        self, importance_scores: dict[tuple[int, int], float]
    ) -> None:
        """Update the importance scores used for eviction decisions."""
        with self._lock:
            for k, v in importance_scores.items():
                self._expert_importance[k] = v

    def _evict_one_unpinned(self, pinned: set[tuple[int, int]]) -> bool:
        """Evict entry not in pinned, prioritizing experts with lowest importance. Returns True if something was evicted."""
        candidates = [k for k in self._lru.keys() if k not in pinned]
        if not candidates:
            return False

        if self._expert_importance:
            # Score combines recency (implicit in candidate list order) and importance.
            # But the simplest is strictly lowest importance first.
            worst_key = min(
                candidates, key=lambda k: self._expert_importance.get(k, 0.0)
            )
        else:
            worst_key = candidates[0]

        self._lru.pop(worst_key, None)
        self._cache.pop(worst_key, None)
        self._expert_importance.pop(worst_key, None)
        self.evictions += 1
        return True

    def _plan_expert_loads(
        self, layer_idx: int, indices: mx.array
    ) -> tuple[mx.array, list[int], list[tuple[tuple[int, int], dict[str, str]]], bool]:
        """Reserve cache slots for any missing experts and return the load plan."""
        with self._lock:
            self._record_gather_call()

        mx.eval(indices)
        flat_mx = indices.reshape(-1)
        unique_eids = sorted(set(flat_mx.tolist()))
        if any(int(eid) < 0 for eid in unique_eids):
            raise ValueError("expert indices must be non-negative")
        pinned = {(layer_idx, int(e)) for e in unique_eids}
        to_load: list[tuple[tuple[int, int], dict[str, str]]] = []
        evicted_any = False

        with self._cond:
            for eid in unique_eids:
                key = (layer_idx, int(eid))
                while key in self._loading:
                    self._cond.wait()
                if key in self._cache:
                    self._record_cache_access(hit=True)
                    self._touch(key)
                    continue

                self._record_cache_access(hit=False)
                while (
                    len(self._cache) + len(self._loading) >= self.max_resident_experts
                    and self._lru
                ):
                    if not self._evict_one_unpinned(pinned):
                        raise MemoryError(
                            f"expert offload: cannot evict enough experts to satisfy --max-resident-experts={self.max_resident_experts} (pinned={len(pinned)} unique experts this layer)"
                        )
                    evicted_any = True
                spec = self.expert_key_table.get(key)
                required_keys = [f"{p}_weight" for p in self._projections]
                if not spec or any(rk not in spec for rk in required_keys):
                    raise ExpertLoadError(
                        f"No expert weight keys indexed for layer={layer_idx} expert={eid}",
                        layer_idx=layer_idx,
                    )
                self._loading.add(key)
                to_load.append((key, spec))

        return flat_mx, unique_eids, to_load, evicted_any

    def prefetch(self, layer_idx: int, expert_ids: list[int]):
        """Asynchronously load experts into cache."""
        for eid in expert_ids:
            key = (layer_idx, int(eid))
            self.prefetch_requests += 1
            with self._cond:
                if key in self._cache or key in self._loading:
                    self.prefetch_already_cached += 1
                    continue

                # Check eviction if we're hitting capacity (best effort for prefetch)
                if (
                    len(self._cache) + len(self._loading) >= self.max_resident_experts
                    and self._lru
                ):
                    self._evict_one_unpinned(set())

                self._loading.add(key)
            self._prefetch_pool.submit(self._prefetch_task, key)

    def _prefetch_task(self, key: tuple[int, int]):
        spec = self.expert_key_table.get(key)
        if not spec:
            with self._cond:
                self._loading.discard(key)
                self._cond.notify_all()
            return
        try:
            tensors = self._load_expert_pair_tensors(spec)
            self._complete_reserved_load(key, tensors)
        except Exception:
            with self._cond:
                self._loading.discard(key)
                self._cond.notify_all()

    def _complete_reserved_load(
        self, key: tuple[int, int], tensors: dict[str, mx.array] | None
    ) -> None:
        with self._cond:
            try:
                if tensors is not None and key not in self._cache:
                    self._cache[key] = tensors
                    self._touch(key)
            finally:
                self._loading.discard(key)
                self._cond.notify_all()

    def prepare_gather(
        self,
        layer_idx: int,
        proj: str,
        indices: mx.array,
        *,
        mode: str = "gather",
    ) -> tuple[mx.array, mx.array]:
        """Return (compact_weight, remapped_indices) for SwitchLinear-style gather_mm."""
        if mode not in ("gather", "loop"):
            mode = "gather"

        flat_mx, unique_eids, to_load, _ = self._plan_expert_loads(layer_idx, indices)

        for key, spec in to_load:
            tensors = None
            try:
                tensors = self._load_expert_pair_tensors(spec)
            finally:
                self._complete_reserved_load(key, tensors)

        with self._lock:
            stacks = [
                self._cache[(layer_idx, int(e))][f"{proj}_weight"] for e in unique_eids
            ]
            compact = mx.stack(stacks, axis=0)

            max_eid = max(unique_eids) + 1
            lut = mx.zeros((max_eid,), dtype=mx.int32)
            unique_arr = mx.array(unique_eids, dtype=mx.int32)
            lut[unique_arr] = mx.arange(len(unique_eids), dtype=mx.int32)
            remapped = lut[flat_mx].reshape(indices.shape)

        return compact, remapped

    def prepare_gather_pair(
        self,
        layer_idx: int,
        indices: mx.array,
        *,
        mode: str = "gather",
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Return (compact_fc1, compact_fc2, remapped_indices) — one LRU pass for both projections."""
        if mode not in ("gather", "loop"):
            mode = "gather"

        flat_mx, unique_eids, to_load, _ = self._plan_expert_loads(layer_idx, indices)

        for key, spec in to_load:
            tensors = None
            try:
                tensors = self._load_expert_pair_tensors(spec)
            finally:
                self._complete_reserved_load(key, tensors)

        with self._lock:
            compacts = {}
            for proj in ("fc1", "fc2"):
                stacks = [
                    self._cache[(layer_idx, int(e))][f"{proj}_weight"]
                    for e in unique_eids
                ]
                compacts[proj] = mx.stack(stacks, axis=0)

            max_eid = max(unique_eids) + 1
            lut = mx.zeros((max_eid,), dtype=mx.int32)
            unique_arr = mx.array(unique_eids, dtype=mx.int32)
            lut[unique_arr] = mx.arange(len(unique_eids), dtype=mx.int32)
            remapped = lut[flat_mx].reshape(indices.shape)

        return compacts["fc1"], compacts["fc2"], remapped

    def prepare_gather_pair_quantized(
        self,
        layer_idx: int,
        indices: mx.array,
        *,
        mode: str = "gather",
    ) -> tuple[
        tuple[mx.array, mx.array, mx.array | None],
        tuple[mx.array, mx.array, mx.array | None],
        mx.array,
    ]:
        """Return ((fc1_w, fc1_s, fc1_b), (fc2_w, fc2_s, fc2_b), remapped_indices) for quantized experts."""
        if mode not in ("gather", "loop"):
            mode = "gather"

        flat_mx, unique_eids, to_load, _ = self._plan_expert_loads(layer_idx, indices)

        for key, spec in to_load:
            tensors = None
            try:
                tensors = self._load_expert_pair_tensors(spec)
            finally:
                self._complete_reserved_load(key, tensors)

        with self._lock:
            compacts = {}
            for proj in ("fc1", "fc2"):
                ws, ss, bs = [], [], []
                for e in unique_eids:
                    entry = self._cache[(layer_idx, int(e))]
                    ws.append(entry[f"{proj}_weight"])
                    ss.append(entry[f"{proj}_scales"])
                    bs.append(entry.get(f"{proj}_biases"))

                compact_w = mx.stack(ws, axis=0)
                compact_s = mx.stack(ss, axis=0)
                if all(b is not None for b in bs):
                    compact_b = mx.stack(bs, axis=0)
                elif any(b is not None for b in bs):
                    raise RuntimeError(
                        f"Mixed presence of biases across experts in layer {layer_idx} proj {proj} is not supported."
                    )
                else:
                    compact_b = None
                compacts[proj] = (compact_w, compact_s, compact_b)

            max_eid = max(unique_eids) + 1
            lut = mx.zeros((max_eid,), dtype=mx.int32)
            unique_arr = mx.array(unique_eids, dtype=mx.int32)
            lut[unique_arr] = mx.arange(len(unique_eids), dtype=mx.int32)
            remapped = lut[flat_mx].reshape(indices.shape)

        return compacts["fc1"], compacts["fc2"], remapped

    def prepare_gather_triple(
        self,
        layer_idx: int,
        indices: mx.array,
        *,
        mode: str = "gather",
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """Return (compact_gate, compact_up, compact_down, remapped_indices) for SwitchGLU."""
        if mode not in ("gather", "loop"):
            mode = "gather"

        flat_mx, unique_eids, to_load, _ = self._plan_expert_loads(layer_idx, indices)

        for key, spec in to_load:
            tensors = None
            try:
                tensors = self._load_expert_pair_tensors(spec)
            finally:
                self._complete_reserved_load(key, tensors)

        with self._lock:
            compacts = {}
            for proj in ("gate", "up", "down"):
                stacks = [
                    self._cache[(layer_idx, int(e))][f"{proj}_weight"]
                    for e in unique_eids
                ]
                compacts[proj] = mx.stack(stacks, axis=0)

            max_eid = max(unique_eids) + 1
            lut = mx.zeros((max_eid,), dtype=mx.int32)
            unique_arr = mx.array(unique_eids, dtype=mx.int32)
            lut[unique_arr] = mx.arange(len(unique_eids), dtype=mx.int32)
            remapped = lut[flat_mx].reshape(indices.shape)

        return compacts["gate"], compacts["up"], compacts["down"], remapped

    def prepare_gather_triple_quantized(
        self,
        layer_idx: int,
        indices: mx.array,
        *,
        mode: str = "gather",
    ) -> tuple[
        tuple[mx.array, mx.array, mx.array | None],
        tuple[mx.array, mx.array, mx.array | None],
        tuple[mx.array, mx.array, mx.array | None],
        mx.array,
    ]:
        """Return ((gate_w, gate_s, gate_b), (up_w, up_s, up_b), (down_w, down_s, down_b), remapped) for quantized SwitchGLU."""
        if mode not in ("gather", "loop"):
            mode = "gather"

        flat_mx, unique_eids, to_load, _ = self._plan_expert_loads(layer_idx, indices)

        for key, spec in to_load:
            tensors = None
            try:
                tensors = self._load_expert_pair_tensors(spec)
            finally:
                self._complete_reserved_load(key, tensors)

        with self._lock:
            compacts = {}
            for proj in ("gate", "up", "down"):
                ws, ss, bs = [], [], []
                for e in unique_eids:
                    entry = self._cache[(layer_idx, int(e))]
                    ws.append(entry[f"{proj}_weight"])
                    ss.append(entry[f"{proj}_scales"])
                    bs.append(entry.get(f"{proj}_biases"))

                compact_w = mx.stack(ws, axis=0)
                compact_s = mx.stack(ss, axis=0)
                if all(b is not None for b in bs):
                    compact_b = mx.stack(bs, axis=0)
                elif any(b is not None for b in bs):
                    raise RuntimeError(
                        f"Mixed presence of biases across experts in layer {layer_idx} proj {proj} is not supported."
                    )
                else:
                    compact_b = None
                compacts[proj] = (compact_w, compact_s, compact_b)

            max_eid = max(unique_eids) + 1
            lut = mx.zeros((max_eid,), dtype=mx.int32)
            unique_arr = mx.array(unique_eids, dtype=mx.int32)
            lut[unique_arr] = mx.arange(len(unique_eids), dtype=mx.int32)
            remapped = lut[flat_mx].reshape(indices.shape)

        return compacts["gate"], compacts["up"], compacts["down"], remapped

    def prepare_gather_quantized(
        self,
        layer_idx: int,
        proj: str,
        indices: mx.array,
        *,
        mode: str = "gather",
    ) -> tuple[mx.array, mx.array, mx.array | None, mx.array]:
        """Return (compact_weight, compact_scales, compact_biases|None, remapped_indices)."""
        fc1, fc2, remapped = self.prepare_gather_pair_quantized(
            layer_idx, indices, mode=mode
        )
        tensors = fc1 if proj == "fc1" else fc2
        w, s, b = tensors
        return w, s, b, remapped

    @property
    def is_quantized(self) -> bool:
        """True if the weight map contains .scales keys for experts."""
        return any(
            (k.endswith(".scales") or k.endswith(".weight_scale"))
            and (
                _NEMOTRON_EXPERT_KEY_RE.match(k)
                or _GEMMA4_EXPERT_KEY_RE.match(k)
                or _QWEN3_EXPERT_KEY_RE.match(k)
                or _KIMI_EXPERT_KEY_RE.match(k)
            )
            for k in self.weight_map
        )

    def reset_metrics(self) -> None:
        with self._lock:
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            self.prefill_evals = 0
            self.decode_evals = 0
            self.prefill_hits = 0
            self.prefill_misses = 0
            self.decode_hits = 0
            self.decode_misses = 0
            self.load_time_ms_total = 0.0
            self.load_count = 0


def attach_expert_offload_manager(
    model: Any, manager: ExpertOffloadManager, *, model_type: str = "nemotron_h"
) -> None:
    """Wire manager into model's expert modules (OffloadSwitchMLP or OffloadSwitchGLU).

    Args:
        model: The loaded model instance.
        manager: The ExpertOffloadManager to attach.
        model_type: Architecture identifier ('nemotron_h' or 'gemma4_text').

    Raises:
        NotImplementedError: If model_type is not yet supported for offloading.
    """
    if model_type not in EXPERT_OFFLOAD_MODEL_TYPES:
        raise NotImplementedError(
            f"Expert offloading is not yet implemented for model_type={model_type!r}. "
            f"Supported types: {sorted(EXPERT_OFFLOAD_MODEL_TYPES)}"
        )

    model.expert_offload_manager = manager

    if model_type == "nemotron_h":
        _attach_nemotron_h(model, manager)
    elif model_type in ("gemma4_text", "gemma4"):
        _attach_gemma4(model, manager)
    elif model_type in ("qwen3_moe", "qwen3_5_moe", "kimi_k25", "kimi_k2"):
        _attach_qwen3_moe(model, manager)


def _attach_qwen3_moe(model: Any, manager: ExpertOffloadManager) -> None:
    model_root = getattr(model, "model", model)
    layers = getattr(model_root, "layers", None)
    if layers is None:
        raise ValueError("_attach_qwen3_moe failed: model.model.layers not found")

    attached = 0
    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        switch_mlp = getattr(mlp, "switch_mlp", None)
        if switch_mlp is not None and hasattr(switch_mlp, "set_expert_manager"):
            switch_mlp.set_expert_manager(manager, i)
            attached += 1

    if attached == 0:
        print(
            "[WARN] _attach_qwen3_moe: no SwitchGLU modules with set_expert_manager found. "
            "Expert offloading requires OffloadSwitchGLU modules."
        )


def _attach_nemotron_h(model: Any, manager: ExpertOffloadManager) -> None:
    """Wire manager into Nemotron-H OffloadSwitchMLP modules."""
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return
    layers = getattr(backbone, "layers", None)
    if not layers:
        return
    for layer in layers:
        if getattr(layer, "block_type", None) != "E":
            continue
        mixer = getattr(layer, "mixer", None)
        smlp = getattr(mixer, "switch_mlp", None)
        if smlp is not None and hasattr(smlp, "set_expert_manager"):
            smlp.set_expert_manager(manager, getattr(layer, "layer_idx", -1))


def _attach_gemma4(model: Any, manager: ExpertOffloadManager) -> None:
    """Wire manager into Gemma 4 SwitchGLU modules.

    Handles both:
      - gemma4 (multimodal): model.language_model.model.layers[L].experts.switch_glu
      - gemma4_text (text-only): model.model.layers[L].experts.switch_glu
    Only layers with enable_moe=True have experts.
    """
    # Navigate to the text model's layers
    lang_model = getattr(model, "language_model", None)
    if lang_model is not None:
        # gemma4 multimodal wrapper
        model_root = getattr(lang_model, "model", lang_model)
    else:
        # gemma4_text direct
        model_root = getattr(model, "model", model)
    layers = getattr(model_root, "layers", None)
    if not layers:
        return
    attached = 0
    for i, layer in enumerate(layers):
        if not getattr(layer, "enable_moe", False):
            continue
        experts = getattr(layer, "experts", None)
        if experts is None:
            continue
        switch_glu = getattr(experts, "switch_glu", None)
        if switch_glu is not None and hasattr(switch_glu, "set_expert_manager"):
            switch_glu.set_expert_manager(manager, i)
            attached += 1
    if attached == 0:
        print(
            "[WARN] _attach_gemma4: no SwitchGLU modules with set_expert_manager found. "
            "Expert offloading requires OffloadSwitchGLU modules."
        )
