# Copyright © 2026 Apple Inc.

"""Expert weight offloading for large MoE models (LRU + per-key safetensors loads)."""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import mlx.core as mx

_EXPERT_KEY_RE = re.compile(
    r"^backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(up_proj|down_proj)\.(weight|scales|biases)$"
)


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
    """Return (layer_idx, expert_id, 'up_proj'|'down_proj', suffix) or None if not a routed expert weight."""
    m = _EXPERT_KEY_RE.match(key)
    if m is None:
        return None
    proj = m.group(3)
    suffix = m.group(4)
    return int(m.group(1)), int(m.group(2)), proj, suffix


def is_nemotron_routed_expert_weight_key(key: str) -> bool:
    return parse_nemotron_expert_key(key) is not None


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
    """Predicts required experts from AttnRes block attention weights."""
    def __init__(self, num_blocks: int, num_experts: int, num_layers: int):
        self.block_expert_affinity = mx.zeros((num_layers, num_blocks, num_experts))
        self.observation_count = mx.zeros((num_layers, num_blocks))

    def record_activation(self, layer_idx: int, block_attention_weights: mx.array, expert_ids: mx.array):
        """
        Record the actual expert activations given the block attention weights.
        block_attention_weights: [num_blocks, B, T]
        expert_ids: [..., top_k]
        """
        # Aggregate alpha over batch and sequence
        alpha = block_attention_weights.mean(axis=(1, 2))  # [num_blocks]
        
        # Flatten expert_ids and get unique
        mx.eval(expert_ids)
        flat_eids_list = list(set(expert_ids.flatten().tolist()))
        flat_eids = mx.array(flat_eids_list, dtype=mx.int32)
        
        affinity = self.block_expert_affinity[layer_idx]
        affinity[:, flat_eids] = affinity[:, flat_eids] + alpha[:, None]
        self.block_expert_affinity[layer_idx] = affinity
        
        self.observation_count[layer_idx] = self.observation_count[layer_idx] + 1

    def predict_experts(self, layer_idx: int, block_attention_weights: mx.array, top_k: int = 16) -> list[int]:
        """
        Predict the top_k experts likely needed.
        """
        alpha = block_attention_weights.mean(axis=(1, 2))  # [num_blocks]
        
        affinity = self.block_expert_affinity[layer_idx]  # [num_blocks, num_experts]
        
        # scores = alpha @ affinity
        scores = mx.matmul(alpha, affinity)
        
        counts = self.observation_count[layer_idx]  # [num_blocks]
        # Avoid division by zero
        total_counts = mx.matmul(alpha, counts)
        
        # If total_counts > 0, normalize
        scores = mx.where(total_counts > 0, scores / total_counts, scores)
        
        predicted = mx.argsort(scores)[-top_k:]
        mx.eval(predicted)
        return predicted.tolist()


class ExpertOffloadManager:
    """LRU-backed resident set for routed expert weights; thread-safe bookkeeping."""

    def __init__(
        self,
        base_path: Path,
        weight_map: dict[str, str],
        expert_key_table: dict[tuple[int, int], dict[str, str]],
        max_resident_experts: int = 16,
        max_cached_shards: int = 4,
    ):
        self.base_path = Path(base_path)
        self.weight_map = weight_map
        self.expert_key_table = expert_key_table
        self.max_resident_experts = max(1, int(max_resident_experts))
        self.max_cached_shards = max(1, int(max_cached_shards))

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._lru: OrderedDict[tuple[int, int], None] = OrderedDict()
        self._cache: dict[tuple[int, int], dict[str, mx.array]] = {}
        self._loading: set[tuple[int, int]] = set()

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

        self._phase: str = "decode"

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
                "resident_slots": len(self._lru),
                "prefill_gather_calls": self.prefill_evals,
                "decode_gather_calls": self.decode_evals,
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
        """Load fc1/fc2 for one expert using persistent shard handles. Quantized = 6 tensors, dense = 2."""
        result = {}
        t0 = time.perf_counter()
        try:
            for proj in ("fc1", "fc2"):
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

    def _evict_one_unpinned(self, pinned: set[tuple[int, int]]) -> bool:
        """Evict LRU entry not in pinned. Returns True if something was evicted."""
        for k in list(self._lru.keys()):
            if k in pinned:
                continue
            self._lru.pop(k, None)
            self._cache.pop(k, None)
            self.evictions += 1
            return True
        return False

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
                if not spec or "fc1_weight" not in spec or "fc2_weight" not in spec:
                    raise ExpertLoadError(
                        f"No expert weight keys indexed for layer={layer_idx} expert={eid}",
                        layer_idx=layer_idx,
                    )
                self._loading.add(key)
                to_load.append((key, spec))

        return flat_mx, unique_eids, to_load, evicted_any

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
            k.endswith(".scales") and is_nemotron_routed_expert_weight_key(k)
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


def attach_expert_offload_manager(model: Any, manager: ExpertOffloadManager) -> None:
    """Wire manager into Nemotron-H OffloadSwitchMLP modules."""
    model.expert_offload_manager = manager
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
