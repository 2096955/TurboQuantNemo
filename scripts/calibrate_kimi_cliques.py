#!/usr/bin/env python3
"""Offline expert-clique calibration for Kimi K2.6.

Records which experts are activated per (layer, task category) during a short
decode run on a small, representative prompt catalog, then writes the top-K
experts per layer for each task to a ``task_expert_cliques.json`` file in the
schema consumed by ``ExpertOffloadManager.pre_populate_task_clique`` and the
``--task-expert-cliques-file`` CLI flag.

Output schema (mirrors ``results/gemma4_task_expert_cliques_min.json``)::

    {
      "code":      {"0": [3, 17, 42, ...], "1": [...], ...},
      "reasoning": {"0": [...], ...},
      "general":   {"0": [...], ...},
      "math":      {"0": [...], ...}
    }

Keys are task names (str). Values map ``str(layer_idx)`` → list of expert ids
(``int``). This is the exact shape consumed by
``ExpertOffloadManager.pre_populate_task_clique(task_name, layer_expert_map)``
in ``mlx-lm/mlx_lm/expert_offload.py`` (the manager iterates ``items()`` of the
dict and calls ``self.prefetch(layer, experts)`` for each entry, so layer keys
are coerced to int by the consumer at lookup time but JSON keys must be
strings — same convention as the gemma4 reference file).

----------------------------------------------------------------------------
Recording mechanism
----------------------------------------------------------------------------
Kimi K2.6 (model_type ``kimi_k25``) routes through ``DeepseekV3Model`` →
``DeepseekV3MoE`` (``mlx-lm/mlx_lm/models/deepseek_v3.py:259``). Unlike
``Qwen3MoeSparseMoeBlock`` (``mlx_lm/models/qwen3_moe.py:159``),
``Gemma4MoE`` (``mlx_lm/models/gemma4_text.py:412``), and Nemotron-H
(``mlx_lm/models/nemotron_h.py:464``), the DeepseekV3 MoE forward does **not**
call ``mgr.dedekimi_observer.record_activation``. So
``manager.dedekimi_observer.activation_ema`` stays all-zero on Kimi K2.6 even
when ``use_dedekimi_observer=True`` is passed at load time.

To stay non-intrusive (per task brief: do not edit ``expert_offload.py``), we
spy at the ExpertOffloadManager level instead. When expert offload is enabled
on Kimi K2.6, ``OffloadQuantizedSwitchGLU.__call__``
(``mlx_lm/models/switch_layers.py:706``) calls
``mgr.prepare_gather_triple_quantized(layer_idx, indices, mode="gather")``
on every MoE forward — which is exactly the per-(layer, indices) signal we
need. We monkey-patch that single bound method on the manager *instance* (not
the class) for the duration of this script and accumulate counts into a
``dict[(layer_idx, expert_id), int]`` table. No mlx-lm files are modified.

----------------------------------------------------------------------------
Decisions logged for the controller
----------------------------------------------------------------------------
* ``--top-k 8`` mirrors Kimi top-K routing; clique stores the 8 most-active
  experts per layer per task.
* Activation counts are per-token-routed appearances (``int(idx)`` count from
  the indices tensor passed to the gather), not raw forward calls. This is
  what ``pre_populate_task_clique`` ultimately wants to bias prefetch.
* Counters are reset between task categories so each clique is task-pure.
* ``manager.dedekimi_observer.activation_ema`` is also captured if it was
  populated — currently it is not for Kimi K2.6 (see gap above), but if the
  recording hook is added later this script will use the EMA path
  automatically.
* If the ``ExpertOffloadManager`` cannot be located on the loaded model
  (``model.expert_offload_manager`` is missing), the script fails fast — the
  whole point of calibration is to record from offload paths.

This script does **not** execute the model; it is run by the controller after
the layered A/B finishes (~30 min on 128 GB M4 Max).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add mlx-lm to path so this script works from a fresh checkout without
# requiring an editable install in every venv.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "mlx-lm"))

import mlx.core as mx  # noqa: E402  (after sys.path mutation)


# Built-in calibration catalog — kept short by design (calibration not benchmark).
TASK_PROMPTS: Dict[str, List[str]] = {
    "code": [
        "Write a Python function that returns the nth Fibonacci number.",
        "Implement quicksort in C++.",
        "Show me a SQL query that joins three tables.",
    ],
    "reasoning": [
        "If all roses are flowers and some flowers fade quickly, "
        "can we conclude all roses fade quickly? Explain.",
        "A train leaves Chicago at 60 mph. Another leaves NY at 80 mph. "
        "They are 800 miles apart. When do they meet?",
        "Why does ice float on water?",
    ],
    "general": [
        "Tell me about the history of the steam engine.",
        "What are the main differences between Python and Rust?",
        "Summarize the plot of Hamlet.",
    ],
    "math": [
        "Solve for x: 3x + 7 = 22.",
        "What is the integral of x^2 from 0 to 5?",
        "Prove that the square root of 2 is irrational.",
    ],
}


# ---------------------------------------------------------------------------
# Spy layer over ExpertOffloadManager
# ---------------------------------------------------------------------------


class ExpertActivationSpy:
    """Non-intrusive activation counter for an ``ExpertOffloadManager``.

    Wraps ``prepare_gather_triple_quantized`` on a single manager instance
    (not the class) so calibration leaves no global side effects. Records
    per-(layer_idx, expert_id) counts of how many routing positions selected
    each expert during the wrapped window.
    """

    def __init__(self, manager: Any):
        self.manager = manager
        # (layer_idx, expert_id) -> count of routed-token appearances.
        self.counts: Dict[Tuple[int, int], int] = defaultdict(int)
        self._original = None
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        target_name = "prepare_gather_triple_quantized"
        original = getattr(self.manager, target_name, None)
        if original is None:
            raise RuntimeError(
                f"ExpertOffloadManager has no method {target_name!r}; "
                "cannot install activation spy."
            )
        self._original = original

        def wrapped(layer_idx, indices, *, mode="gather"):
            try:
                self._record(int(layer_idx), indices)
            except Exception as exc:  # noqa: BLE001
                # Never let calibration spying corrupt the forward pass.
                print(f"[spy] record failed (ignored): {exc}", file=sys.stderr)
            return original(layer_idx, indices, mode=mode)

        # Bind on the instance so we don't perturb other manager instances or
        # any concurrent users (there are none in calibration, but be safe).
        setattr(self.manager, target_name, wrapped)
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        if self._original is not None:
            setattr(self.manager, "prepare_gather_triple_quantized", self._original)
        self._original = None
        self._installed = False

    def reset(self) -> None:
        self.counts.clear()

    def _record(self, layer_idx: int, indices: mx.array) -> None:
        # ``indices`` shape on the Kimi/DeepseekV3 path is roughly
        # [batch, seq_len, top_k]; we flatten and accumulate per-id counts.
        flat = mx.reshape(indices, (-1,))
        mx.eval(flat)
        ids: List[int] = flat.tolist()  # type: ignore[assignment]
        if not isinstance(ids, list):
            # Single scalar -> wrap.
            ids = [int(ids)]
        for eid in ids:
            self.counts[(layer_idx, int(eid))] += 1

    # ------------------------------------------------------------------
    # Snapshot / clique extraction
    # ------------------------------------------------------------------

    def snapshot_clique(self, top_k: int) -> Dict[str, List[int]]:
        """Return ``{str(layer_idx): [expert_id, ...]}`` for the current counts.

        Per-layer experts are sorted by descending count; ties broken by
        expert id ascending (stable across runs). Layers never observed are
        omitted from the output (matches the gemma4 reference file convention,
        which only includes layers with non-empty cliques).
        """
        per_layer: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for (layer, eid), count in self.counts.items():
            per_layer[layer].append((count, eid))

        out: Dict[str, List[int]] = {}
        for layer in sorted(per_layer):
            entries = per_layer[layer]
            # Sort by (-count, eid) for stable, descending-importance order.
            entries.sort(key=lambda ce: (-ce[0], ce[1]))
            top = [eid for (_count, eid) in entries[:top_k]]
            if top:
                out[str(layer)] = top
        return out


# ---------------------------------------------------------------------------
# Model loading + decode helpers
# ---------------------------------------------------------------------------


def _resolve_manager(model: Any) -> Any:
    """Return the ``ExpertOffloadManager`` attached to ``model``, or raise."""
    mgr = getattr(model, "expert_offload_manager", None)
    if mgr is None:
        raise RuntimeError(
            "Loaded model has no ``expert_offload_manager`` attribute. "
            "Calibration requires expert_offload=True (see --max-resident)."
        )
    return mgr


def _build_prompt_ids(tokenizer: Any, prompt: str) -> mx.array:
    """Encode ``prompt`` via the tokenizer's chat template if available."""
    apply_chat = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat):
        try:
            ids = apply_chat(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
            )
        except Exception:  # noqa: BLE001
            ids = tokenizer.encode(prompt)
    else:
        ids = tokenizer.encode(prompt)
    if isinstance(ids, mx.array):
        return ids
    return mx.array(list(ids))


def _decode_one(
    model: Any,
    tokenizer: Any,
    prompt: str,
    decode_tokens: int,
    generate_step,
) -> int:
    """Run prefill + ``decode_tokens`` decode steps; return tokens decoded."""
    prompt_ids = _build_prompt_ids(tokenizer, prompt)
    decoded = 0
    for n, (token, _logprobs) in enumerate(
        generate_step(
            prompt_ids,
            model,
            max_tokens=decode_tokens,
        )
    ):
        mx.eval(token)
        decoded = n + 1
        if decoded >= decode_tokens:
            break
    return decoded


def _capture_observer_ema(manager: Any) -> Dict[str, List[int]] | None:
    """If the DedeKimi observer is wired AND populated, return top-K-ready EMA.

    Returns ``None`` when the observer is absent or all-zero (the current
    Kimi K2.6 reality — see module docstring). Caller falls back to spy counts.
    """
    observer = getattr(manager, "dedekimi_observer", None)
    if observer is None:
        return None
    ema = getattr(observer, "activation_ema", None)
    if ema is None:
        return None
    try:
        import numpy as np

        if not isinstance(ema, np.ndarray):
            return None
        if float(np.abs(ema).sum()) == 0.0:
            return None
        # Hand back a per-layer ranked list (caller slices to top-K).
        ranked: Dict[str, List[int]] = {}
        num_layers = ema.shape[0]
        for layer in range(num_layers):
            order = np.argsort(-ema[layer])
            ranked[str(layer)] = [int(e) for e in order if ema[layer, int(e)] > 0]
        return ranked
    except Exception as exc:  # noqa: BLE001
        print(f"[observer] EMA capture failed (ignored): {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline expert-clique calibration for Kimi K2.6"
    )
    p.add_argument(
        "--model",
        default="/Volumes/Samsung9904tb/Kimi-K2.6",
        help="Path or HF id of the Kimi K2.6 checkpoint.",
    )
    p.add_argument(
        "--max-resident",
        type=int,
        default=200,
        help="max_resident_experts for ExpertOffloadManager (calibration only).",
    )
    p.add_argument(
        "--decode-tokens",
        type=int,
        default=64,
        help="Tokens to decode per prompt (calibration window).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Top-K experts per layer to record into each task clique.",
    )
    p.add_argument(
        "--output",
        default=(
            "/Users/anthonylui/QwenCoderLocal/artifacts/"
            "kimi_k26_cliques/kimi_cliques.json"
        ),
        help="Where to write the task_expert_cliques.json file.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (mx.random.seed) for reproducible decoding.",
    )
    p.add_argument(
        "--expert-offload-dir",
        default=None,
        help=(
            "Optional repacked-experts directory. If unset, the manager loads "
            "from the model checkpoint dir."
        ),
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)

    # Imported lazily so --help works even if mlx-lm isn't fully importable.
    from mlx_lm import load
    from mlx_lm.generate import generate_step

    mx.random.seed(args.seed)

    model_config: Dict[str, Any] = {
        "expert_offload": True,
        "max_resident_experts": int(args.max_resident),
        "use_dedekimi_observer": True,  # harmless even when unused on Kimi
    }
    if args.expert_offload_dir:
        model_config["expert_offload_dir"] = args.expert_offload_dir

    print(
        f"[calibrate] loading {args.model} (expert_offload=True, "
        f"max_resident={args.max_resident})..."
    )
    t0 = time.perf_counter()
    model, tokenizer = load(args.model, model_config=model_config)
    print(f"[calibrate] load complete in {time.perf_counter() - t0:.2f}s")

    manager = _resolve_manager(model)
    spy = ExpertActivationSpy(manager)
    spy.install()

    cliques: Dict[str, Dict[str, List[int]]] = {}
    metadata: Dict[str, Any] = {
        "model": args.model,
        "max_resident_experts": args.max_resident,
        "decode_tokens_per_prompt": args.decode_tokens,
        "top_k": args.top_k,
        "seed": args.seed,
        "task_categories": list(TASK_PROMPTS.keys()),
        "recording_mechanism": (
            "ExpertOffloadManager.prepare_gather_triple_quantized spy "
            "(monkey-patched per-instance; non-intrusive)."
        ),
    }

    try:
        for task, prompts in TASK_PROMPTS.items():
            print(
                f"\n[calibrate] === task={task!r} "
                f"({len(prompts)} prompts × {args.decode_tokens} tok) ==="
            )
            spy.reset()
            t_task = time.perf_counter()
            for i, prompt in enumerate(prompts):
                t_p = time.perf_counter()
                n = _decode_one(
                    model, tokenizer, prompt, args.decode_tokens, generate_step
                )
                dt = time.perf_counter() - t_p
                print(
                    f"  [{i + 1}/{len(prompts)}] decoded {n} tok in "
                    f"{dt:.2f}s — {n / max(dt, 1e-6):.2f} tok/s"
                )

            # Prefer DedeKimi EMA if the model populated it (it currently
            # doesn't on Kimi K2.6 — see module docstring). Spy counts are
            # the working source of truth.
            ema_clique = _capture_observer_ema(manager)
            if ema_clique is not None:
                clique = {
                    layer: ranked[: args.top_k]
                    for layer, ranked in ema_clique.items()
                    if ranked
                }
                source = "dedekimi_observer.activation_ema"
            else:
                clique = spy.snapshot_clique(top_k=args.top_k)
                source = "spy.prepare_gather_triple_quantized"
            cliques[task] = clique
            print(
                f"[calibrate] task={task!r} done in "
                f"{time.perf_counter() - t_task:.2f}s — "
                f"{len(clique)} layers populated (source={source})"
            )
    finally:
        spy.uninstall()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Match the gemma4 reference shape exactly: top-level dict keyed by task
    # name, values are dict[str(layer_idx), list[int expert_id]]. Metadata is
    # written to a sidecar file so the consumed JSON stays schema-pure.
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(cliques, fh, indent=2, sort_keys=True)
    print(f"\n[calibrate] wrote cliques -> {out_path}")

    sidecar_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    metadata["task_layer_counts"] = {
        task: len(layer_map) for task, layer_map in cliques.items()
    }
    metadata["task_expert_counts"] = {
        task: sum(len(v) for v in layer_map.values())
        for task, layer_map in cliques.items()
    }
    with sidecar_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)
    print(f"[calibrate] wrote metadata -> {sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
