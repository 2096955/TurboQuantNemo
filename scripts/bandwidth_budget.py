"""
bandwidth_budget.py — predict and measure decode throughput ceilings for
RotaryQuant pathway models on Apple Silicon.

The point of this script is to make the bandwidth-bound argument testable
rather than rhetorical. Two modes:

  1. Predict: given a model + hardware + context + bits + hit-rate, print the
     theoretical ceiling under sensible assumptions about active parameters,
     KV traffic, and expert-miss NVMe traffic.

  2. Measure: import BandwidthMeter from this module into the mlx-lm fork,
     wrap a decode loop, emit per-token byte counts categorised as
     kv_compress / kv_read / weight_load / expert_miss / other.
     The meter prints predicted-vs-measured headroom every N tokens.

Usage:
    python bandwidth_budget.py --model kimi-k2-6 --hardware m4-max
    python bandwidth_budget.py --model kimi-k2-6 --hardware m4-max --sweep
    python bandwidth_budget.py --model nemotron-h-120b --hardware m4-max-32gb \
        --context 8192 --hit-rate 0.95 --bits-expert 4

To plug into mlx-lm decode loop:
    from bandwidth_budget import BandwidthMeter
    meter = BandwidthMeter("kimi-k2-6", "m4-max")
    for tok_id in range(n_tokens):
        with meter.token():
            meter.add("weight_load", weight_bytes)
            meter.add("kv_read", kv_bytes)
            meter.add("expert_miss", miss_bytes)
            ...  # actual decode step
    print(meter.report())
    meter.to_json("results/bandwidth_kimi_k26_m4max.json")

All numbers are deliberately conservative back-of-envelope. Fix the model
specs to your actual checkpoint configs before quoting them publicly.
"""

from __future__ import annotations
import argparse
import contextlib
import json
import sys
import time
from dataclasses import dataclass
from typing import Optional


# -----------------------------------------------------------------------------
# Hardware profiles — Apple Silicon. Extend as needed.
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class HardwareProfile:
    name: str
    mem_bandwidth_gbs: float       # LPDDR5x peak, GB/s
    sustained_fraction: float      # realistic fraction of peak under decode load
    nvme_read_gbs: float           # sustained sequential NVMe read, GB/s
    total_ram_gb: float
    gpu_cores: int
    note: str = ""

    @property
    def usable_bandwidth_gbs(self) -> float:
        return self.mem_bandwidth_gbs * self.sustained_fraction


HARDWARE: dict[str, HardwareProfile] = {
    "m4-max": HardwareProfile(
        name="M4 Max (128 GB, 40 GPU cores)",
        mem_bandwidth_gbs=546.0,        # Apple spec: 546 GB/s LPDDR5x
        sustained_fraction=0.65,        # decode workloads rarely exceed this
        nvme_read_gbs=6.0,              # sustained sequential, post-cache
        total_ram_gb=128.0,
        gpu_cores=40,
        note=("Reference Kimi K2.6 host. Apple's published 546 GB/s is peak; "
              "real workloads see 60-70% sustained."),
    ),
    "m4-max-32gb": HardwareProfile(
        name="M4 Max (32 GB, 30 GPU cores)",
        mem_bandwidth_gbs=410.0,
        sustained_fraction=0.65,
        nvme_read_gbs=6.0,
        total_ram_gb=32.0,
        gpu_cores=30,
    ),
    "m3-max": HardwareProfile(
        name="M3 Max (128 GB)",
        mem_bandwidth_gbs=400.0,
        sustained_fraction=0.65,
        nvme_read_gbs=5.5,
        total_ram_gb=128.0,
        gpu_cores=40,
    ),
    "m2-ultra": HardwareProfile(
        name="M2 Ultra (192 GB)",
        mem_bandwidth_gbs=800.0,
        sustained_fraction=0.55,        # fusion drops sustained fraction
        nvme_read_gbs=7.0,
        total_ram_gb=192.0,
        gpu_cores=76,
    ),
}


# -----------------------------------------------------------------------------
# Model specs — pathway models + Kimi targets
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    name: str
    total_params_b: float          # billions
    active_params_b: float         # billions per token (dense + active experts + shared)
    layers: int
    moe_layers: int                # subset of layers that are MoE
    experts_per_layer: int
    experts_per_token: int         # top-k routing
    shared_experts: int
    hidden: int
    head_dim: int
    n_q_heads: int
    n_kv_heads: int
    attention_kind: str            # "gqa" | "mla" | "deltanet_hybrid"
    expert_size_mb: float = 0.0    # per single routed expert at INT4
    # MLA-specific:
    mla_content_dim: int = 0       # compressible content portion of cache entry
    mla_rope_dim: int = 0          # FP16-preserved RoPE portion
    note: str = ""

    @property
    def mla_total_dim(self) -> int:
        return self.mla_content_dim + self.mla_rope_dim


MODELS: dict[str, ModelSpec] = {
    # Kimi K2.6 - architecture not yet published; treated as K2.5-class until confirmed
    "kimi-k2-6": ModelSpec(
        name="Kimi K2.6 (provisional, K2.5-class)",
        total_params_b=1040.0, active_params_b=32.0,
        layers=61, moe_layers=60,
        experts_per_layer=384, experts_per_token=8, shared_experts=1,
        hidden=7168, head_dim=128, n_q_heads=128, n_kv_heads=128,
        attention_kind="mla",
        expert_size_mb=17.6,        # 3 * 7168 * 2048 * 4/8 / 1e6
        mla_content_dim=448, mla_rope_dim=64,
        note=("Architecture assumed identical to K2.5 until K2.6 spec is "
              "published. Re-validate dims before committing kernels."),
    ),
    "kimi-k2-5": ModelSpec(
        name="Kimi K2.5",
        total_params_b=1040.0, active_params_b=32.0,
        layers=61, moe_layers=60,
        experts_per_layer=384, experts_per_token=8, shared_experts=1,
        hidden=7168, head_dim=128, n_q_heads=128, n_kv_heads=128,
        attention_kind="mla",
        expert_size_mb=17.6,
        mla_content_dim=448, mla_rope_dim=64,
    ),
    "nemotron-h-120b": ModelSpec(
        name="Nemotron-H 120B (hybrid Mamba+MoE)",
        total_params_b=120.0, active_params_b=12.0,
        layers=96, moe_layers=48,
        experts_per_layer=64, experts_per_token=6, shared_experts=1,
        hidden=4096, head_dim=128, n_q_heads=32, n_kv_heads=8,
        attention_kind="gqa",
        expert_size_mb=5.0,
        note="SSM layers reduce KV bandwidth materially.",
    ),
    "gemma-4-26b": ModelSpec(
        name="Gemma 4 26B-A4B",
        total_params_b=26.0, active_params_b=4.0,
        layers=30, moe_layers=30,
        experts_per_layer=128, experts_per_token=8, shared_experts=0,
        hidden=2816, head_dim=256, n_q_heads=24, n_kv_heads=8,
        attention_kind="gqa",
        expert_size_mb=3.0,
        note=("25/30 layers are sliding-window (1024 tokens). KV traffic is "
              "dominated by the 5 global-attention layers at long context."),
    ),
    "qwen-3-6-27b": ModelSpec(
        name="Qwen 3.6-27B (dense hybrid)",
        total_params_b=27.0, active_params_b=27.0,
        layers=64, moe_layers=0,
        experts_per_layer=0, experts_per_token=0, shared_experts=0,
        hidden=5120, head_dim=256, n_q_heads=24, n_kv_heads=4,
        attention_kind="deltanet_hybrid",
        expert_size_mb=0.0,
        note=("Only 16/64 layers carry conventional KV cache. The other 48 "
              "are Gated DeltaNet with fixed-size state."),
    ),
    "qwen-3-6-35b-a3b": ModelSpec(
        name="Qwen 3.6-35B-A3B",
        total_params_b=35.0, active_params_b=3.0,
        layers=40, moe_layers=10,
        experts_per_layer=256, experts_per_token=8, shared_experts=1,
        hidden=4096, head_dim=128, n_q_heads=32, n_kv_heads=4,
        attention_kind="gqa",
        expert_size_mb=2.5,
        note="30 DeltaNet + 10 full-attention layers; KV only on the 10.",
    ),
}


# -----------------------------------------------------------------------------
# Budget computation
# -----------------------------------------------------------------------------

@dataclass
class BandwidthBudget:
    """
    Per-token byte traffic broken into the three big consumers:
      - active_weights : weights moved through compute every token (dense +
                         active routed experts + shared expert), all at
                         bits_expert precision for simplicity
      - kv_traffic     : cached K/V bytes read each decode step
      - expert_miss    : NVMe-streamed expert weights on cold cache misses

    RAM and NVMe transfers can overlap on Apple Silicon, so the predicted
    ceiling is set by max(ram_seconds, nvme_seconds), not the sum.
    """
    model: ModelSpec
    hardware: HardwareProfile
    context_len: int
    bits_dense: int
    bits_expert: int
    bits_kv: int                     # effective KV bits after IsoQuant
    expert_hit_rate: float           # 0.0..1.0; 1.0 = all active experts resident

    def active_weight_bytes(self) -> float:
        # Active params * bits/8. Uses bits_expert as the dominant precision
        # because routed experts vastly outnumber dense weights in active count.
        return self.model.active_params_b * 1e9 * self.bits_expert / 8

    def kv_bytes_per_token(self) -> float:
        m = self.model
        if m.attention_kind == "mla":
            # Content compressed via IsoQuant, RoPE preserved at FP16.
            # See DKV_KERNEL_SPEC.md for the rationale.
            per_layer = (
                m.mla_content_dim * self.bits_kv / 8
                + m.mla_rope_dim * 2          # FP16 = 2 bytes
            )
            n_layers = m.layers
        elif m.attention_kind == "gqa":
            per_layer = 2 * m.n_kv_heads * m.head_dim * self.bits_kv / 8
            n_layers = m.layers
        elif m.attention_kind == "deltanet_hybrid":
            # Only ~1/4 of layers carry conventional KV
            per_layer = 2 * m.n_kv_heads * m.head_dim * self.bits_kv / 8
            n_layers = max(1, m.layers // 4)
        else:
            per_layer = 2 * m.n_kv_heads * m.head_dim * self.bits_kv / 8
            n_layers = m.layers
        return per_layer * n_layers * self.context_len

    def expert_miss_bytes(self) -> float:
        m = self.model
        if m.experts_per_layer == 0 or m.expert_size_mb == 0:
            return 0.0
        misses_per_token = (
            m.experts_per_token * m.moe_layers * (1.0 - self.expert_hit_rate)
        )
        bits_factor = self.bits_expert / 4.0   # expert_size_mb is at INT4
        return misses_per_token * m.expert_size_mb * 1e6 * bits_factor

    def total_bytes_per_token(self) -> float:
        return (
            self.active_weight_bytes()
            + self.kv_bytes_per_token()
            + self.expert_miss_bytes()
        )

    def predicted_tok_per_sec(self) -> float:
        ram_s = (self.active_weight_bytes() + self.kv_bytes_per_token()) / (
            self.hardware.usable_bandwidth_gbs * 1e9
        )
        nvme_s = self.expert_miss_bytes() / (self.hardware.nvme_read_gbs * 1e9)
        bottleneck = max(ram_s, nvme_s)
        return 1.0 / bottleneck if bottleneck > 0 else float("inf")

    def report(self) -> str:
        a = self.active_weight_bytes() / 1e9
        k = self.kv_bytes_per_token() / 1e9
        e = self.expert_miss_bytes() / 1e9
        ceil = self.predicted_tok_per_sec()
        ram_demand = (a + k) * ceil
        nvme_demand = e * ceil
        ram_frac = ram_demand / self.hardware.usable_bandwidth_gbs
        nvme_frac = (nvme_demand / self.hardware.nvme_read_gbs
                     if self.hardware.nvme_read_gbs > 0 else 0)
        bottleneck = "NVMe (expert misses)" if nvme_frac > ram_frac else "RAM (weights+KV)"
        return (
            f"\n=== Bandwidth budget: {self.model.name} on {self.hardware.name} ===\n"
            f"Context: {self.context_len:>10,d} tokens   "
            f"bits dense/expert/kv: {self.bits_dense}/{self.bits_expert}/{self.bits_kv}   "
            f"hit rate: {self.expert_hit_rate:.0%}\n\n"
            f"Per-token bytes moved:\n"
            f"  active weights : {a:7.3f} GB/token\n"
            f"  KV cache reads : {k:7.3f} GB/token   (T={self.context_len:,})\n"
            f"  expert misses  : {e:7.3f} GB/token   (NVMe path)\n\n"
            f"Predicted ceiling: {ceil:6.2f} tok/s\n"
            f"  RAM demand     : {ram_demand:6.1f} GB/s "
            f"(of {self.hardware.usable_bandwidth_gbs:.0f} GB/s usable)\n"
            f"  NVMe demand    : {nvme_demand:6.2f} GB/s "
            f"(of {self.hardware.nvme_read_gbs:.0f} GB/s sustained)\n"
            f"  bottleneck     : {bottleneck}\n"
        )


# -----------------------------------------------------------------------------
# Live measurement instrumentation
# -----------------------------------------------------------------------------

class BandwidthMeter:
    """
    Drop-in instrumentation for a decode loop. Categories:
      kv_compress  : bytes written to packed KV cache (compress + pack)
      kv_read      : bytes read from packed KV cache per attention call
      weight_load  : bytes read from RAM for active expert/dense weights
      expert_miss  : bytes read from NVMe for cold expert loads
      other        : everything else (activations, RoPE, buffers, etc.)

    Usage:
        meter = BandwidthMeter("kimi-k2-6", "m4-max")
        for _ in range(n_tokens):
            with meter.token():
                meter.add("weight_load", n_bytes)
                meter.add("kv_read",     n_bytes)
                ...
        print(meter.report())
    """

    CATEGORIES = ("kv_compress", "kv_read", "weight_load", "expert_miss", "other")

    def __init__(
        self,
        model: str,
        hardware: str,
        expected_hit_rate: float = 0.95,
        bits_dense: int = 4,
        bits_expert: int = 4,
        bits_kv: int = 3,
        context_len: int = 4096,
        log_every: int = 32,
    ):
        if model not in MODELS:
            raise KeyError(f"Unknown model {model!r}; have {sorted(MODELS)}")
        if hardware not in HARDWARE:
            raise KeyError(f"Unknown hardware {hardware!r}; have {sorted(HARDWARE)}")
        self.budget = BandwidthBudget(
            model=MODELS[model], hardware=HARDWARE[hardware],
            context_len=context_len, bits_dense=bits_dense,
            bits_expert=bits_expert, bits_kv=bits_kv,
            expert_hit_rate=expected_hit_rate,
        )
        self.log_every = log_every
        self.tokens: list[dict] = []
        self._cur: Optional[dict] = None
        self._t0: Optional[float] = None

    @contextlib.contextmanager
    def token(self):
        self._cur = {c: 0 for c in self.CATEGORIES}
        self._t0 = time.perf_counter()
        try:
            yield self
        finally:
            elapsed = time.perf_counter() - (self._t0 or 0)
            self._cur["elapsed_s"] = elapsed
            self.tokens.append(self._cur)
            self._cur = None
            if self.log_every and len(self.tokens) % self.log_every == 0:
                print(self._rolling_status(), flush=True)

    def add(self, category: str, n_bytes: int) -> None:
        if self._cur is None:
            raise RuntimeError("BandwidthMeter.add() called outside token() context")
        if category not in self.CATEGORIES:
            raise ValueError(
                f"Unknown category {category!r}; expected one of {self.CATEGORIES}"
            )
        self._cur[category] += int(n_bytes)

    def _rolling_status(self) -> str:
        recent = self.tokens[-self.log_every:]
        avg_bytes = sum(
            sum(t[c] for c in self.CATEGORIES) for t in recent
        ) / len(recent)
        avg_elapsed = sum(t["elapsed_s"] for t in recent) / len(recent)
        measured = 1.0 / avg_elapsed if avg_elapsed > 0 else float("inf")
        predicted = self.budget.predicted_tok_per_sec()
        headroom = (measured / predicted) if predicted > 0 else 0
        return (
            f"[meter] tok={len(self.tokens):>5} "
            f"measured={measured:6.2f} tok/s  "
            f"predicted={predicted:6.2f} tok/s  "
            f"headroom={headroom:.0%}  "
            f"avg={avg_bytes/1e9:.3f} GB/tok"
        )

    def report(self) -> str:
        if not self.tokens:
            return "No tokens recorded."
        n = len(self.tokens)
        sums = {c: sum(t[c] for t in self.tokens) for c in self.CATEGORIES}
        elapsed = sum(t["elapsed_s"] for t in self.tokens)
        measured = n / elapsed if elapsed > 0 else float("inf")
        predicted = self.budget.predicted_tok_per_sec()
        out = [self.budget.report()]
        out.append(f"=== Measured over {n} tokens ({elapsed:.1f}s) ===\n")
        out.append(
            f"Throughput: {measured:6.2f} tok/s "
            f"(predicted {predicted:6.2f} tok/s; headroom {measured/predicted:.0%})\n\n"
        )
        out.append(f"Per-token byte breakdown (mean across {n} tokens):\n")
        for c in self.CATEGORIES:
            mean = sums[c] / n / 1e9
            total = sums[c] / 1e9
            out.append(f"  {c:14s}: {mean:7.3f} GB/token   ({total:7.1f} GB total)\n")
        return "".join(out) + "\n"

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({
                "model": self.budget.model.name,
                "hardware": self.budget.hardware.name,
                "context_len": self.budget.context_len,
                "expected_hit_rate": self.budget.expert_hit_rate,
                "predicted_tok_per_sec": self.budget.predicted_tok_per_sec(),
                "tokens": self.tokens,
            }, f, indent=2)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _cli() -> int:
    ap = argparse.ArgumentParser(
        description="Predict decode throughput ceilings under bandwidth constraints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", choices=sorted(MODELS), default="kimi-k2-6")
    ap.add_argument("--hardware", choices=sorted(HARDWARE), default="m4-max")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--bits-dense", type=int, default=4)
    ap.add_argument("--bits-expert", type=int, default=4,
                    help="4 = native INT4. 2 = aggressive (Kimi double-quant risk).")
    ap.add_argument("--bits-kv", type=int, default=3,
                    help="Effective KV bits after IsoQuant compression.")
    ap.add_argument("--hit-rate", type=float, default=0.95,
                    help="LRU expert cache hit rate. Below ~0.85 NVMe dominates.")
    ap.add_argument("--sweep", action="store_true",
                    help="Sweep over context length and hit rate.")
    args = ap.parse_args()

    model = MODELS[args.model]
    hw = HARDWARE[args.hardware]

    if not args.sweep:
        b = BandwidthBudget(
            model=model, hardware=hw, context_len=args.context,
            bits_dense=args.bits_dense, bits_expert=args.bits_expert,
            bits_kv=args.bits_kv, expert_hit_rate=args.hit_rate,
        )
        print(b.report())
        if model.note:
            print(f"  model note    : {model.note}")
        if hw.note:
            print(f"  hardware note : {hw.note}")
        return 0

    print(f"\nSweep for {model.name} on {hw.name}")
    print(f"bits_dense={args.bits_dense} bits_expert={args.bits_expert} "
          f"bits_kv={args.bits_kv}\n")
    print(f"{'context':>10}  {'hit-rate':>9}  {'GB/tok':>8}  {'tok/s':>8}  "
          f"{'ram GB/s':>9}  {'nvme GB/s':>9}  {'bottleneck':>10}")
    print("-" * 78)
    for ctx in (1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144):
        for hr in (0.99, 0.95, 0.90, 0.85, 0.70):
            b = BandwidthBudget(
                model=model, hardware=hw, context_len=ctx,
                bits_dense=args.bits_dense, bits_expert=args.bits_expert,
                bits_kv=args.bits_kv, expert_hit_rate=hr,
            )
            ceil = b.predicted_tok_per_sec()
            ram_demand = (b.active_weight_bytes() + b.kv_bytes_per_token()) / 1e9 * ceil
            nvme_demand = b.expert_miss_bytes() / 1e9 * ceil
            bot = "NVMe" if nvme_demand / max(hw.nvme_read_gbs, 1e-9) > \
                            ram_demand / max(hw.usable_bandwidth_gbs, 1e-9) else "RAM"
            print(f"{ctx:>10,d}  {hr:>9.2f}  "
                  f"{b.total_bytes_per_token()/1e9:>8.3f}  "
                  f"{ceil:>8.2f}  "
                  f"{ram_demand:>9.1f}  {nvme_demand:>9.2f}  {bot:>10}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
