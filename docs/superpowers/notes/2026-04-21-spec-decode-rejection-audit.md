# Spec-Decode Rejection Audit (δ memo)

**Date:** 2026-04-21
**Audit scope:** `docs/FROM_ATTENTION_TO_CONSUMER_HARDWARE.md:523` and related history.
**Triggered by:** Phase 0 §0.4 of `2026-04-20-research-reality-program-design.md`.

## Original rejection text

> **What we chose not to do:** Speculative decoding was considered and rejected — it is incompatible in the general case with expert offloading on memory-constrained hardware without routing-aware draft models. Each speculative token can route to different experts, turning a predictable prefetch stream into a chaotic SSD stampede that drops hit rate below the 70% threshold. QJL residual correction is off by default — the bit budget is better spent on more Lloyd-Max centroids (Section 5.1).

(`docs/FROM_ATTENTION_TO_CONSUMER_HARDWARE.md`, current line 523, in the "Three layers of importance" framing block following §10.)

## Provenance

- Introduced in commit `d486b7f` on 2026-04-13 by 2096955 ("docs: update hardware inference paper with Phase 3 proof and llama.cpp hardening") as part of the original 709-line draft of the paper. Born verbatim, written as authorial framing, not derived from a prior measurement note.
- Two later commits touched the same file but did **not** modify the rejection paragraph: `78e5ac0` (Mojo vs MLX section M6) and `7179a9f` (10-item peer review honesty pass). The text has stood unchanged across both edit passes.

## Evidence basis

- **Measured:** No. Repository contains no benchmark, log, or note recording a hit-rate measurement under speculative decoding + expert offload. `expert_offload.py:459-479` exposes `hit_rate`, `prefill_hit_rate`, `decode_hit_rate` instrumentation, but no script invokes it under speculative draft conditions.
- **Projected:** The "chaotic SSD stampede" argument is a plausibility argument, not a model. It assumes (a) draft tokens route uniformly across experts (worst case), (b) the resident set cannot absorb the extra fan-out, (c) speculative gain is bounded by hit rate.
- **70% threshold derivation:** Asserted, not derived. The string "70%" appears nowhere else in `docs/`, `scripts/`, or `mlx-lm/` in connection with speculative decoding or expert offload. The closest empirical anchor is `scripts/benchmark_moe_offload.py:360` which uses a **50%** decode-hit-rate as a sanity floor for "is offload even worth running" — a different threshold for a different purpose.
- **Code reality:** `mlx-lm/mlx_lm/generate.py:530` (`speculative_generate_step`) and `tests/test_generate.py:85` (`test_stream_generate_speculative`) confirm speculative decoding works in stock MLX. The rejection is specifically about the **combination** of speculative + offload, not speculative as a feature.

## Scope analysis

**Rejection covers:**
- Expert-offloaded models on memory-constrained hardware (32 GB Apple Silicon class)
- Draft models without routing awareness (i.e., draft model selects experts independently from target)
- Decode-time speculative generation (prefetch interaction is the failure mode cited)

**Rejection does NOT address:**
1. Models that fit fully in memory (no offload at all) — speculative is plausibly a pure win, no expert prefetch concern.
2. Draft models constrained to **shared experts only** — by construction these never trigger expert-specific prefetch divergence.
3. Speculative decoding scoped to **prefill** rather than decode — different memory access pattern; KV-cache prefill already does bulk loading.
4. Routing-aware draft models (the rejection text literally exempts these but the doc treats this as out of scope).
5. Smaller models on the same hardware where the resident set covers ≥95% of routed experts (offload exists nominally but rarely fires).

## Verdict

**`reopen_for_narrow_case_X`** — where X = "speculative decoding using a shared-expert-only draft model, on a target model that pays expert offload cost in steady state".

Rationale: The rejection's argument structure is sound for the general case it scopes to (uniform-routing draft + offloaded target + memory-constrained host), but it is overly broad as documentation. There is at least one narrow case (shared-expert draft, where every speculative token routes through the always-resident shared expert and incurs zero new prefetch demand) where the chaos-stampede argument cannot apply by construction. The 70% threshold is also asserted rather than derived, so even the general case lacks a binding measurement.

The rejection paragraph should not be deleted or weakened in the published paper — it correctly describes the default behavior for the configurations the paper measures. But the project's mental model should treat δ (speculative pathway) as deferred-with-conditions, not closed.

## Future-work implication

If pursued as a future sprint:

- **Scope estimate:** ~3-5 engineering days to (a) instrument hit-rate under speculative draft generation in `expert_offload.py`, (b) build a shared-expert-only draft adapter that constrains routing to layer 0's shared expert, (c) measure end-to-end speedup vs baseline on Nemotron-H 120B (the offload-heavy reference model).
- **Success criteria:** speculative + offload achieves ≥1.4× decode tok/s on Nemotron-H 120B (32 GB budget) without dropping hit rate below 95% of baseline. Failure case: hit rate falls more than 5pp, kill δ permanently with measurement-backed rationale.
- **Decision authority:** Whoever owns the offload pathway at the time. This memo lowers the activation energy for revisiting δ but does not authorise the work itself.

This memo does **not** close B-δ; it scopes the conditions under which it could be reopened.
