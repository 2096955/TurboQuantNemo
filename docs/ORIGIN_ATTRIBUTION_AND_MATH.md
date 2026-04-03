# Origin, attribution, and math (TurboQuantNemo)

This document gives a **reusable narrative**, **correct citations**, and a **clear separation of ideas** so public-facing text does not conflate unrelated techniques. The public repo is [TurboQuantNemo](https://github.com/2096955/TurboQuantNemo).

---

## 1. Compelling backstory (what to say, honestly)

**One-sentence pitch:** *Bring a large Nemotron-H MoE model down to a single-user Apple Silicon laptop by combining weight quantization, on-demand expert loading, and (optionally) research-grade KV cache compression—without pretending the whole stack is one invention.*

**Story arc (use in blog posts or README intros):**

1. **Constraint:** Useful coding models at 100B+ scale are out of reach for most developers if every parameter must sit in BF16 in RAM.
2. **Observation:** MoE models spend most of the time on a *subset* of experts per token; the rest can live on disk and be loaded on demand.
3. **Engineering bet:** Mixed quantization (fewer bits on routed experts than on dense layers) plus a correct MLX load path (`mx.load`) recovers usable quality at 2–4 bit expert widths.
4. **Adjacent research:** **TurboQuant** (see §3) targets **KV cache** compression, not MoE weights; this repo bundles a **TurboQuant–style MLX port** for attention layers while the **validated 120B path** here is **quantized expert offload on Nemotron-H**, not a claim that Nemotron hybrid attention is fully validated through TurboQuant KV (see `README_TurboQuantNemo.md`).
5. **Orchestration:** Claude Code (or any coordinator) can plan and delegate; the local `mlx_lm.server` + MCP stack runs the heavy generation loop.

That framing avoids overstating: it names **three separate contributions** (MoE offload + mixed quant, TurboQuant KV line, orchestration) and **one validated result** (expert offload + quality gate on real checkpoints).

---

## 2. Attribution (cite people and projects)

| Topic | What to cite | Notes |
|--------|----------------|--------|
| **TurboQuant (KV cache)** | Frantar et al., *TurboQuant: Accelerating Large Language Models with KV Cache Quantization*, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026). | Use when describing **KV cache** compression, Lloyd–Max codebooks, QJL residual, asymmetric score estimator. |
| **QJL (sign residual)** | Related work used inside TurboQuant; see [arXiv:2406.03482](https://arxiv.org/abs/2406.03482) (AAAI 2025). | Cite when discussing **1-bit residual sign** correction. |
| **PolarQuant** | [arXiv:2502.02617](https://arxiv.org/abs/2502.02617) (AISTATS 2026). | Optional related rotation/quantization context in TurboQuant line. |
| **MLX implementation** | `turboquant-mlx` in this repo; [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) upstream; MLX [ml-explore/mlx](https://github.com/ml-explore/mlx). | Apple Silicon inference stack. |
| **Nemotron-H weights** | NVIDIA model terms (e.g. `nvidia/Nemotron-3-Super-120B` on Hugging Face); follow NVIDIA license for redistribution. | Not a claim of endorsement. |
| **This repo’s MoE work** | Engineering on top of **mlx-lm fork**: expert repack, `ExpertOffloadManager`, quantized offload path, server hardening, scripts. | Describe as **implementation / integration**, not a new quantization theory. |

**PyTorch reference implementation** (for TurboQuant algorithm comparison): [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch).

**Avoid:** Calling the whole project “TurboQuant” without clarifying that **TurboQuant** is the **KV paper**; the **repo name** is a **portmanteau** (TurboQuant + Nemotron).

---

## 3. Math: TurboQuant (KV cache compression)

This is **orthogonal** to MoE expert weight quantization. Summary follows the [TurboQuant paper](https://arxiv.org/abs/2504.19874) and the in-repo `turboquant-mlx/README.md`.

The core idea: during inference, attention keys and values grow linearly with sequence length. TurboQuant compresses them to 2-4 bits per element while keeping attention scores accurate enough that output quality barely changes.

### 3.1 Codebooks (Lloyd–Max)

**Problem:** You need to map a continuous key value to one of 2^b discrete levels (e.g., 8 levels for 3-bit).

**Solution:** Precompute optimal quantization levels (centroids) and decision boundaries using the Lloyd-Max algorithm, which minimizes mean squared error for a given distribution. Each coordinate dimension and bit width gets its own codebook, stored as `dim_{D}_{b}bit.npz`.

**Why it works:** Lloyd-Max places centroids where the data is dense, so common values get precise representation while rare extremes share levels. This beats uniform quantization significantly.

### 3.2 Random rotation

**Problem:** Real key vectors have uneven coordinate distributions — some dimensions carry most of the signal, others are near-zero. Scalar quantization wastes bits on the quiet dimensions.

**Solution:** Multiply keys by a random orthogonal matrix before quantizing. This spreads the signal evenly across all coordinates, making each dimension roughly Gaussian with similar variance.

**Why it works:** The rotation is information-preserving (orthogonal = no loss) but makes every coordinate equally important, so the codebook can treat them uniformly.

### 3.3 QJL 1-bit residual

**Problem:** Even with good codebooks, quantization introduces error. That error biases the attention score estimate.

**Solution:** Compute the quantization residual (true key minus codebook reconstruction), project it through a random Gaussian matrix S, and store only the **signs** (1 bit per projection dimension). This is the QJL (Quantized Johnson-Lindenstraum) technique from [arXiv:2406.03482](https://arxiv.org/abs/2406.03482).

**Why it works:** The sign of a random projection preserves directional information about the residual. At query time, the correction term cancels the systematic bias from codebook quantization, using only 1 extra bit per dimension.

### 3.4 Asymmetric attention score estimator

Putting it all together, the attention score between query q and key k is estimated as:

$$\langle q, k \rangle \approx \underbrace{\langle q, k_{\text{mse}} \rangle}_{\text{codebook term}} + \underbrace{\frac{\|r\| \sqrt{\pi/2}}{m} \langle Sq, \text{sign}(S \cdot r) \rangle}_{\text{QJL residual correction}}$$

where:
- **k_mse** = codebook reconstruction of k (the "best guess" from quantization)
- **r** = k - k_mse (the quantization residual)
- **S** = random Gaussian projection matrix (shared across all keys)
- **m** = number of projection dimensions
- The first term is the standard dot product with the quantized key
- The second term corrects for quantization error using only the stored signs

**"Asymmetric"** means the query q is used at full precision — only the key is quantized. This is natural for attention: queries are computed fresh each forward pass, while keys accumulate in the cache.

**Current status:** The MLX implementation in `turboquant-mlx/` covers codebooks, rotation, QJL signs, and the asymmetric estimator. Value compression may still use dense buffers in places; full theoretical memory savings require packed storage — see `turboquant-mlx/README.md`.

---

## 4. Math: MoE weight quantization and offload (this repo’s validated path)

This is **not** TurboQuant. It is standard **affine (group) quantization** of linear layers plus **routing** and **on-demand expert tensor loads**.

### 4.1 Group-wise affine quantization (intuition)

For a weight matrix \(W\), affine quantization uses group size \(g\), bit width \(b\), scales \(s\), and optional zero-points / biases depending on mode. Each group is mapped to a small integer grid; dequantization approximates:

\[
W \approx \operatorname{Dequant}(W_{\text{packed}}, s, \ldots)
\]

MLX’s `nn.quantize` and `gather_qmm` paths use **packed** weights and **scales** consistent with the checkpoint (see `mlx_lm` sources).

### 4.2 MoE routing

For token \(t\), a gate selects top-\(k\) expert indices \(e_1,\ldots,e_k\). Only those experts’ weights are needed for the forward pass. **Offload** stores non-resident experts on disk and loads them into an LRU cache; **quantized** experts store **weight, scales,** and optional **biases** per expert.

### 4.3 Per-token expert gather

Active experts are gathered into a batch and run through quantized linear ops (e.g. `gather_qmm` in MLX), which is how the implementation avoids materializing all experts at once.

### 4.4 Critical implementation detail (quality)

The **numerical** recipe is only as good as the **loader**: quantized weights must be read with **`mx.load`**-compatible paths so **uint32** packed weights and **bfloat16** tensors are not corrupted (see `README_TurboQuantNemo.md` root-cause section). Nemotron-H **`time_step_limit`** must match Hugging Face defaults for correct Mamba/SSM behavior.

---

## 5. Suggested “citation paragraph” for README or talks

> **TurboQuantNemo** combines an [MLX](https://github.com/ml-explore/mlx)-based [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) fork with **quantized routed-expert offload** for **NVIDIA Nemotron-H** checkpoints, optional **TurboQuant**-style KV cache compression ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874)) following the `turboquant-mlx` port, and a small MCP/tooling layer for local coding assistants. The **TurboQuant** name refers to the **KV cache** method from Frantar et al.; **MoE offload and mixed-precision weights** are separate engineering contributions validated in this repository.

---

## 6. Related reading in this tree

- `docs/RELEASE_CANDIDATE_CHECKLIST.md` — reproducible gates for third parties.
- `README_TurboQuantNemo.md` — scope: what is validated vs experimental.
- `turboquant-mlx/README.md` — TurboQuant phases, benchmarks, and equations.
- `docs/TURBOQUANT_CHANGELOG.md` — historical MLX integration notes.
