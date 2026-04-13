# From attention to consumer hardware

**How MoE routing sparsity, isometric KV compression, compressed sensing, and cross-layer attention signals compose into a unified inference system**

---

1. Standard attention
2. Mixture-of-experts
3. The memory budget problem
4. KV cache compression — the shared pipeline
4b. Compressed sensing — why aggressive compression works
5. TurboQuant — dense global rotation
6. IsoQuant — isometric rotation via WHT + SO(4)
7. Deferred prefill
8. MLA — when KV is already compressed
9. AttnRes — the depth dimension (optional predictor)
10. The full stack
10b. Empirical anchors
11. QES — background optimisation
12. Contribution boundaries

---

> **How to read this document.** Each section opens with the maths, then a grey box like this one explains the intuition. All the analogies use a single running metaphor: *a yum cha kitchen preparing 384 dim sum dishes from a tiny service area*. The AI model is the kitchen. The specialist experts are station chefs. Incoming tokens are customer orders. RAM is counter and steamer-basket space. Disk is the back alley where the off-duty chefs wait. The star dish — siu loong bao (soup dumplings) — stands in for the most demanding operation: KV cache compression. If you're comfortable with the equations, skip the grey boxes. If you want the intuition first, read only the grey boxes for a complete story, then come back to the maths.

> **Note: Implementation status (April 2026).** The core stack (expert offload + IsoQuant KV + deferred prefill + weight allocation) is now **proven and validated** for Qwen3, Gemma 4, and Nemotron-H. Phase 3 (16GB Gemma pathway) is closed: Gemma 4 layer-aware achieves 12/12 on the quality gate and 12.85 tok/s on constrained hardware. AttnRes predictor and task-aware pinning remain optional enhancements. Our fused Metal decode pipeline (Section 6.3) is verified by 9 correctness tests and eliminates KV materialisation overhead. A parallel `llama.cpp` track provides a **hardened** `GGML_TYPE_ISOQUANT3_0` implementation, featuring a dedicated Metal write kernel, corrected GQA expansion (4D reshape), and fixed GGML context memory allocation for per-layer rotation tensors. QES is documented design. See Section 10b for measured results.

---

## 1. Standard attention — the starting point

Every transformer layer begins here. Given a sequence of token embeddings, we project into queries, keys, and values:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

where $X \in \mathbb{R}^{T \times d_{\text{model}}}$ and each projection $W \in \mathbb{R}^{d_{\text{model}} \times d_k}$. The attention output is:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

The softmax operates row-wise. For query position $i$, the attention weights over all key positions $j = 1, \ldots, T$ are:

$$a_{ij} = \frac{\exp(q_i^\top k_j / \sqrt{d_k})}{\sum_{m=1}^{T} \exp(q_i^\top k_m / \sqrt{d_k})}$$

The output for position $i$ is the weighted sum $o_i = \sum_j a_{ij} \, v_j$.

**The memory problem is immediate.** During autoregressive generation, every past token's key and value vectors must be retained — the *KV cache*. For $L$ layers, $H$ heads, sequence length $T$, and head dimension $d_k$:

$$\text{KV memory} = 2 \times L \times H \times T \times d_k \times \text{bytes per element}$$

At FP16 (2 bytes), a 60-layer model with $d_k = 128$ and 8K context already demands gigabytes of KV storage alone.

> **The main prep area.** Every time a new order comes in, the head chef must check the main prep area — the containers of prepped fillings already on the counter. How was the pork-and-ginger mix from table 4? Is the prawn paste from the first batch of har gow still fresh? That check across every prepped container is *attention*. The full set of prep containers is the *KV cache*. As the day goes on, the counter fills up with bowls. Our problem: we're running out of counter space.

---

## 2. Mixture-of-experts — the parameter explosion

In a standard transformer, each layer's feed-forward network (FFN) is:

$$\text{FFN}(x) = W_2 \, \sigma(W_1 x)$$

where $W_1 \in \mathbb{R}^{d_{\text{ff}} \times d}$, $W_2 \in \mathbb{R}^{d \times d_{\text{ff}}}$, and $\sigma$ is typically SiLU/GELU.

A Mixture-of-Experts layer replaces this single FFN with $E$ parallel expert networks $\{f_1, \ldots, f_E\}$ and a gating (routing) function $G$:

$$G(x) = \text{softmax}(W_g \, x) \in \mathbb{R}^E$$

$$\text{MoE}(x) = \sum_{e \in \text{TopK}(G(x))} G(x)_e \cdot f_e(x)$$

Only the top-$K$ experts (typically $K = 2$ or $K = 8$) are activated per token. This is the critical sparsity property: a model with $E = 384$ experts and $K = 8$ activates only ~2% of expert parameters per token. The model has 1 trillion total parameters, but only ~3B are active at any moment. The remaining 97% are dead weight in RAM — *unless we can offload them*.

### 2.1 Shared experts

Some architectures (Kimi-K2.5, DeepSeek) include a *shared expert* $f_{\text{shared}}$ that is always active regardless of routing:

$$\text{MoE}(x) = f_{\text{shared}}(x) + \sum_{e \in \text{TopK}(G(x))} G(x)_e \cdot f_e(x)$$

The shared expert carries the "common knowledge" load. Empirically, its weight distributions are heavy-tailed: shared expert kurtosis measures 13.10 versus 3.41 for routed experts — a 3.8× gap. This means the shared expert's weight distribution has far more outlier values that aggressive quantisation would destroy. We pin it at Q8_0 — a structural decision derived from the kurtosis gap, not a positional heuristic.

> **The station chefs.** Your kitchen has 384 specialised station chefs, but any given order only needs 8. The shared expert is the kitchen si fu — the master chef who inspects, adjusts, and signs off on every single dish before it leaves the pass. Because the si fu's hands touch everything, they get the best equipment and prime counter space (Q8_0 precision). The other 376 specialists are idle most of the time. We keep the 8 active ones at their stations and send the rest outside — they're out the back playing cards, but the floor manager (the router) can yell their name and have them back at their station in seconds.

---

## 3. The memory budget problem

For a 1T-parameter MoE on a 128GB machine:

| Component | Uncompressed | Target |
|---|---|---|
| All expert weights | ~800 GB | Can't be resident |
| Dense/shared weights | ~40 GB | ~10 GB (4-bit) |
| KV cache (8K context) | ~8 GB | ~1–2 GB (3-bit) |
| Activations + overhead | ~5 GB | ~5 GB |

Three independent compression axes address three independent memory consumers:

**Weight quantisation** — compress the model parameters (expert and dense weights). **KV cache compression** — compress the attention state (keys and values stored per token). **Expert offloading** — exploit routing sparsity to keep only active experts in RAM.

> **Note:** These axes are implemented independently. Weight quantisation operates on $W$ matrices. KV compression operates on the $k_j, v_j$ vectors stored during generation. Expert offloading operates on the residency policy — which weights are in RAM vs. on disk. They interact through model dynamics (e.g., quantised weights produce slightly different KV vectors), but each can be enabled, disabled, or tuned without modifying the others.

> **Three ways to fit 384 dishes into a tiny kitchen.** (1) *Shrink the recipe cards* — write shorthand the chefs can still follow, like "B3" instead of a full-page recipe (weight quantisation). (2) *Stack the fillings in small tubs in the fridge* — instead of 128 open bowls covering the entire counter, portion each filling into a small labelled tub and stack them in the walk-in. When the chef needs one, they pull the tub and reconstitute (KV cache compression). (3) *Send most chefs outside* — only the 8 active ones stay at their stations. The other 376 are out the back playing cards or smoking, but they can hear the floor manager yell their name and be back at their station in seconds (expert offloading). These three don't conflict. Each targets a different space problem: recipe storage, counter space, and how many bodies are in the kitchen.

---

## 4. KV cache compression — the shared pipeline

Both TurboQuant and IsoQuant follow the same four-stage pipeline for compressing key vectors. We describe it generically, then show where they diverge.

Given a key vector $k \in \mathbb{R}^{d_k}$ (one head, one token):

**Step 1. Normalise.** Map to the unit sphere: $\hat{k} = k / \|k\|_2$. Store $\|k\|$ separately (1 scalar per vector). In yum cha terms: weigh each filling and express it as a proportion of the total batch. Record the original weight on the wrapper's label so you can scale back later.

**Step 2. Rotate** (the divergence point). Apply an isometric transformation $\Pi$ to spread correlated dimensions uniformly: $\tilde{k} = \Pi(\hat{k})$. The rotation must preserve inner products: $\langle \Pi(a), \Pi(b) \rangle = \langle a, b \rangle$. This is essential — the attention score $q^\top k$ must survive the round-trip through compression. Think of this as portioning the filling so every dumpling gets the same amount before wrapping. If some dumplings are overstuffed and others are empty, the standard wrapper won't fit properly. "Isometric" means no filling is lost or created — you're just redistributing between dumplings until they're all equal. "Orthogonal" means it's perfectly reversible — you can always scoop the filling back out and recover the original portions exactly.

**Step 3. Scalar quantise.** Each dimension of $\tilde{k}$ is independently quantised using Lloyd-Max optimal codebooks. Lloyd-Max is a dual-optimisation: you simultaneously find the best *decision boundaries* $b_i$ (where to split the range) and *reconstruction centroids* $c_i$ (what value to store for each bin). The objective is to minimise the total distortion:

$$D = \sum_{i=1}^{2^b} \int_{b_{i-1}}^{b_i} (x - c_i)^2 \, p(x) \, dx$$

where $p(x)$ is the probability density of the rotated vector components and $b$ is the bit-width. This is solved by two interlocking conditions:

**Nearest-neighbour condition (optimal boundaries).** For fixed centroids, boundaries sit exactly halfway between adjacent centroids:

$$b_i = \frac{c_i + c_{i+1}}{2}$$

**Centroid condition (optimal reconstruction values).** For fixed boundaries, each centroid is the conditional mean — the centre of mass of the distribution within its bin:

$$c_i = \frac{\int_{b_{i-1}}^{b_i} x \, p(x) \, dx}{\int_{b_{i-1}}^{b_i} p(x) \, dx}$$

Because these depend on each other, they're solved iteratively (Lloyd's algorithm): initialise centroids, update boundaries, update centroids, repeat until distortion converges.

Why does this matter for a 1T model? LLM activations aren't uniformly distributed — they're peaky near zero with long tails. Uniform quantisation (like rounding to the nearest 0.1) wastes most of its bins on the empty tails and under-resolves the dense centre. Lloyd-Max puts more centroids where the data actually lives — high-density regions get more bins and higher precision, low-density tails share fewer bins.

In yum cha terms: you have 8 different-sized steamer baskets (the centroids) for packing 128 dumplings. Uniform basket sizes mean most baskets are too big for the common small dumplings and too small for the occasional large one — wasted space everywhere. Lloyd-Max sizes the baskets to match how your dumplings are actually distributed: lots of small baskets for the common portions, a few large ones for the outliers. The rotation in Step 2 ensures the dumplings are portioned evenly enough for this basket-sizing to work optimally. Now that every dumpling is the same size, place each one into the best-fitting basket and stamp it with a code — "P3" instead of "pork-ginger-scallion-gelatin 12g."

**Step 4. Bit-pack and store.** Quantised indices are packed into integers. At 3-bit, 128 dimensions pack into 48 bytes — versus 256 bytes at FP16. A ~5× compression. Uniform dumplings stack tightly in the steamer basket.

### Score estimation at decode time

When computing attention, the query $q$ remains at full precision. The compressed key must be decompressed to estimate $q^\top k$:

$$\widehat{q^\top k} = \|k\| \cdot q^\top \Pi^{-1}(Q_b(\Pi(\hat{k})))$$

The estimation error is:

$$\epsilon = q^\top k - \widehat{q^\top k} = \|k\| \cdot q^\top \Pi^{-1}(\Pi(\hat{k}) - Q_b(\Pi(\hat{k})))$$

This is the quantisation residual, projected back through the inverse rotation. Softmax is invariant to additive shifts but highly sensitive to rank ordering — making top-$k$ preservation more critical than MSE.

> **Wrapping and stacking the siu loong bao.** You need to fit 128 prepped dumplings into a tiny fridge. The four steps: (1) *Weigh* each dumpling and note the weight (normalise). (2) *Even out the portions* — if some dumplings are overstuffed and others are nearly empty, a standard wrapper won't fit them all. So you redistribute: scoop filling from the fat ones into the skinny ones until every dumpling is the same size (rotate). (3) *Wrap* — now that they're all uniform, fold each one into a standard skin with a stamped code (quantise). (4) *Stack in the steamer* — uniform dumplings stack tightly (bit-pack). When the chef needs one, they unwrap and reconstitute. The question TurboQuant and IsoQuant disagree on: how do you do step 2 — the evening-out?

---

## 4b. Compressed sensing — why aggressive compression works

The pipeline above compresses 128-dimensional vectors to 3 bits per dimension — a 5× reduction. Why doesn't this destroy the model's ability to attend to the right tokens? The answer draws on compressed sensing theory (Candès & Tao, 2006).

### The Restricted Isometry Property (RIP)

A measurement matrix $\Phi$ satisfies the RIP of order $s$ with constant $\delta_s$ if, for all $s$-sparse vectors $x$:

$$(1 - \delta_s)\|x\|_2^2 \leq \|\Phi x\|_2^2 \leq (1 + \delta_s)\|x\|_2^2$$

In plain terms: the measurement preserves the energy (length) of sparse signals to within a factor of $\delta_s$. If $\delta_s$ is small, the compressed version faithfully represents the original — no hidden distortions.

Our rotation matrix $\Pi$ in the KV compression pipeline acts as a structured sensing matrix. It satisfies an empirical RIP-like condition on real KV vectors: the isometric rotation preserves inner products (and therefore attention scores) even after quantisation. This is an empirical observation on our specific vectors, not a formal RIP proof — real KV vectors are not exactly sparse in the classical sense. But the analogy is precise enough to explain *why* the pipeline works and *when* it might fail. We measure three proxies — isometry error, cosine similarity, and top-5 retrieval agreement — that would be zero, one, and one respectively for an ideal RIP matrix. Their small deviations ($\delta < 0.05$, cosine $> 0.98$, top-5 $> 0.90$) indicate that the rotation behaves similarly to a random orthogonal matrix on the distribution of real KV vectors.

> **The "taste a few dumplings" rule.** You've steamed 100 siu loong bao and need to check the whole batch is right — proper soup-to-meat ratio, wrapper not too thick, no bursting. Tasting every single one would take forever. Compressed sensing says: if you pick a few representative dumplings (one from the first steamer, one from the middle, one from the last) and they taste right, you can *guarantee* the rest are right too — **but only if** your sampling method satisfies the "no surprises" rule (RIP). That rule says: the way you pick your samples must faithfully represent the whole batch. No hidden pockets of bad dumplings that your sampling misses. RIP is the mathematical guarantee that your taste test doesn't lie to you.

### How compressed sensing maps to our stack

Three elements of the classical compressed sensing framework appear directly in our system:

**The sensing matrix satisfying RIP.** The isometric rotation $\Pi$ (whether TurboQuant's dense QR or IsoQuant's WHT + SO(4)) preserves the geometry of the attention space. We verify this empirically rather than proving RIP formally: isometry error $\delta \leq 0.05$, cosine similarity $> 0.97$, top-5 retrieval agreement $> 0.90$. These thresholds are measured on real KV vectors from the target model, not derived from theory.

**The sparsity structure.** MoE routing provides sparse structure — only 8 of 384 experts are active per token. The attention distribution itself is sparse (softmax concentrates mass on a few tokens). These are the structural properties that make compression viable. The relevant property is low entropy in attention distributions, not strict $\ell_0$ sparsity — compression degrades when attention entropy approaches uniform.

**The recovery method.** Classical compressed sensing uses expensive $\ell^1$ minimisation to recover the full signal. We don't need full recovery — we only need the scalar attention score $q^\top k$ and the sparse expert routing indices. The asymmetric score estimator (Section 4, score estimation) recovers exactly what we need, at the cost of a single dot product. This is why our system is fast: we bypass full vector reconstruction entirely.

> **Note:** Compressed sensing provides the theoretical *framework* for understanding why 5× compression doesn't destroy attention quality. The rotation promotes approximate isotropy (empirical RIP-like behaviour). The sparsity comes from MoE routing and softmax concentration. And we only recover the specific numbers we need (attention scores, expert indices), not the full original vectors. This is an analogy to classical compressed sensing, not a formal proof — but it correctly predicts where the pipeline works (high-sparsity attention) and where it degrades (uniform attention distributions at long context).

---

## 5. TurboQuant — dense global rotation

TurboQuant (Zandieh & Mirrokni, Google Research; ICLR 2026; arXiv:2504.19874) implements stage 2 as:

$$\Pi_{\text{TQ}}(\hat{k}) = \Phi \, \hat{k}$$

where $\Phi \in \mathbb{R}^{d_k \times d_k}$ is a random orthogonal matrix constructed via QR decomposition of a Gaussian matrix. The original TurboQuant paper describes this as a randomised Hadamard transform $\Phi = H_{d_k} \cdot D$ (Hadamard matrix times random sign-flip diagonal), which has $O(d_k \log d_k)$ asymptotic complexity. In practice — and this applies equally to our IsoQuant implementation — the rotation is stored and applied as a dense $d_k \times d_k$ matrix, requiring $d_k^2$ stored parameters and $d_k^2$ FMAs per vector — for $d_k = 128$, that is 16,384 of each. The theoretical method is structured (Hadamard), but current implementations materialise it as dense matrices, eliminating asymptotic gains. We evaluate the implementation, not the algorithm.

**Why it works:** The full-rank rotation transforms any input distribution into one where the per-dimension marginals converge to a Beta distribution on $[-1, 1]$ — precisely the distribution Lloyd-Max codebooks are optimised for. A dense random orthogonal matrix also acts as a near-ideal compressed sensing matrix (Section 4b) — it promotes the approximate isotropy that makes scalar quantisation effective. In yum cha terms: the chef weighs every dumpling against every other and redistributes filling until they're all exactly the same size. Perfect portioning — but you had to handle all 128 dumplings at once.

**Why it's slow on Apple Silicon:** For $d_k = 128$, each dense rotation costs $d_k^2 = 16{,}384$ FMAs per vector — every dumpling weighed against every other. At prefill time, the dense matrix-vector multiply maps poorly to Metal's SIMD architecture. Measured: 86.46 ms for 65K vectors on M4 (upstream benchmark from scrya-com).

### 5.1 QJL residual correction

TurboQuant adds an unbiased correction via the Quantised Johnson-Lindenstrauss (QJL) lemma — itself a 1-bit compressed sensing measurement of the quantisation residual. The residual $r = k - k_{\text{reconstructed}}$ is projected through a shared random Gaussian matrix $S \in \mathbb{R}^{m \times d_k}$:

$$b = \text{sign}(S\,r) \in \{-1, +1\}^m$$

Only the sign bits are stored (1 bit each). The corrected score estimate is:

$$\widehat{q^\top k}_{\text{corrected}} = \widehat{q^\top k}_{\text{Lloyd-Max}} + \frac{\|r\|}{m} \sum_{i=1}^{m} b_i \cdot (S\,q)_i$$

This is provably unbiased. However, the variance of the correction is $O(\|r\|^2 / m)$, and softmax *amplifies* variance — a high-variance unbiased estimator can produce worse attention distributions than a slightly biased low-variance one. Imagine keeping scribbled correction notes on each dumpling wrapper — on average correct, but for any individual dumpling wildly off. The head chef's palate (softmax) is extremely sensitive to which filling dominates; a consistently slightly-off correction is less disruptive than a wildly fluctuating one. Empirically, allocating the QJL bit budget to more Lloyd-Max centroids — better wrappers rather than scribbled notes — yields better perplexity. QJL is therefore **off by default** in our stack.

---

## 6. IsoQuant — isometric rotation via WHT + SO(4)

IsoQuant (arXiv:2603.28430) replaces the rotation step in the KV compression pipeline. Its development went through three iterations, each a response to measured failure.

### 6.1 The SO(4) rotation

The IsoQuant paper proposes partitioning the $d_k$-dimensional vector into groups of 4 and rotating each block independently using paired quaternions:

$$\hat{k} = (\hat{k}_{[1:4]}, \, \hat{k}_{[5:8]}, \, \ldots, \, \hat{k}_{[d_k-3:d_k]})$$

Each 4D block is rotated by *two* independent unit quaternions — a left factor $\mathfrak{q}_L$ and a right factor $\mathfrak{q}_R$:

$$\tilde{k}_{[i]} = \mathfrak{q}_{L,i} \otimes \hat{k}_{[i]} \otimes \bar{\mathfrak{q}}_{R,i}$$

With two independent quaternions, the transformation spans the full $\text{SO}(4)$. A single-quaternion conjugation $q \otimes v \otimes \bar{q}$ only reaches a subgroup (left-isoclinic rotations). In yum cha terms: with one set of hands you can tilt a jug and mix between neighbouring bowls. With two sets of hands working independently, you can redistribute between all four bowls in every possible combination — more flexibility, better evening-out of the filling.

### 6.2 What we tried and what failed — the v1→v2→v3 evolution

The paper's benchmarks (PPL 6.91 vs TurboQuant's 7.07; top-5 retrieval 93.8% vs 87.5%) are from the upstream scrya-com repository on controlled benchmarks. When we implemented IsoQuant on real models, the story was more complicated:

**v1: Single-quaternion sandwich (collapsed).** The left-isoclinic-only form $\mathfrak{q}_L \otimes v \otimes \bar{\mathfrak{q}}_L$. On Qwen3 with real KV vectors, this scored **0/5** on our quality harness. Complete collapse — the dumplings were inedible.

> **Warning: Negative result.** IsoQuant-Fast (single quaternion, 512 FMAs) was tried and failed on real anisotropic KV vectors. Qwen3 scored 0/5 while TurboQuant scored 2/5 on the same harness. The single-quaternion form is SO(3) embedded in 4D — one axis is fixed per block, leaving 25% of dimensions unmixed. This variant is **not viable**.

**v2: Block-only SO(4) (degraded).** Paired quaternions with full $\text{SO}(4)$ per block, but no global mixing. Scored **1/5** — worse than TurboQuant's 2/5. Block-diagonal rotation decorrelates within 4D blocks but not between them. With 32 independent blocks, cross-block correlations in real KV vectors pass through to Lloyd-Max unhandled.

**v3: WHT + SO(4) (current default).** Global Walsh-Hadamard pre-mixing, *then* SO(4) block rotation:

$$\tilde{k} = \Pi_{\text{SO}(4)}(H_d \cdot \hat{k})$$

where $H_d$ is the normalised Walsh-Hadamard matrix. The WHT handles global decorrelation across all dimensions; the SO(4) blocks handle fine-grained per-block rotation for Lloyd-Max. Scored **2/5** on Qwen3 — matching TurboQuant and the uncompressed default on our narrow, short-generation harness. This is the working architecture.

> **The siu loong bao filling trials.** The head chef tried three ways to mix the filling for siu loong bao. The filling needs perfect balance of pork, ginger, scallion, and soup gelatin. *v1: One-handed mixing.* Lumpy, uneven, inedible — 0/5 on the tasting panel. *v2: Two hands, but isolated batches.* Pork-ginger in one bowl, scallion-gelatin in another. Some batches right, others wrong — 1/5. *v3: Global rough mix first, then two-handed batches.* Pour everything into one big bowl and roughly stir (WHT), then split into batches of four and use two hands per batch for the fine work (SO(4)). The filling was balanced — 2/5, matching the control batch.

### 6.3 Current runtime reality

The IsoQuant paper reports a 31× prefill speedup on Apple Silicon Metal using structured kernels (upstream benchmark from scrya-com). Our current MLX implementation has two execution paths — a dense write path and a fused Metal read path:

**Write path (KV insertion — dense).** The rotation is materialised as a single dense $d_k \times d_k$ matrix — runtime cost is $O(d_k^2)$, identical to TurboQuant's dense rotation. This is applied during KV insertion (bulk compress after deferred prefill, or incremental compress during decode). Fusing the write path into a single Metal kernel (FP16 → WHT butterfly → SO(4) block rotation → Lloyd-Max quantise → 3-bit pack) is designed but not yet implemented.

**Read path (decode attention — fused Metal pipeline).** The decode-time attention is computed via 4 fused Metal kernels (`fused_kv_decode_kernels.py`) that operate directly on 3-bit packed KV data without materialising full FP16 K or V tensors:

```text
[Kernel A]  fused_qk_dot:       packed 3-bit K → unpack in-register → centroid lookup → dot(q,k) → SIMD reduce → scores
[Kernel B]  mx.softmax:         standard MLX softmax
[Kernel C]  fused_value_accum:  packed 3-bit V → unpack in-register → dequant → weighted sum → output (rotated)
[Kernel D]  metal_rotate_inverse: WHT + SO(4) structured inverse rotation (applied once on aggregated output)
```

Cost: $O(T \times d_k)$ for Kernels A and C (linear scan, no rotation) + $O(d_k \log d_k)$ for Kernel D (once). This replaces the original path that materialised full K and V tensors at $O(T \times d_k^2)$ cost.

The *logical* operation count for the structured inverse pass: WHT butterfly requires 896 FMAs ($d_k \log_2 d_k$ for $d_k = 128$), and the SO(4) block matvec requires 512 FMAs (32 blocks × 16 FMAs per 4×4 matvec) = **1,408 total** for Kernel D, versus 16,384 for a dense inverse rotation. The forward query rotation remains dense ($O(d_k^2)$) but is applied to a single vector per head, so the absolute cost is negligible.

**Kernel A: Fused Q·Kᵀ with 3-bit decode.** One threadgroup per token, 32 threads cooperating over $d_k = 128$ dimensions. Query cached in threadgroup memory (512 bytes). For each of the 16 packed 24-bit words: extract 8 × 3-bit indices via bit shifts, look up centroids, compute per-element dot product with the query, accumulate. Multiply by stored norm. Partial sums reduced via `simd_sum()`. Write score. No intermediate K buffer is ever written. Native GQA support via `kv_head_map` — each query head maps to its KV head without materialising repeated K tensors.

**Kernel C: Fused value accumulation in rotated space.** One threadgroup per query head, threads striped over dimensions. For each token $j$: unpack the 3-bit packed V index for the thread's dimension, centroid lookup, multiply by attention weight $a_j$ and stored norm, accumulate into thread-local register. The accumulated output remains in **rotated space** — no per-token inverse rotation. One inverse rotation is applied after the sum (Kernel D). Same `kv_head_map` GQA support as Kernel A.

**3-bit packing format.** Quantisation indices (0–7) are packed as 8 indices → 24 bits → 3 bytes. Packing is performed once via `pack_indices_3bit()` and cached on `IsoQuantKVCache`. The packed cache is invalidated when new tokens are appended (decode step) and re-packed on next access.

**Critical implementation detail: `mx.fast.metal_kernel` grid semantics.** The `grid` parameter specifies **total threads**, not threadgroup count — unlike Metal's `dispatchThreadgroups`. Original dispatch `grid=(seq_len, num_heads, 1)` with `threadgroup=(32, 1, 1)` placed all tokens in one threadgroup of `seq_len` threads instead of `seq_len` threadgroups of 32 threads. Token 0 was correct (happened to be in the right threadgroup position), tokens 1+ were wrong (96.9% output mismatch). Fix: `grid=(32 * seq_len, num_heads, 1)` for Kernel A, `grid=(min(head_dim, 128) * num_heads, 1, 1)` for Kernel C. This is a general hazard for any `mx.fast.metal_kernel` user.

**Execution path selection.** `IsoQuantKVCache._fused_metal_ok` flag: `None` → untested, `True` → latched (Metal kernels succeed), `False` → latched (fallback to MLX-ops for all subsequent calls). First `fused_attention()` call tries Metal; on success latches `True`, on failure latches `False`. Verified: `_fused_metal_ok = True` after first decode on Apple Silicon. The MLX-ops fallback is algorithmically correct (rotated-space attention with single inverse rotation) but uses `mx.matmul` — no bandwidth advantage over the reconstruct path.

**Wiring.** `fused_attention()` is called from 5 model files: `qwen3_moe.py`, `gemma4_text.py`, `qwen2.py`, `nemotron_h.py`, `qwen3_next.py`. Each checks `isinstance(cache, IsoQuantKVCache) and cache.supports_fused_attention` before the standard TurboQuant reconstruct+SDPA path. Verified by 9 correctness tests (`tests/test_fused_isoquant_attention.py`): fused vs. dense without mask, with mask, GQA (8 query heads / 2 KV heads), longer sequence (T=64), fallback for unsupported bit-width, inner product preservation proof, packed cache reuse and invalidation. All tests pass at atol=1e-2, rtol=1e-2.

> **Note:** The 31× speedup is an upstream benchmark from structured Metal kernels applied to the rotation step. Our fused decode pipeline eliminates the materialisation and per-token rotation overhead, but the end-to-end speedup depends on the fraction of decode time spent in KV attention vs. expert I/O — which has not yet been profiled. The honest assessment: IsoQuant has correct maths expressed in the correct execution primitives for the decode read path. Whether this translates to measurable end-to-end improvement depends on the regime (context length, memory pressure).

| Property | TurboQuant | IsoQuant v3 (WHT + SO(4)) |
|---|---|---|
| Logical FMAs (structured) | 16,384 | 1,408 |
| Write path FMAs | 16,384 | 16,384 (baked dense) |
| Decode read path | Dense reconstruct + SDPA | Fused Metal (no K/V materialisation) |
| Stored parameters | 16,384 | 256 |
| Quality (Qwen3 5-prompt) | 2/5 | 2/5 |

### 6.4 WHT is required, not optional

The v2→v3 evolution demonstrates this: without WHT, block-only SO(4) scored 1/5 vs TurboQuant's 2/5. The WHT provides the global mixing that block-diagonal rotation cannot. **Do not skip the WHT pre-pass** — this is an architectural constraint, not a performance toggle.

### 6.4b llama.cpp GGML type: GGML_TYPE_ISOQUANT3_0

The IsoQuant rotation is integrated into a llama.cpp fork (`_third_party/llama-cpp-turboquant/`) as a real GGML type with a dedicated Metal write kernel — not a wrapper around TurboQuant, but a distinct enum with its own Metal shader specialisations, write path, and runtime wiring.

**Type identity.** `GGML_TYPE_ISOQUANT3_0 = 44` with `block_isoquant3_0`: 14-byte struct identical in layout to `block_turbo3_0` (2B norm + 8B 2-bit indices + 4B sign extension, QK=32). The block packing is the same — what differs is the rotation applied before quantisation.

**Metal shaders (read path).** 9 symmetric non-vec and 6 symmetric vec flash-attention template instantiations (`kernel_flash_attn_ext_kisoquant3_visoquant3_dk{N}_dv{N}`). Dequantise functions delegate to the turbo3 implementations via reinterpret-cast (identical block layout). **Only symmetric K/V pairs are instantiated** — mixed isoquant3 × other-type pairs are excluded from the asymmetric capability checks and will assert at runtime if attempted.

**Metal write path (dedicated kernel).** `kernel_set_rows_isoquant3` is a standalone Metal kernel — not a template instantiation of `kernel_set_rows_turbo`. It applies a two-stage rotation per 128-element WHT group: (1) the shared WHT forward transform via `turbo_rotate_forward()`, then (2) 32 independent 4×4 SO(4) block rotations via `isoquant_rotate_group_128()` reading per-head metadata from a bound rotation tensor. The helper `isoquant_apply_so4_4()` performs each 4×4 matvec in 16 FMAs.

**Rotation metadata plumbing.** The GGML op `ggml_set_rows_ext()` extends the standard `ggml_set_rows()` with an optional fourth source tensor (`src[3]`) carrying rotation metadata. The Metal dispatch in `ggml-metal-ops.cpp` binds this tensor at buffer slot 4 and passes per-head layout parameters (`rot_blocks_per_head`, `rot_n_head`, `rot_nb1`, `rot_nb2`) through the kernel args struct `ggml_metal_kargs_set_rows`. The existing `ggml_set_rows()` passes `NULL` for the metadata tensor, preserving backward compatibility for all non-IsoQuant callers.

**Per-layer rotation tensors.** The KV cache allocates two F32 tensors per layer when the cache type is ISOQUANT3_0: `k_isoquant_rot` and `v_isoquant_rot`, each shaped `[16, head_dim/4, n_head_kv]` — storing 4×4 row-major SO(4) blocks per quaternion group per head. These are initialised to identity matrices at cache construction (via `init_isoquant_rotation_tensor()`), making the default behaviour equivalent to WHT-only rotation. Non-identity blocks activate the real IsoQuant rotation. The KV write call sites in `llama-kv-cache.cpp` switch to `ggml_set_rows_ext()` only for ISOQUANT3_0, passing the per-layer rotation tensor.

**Runtime wiring.** ISOQUANT3_0 is integrated into all turbo-style checks across `llama-context.cpp` (K/V padding, flash-attention auto-enable), `llama-kv-cache.cpp` (layer-adaptive mode guards for all 7 modes, zero-padding, rotation matrix creation, get/cpy K/V), and `llama-graph.cpp` (10 sites: Q pre-rotate WHT, V inverse WHT, pad/unpad). CLI accepts `--cache-type-k isoquant3` / `--cache-type-v isoquant3` via `arg.cpp` and `llama-bench`.

**TURBO_LAYER_ADAPTIVE guard.** All adaptive modes (1–7) are guarded to preserve isoquant3 type identity — modes 1/2 skip K/V rewriting to Q8_0, modes 5/6/7 skip V rewriting to turbo2/turbo4/q8_0, when the original type is ISOQUANT3_0.

**Hardening patches.** Three post-review cleanup patches address edge cases and correctness gaps in the write path: (1) **Transposed-V hard assert.** The transposed-V code path (used when flash-attention is disabled) reshapes V to `ne[0]=1`, making per-element rotation semantically wrong. A hard `GGML_ASSERT(v->type != GGML_TYPE_ISOQUANT3_0 && "ISOQUANT3_0 is incompatible with transposed V cache")` prevents this path from being reached. (2) **Group-size constant.** The Metal kernel helper `isoquant_rotate_group_128()` now uses `QK_ISOQUANT3_GROUP` (not `QK_TURBO3_GROUP`) with `static_assert(QK_ISOQUANT3_GROUP == 128, "isoquant SO(4) mapping assumes 128-wide WHT groups")`. (3) **Cached identity guard.** A `bool isoquant_rot_is_identity` flag on the `kv_layer` struct is checked at both K and V write sites. The initial implementation used a per-write `ggml_backend_tensor_get()` readback over the full rotation tensor — review identified this as a device-to-host synchronisation on the Metal write hot path. Replaced with a cached boolean set at init time: the flag defaults to `true` (matching the identity initialisation), and any future code that loads non-identity SO(4) blocks must set the flag to `false`, at which point the assert fires with a message directing implementers to the missing graph-side Q/output rotation ops. This is a **use-time** guard, not a load-time one — it prevents non-identity data from being *consumed*, not from being written to the rotation tensors.

**Current functional state.** The dedicated Metal write kernel exists, compiles, and is **fully hardened**. GGML context memory allocation now accounts for per-layer rotation tensors (+2*n_layer_kv slots), preventing initialization crashes. GQA expansion in `apply_so4_blocks_forward` is corrected via a 4D reshape ([4, 4, n_blocks, n_head_kv] → [4, 4, n_blocks, n_head]), ensuring contiguous repetition of KV heads for their query groups and avoiding tiling artifacts. CPU fallback logic includes a loud warning for non-identity SO(4) metadata. Smoke tests on Metal (Qwen2.5-1.5B) verify generation at ~90 tok/s with active rotations.

**What remains.** (1) Loading non-identity SO(4) blocks into the rotation tensors from GGUF model metadata or configuration files. (2) Graph-side Q/output rotation must be enabled for non-identity blocks (currently trips the identity guard if consumed). (3) CPU `set_rows` path awareness for metadata. (4) Large-scale inference validation with a 120B-class GGUF model. The architecture is proven; the pipeline is ready for production weights.

### 6.5 Inverse rotation at read time

TurboQuant's dense rotation is self-cancelling in the attention weighted sum — the inverse folds into the query projection. IsoQuant's block-diagonal quaternion rotations are *not* self-cancelling. The inverse must be applied explicitly when reading values back:

$$\hat{v}_{\text{reconstructed}} = \bar{\mathfrak{q}}_{R,i} \otimes \tilde{v}_{[i]} \otimes \mathfrak{q}_{L,i}$$

The inverse rotation is like scrubbing the steamer trays clean between sweet and salty dishes. The same steamer stack that just held har gow (shrimp dumplings) now steams your coconut tarts. If you don't clean between batches, salt clings to the trays and contaminates the sweet filling. The inverse rotation scrubs the trays clean before the next load — restoring the original purity of each dish.

The maths guarantees this works because the rotation is isometric — nothing was ever lost during the forward rotation, so the inverse perfectly restores the original. The same isometry and cosine similarity checks applied to keys must also be applied to the value path: reconstruction errors in either path directly affect the softmax-weighted sum in attention and must satisfy cosine similarity $> 0.97$.

> **Warning:** Without the explicit inverse — without scrubbing the trays — perplexity explodes from 7.05 to 15,369. The coconut tarts taste of shrimp. The inverse is cheap (same FMA budget as the forward rotation), but skipping it is catastrophic.

**Implementation note: inverse rotation after the sum, not per value.** The original decode path applied the inverse rotation to **every token's K and V vector individually** ($O(T \times d_k^2)$). The `fused_attention()` method on `IsoQuantKVCache` computes attention in the rotated space instead — rotating the query forward rather than rotating all keys inverse, then applying the inverse **once** on the aggregated attention output ($O(d_k^2)$). This is valid because isometric rotations preserve inner products: $q^\top k = (Rq)^\top (Rk)$. At $T = 500$, this eliminates 500× of rotation work. See Section 6.3 for the full fused pipeline.

### 6.6 DKV constraint for MLA

For MLA-based models (Kimi-K2.5), the latent vector $c_t$ splits into content ($\mathbb{R}^{448}$) and RoPE position ($\mathbb{R}^{64}$). Rotating the RoPE dimensions smears positional phase into content coordinates, destroying long-context awareness — the model attends to the right dish but forgets which table ordered it. **IsoQuant is applied only to content dimensions; RoPE dimensions are never rotated or quantised.** This is the DKV constraint — a non-negotiable architectural rule.

---

## 7. Deferred prefill — eliminating compounding error

Both TurboQuant and IsoQuant introduce quantisation error on every insertion. During prefill, this means $T$ successive insertions with independent error. The errors compound through the softmax — every time you wrap a dumpling, a tiny amount of flavour is lost. Wrap 4,000 one by one during the rush and the head chef can taste the accumulated degradation.

The solution: don't compress during prefill at all. Keep everything fresh on the counter during the rush, then wrap the lot once the initial burst is over.

```
Prefill (tokens 1…T):
  Store K, V in FP16 — zero compression error
  Memory: 2 × L × d × T × 2 bytes

Transition (after prefill completes):
  Bulk-compress entire FP16 buffer:
    For each vector: normalise → rotate → Lloyd-Max quantise
  Free the FP16 buffer

Decode (each new token):
  Compress incrementally on insertion
```

For $T = 4096$, $L = 61$, $d = 512$: the FP16 buffer is ~512 MB — manageable. Beyond ~8K tokens, fall back to incremental compression.

**Sliding-window caveat (Gemma 4).** Some architectures use sliding-window attention on most layers (e.g., Gemma 4: 1024-token window on 25 of 30 layers, full attention every 6th layer). KV entries in sliding-window layers are evicted within 1024 tokens — compressing them with IsoQuant wastes compute, since the compression cost is not amortised before eviction. Compress only **global-attention** KV (long-lived entries). Sliding-window KV can stay FP16 or use lightweight FP8.

---

## 8. MLA — when KV is already compressed

Kimi-K2.5 uses Multi-Head Latent Attention (MLA), which *already* compresses the KV representation architecturally:

$$c_t = W_{\text{DKV}} \, x_t \in \mathbb{R}^{d_c}$$

where $d_c \ll H \cdot d_k$. Keys and values are reconstructed on the fly:

$$k_t^{(h)} = W_K^{(h)} c_t, \qquad v_t^{(h)} = W_V^{(h)} c_t$$

This is a learned compression — like the kitchen already using concentrated stock cubes instead of fresh broth. The question: does portioning the concentrate into smaller tubs (IsoQuant on top of MLA) save meaningful fridge space?

> **Warning: The DKV constraint.** The latent splits into content ($\mathbb{R}^{448}$) and RoPE position ($\mathbb{R}^{64}$). Rotating RoPE dimensions smears positional phase — the model attends to the right dish but forgets which table ordered it. **IsoQuant on content dimensions only. Never rotate RoPE.** (See also Section 6.6.)

**Decision gate:** The stack is conditional: if MLA already compresses KV sufficiently, IsoQuant becomes unnecessary. If additional compression <10% over MLA alone or PPL increase >0.5, skip IsoQuant entirely. MLA collapses the KV compression axis, removing the need for a second compression layer.

---

## 9. Attention residuals (AttnRes) — the depth dimension (optional predictor)

Everything above operates along the *sequence dimension* (compressing KV across tokens within a layer). AttnRes operates along the *depth dimension* (across layers) — providing a signal for **which chefs should be in the kitchen**.

### 9.1 Standard residual stream

In a vanilla transformer, each layer adds to a residual stream:

$$h_l = h_{l-1} + f_l(h_{l-1})$$

Every layer contributes equally — every station adds its component to the dish, even if the garnish station adds nothing to this particular order.

### 9.2 Block attention residuals

AttnRes (Moonshot AI / Kimi Team, arXiv:2603.15031) replaces the additive residual with learned depth-wise attention. Group $L$ layers into $N \approx 8$ blocks:

$$h_l = \sum_{n=0}^{N} \alpha_{n \to l} \cdot B_n$$

$$\alpha_{n \to l} = \frac{\exp(w_l^\top \, \text{RMSNorm}(B_n))}{\sum_{n'} \exp(w_l^\top \, \text{RMSNorm}(B_{n'}))}$$

Here $w_l \in \mathbb{R}^d$ is a single learned pseudo-query per layer. The softmax is over the *depth* dimension — which block to attend to, not which token.

> **Note: The critical causal property.** The $\alpha$ weights are computed *before* the MoE router fires. We know which blocks matter for this token before deciding which chefs to call in. AttnRes is not just a modelling improvement — it is a runtime control signal that collapses multiple systems problems (prefetch, eviction, precision allocation) into one observable.

> **Tasting as you go.** Before the next station adds its ingredient, the head chef tastes the dish (AttnRes $\alpha$). From the flavour, they know which specialist chef is needed next. They send a runner outside to fetch them (`madvise` prefetch) while the current station is still stir-frying. By the time the wok is empty, the next specialist is already walking through the door with their ingredients. Previous approaches used static rules like "the wrapper station and the steaming station are always the most important" (APEX) or checked last week's sales (DynaExQ). AttnRes is the chef tasting in real time, per dish, at zero extra cost — because tasting is already part of how a kitchen works.

### 9.3 From block importance to expert management

The $\alpha$ weights can drive three things:

**Expert prefetching (optional).** Affinity matrix $A \in \mathbb{R}^{L \times N \times E}$. For layer $l+2$:

$$\text{score}(e) = \sum_{n=1}^{N} \alpha_{n \to l+2} \cdot A[l+2, n, e]$$

Predicted experts = top-$K$. Issue `madvise(WILLNEED)` 2 layers ahead — the NVMe page-in overlaps with current computation.

**Importance-aware eviction.** Low-$\alpha$ blocks → their experts are safer to evict.

**Dynamic precision allocation** (optional). Low-$\alpha$ blocks tolerate 2-bit; high-$\alpha$ warrant 4-bit.

> **Warning: Empirical note (April 2026).** The AttnRes predictor is implemented (`--use-predictor`) and achieves 97% decode hit rate on Gemma 4, but reduces throughput by 24–33% due to per-layer prediction overhead and crashes at low `max_resident_experts` (16/32). A top-2 variant reduces the penalty but does not eliminate it. Task-aware pinning showed 0% hit-rate improvement over baseline LRU. The predictor is **optional and not enabled by default** — it is not required for pathway validation. The core stack uses pure LRU with `ensure_loaded()`.

**DedeKimi observer constraint.** Activation patterns logged by the DedeKimi observer (EMA-based expert frequency + per-layer entropy tracking) are for observation only. They must not be used for prefetch or eviction control until offline validation proves that observer-driven predictions improve hit rates over AttnRes alone — otherwise, biased activations from prefetch-driven cache residency create self-reinforcing feedback loops.

### 9.4 What AttnRes replaces

| Prior approach | Signal | Limitation |
|---|---|---|
| APEX | Layer position | Static; same for all inputs |
| DynaExQ | Activation frequency | Historical; one token behind |
| MoPEQ | Hessian curvature | Offline; expensive |
| **AttnRes** | **Block attention $\alpha$** | **Runtime, input-dependent, zero cost** |

AttnRes collapses three separate heuristics into a single signal already computed as part of the forward pass. Whether that signal can be profitably exploited for prefetch without throughput regression remains an open engineering question (see Section 10b.6).

---

## 10. The full stack — how it composes

For a single token during decode on Kimi-K2.5 (1T parameters, 384 experts, 60 layers, 128GB Apple Silicon). The core stack (required) is marked; optional components are labelled:

**Step 1.** **AttnRes** computes $\alpha_{n \to l}$ — which blocks matter for this token (if the model exposes AttnRes weights; otherwise pure LRU).

**Step 2.** **(Optional) Expert predictor** uses $\alpha_{n \to l+2}$ and affinity matrix to predict experts for layer $l+2$. Async `madvise(WILLNEED)`. Currently disabled by default due to throughput regression (see Section 9.3).

**Step 3.** **Attention** retrieves compressed KV from IsoQuant cache via fused Metal pipeline (Section 6.3): packed 3-bit → fused Q·Kᵀ → softmax → fused V accumulation → single inverse rotation. No full K or V tensor is materialised. New $k_l, v_l$ compressed on insertion.

**Step 4.** **MoE routing** selects top-8 + shared expert. `ensure_loaded()` from disk. Shared expert pinned at Q8_0. Routed experts at native INT4.

**Step 5.** **Expert computation** on quantised weights. `mx.eval()` fence forces memory reuse.

**Step 6.** **LRU eviction** of experts (importance-weighted if AttnRes available, otherwise pure LRU) via `madvise(DONTNEED)`.

### Runtime memory budget

Hard ceiling: 110 GB committed (128 GB minus macOS stability margin — aggressive page eviction begins at ~85% utilisation).

| Component | Allocation |
|---|---|
| Non-expert layers (INT4 dense, Q8_0 shared) | 40 GB |
| OS + Metal + stability margin | 10 GB |
| KV cache (MLA compressed) | 8 GB |
| Activations, buffers, deferred prefill spike | 2 GB |
| **Available for LRU expert cache** | **68 GB** |

At native INT4 (17.6 MB/expert), 68 GB holds ~3,863 slots — ~64 experts/layer out of 384 (17% resident). The rest live on the ~500 GB disk checkpoint.

---

## 10b. Empirical anchors — what we have measured

This section reports the results we have, honestly separated from the results we still need. The system described above is a design — these numbers ground specific parts of it.

### 10b.1 TurboQuantNemo baseline (validated, published)

The expert offloading and mixed-precision quantisation stack has been validated end-to-end on Nemotron-H 120B (github.com/2096955/TurboQuantNemo):

| Metric | Result |
|---|---|
| Model | Nemotron-H 120B (4-bit dense, 2-bit experts) |
| Hardware | 32 GB M-series Apple Silicon |
| Peak memory | 17.4 GB |
| Decode throughput | 18.7 tok/s |
| Prefill (1024 tokens) | ~8.5 s |
| Quality gate | Pass (coding, reasoning, instruction-following) |

Memory breakdown: ~9 GB resident dense weights, ~6 GB LRU expert cache (hot set), ~2 GB KV cache (uncompressed), ~0.4 GB activations/buffers.

This validates the expert offloading axis (Section 3) and mixed-precision weight quantisation on consumer hardware. KV cache compression (TurboQuant) was ported but not validated on this model's hybrid Mamba-attention architecture due to SSM cache corruption — KV compression is applied only to pure attention layers.

### 10b.2 Gemma 4 26B-A4B weight quantisation (exploratory, pinned)

Three-way self-quantised comparison from BF16 source (pinned artefacts in repo):

| Variant | Quality (11-prompt) | Peak memory (4096 tokens) | Disk |
|---|---|---|---|
| Uniform 4-bit | 5/11 (45%) | 5.37 GB | 13 GB |
| Layer-aware (APEX bands) | 6/11 (55%) | 4.53 GB | 10 GB |
| 2-bit experts | 5/11 (45%) | 4.30 GB | 8 GB |

Caveats: max_tokens=200 truncates many answers mid-code-block — low absolute pass rates are partly harness-limited. Single machine. These prove conversion plumbing and memory behaviour, not quality validation. The quality bar (agreed absolute floor + non-regression) must be met before declaring any variant validated.

## 10b. Empirical anchors — what we have measured

This section reports the results we have, honestly separated from the results we still need. The system described above is a design — these numbers ground specific parts of it.

### 10b.1 Phase 3: 16GB Pathway Proof (Validated)

The 16GB pathway for Gemma 4-26B has been validated end-to-end on constrained Apple Silicon (capped 128GB M4 Max at 12.8GB envelope):

| Metric | Gemma 4-26B-A4B (Layer-aware) | Acceptance | Status |
|---|---|---|---|
| Quality Gate | **12/12** | All pass | ✅ PASS |
| Decode Throughput | **12.85 tok/s** | ≥ 5 tok/s | ✅ PASS |
| Peak Memory | **5,420 MB** | ≤ 12,800 MB | ✅ PASS |
| 2h Stability Soak | RSS drift 1.18x, P99/P50 1.29 | < 1.5x drift | ✅ PASS |

Qwen3-30B-A3B is currently **blocked on quality** (8/12), though it passes the benchmark (9.87 tok/s, 9489 MB peak) and soak. Failures (repetition, formatting) are attributed to the current 4-bit serving stack rather than IsoQuant KV.

### 10b.2 Phase 1: Canonical KV Fidelity (Pinned)

IsoQuant consistently preserves attention scores across all three architectures, outperforming TurboQuant by 10-50x in PPL retention. Results are multi-depth at 512 and 2048 tokens:

| Model | backend | PPL @ 512 | PPL @ 2048 | Delta @ 2048 |
|---|---|---|---|---|
| **Qwen3-30B-A3B** | default | 1.3829 | 1.0844 | — |
| | turboquant | 1.4497 | 1.1249 | +0.0405 |
| | isoquant | 1.3872 | 1.0853 | **+0.0009** |
| **Gemma 4-26B-A4B**| default | 1.3829 | 1.0844 | — |
| | turboquant | 1.6980 | 1.1466 | +0.0622 |
| | isoquant | 1.3829 | 1.0844 | **+0.0000** |
| **Nemotron-30B** | default | 1.3911 | 1.0866 | — |
| | turboquant | 1.4086 | 1.0905 | +0.0039 |
| | isoquant | 1.3961 | 1.0878 | **+0.0012** |

### 10b.3 Phase 0: Kurtosis & Structural Decisions

Pearson kurtosis measurements justify the mixed-precision weight allocation:
- **Shared expert kurtosis:** 13.10 (Heavy-tailed outliers)
- **Routed expert kurtosis:** 3.41 (Narrower distribution)
- **Gap:** 3.8x

This confirms the Q8_0 shared expert pinning as a structural necessity rather than a heuristic choice.

### 10b.4 What is still missing (and what would make this definitive)

Two things would elevate this from "well-argued system that probably works" to "system that demonstrably works":

**Real 16GB Hardware Rerun.** Native verification on 16GB physical RAM to confirm OS swap pressure and Metal resource contention matches our simulated envelope results.

**End-to-end decode profiling.** A per-token time breakdown: what fraction of decode time is spent in KV attention vs. expert I/O vs. other? The fused Metal decode pipeline (Section 6.3) eliminates KV overhead from the attention path, but we must confirm this translates to measurable end-to-end gains in the context-heavy regime.

### 10b.5 Go/No-go decisions (April 2026)

| Component | Decision | Rationale |
|---|---|---|
| IsoQuant (WHT + SO(4)) | **Go** | Quality parity with default (delta PPL ≈ 0), 64× fewer parameters |
| Fused Metal pipeline | **Go** | Verified by 9 correctness tests, eliminated materialization |
| Deferred prefill | **Go** | Eliminates compounding error; ~512 MB buffer is manageable |
| Gemma4 pathway | **Go** | All gates pass at 12.85 tok/s within 16GB budget |
| Qwen3 pathway | **Blocked** | Quality issues (8/12) likely inherent to 4-bit stack or base model |
| AttnRes predictor | **No-go** | 24–33% throughput regression, crashes at low resident count |
| Task-aware pinning | **No-go** | 0% hit-rate improvement over baseline LRU |
| QES | **Planned** | Background evolution strategies for gate-weight optimization |


Detailed benchmark artefacts, stability soak results, and the full go/no-go evidence are maintained in the companion engineering roadmap: *Efficient MoE quantization for 16GB/32GB Apple Silicon — pathway to Kimi K2.5*.

---

## 11. QES — background optimisation via evolution strategies (proposed)

Everything above describes inference. QES is a *proposed* tuning layer — a background process that would adapt the model's **gate weights** (the routing function $W_g$) to perform better on the actual hardware. Empirical validation is ongoing.

> **Tweaking the seating chart, not the recipes.** QES doesn't change the chefs' recipes (expert weights) or the head chef's tasting method (AttnRes pseudo-queries). It tweaks the floor manager's seating chart — the rules for which chef handles which order. On the chef's night off, the manager tests 10–20 slightly different seating arrangements on a practice service and keeps the ones that run most smoothly.

### 11.1 The reward function

$$R_{\text{total}} = \underbrace{R_{\text{accuracy}}}_{\text{primary}} - \sum_{c} \lambda_c \cdot \text{penalty}_c$$

Four penalties (squared, task-conditioned): **expert entropy collapse** (too few chefs used), **expert churn** (chefs swapping stations chaotically), **cache hit rate** (chefs not where they're needed), **memory pressure** (kitchen overflowing — hard constraint, $\lambda = 10$).

### 11.2 The loop

QES perturbs **gate weights only** — the router's $W_g$. Expert weights and AttnRes pseudo-queries are never touched. Gradient-based methods (RL, policy gradients) are unstable under discrete quantised routing and hardware-coupled rewards — evolution strategies handle discrete INT4 weights and non-differentiable system metrics (cache hit rate, memory pressure) naturally.

```
For each generation (weekly):
  1. Sample 10–20 perturbed gate weight sets
     (small noise to W_g in 30% of MoE layers)
  2. Run inference on 16 validation prompts
  3. Rank by reward, keep elites
  4. Safety: accuracy drop >5% or hit rate drop >10 pts → revert
  5. Update from best elite
```

> **Recipe tournament.** Each round, 10–20 tweaked seating charts are tested on the same orders. Judged on customer satisfaction (accuracy) plus kitchen flow (cache hits, churn, counter space). Best charts survive. Over weeks, the routing drifts toward patterns that are accurate *and* practical for this specific kitchen. Trial-and-error, not recipe-book (gradient-free), because the chart values are in whole numbers — you can't assign "2.7 orders" to a chef (INT4 is discrete).

### 11.3 Task-conditioned weights

| Task | λ_entropy | λ_churn | λ_hit | λ_mem |
|------|-----------|---------|-------|-------|
| Code | 0.5 | 0.3 | 1.0 | 10.0 |
| Reasoning | 1.0 | 0.5 | 0.5 | 10.0 |
| Multilingual | 0.8 | 0.4 | 0.7 | 10.0 |

**Status:** QES is designed but not yet evaluated. No gate-weight perturbation experiments have been run. The loop, reward function, and KL constraint above are the specification; empirical validation (cache hit rate improvement, quality retention under perturbation) is ongoing work.

---

## 12. Contribution boundaries

**Core stack** (implemented, produces artefacts): Expert offloading with LRU and `ensure_loaded()`. IsoQuant (WHT + SO(4)) KV compression on Apple Silicon Metal. Fused 4-kernel Metal decode pipeline (`fused_kv_decode_kernels.py`) operating directly on 3-bit packed data. Inverse rotation moved after attention sum. Deferred prefill with bulk compression. Mixed-precision weight quantisation (4-bit dense, 2-bit experts, Q8_0 shared). Compressed sensing as theoretical framework for understanding compression viability. The core stack end-to-end on consumer Apple Silicon. llama.cpp track: `GGML_TYPE_ISOQUANT3_0` (enum 44) with Metal flash-attention template specialisations, dedicated `kernel_set_rows_isoquant3` write kernel applying WHT + per-head SO(4) block rotations from metadata, `ggml_set_rows_ext()` API for metadata-bearing writes, per-layer rotation tensors in KV cache, full runtime wiring across `llama-context`, `llama-kv-cache`, and `llama-graph`, and CLI access (`--cache-type-k isoquant3`). **Hardened pipeline** ready for real SO(4) data; corrected for GQA expansion (4D reshape) and GGML context memory allocation; defaults to identity (WHT-only) pending rotation tensor initialisation and graph-side Q/output rotation matching.

**Optional enhancements** (implemented, not enabled by default): AttnRes predictor (`--use-predictor`) — throughput regression prevents it from being a net win on constrained hardware. Task-aware pinning — 0% hit-rate improvement.

**Proposed** (documented design, not yet implemented): QES gate-weight optimisation.

**Individual prior art:** LRU expert offloading (Eliseev & Mazur 2023). Lloyd-Max KV quantisation (TurboQuant, ICLR 2026). Mixed-precision weight quantisation (APEX, MxMoE). Quaternion rotation (RotorQuant/scrya-com; arXiv:2603.28430). Block attention residuals (Moonshot AI, arXiv:2603.15031). Compressed sensing theory (Candès & Tao 2006).

**Empirical question marks:** Whether IsoQuant adds meaningful compression on top of MLA (<10% or PPL >+0.5 → skip). Long-context stability under combined compression. Whether the fused Metal decode pipeline delivers wall-clock speedup end-to-end — the kernels eliminate KV materialisation overhead, but if KV attention is <20% of total decode time, the impact is negligible. End-to-end profiling is the remaining gate.

**What we chose not to do:** Speculative decoding was considered and rejected — each speculative token can route to different experts, turning a predictable prefetch stream into a chaotic SSD stampede that drops hit rate below the 70% threshold. It is incompatible with expert offloading on memory-constrained hardware. QJL residual correction is off by default — the bit budget is better spent on more Lloyd-Max centroids (Section 5.1).

**The unifying invariant.** The entire system is designed around one principle: *preserve the ordering of attention scores under constrained memory and bandwidth*. Softmax is invariant to additive shifts but highly sensitive to rank ordering — making top-$k$ preservation more critical than MSE. Every component serves this invariant: KV compression preserves approximate dot products, compressed sensing explains why the approximation holds, IsoQuant/TurboQuant preserve geometry, AttnRes identifies which computations matter, and MoE sparsity reduces the active parameter set.

**Three layers of importance.** The system implicitly defines a hierarchy of signals: (1) *token importance* — softmax sparsity determines which past tokens matter for the current one; (2) *expert importance* — MoE routing determines which specialist computations fire; (3) *layer importance* — AttnRes $\alpha$ weights determine which depth blocks carry meaning for the current computation. The composition works because these three signals are orthogonal: token importance drives KV cache policy, expert importance drives loading and eviction, layer importance drives precision allocation. No joint optimisation is required — each operates independently on a different resource dimension.

---

## Appendix: The Math of the Kitchen (A Yum Cha Guide to the Formulas)

This appendix translates the mathematical formulas from the main text into the daily operations of our yum cha kitchen. If you've been following the grey boxes, these mappings will help you see exactly how the "chef's intuition" connects to the formal equations.

### 1. Standard Attention: Finding the Right Filling

**The Formula:**
$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

**The Kitchen Mapping:**
*   **$Q$ (Query):** The current **order** (dough wrapper) waiting for a filling.
*   **$K$ (Key):** The **labels/tags** on every prepped filling bowl in the kitchen.
*   **$V$ (Value):** The **actual fillings** inside those bowls.
*   **$QK^\top$:** The head chef **checking the match**. For each bowl, the chef compares the dough ($Q$) to the label ($K$) to see how well they go together.
*   **$\text{softmax}$:** The chef's **final decision**. This turns the match scores into precise percentages (e.g., "This order is 90% Pork-and-Ginger and 10% Prawn-and-Chive"). All percentages must add up to 100%.
*   **Multiplying by $V$:** The **scooping**. The chef scoops the exact proportions of each filling ($V$) into the wrapper ($Q$) to make the perfect dumpling.

---

### 2. Mixture-of-Experts (MoE): Calling the Station Chefs

**The Formula:**
$$\text{MoE}(x) = \sum_{e \in \text{TopK}(G(x))} G(x)_e \cdot f_e(x)$$

**The Kitchen Mapping:**
*   **$x$:** The incoming **customer order**.
*   **$G(x)$:** The **Floor Manager** (Gating function). They look at the order and decide which chefs are needed.
*   **$\text{TopK}$:** The **Active List**. Out of 384 chefs playing cards in the back alley, the manager only yells for the **Top $K$** (usually 8) whose skills best match this specific order.
*   **$f_e(x)$:** The **Station Chef** ($e$). They receive the order and prepare their specific part of the dish.
*   **The sum ($\sum$):** The **Dish Assembly**. Combining the work of all 8 active chefs into one final plate.

---

### 3. Lloyd-Max Quantization: Sizing the Steamer Baskets

**The Formula:**
$$D = \sum_{i=1}^{2^b} \int_{b_{i-1}}^{b_i} (x - c_i)^2 \, p(x) \, dx$$

**The Kitchen Mapping:**
*   **$x$:** The **actual size** of a raw dumpling.
*   **$c_i$ (Centroids):** The **fixed sizes** of steamer baskets we have.
*   **$b_i$ (Boundaries):** The **sorting rule** (e.g., "Any dumpling between 12g and 15g goes into the Medium basket").
*   **$(x - c_i)^2$:** The **deformation**. If a 14g dumpling is forced into a 12g basket, it gets squashed. This "squared error" measures how much the dumpling is damaged.
*   **$p(x)$:** The **dumpling distribution**. Knowing that 80% of our dumplings are "Standard" size and only 1% are "Giant."
*   **Minimizing $D$ (Distortion):** This math finds the **perfect basket sizes** ($c_i$) and **sorting rules** ($b_i$) so that the average dumpling across the whole day is squashed as little as possible.

---

### 4. IsoQuant Rotation (WHT + SO(4)): Mixing the Filling

**The Formula:**
$$\tilde{k} = \Pi_{\text{SO}(4)}(H_d \cdot \hat{k}) \quad \text{where} \quad \tilde{k}_{[i]} = \mathfrak{q}_{L,i} \otimes \hat{k}_{[i]} \otimes \bar{\mathfrak{q}}_{R,i}$$

**The Kitchen Mapping:**
*   **$H_d$ (Walsh-Hadamard Transform):** The **Global Mix**. Dumping all ingredients into one massive bowl and giving it a rough, high-speed stir to spread everything out.
*   **$\mathfrak{q}_{L}, \mathfrak{q}_{R}$ (Quaternions):** The **Two-Handed Fine Mix**. For small batches of 4 ingredients, the chef uses two independent hands to rotate and fold them perfectly.
*   **Why do this?** If the pork is in one lump and the ginger in another, the Lloyd-Max baskets (quantization) won't work—one basket gets all pork, the other all ginger. The rotation ensures the filling is **isometrically balanced**—no ingredient is lost, but every bite tastes exactly the same, making it easy to pack them into standard baskets.

---

### 5. Compressed Sensing (RIP): The Taste Test Guarantee

**The Formula:**
$$(1 - \delta_s)\|x\|_2^2 \leq \|\Phi x\|_2^2 \leq (1 + \delta_s)\|x\|_2^2$$

**The Kitchen Mapping:**
*   **$x$:** The **entire batch** of 1,000 dumplings.
*   **$\Phi$:** The **sampling spoon**.
*   **$\Phi x$:** The **few bites** the chef actually tastes.
*   **RIP (Restricted Isometry Property):** The **"No Surprises" Rule**. This formula is the mathematical proof that the bites you tasted ($\Phi x$) have almost the same "energy" and flavour profile as the whole batch ($x$).
*   **Conceptual takeaway:** If your sampling method satisfies RIP, you can **mathematically guarantee** the quality of 1,000 dumplings by only tasting 5. You don't have to guess; the math says there are no hidden "bad dumplings" that your spoon missed.

---

### 6. Attention Residuals (AttnRes): Tasting as You Go

**The Formula:**
$$h_l = \sum_{n=0}^{N} \alpha_{n \to l} \cdot B_n$$

**The Kitchen Mapping:**
*   **$B_n$:** The **dish-in-progress** as it leaves previous kitchen blocks ($n$).
*   **$\alpha_{n \to l}$:** The **Mid-Prep Taste Test**.
*   **The Result:** The head chef tastes the dish *before* it reaches the next station. Based on that taste ($\alpha$), they know exactly which specialist chef ($B_n$) had the most impact so far.
*   **The Benefit:** This signal is like the chef yelling to the back alley: "I can taste the ginger is weak, get the Ginger Specialist ready now!"—allowing the next chef to be **pre-fetched** and ready at their station before the plate even arrives.
