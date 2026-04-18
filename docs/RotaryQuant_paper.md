# RotaryQuant: Structured-Rotation KV Compression for Large MoE Models on Consumer Hardware

> **Status: Draft v1 (skeleton — sections to be filled in sequentially).**
> File `RotaryQuant_paper.md` is the supervisor-restructured successor to `FROM_ATTENTION_TO_CONSUMER_HARDWARE.md`. The yum-cha-kitchen intuitive companion has been moved to Appendix C; the academic structure (Introduction → Literature Review → Theory → Results → Conclusion) follows the supervisor's prescribed paragraph plan.

---

## Abstract

Large MoE LLMs remain prohibitively expensive to operate on consumer devices with 16-32 GB unified memory because resident weights, KV state, and buffers exceed both capacity and bandwidth budgets. This paper presents RotaryQuant, a structured-rotation KV-compression stack that replaces dense per-head rotations with a Walsh-Hadamard pre-mix followed by blockwise isoclinic SO(4) rotations, preserving the isotropy benefits of rotation-based quantisation while storing 64× fewer rotation parameters than dense baselines. The method adds a deferred-prefill protocol that preserves FP16 KV during prefill and bulk-compresses only at the prefill-to-decode boundary, a fused decode pipeline that attends directly in rotated space without repeated materialisation, and end-to-end composition with low-bit weight quantisation and expert offloading as three independent compression axes. Across the main fidelity pathways, structured rotation matches dense rotation quality at fractional storage and compute cost: at 2,048 context, Qwen-3 PPL shifts by only +0.0009 under RotaryQuant (versus +0.0405 under TurboQuant) and Nemotron-H by +0.0012 (versus +0.0039), with the deltas inside the practical noise band on these architectures; the Gemma-4 row is effectively a no-op because only 5 of 30 layers engage the compressor. End-to-end pathway proofs reach 12.85 tok/s for Gemma-4-26B inside a 16 GB-constrained envelope and 14.85 tok/s for Nemotron-H 120B inside a 32 GB-constrained envelope. The stack is conditionally beneficial: when a model already fits in RAM, expert offload can impose an approximately 100× decode penalty, so the full composition should be used only where memory pressure genuinely forces offload.

---

## Acronyms and Notation

The following acronyms appear throughout the paper and are expanded on first use thereafter. Paper-specific terms are tagged with *(this paper)* or *(prior work)* where the meaning differs from common usage.

- **AMX** — Apple Matrix coprocessor; a vector unit in Apple Silicon used for matrix-vector kernels.
- **AttnRes** — Attention Residuals; a routing-prediction head explored in this paper *(this paper)*.
- **AWQ** — Activation-aware Weight Quantisation; a post-training weight-quantisation method that down-weights activation outliers.
- **BPE** — Byte-Pair Encoding; subword tokenisation algorithm used by most modern LLMs.
- **DKV** — Down-projected Key-Value cache; the compressed KV representation used by Multi-head Latent Attention.
- **ESG** — Environmental, Social, and Governance; reporting framework used to assess sustainability and societal impact.
- **FFN** — Feed-Forward Network; the per-token MLP block in a transformer layer.
- **FMA** — Fused Multiply-Add; a single hardware instruction performing $a \times b + c$, the unit cost we use to compare rotation methods.
- **FP16 / FP32** — half-precision and single-precision IEEE-754 floating-point formats.
- **GEAR** — A KV-cache compression method combining low-rank decomposition with sparse residuals *(prior work)*.
- **GELU** — Gaussian Error Linear Unit; the activation function used in many transformer FFNs.
- **GGML** — Georgi Gerganov Machine Learning; the C tensor library underpinning `llama.cpp`.
- **GGUF** — GGML Universal Format; the on-disk checkpoint format used by `llama.cpp`.
- **GPTQ** — Post-training weight quantisation method using second-order curvature approximations.
- **GQA** — Grouped-Query Attention; an attention variant where several query heads share a smaller set of key/value heads.
- **HBM** — High-Bandwidth Memory; the memory class used by datacentre GPUs (e.g., HBM2e/HBM3 on H100/B200).
- **IoT** — Internet of Things; broadly, networked devices with constrained compute and memory budgets.
- **IsoQuant** — Isometric Quantisation; the published family of structured-rotation KV-cache methods, including the earlier IsoQuant v1 / IsoQuant v2 attempts and the WHT + SO(4) lineage cited in [6] *(prior work / family name)*.
- **JL** — Johnson-Lindenstrauss; the lemma that bounds distortion under random low-dimensional projection.
- **KIVI** — A 2-bit KV-cache quantiser using per-channel keys and per-token values *(prior work)*.
- **KV** — Key-Value; the per-token tensors stored across attention layers during autoregressive decoding.
- **KVQuant** — A 2-bit non-uniform KV-cache quantiser with per-channel sensitivity weights *(prior work)*.
- **LLM** — Large Language Model.
- **LRU** — Least Recently Used; a cache-eviction policy.
- **MLA** — Multi-head Latent Attention; an attention variant that stores a low-dimensional latent (down-projected KV) state from which K and V are reconstructed at attention time.
- **MLP** — Multi-Layer Perceptron.
- **MLX** — Apple's MLX framework for machine-learning on Apple Silicon (unified memory, lazy evaluation).
- **MMLU** — Massive Multitask Language Understanding; a 57-subject multiple-choice benchmark used as a quality proxy.
- **MoE** — Mixture-of-Experts; an architecture where each token activates only a subset of available expert sub-networks.
- **MSE** — Mean Squared Error.
- **NVMe** — Non-Volatile Memory Express; the protocol used by modern SSDs over PCIe.
- **OOM** — Out Of Memory.
- **PCM** — Pulse-Code Modulation; cited only in the metaphor appendix.
- **PPL** — Perplexity; standard intrinsic-quality metric, $\exp(\text{cross-entropy})$.
- **PRISMA** — Preferred Reporting Items for Systematic Reviews and Meta-Analyses; reporting standard cited in Section 2.
- **PTQ** — Post-Training Quantisation.
- **QES** — Quality-Equivalent Sampling; a planned conditional safeguard for 2-bit expert quality *(this paper, planned)*.
- **QJL** — Quantised Johnson-Lindenstrauss; a residual-correction step in TurboQuant *(prior work)*.
- **QR** — QR decomposition; factors a matrix into an orthogonal $Q$ and an upper-triangular $R$. Used to draw random orthogonal rotation matrices in TurboQuant.
- **Q8_0** — A `llama.cpp`/GGUF 8-bit weight-quantisation format with a single per-block scale; cited here as the higher-precision tier reserved for shared experts.
- **QuaRot** — A rotation-based weight-and-activation quantisation method *(prior work)*.
- **QuIP#** — A weight-quantisation method using lattice codebooks and incoherence pre-processing *(prior work)*.
- **RAG** — Retrieval-Augmented Generation.
- **RAM** — Random-Access Memory (here, the unified-memory pool on Apple Silicon).
- **RMSE** — Root-Mean-Square Error.
- **RoPE** — Rotary Position Embedding; the positional-encoding scheme used by most modern transformers.
- **RotaryQuant** — The working WHT + SO(4) variant of the IsoQuant family, integrated end-to-end on consumer hardware *(this paper)*. Earlier IsoQuant variants (v1 single-quaternion sandwich, v2 block-only) are documented as failed predecessor designs in §4.2.
- **RotorQuant** — A four-dimensional Cayley-rotor variant in the broader structured-rotation line, retained as an experimental fall-back *(this paper)*.
- **RSS** — Resident Set Size; the portion of process memory held in physical RAM.
- **SDPA** — Scaled Dot-Product Attention.
- **SIMD** — Single-Instruction Multiple-Data; the vector-instruction class used by ARM/AMX kernels.
- **SO(4)** — The special orthogonal group in four dimensions; here, the per-block rotation used by RotaryQuant.
- **SSD** — Solid-State Drive.
- **SSM** — State-Space Model; the recurrence used by Mamba-class layers in Nemotron-H.
- **TurboQuant** — Dense-rotation 3-bit KV-cache quantiser; the closest prior art to RotaryQuant *(prior work)*.
- **WHT** — Walsh-Hadamard Transform; the structured orthogonal $d \times d$ pre-mix used as the first stage of a RotaryQuant rotation.

### Notation

The following symbols are used throughout the paper. Where a symbol is overloaded, the meaning is given in context.

- $d$ or $d_k$ — head dimension (typically 128 in this paper); introduced in Section 1.
- $H$ — number of attention heads; introduced in Section 1.
- $H_q, H_{kv}$ — number of query heads and key/value heads under GQA; introduced in Section 2.
- $T$ — sequence length (token positions stored in the KV cache); introduced in Section 1.
- $B$ — batch size (typically 1 in this paper); introduced in Section 1.
- $E$ — total number of experts in an MoE layer; introduced in Section 2.1.
- $k$ — top-$k$ active experts per token; introduced in Section 2.1.
- $q$ — query vector (dimension $d$); introduced in Section 1.
- $k_j, v_j$ — key and value vectors at sequence position $j$; introduced in Section 2.2.
- $W$ — weight matrix; introduced in Section 2.1.
- $R$ — rotation matrix used by KV-cache compression (orthogonal, $d \times d$); introduced in Section 2.4.
- $H_d$ — the $d \times d$ Walsh-Hadamard matrix; subscripted to disambiguate from $H$ (number of heads). Introduced in Section 4.2.
- $q_L, q_R$ — left and right unit quaternions parameterising one isoclinic SO(4) block; introduced in Section 4.2.
- $\alpha_j$ — softmax attention weight at position $j$; introduced in Section 1.
- $\sigma_q$ — query-vector standard deviation; appears in the rank-preservation bound in Section 2.4.

---

## 1. Introduction

### 1.1 The hardware ceiling on large language model inference

Recent MoE releases have widened the gap between published model scale and the memory budgets of ordinary client devices. Gemma-4 26B-A4B, Nemotron-H 120B with roughly 12 B active parameters per token, and Qwen-3-MoE are presented at sizes that presume far more headroom than the 16-32 GB unified-memory class common on Apple Silicon, the 8-16 GB range typical of mainstream Windows laptops, or the 6-12 GB budgets of current phones [31][32][33]. Even under aggressive compression, the arithmetic is unfavourable: a 30-B-parameter MoE at 4-bit still needs roughly 15 GB merely to hold weights, while an 8 K context can add a further 4-8 GB of KV state; both must still coexist with the operating system, runtime buffers, and allocator slack. The obstacle is therefore simultaneously one of capacity and bandwidth. Datacentre accelerators are provisioned around HBM (high-bandwidth memory), whereas consumer systems depend on LPDDR5x (low-power DDR5x), making every unnecessary transfer of weights or cache state a first-order inference cost.

### 1.2 The infrastructure and ESG cost of cloud-only inference

The consequence of this mismatch is not merely local inconvenience; it reinforces a cloud-centric inference regime whose costs are borne at infrastructural and environmental scale. Datacentre inference is increasingly concentrated within a small number of hyperscale operators, depends on energy-intensive accelerator fleets, and relies on cooling systems whose water demand is already material in the sustainability literature on hyperscale facilities and scope-2 electricity emissions [34][35]. Because accelerator supply chains remain constrained, growth in inference demand also competes for scarce high-end silicon, packaging capacity, networking equipment, and power-delivery headroom [36]. Meanwhile, the aggregate carbon burden of cloud inference is not declining commensurately with per-chip efficiency gains, because model sizes, context windows, and always-on serving footprints are scaling faster than those gains can offset [37]. Shifting a meaningful fraction of inference onto already-deployed personal devices is therefore relevant not only to convenience or privacy, but also to attenuating centralised infrastructure growth; Section 1.3 treats that shift as one practical lever.

### 1.3 The everyday-use cost of cloud-only inference

Cloud-only inference also imposes direct costs on ordinary use. Every request incurs network latency, depends on stable connectivity, and crosses an external service boundary, so privacy exposure is not abstract but literal: each prompt leaves the device. This architecture disadvantages users in low-bandwidth or intermittently connected regions, limits deployments on mobile and field equipment, and converts routine assistance into a recurring subscription expense that can exclude users in lower-income settings. The relevant hardware substrate, however, is already widely deployed. Phones, laptops, IoT sensors, and edge-robotics platforms now ship with heterogeneous compute stacks that include Apple M-series parts, Snapdragon-class accelerators, and Tensor-class mobile silicon capable of meaningful local inference workloads [38]. The limiting factor is therefore less the absence of hardware than the absence of software that uses it efficiently. If compression, scheduling, and memory management improve sufficiently, these devices need not remain thin clients for remote models; they can become first-class inference hosts in their own right.

### 1.4 How the field currently addresses the gap

The literature addresses this deployment gap through several partially complementary compression directions rather than a single dominant remedy. On the weight side, 4/3/2-bit PTQ reduces storage and memory traffic without end-to-end retraining, with GPTQ-style second-order fitting and related activation-aware calibration methods forming a widely used practical lineage [22][29]. A second line compresses the KV cache itself, whether through asymmetric low-bit schemes such as KIVI [3] and KVQuant [4], residual or low-rank hybrids such as GEAR [5], or rotation-based methods such as TurboQuant [1]. A third line exploits MoE sparsity at runtime through expert offloading, prefetch, and residency control so that inactive experts need not remain memory-resident [9]. Alongside these, pruning and structured sparsity shrink parameter count directly, distillation transfers behaviour into smaller student models, and speculative decoding trades auxiliary draft-model computation for lower end-to-end latency [29][30]. These methods attack different resource terms and therefore combine imperfectly rather than collapsing into one best practice. Section 2 revisits the strands most relevant here in greater depth: weight quantisation, KV compression, expert offloading, and the TurboQuant line.

### 1.5 Where existing methods fall short

Despite substantial progress, the rotation-quantisation pipeline most relevant to this paper still leaves three concrete inefficiencies unresolved. First, TurboQuant stores a dense rotation matrix per head, so both parameter storage and application cost scale as $O(d^2)$, requiring dense per-head transforms and $O(d^2)$ FMA work at decode time [1]. Secondly, in existing rotation-quantisation pipelines, the prefill-to-decode boundary is handled only after sequence positions have already accumulated dependence on one another, so quantisation error introduced at that boundary is subsequently propagated through later attention steps instead of being confined to newly appended tokens [1]. Thirdly, current fused attention paths continue to materialise K and V during decode, so inverse rotation and reconstruction scale as $O(T \times d^2)$ over sequence length $T$ rather than remaining a fixed overhead [1]. We stress that these are not universal defects of KV compression as a category; they are the specific TurboQuant-side weaknesses that this paper addresses.

### 1.6 Our contribution

This paper advances RotaryQuant as a structured alternative to dense rotation-based KV compression. A Walsh-Hadamard pre-mix followed by blockwise isoclinic SO(4) rotations addresses the storage and compute burden of dense rotations by replacing 16,384 stored entries per head with 256 structured entries and reducing application cost to $O(d \log d)$ FMAs rather than $O(d^2)$ [6]. A deferred-prefill protocol addresses boundary error by retaining FP16 KV throughout prefill and bulk-compressing only once at the transition to decode. A fused decode pipeline addresses materialisation cost by performing attention in rotated space rather than reconstructing K and V at every step. These mechanisms are integrated end-to-end with weight quantisation and expert offloading as three independent compression axes, then validated on real consumer hardware. Section 4 presents the method, and Section 5 reports the empirical evidence.

---

## 2. Compression Methods (Literature Review)

Section 2 reviews three independent compression axes that target three distinct memory consumers in large-model inference. Weight quantisation reduces resident parameter storage by encoding dense and expert weight matrices at lower precision; KV-cache compression reduces the attention-state footprint by compressing the $(k_j, v_j)$ vectors accumulated for each token during autoregressive generation; expert offloading reduces live RAM occupancy by exploiting routing sparsity so that inactive experts need not remain resident. These mechanisms are implemented independently. Weight quantisation operates on $W$ matrices, KV compression operates on the stored $(k_j, v_j)$ tensors, and expert offloading operates on the residency policy that decides which expert shards are in RAM versus on non-volatile storage. They nevertheless interact through model dynamics, because quantised weights perturb activations and therefore the KV vectors subsequently written to cache, but each axis can still be enabled, disabled, or retuned without rewriting the others. Other compression strands, including pruning, distillation, and speculative decoding, address adjacent constraints and are surveyed elsewhere [29][30]. The present review therefore narrows to the three axes whose composition is most directly responsible for making large MoE inference feasible on consumer hardware, while deferring the paper's net-new contribution to Section 4.

### 2.1 Weight Quantisation and Sparse Mixture-of-Experts

Weight quantisation reduces resident parameter memory by replacing FP16 or FP32 blocks with low-bit PTQ representations, typically at 4, 3, or 2 bits, plus small per-block scale and zero-point metadata. In the now-standard block-wise regime, weights are partitioned into fixed groups and each group is reconstructed from quantised integers and local calibration parameters, so storage and bandwidth fall roughly in proportion to bit-width while quantisation error remains local to the block. GPTQ uses a second-order curvature-sensitivity approximation to allocate quantisation error across weights; AWQ biases calibration towards channels with high activation magnitude; SmoothQuant shifts activation outliers into weights before quantisation, reducing dynamic-range imbalance ahead of low-bit deployment [22][23][24]. The trade-off is standard: lower precision saves memory traffic and parameter storage, but perturbs the function implemented by the layer. Mixed-precision tiers therefore reserve a higher format such as Q8_0 for always-active shared components and a lower format such as Q4 for the much larger pool of conditionally used routed experts [10][11].

Sparse MoE changes not the bit-width but the number of parameters that must be active for any one token. Each MoE layer hosts $E$ experts and a learned gating function that scores them, after which only the top-$k$ experts are executed for the current token. Total parameters therefore scale as $P = N_{dense} + E \times P_{expert}$, whereas the active parameters seen by one token are only approximately $N_{dense} + k \times P_{expert}$ with $k \ll E$ [25][26]. This conditional-computation template underlies modern systems such as Qwen-3-MoE, Nemotron-H 120B with roughly 12 B active parameters per token, and Gemma-4 26B-A4B [33][32][31]. It is useful at inference time to distinguish shared experts, which are always on and therefore commonly pinned at the higher-precision tier, from routed experts, which are conditionally activated and are natural candidates for lower-bit storage. Taken together, quantisation and sparsity determine resident weight memory: up to block metadata and conversion from bits to bytes, the footprint scales as $N_{dense} \times bits_{dense} + (k \text{ or } E) \times P_{expert} \times bits_{expert}$ depending on whether offloading allows only the active working set, rather than the full expert pool, to remain resident. That observation motivates the third axis reviewed in Section 2.3.

### 2.2 Key-Value Cache Compression

During autoregressive decoding, each layer stores for every past position $j$ the key and value vectors $k_j$ and $v_j$ rather than recomputing them for every new query. The cache turns naive quadratic recomputation into a linear append-and-read process, but its memory grows directly with context length. For batch size $B$, $L$ layers, $H_{kv}$ key/value heads, head dimension $d$, and context length $T$, per-request KV memory is $B \times L \times H_{kv} \times d \times T \times 2 \times \text{dtype\_bytes}$. At long context, especially once $T \geq 32$ K, that term can rival or exceed resident weight memory.

KIVI compresses keys and values asymmetrically: keys are quantised per channel and values per token, both at 2-bit, reflecting the claim that key outliers are more channel-stable whereas value outliers are more position-dependent [3]. Its appeal is simplicity, but the method remains tied to that asymmetry assumption.

KVQuant also operates at 2-bit, but uses non-uniform quantisation and per-channel sensitivity weights to allocate distortion unevenly across dimensions [4]. This is more robust when a few channels dominate the score, at the cost of extra calibration logic and metadata.

GEAR instead uses a hybrid representation: a low-rank component preserves the bulk signal, while a sparse residual stores the most important leftovers explicitly [5]. It is therefore closer to factorised approximation with corrective structure than to pure scalar quantisation.

TurboQuant is the closest prior art because it retains the same four-stage vector pipeline while replacing only the rotation stage: a dense global orthogonal transform pushes each normalised vector towards approximate isotropy before 3-bit Lloyd-Max scalar quantisation [1]. Section 2.4 returns to it in detail because the present work inherits this logic while changing the rotation structure and boundary handling.

QuaRot and QuIP# are not KV-cache methods in the narrow sense; they primarily target weights and activations, using orthogonal pre-mixing to suppress outliers or increase incoherence before low-bit encoding [7][8]. They matter here as theoretical context because they show why rotation can stabilise otherwise brittle scalar quantisation.

A vector distribution is approximately isotropic if its covariance is close to a scalar multiple of the identity. Aggressive scalar quantisation at 1-3 bit relies on this because the bit allocation is effectively uniform across components; without isotropy, a few outlier coordinates dominate distortion. Rotation-based methods such as TurboQuant and the IsoQuant family [6], of which RotaryQuant is the working variant here, therefore impose approximate isotropy by pre-mixing before quantisation. The Hanson-Wright concentration bound provides the theoretical justification by showing how quadratic forms of orthogonally mixed random vectors concentrate around their isotropic expectation [27].

### 2.3 Expert Offloading

Expert offloading addresses a separate memory term: not the precision of stored weights, but which expert weights must be resident at all. In MoE inference, only $k$ of $E$ experts are active for a given token, so the inactive majority can in principle be streamed from NVMe or other non-volatile storage on demand and evicted afterwards, rather than occupying RAM continuously. In that sense, offloading turns routing sparsity into a residency policy. The core design question is not whether the full expert pool exists on disk, but how large a resident working set must be kept in memory to absorb the next few routing decisions without stalling decode.

The implementation patterns surveyed in the literature are broadly consistent. A resident cache of expert shards is maintained in RAM, typically under an LRU policy, and expert loads are issued only when routing demands a shard that is not already present [9]. More aggressive systems add prefetch via routing predictors, explicit residency-set sizing, or batching-aware schedulers that try to align decode steps so that multiple requests can reuse the same loaded experts [10][11][28]. These mechanisms differ in policy, but all treat expert residency as a runtime systems problem layered on top of the model's own gating behaviour rather than as a modification to the FFN mathematics.

The trade-off is that an expert miss incurs storage latency on the order of milliseconds per shard, so resident-set sizing dominates throughput. Offloading is therefore a memory-latency exchange rather than a free win: if the cache is too small, sparse routing still degenerates into repeated I/O. The concrete boundary is visible in this paper's own ablation: with 16 resident slots over 30,960 gather calls on Gemma-4-26B-A4B, the cache hit rate falls to 0%, illustrating that offloading only helps when the available resident set is large enough to capture the actual routing working set. Section 5.5 returns to this trade-off empirically.

### 2.4 TurboQuant — Process, Strengths, and Weaknesses

TurboQuant is the closest prior art because it follows the same overall KV-compression logic as RotaryQuant while instantiating the rotation stage in the densest possible way. Its pipeline can be read as four stages executed on each vector: $(i)$ project the key or value to unit norm so that direction is separated from magnitude; $(ii)$ apply a dense orthogonal rotation $R$, typically drawn by taking the $Q$ factor from the QR decomposition of a Gaussian matrix, in order to redistribute variance across coordinates; $(iii)$ quantise the rotated coordinates independently with a Lloyd-Max scalar codebook matched to the chosen bit-width, typically 3-bit; and $(iv)$ apply the inverse rotation $R^{\top}$ at decode time to reconstruct an approximation to the original direction before attention is evaluated [1]. The attraction of this process is its conceptual simplicity: it is a single-file design that can be inserted into an existing attention pipeline without changing the surrounding model.

Its strengths follow directly from that dense rotation. Because $R$ mixes the full $d$-dimensional vector rather than only a local block, TurboQuant empirically promotes isotropy across the entire head space, which is precisely the condition under which independent scalar quantisation is least brittle. The associated score-error analysis yields the standard rank-preservation condition: if the attention-score gap is comfortably larger than the noise scale $\sigma_q \sqrt{d_k}\,\|q\|$, then the ordering of the most important tokens is preserved with high probability. In practice, this makes TurboQuant an unusually clean baseline against which to compare any structured alternative [1].

The weaknesses are equally clear. First, storage: once $R$ is materialised densely, it requires $d^2$ stored entries per head, so at $d = 128$ each head carries 16,384 floats, totalling tens of megabytes across many layers and heads; the structured alternatives in Section 4.2 require only 256 entries per head. Secondly, compute: applying or inverting a dense rotation costs $O(d^2)$ FMAs per vector, which dominates the per-step rotation overhead when fused-attention paths are unavailable. Thirdly, boundary error: TurboQuant's incremental compression starts from the first prefill token, so quantisation error introduced early in prefill is then re-used by every later attention step instead of being confined to newly appended tokens. Section 4.3 returns to that prefill-to-decode boundary as the specific weakness addressed by deferred prefill.

---

## 3. The Three-Axis Composition (Recap)

Section 2 established three independent compression axes addressing three independent memory consumers: weight quantisation acts on the $W$ matrices, KV-cache compression acts on the stored $(k_j, v_j)$ vectors, and expert offloading acts on the residency policy for expert shards. The axes are separable in code but coupled in dynamics, because quantised execution writes slightly different KV states. The remainder of the paper therefore presents the net-new contribution: the ordering constraint on their composition, followed by the structured rotation, deferred-prefill, fused-decode, empirical-quality, and open-issues analyses of Section 4.

---

## 4. Theory — RotaryQuant (Net-New Contribution)

This section develops the mechanisms that replace the three TurboQuant weaknesses identified in Section 2.4 [1]. Dense per-head rotation incurs $O(d^2)$ storage; dense application and inversion incur $O(d^2)$ work and naive decode scales as $O(T \times d^2)$; incremental compression from token 0 allows early prefill error to propagate through later prompt positions. RotaryQuant replaces these with an explicit ordering rule for the three axes (§4.1), RotaryQuant's structured WHT-plus-SO(4) rotation (§4.2), deferred prefill (§4.3), and a fused decode path that amortises the inverse (§4.4). Section 5 reports the empirical evidence; the present section focuses on the mechanisms and their cost analysis.

### 4.1 Operational non-commutativity of the compression pipeline

The three axes of Section 2 are independent in scope but not commutative in operation. This paper uses the canonical order weight quantisation $\rightarrow$ KV compression $\rightarrow$ expert offload. Weight quantisation first defines the tensors from which expert shards are serialised, so the offloaded representation on disk is already a function of $Q_W(W)$ rather than of the original FP16 checkpoint. KV compression is then applied to the $(k_j, v_j)$ stream produced by those quantised layers, so the cache state depends on the fact that the FFN and attention blocks were executed with quantised weights before the KV vectors were written.

Let $Q_W$ denote weight quantisation, $C_{KV}$ denote KV compression, and $O_E$ denote expert offload. These are state-modifying operators over a runtime state comprising in-memory weights, offloaded expert shards, residency metadata, and the accumulated KV stream. In general,

$$
O_E \circ C_{KV} \circ Q_W \neq C_{KV} \circ O_E \circ Q_W,
$$

because $C_{KV}$ writes into a stream whose contents depend on the execution path, while $O_E$ constrains that path by deciding which quantised expert shards are resident and which must be fetched. The reverse ordering is therefore operationally distinct even when the nominal model architecture is unchanged.

This matters for benchmarking as well as for theory. The Gemma-4 ablation in Section 5.5 shows that enabling expert offload on a model that otherwise fits in RAM shifts decode throughput from about 110 tok/s to about 1 tok/s, roughly a 100-fold change. The ordering of axes must therefore be reported as part of the experimental configuration rather than treated as an implicit detail.

### 4.2 RotaryQuant: Walsh-Hadamard Pre-Mix + Isoclinic SO(4) Rotation + Structured Inverse

The central mechanism replacing TurboQuant's dense rotation is RotaryQuant's structured factorisation, which adopts the IsoQuant family of structured rotations [6] and selects the WHT + SO(4) variant after the v1 / v2 implementation failures documented below. Instead of storing one dense orthogonal matrix $R \in \mathbb{R}^{d \times d}$, we decompose the rotation as

$$
R = R_{\text{block}} \circ H_d,
$$

where $H_d$ is the Walsh-Hadamard pre-mix and $R_{\text{block}}$ is a block-diagonal stack of $d/4$ independent SO(4) rotations. The first stage is the rough mix: $H_d$ spreads energy across all $d$ coordinates and addresses the global anisotropy for which Section 2.2 invoked the Hanson-Wright argument [27]. The second stage is the fine mix: within each 4-coordinate group, an isoclinic SO(4) rotation refines the local geometry before scalar quantisation. Each block is parameterised by two unit quaternions, $q_L$ and $q_R$, so the action spans the full SO(4) rather than the single-quaternion subgroup.

The storage advantage follows directly. A head of width $d$ contains $d/4$ SO(4) blocks; each block stores two quaternions; each quaternion has four components. The explicit SO(4) parameter count is therefore $(d/4) \times 2 \times 4 = d/2$ scalars, which is 64 at $d = 128$. Counting the key/value metadata carried by the current implementation yields 256 stored entries per head, versus TurboQuant's $d^2 = 16{,}384$. The result is a 64-fold reduction in stored rotation parameters, while the WHT contributes no dense matrix of its own.

The compute advantage is equally structural. Applying $H_d$ via the fast WHT costs $O(d \log d)$ FMAs per vector; applying $R_{\text{block}}$ costs $O(d)$ because it decomposes into $d/4$ independent 4D matvecs. At $d = 128$, the total is 1,408 FMAs (896 for the WHT plus 512 for the SO(4) blocks; see Appendix A.3 for the derivation) rather than 16,384 for a dense rotation, so the asymptotic change from $O(d^2)$ to $O(d \log d)$ is also a large constant-factor reduction. Because the normalised Hadamard matrix is symmetric, $H_d^\top = H_d$, and because each SO(4) block inverts by quaternion conjugation, the inverse $R^\top = H_d^\top \circ R_{\text{block}}^\top$ is structured as well and inherits the same cost class.

This reduced cost does not abandon the isotropy objective. The Hadamard pre-mix preserves the expectation-level variance-spreading property sought by dense rotation, while the SO(4) refinement reduces residual anisotropy inside each 4-coordinate group without undoing the global mixing. The implementation history is instructive here: IsoQuant v1 used a single-quaternion sandwich and scored 0/5 correctness on the bench; IsoQuant v2 used block-only SO(4) without global mixing and scored 1/5; v3 added the WHT pre-mix, passed the bench, and is the configuration branded RotaryQuant in this paper. The distinction from TurboQuant is therefore exact. TurboQuant uses one dense $R$; RotaryQuant replaces it with a rough global mix, a two-handed fine mix, and a structured inverse. The net-new contribution of this paper is to integrate that rotation into a full inference stack with weight quantisation, expert offloading, deferred prefill, and fused decode on consumer hardware.

| Metric | TurboQuant (dense rotation) | RotaryQuant (WHT + SO(4)) |
|---|---:|---:|
| Theoretical FMAs at $d=128$ | 16,384 | **1,408** (≈11× reduction; see Appendix A.3) |
| Stored rotation parameters | $d^2 = 16{,}384$ | $d/2 = 64$ raw + metadata = **256** (64× reduction) |
| Asymptotic rotation cost | $O(d^2)$ | $O(d \log d)$ |
| Approximate isotropy | yes (dense $R$) | yes (Hanson-Wright on Hadamard-mixed Gaussians; §A.1) |
| Application path | one dense matvec | fast Walsh-Hadamard transform + $d/4$ blockwise SO(4) matvecs |

### 4.3 Deferred prefill — eliminating compounding error at the prefill→decode boundary

The third weakness identified in Section 2.4 is boundary error rather than rotation cost. If TurboQuant compresses incrementally from token 0, then error introduced at position $j$ is already present when attention is computed for positions $j+1, j+2, \ldots, T$. Prefill therefore re-reads its own compression noise throughout the prompt, precisely when the sequence is longest and numerically most sensitive.

Deferred prefill changes the protocol. During prefill, the cache retains every $(k_j, v_j)$ pair in FP16 inside a transient buffer and performs no KV compression. Attention over the prompt is therefore computed on exact pre-compression states. Only at the prefill-to-decode boundary, just before the first generated token, is the entire buffered sequence bulk-compressed in one pass. After that transition, decode proceeds incrementally in the usual way, with each new token compressed on insertion.

The memory cost is explicit and bounded. The transient buffer occupies approximately $L \times H_{kv} \times d \times T_{\text{prefill}} \times 2 \times 2$ bytes. At $L = 28$, $H_{kv} = 8$, $d = 128$, and $T_{\text{prefill}} = 2048$, this is roughly 230 MB, which is acceptable within the 16-32 GB unified-memory budgets targeted here. The benefit is that compression error does not compound during prefill at all; it appears only on decode-side appends, where the compounding distance is bounded by the generated continuation. Deferred prefill is therefore a protocol-level correction to TurboQuant's boundary weakness, not a change in the rotation mathematics.

### 4.4 Fused decode pipeline — eliminating per-step inverse rotation

The second weakness of Section 2.4 concerned decode-time reconstruction cost. A naive implementation applies the inverse rotation $R^\top$ to every cached key and value at every decode step, giving $O(T \times d^2)$ rotation work per step under a dense formulation. The fused decode pipeline avoids that pattern by performing attention directly in rotated space: the query is rotated forward by $R$, the score computation is evaluated against rotated cached keys, the weighted sum is accumulated against rotated cached values, and only the final aggregated output is mapped back once through $R^\top$.

This changes the rotation side of decode from repeated inverse application over $T$ cached positions to one inverse on the post-attention result, reducing the dense comparison from $O(T \times d^2)$ to $O(d^2)$ and preserving RotaryQuant's structured $O(d \log d)$ benefit. On Apple Silicon the production path is the fused Metal implementation. Section 5.9 reports `fused_metal_success_rate = 1.0` across 3,612 observed decode attempts. The same instrumentation reports `packed_cache_hit_rate = 0.0`, but this is by design: the packed 3-bit cache is invalidated after each cache write and rebuilt lazily on the next fused read. Incremental append into the packed format would amortise that cost further, but it is out of scope here.

### 4.5 Quality preservation — empirical isotropy and rank-preservation under MoE routing

The theoretical motivation for structured rotation matters only if the rank-preservation intuition of Section 2.4 survives real MoE traces. Section 5.2 shows that it does. At 2,048 context, RotaryQuant preserves perplexity to within negligible absolute deltas on the architectures where the KV pathway is genuinely exercised: Qwen-3 shows a PPL increase of +0.0009 relative to default, and Nemotron-H shows +0.0012, while both improve on the corresponding TurboQuant deltas. Structured rotation therefore appears to preserve the same practical isotropy condition as dense rotation at fractional storage cost.

One caveat must be stated explicitly. Gemma-4 reports a delta of +0.0000, but that figure is not strong evidence for fidelity because the layer-aware pathway applies RotaryQuant to only 5 of the model's 30 attention layers; the other 25 are sliding-window layers whose KV writes are largely unaffected. The Gemma row is therefore a near no-op measurement. The meaningful quality evidence comes from Qwen-3 and Nemotron-H.

### 4.6 Open issues

Several issues remain open and should be stated plainly. First, the `llama.cpp` fused kernel has not yet reached numerical equivalence with the composed FP32 reference path. On a single-vector probe the observed divergence is $O(1)$ rather than rounding-scale, with RMSE approximately 0.95 and top-10 overlap 7/10. A 30-step greedy decoding check still agrees token-for-token, but that is only a necessary sanity check; it is not sufficient evidence for production correctness, because short-horizon top-1 agreement can survive materially different lower-ranked logits. Longer-horizon validation and closer logit-space analysis are still required before the fused kernel can be treated as fully closed.

Secondly, the packed 3-bit cache is currently rebuilt on every decode step under the existing invalidation policy. This follows directly from the current design, which invalidates the packed representation after prefill finalisation, shape reset, and incremental append, then reconstructs it lazily for the next fused read. The behaviour is intentional and functionally correct, but it leaves throughput on the table. An incremental-append packing scheme would amortise this cost more effectively and is an obvious future optimisation target.

Thirdly, the current RotaryQuant instrumentation singleton is not multi-process safe. That limitation is acceptable for the single-process benchmark harness used in this paper, where the counters are intended only to characterise one run at a time, but it would require redesign before deployment in a long-running multi-tenant server such as `mlx-lm.server`. Finally, the per-step counters still do not attribute byte-level packed-read and packed-write volumes. Those measurements are not required to establish correctness or the main throughput story, but they would materially improve future proportional comparisons against alternative KV compressors.

---

## 5. Results

Sections 1-4 set out the problem, surveyed prior compression work, and presented the RotaryQuant mechanisms theoretically. This section reports what happens when those mechanisms are run. The subsections move from the broadest evidence — pathway proofs that the full stack runs at all (§5.1) — through quality (KV cache fidelity at fixed context depth, §5.2), throughput (decode profiling, §5.3), repeatability (inter-run variance, §5.4), causal attribution (the three-axis ablation, §5.5), failure characterisation (§5.6), comparison against the upstream stock baseline (§5.7), concurrent-request scaling (§5.8), the new RotaryQuant per-step instrumentation (§5.9), prior validations (§5.10), and the project's go / no-go disposition (§5.11). Each subsection ends with an explicit better / worse / no-change call against the relevant baseline.

### 5.1 Pathway proofs (does the full stack run end-to-end?)

The pathway-proof view is a smoke test, not a benchmark leaderboard: can the combined stack of mixed-precision weights, expert offload, and compressed KV state run end-to-end on the target pathways? The numbers in the table below are reproduced from the project's prior end-to-end runs [18].

| Model | Quality | tok/s | Peak Memory | Budget | 2 h Soak | Status |
|---|---|---|---|---|---|---|
| Gemma 4-26B (layer-aware) | 12/12 | 12.85 ████░░░░░░ | 5.4 GB | 16 GB | P99/P50 1.29, RSS 1.18× | Proven |
| Nemotron-H 120B (mixed) | 12/12 | 14.85 █████░░░░░ | 17.2 GB | 32 GB | P99/P50 1.14, RSS 0.994× | Proven |
| Nemotron-H 30B (mixed) | 10/12 | 35.5 ██████████ | 4.3 GB | 32 GB | P99/P50 1.16, RSS 1.03× | Blocked on quality |
| Qwen3-30B-A3B (4-bit) | 8/12 | 9.87 ███░░░░░░░ | 9.5 GB | 16 GB | — | Blocked on quality |

Here, `Proven` means the full stack met the quality, budget, and soak gates on that pathway; `Blocked on quality` means the system ran within budget but did not clear the prompt-quality gate.

The call is therefore narrow but important: better in the sense that the full composition is proven on Gemma-4 and Nemotron-H 120B, while Nemotron-H 30B and Qwen-3 remain blocked on quality rather than on systems viability [18].

### 5.2 KV cache fidelity (PPL at fixed context depth)

Fixed-depth PPL is the direct quality check for the KV path. The corrected comparison is in absolute PPL deltas at 2048 context, not ratios. Ordered by relevance, Qwen-3 and Nemotron-H are the meaningful rows because RotaryQuant compresses the full attention pathway there; Gemma-4 is mainly a caveat [18].

| Model | Backend | PPL @ 512 | PPL @ 2048 | Δ PPL @ 2048 |
|---|---|---:|---:|---:|
| Qwen3-30B-A3B | default | 1.3829 | 1.0844 | — |
|  | TurboQuant | 1.4497 | 1.1249 | +0.0405 |
|  | RotaryQuant | 1.3872 | 1.0853 | **+0.0009** |
| Gemma 4-26B-A4B | default | 3.2029 | 1.3483 | — |
|  | TurboQuant | 3.5180 | 1.4105 | +0.0622 |
|  | RotaryQuant | 3.2029 | 1.3483 | **+0.0000** |
| Nemotron-H 120B | default | 1.3911 | 1.0866 | — |
|  | TurboQuant | 1.4086 | 1.0905 | +0.0039 |
|  | RotaryQuant | 1.3961 | 1.0878 | **+0.0012** |

At 2048 context, RotaryQuant moves Qwen-3 by only +0.0009 PPL versus +0.0405 for TurboQuant, and Nemotron-H by +0.0012 versus +0.0039. Those are the rows that actually test the compressor, and both show negligible quality movement at the measured depth. Gemma-4's +0.0000 should not be over-interpreted: 25 of its 30 layers use sliding-window attention with an unmodified `RotatingKVCache`, so the result is effectively a near-no-op because only 5 of 30 layers engage the compressor [18].

The source does not report repeated-trial significance testing, so the conservative reading is practical rather than inferential: on the meaningful rows, RotaryQuant shows no practically meaningful quality loss relative to default and no substantiated difference versus TurboQuant, while achieving that parity at fractional storage cost. The call is therefore no change in quality on Qwen-3 and Nemotron-H, with Gemma-4 treated as a caveated no-op rather than a win [18].

### 5.3 End-to-end decode profiling

Per-component decode time attribution via `mx.eval()` timing fences (warm run, 64 decode tokens) is shown below [18].

| Component | Gemma4 (ms/tok) | Qwen3 (ms/tok) | Nemotron-H 120B (ms/tok) |
|---|---|---|---|
| kv_attention | 65.3 █████░░░░░ (51%) | 58.1 █████░░░░░ (54%) | 6.6 █░░░░░░░░░ (14%) |
| routed_expert | 47.5 ████░░░░░░ (37%) | 48.3 █████░░░░░ (45%) | 28.7 ██████░░░░ (60%) |
| dense_ffn | 11.5 █░░░░░░░░░ (9%) | 0.0 ░░░░░░░░░░ (0%) | 0.0 ░░░░░░░░░░ (0%) |
| other (Mamba/SSM) | 0.0 ░░░░░░░░░░ (0%) | 0.0 ░░░░░░░░░░ (0%) | 11.3 ██░░░░░░░░ (24%) |
| uninstrumented | 3.9 ░░░░░░░░░░ (3%) | 1.3 ░░░░░░░░░░ (1%) | 1.5 ░░░░░░░░░░ (3%) |

KV attention is 51-54% of decode time on standard MoE architectures (Gemma 4, Qwen 3), confirming it as the single largest cost centre and justifying KV-compression work. On hybrid Mamba+MoE (Nemotron-H), attention drops to 14% and expert routing dominates at 60%.

The 14.85 tok/s Nemotron-H result is therefore driven mainly by expert offload and mixed-precision weights, with KV compression helping only on a minority cost centre. This profile also explains why further KV micro-optimisation cannot plausibly deliver headline-scale gains on Nemotron-H unless expert I/O is reduced first. The call is clear: for the 120B pathway, expert offload is the dominant lever and KV compression is a modest contributor [18].

### 5.4 Variance across independent runs

Inter-run variance was measured on `gemma4-layer-aware`, profile A, with three independent runs per configuration, seed 42, and a full model reload between runs. Under `--expert-offload`, the default KV path reaches $1.048 \pm 0.056$ tok/s with a mean peak of 2,445 MB, while RotaryQuant reaches $1.054 \pm 0.003$ tok/s with a mean peak of 2,804 MB. The reported coefficients of variation are 5.4% and 0.3% respectively, so both settings are stable enough that the single-run headline numbers lie inside a small noise floor [18].

The important point is what does not change: the means differ by only 0.006 tok/s in a regime where both settings have already collapsed to roughly 1 tok/s because expert offload dominates runtime. What does change is memory, with RotaryQuant adding about 359 MB of peak usage. Because the model can fit without offload on the measurement host, that overhead appears without any compensating decode gain. The low variance matters because it rules out the easy objection that the memory delta is just run-to-run noise. The call is therefore no change in throughput and worse peak memory for RotaryQuant in this small-model-with-offload configuration [18].

### 5.5 Ablation across the three axes

The ablation on `gemma4-layer-aware` holds the mixed-precision weight checkpoint fixed and varies only the two runtime axes, expert offload and KV mode. It is therefore a 16 GB-constrained scenario, not a general recommendation for Gemma-4 [18].

| Expert offload | KV mode | tok/s | Peak MB |
|---|---|---:|---:|
| Yes | RotaryQuant | 1.05 | 2,804 |
| Yes | Default | 1.01 | 2,445 |
| No | RotaryQuant | 20.6 | 10,748 |
| No | Default | **109.8** | **10,649** |

Two conclusions follow. First, `--expert-offload` is extraordinarily expensive on a model that already fits in RAM: relative to the no-offload default path, throughput falls from 109.8 tok/s to about 1 tok/s, roughly a two-orders-of-magnitude penalty, while peak memory drops from about 10.6 GB to 2.4-2.8 GB. Secondly, RotaryQuant on the no-offload path is a net loss in this Gemma-4 measurement: throughput falls from 109.8 tok/s to 20.6 tok/s, a 5.3× slowdown, while peak memory is slightly higher at 10,748 MB versus 10,649 MB [18].

Within the offload rows, RotaryQuant changes throughput only from 1.01 tok/s to 1.05 tok/s, effectively no change, but raises peak memory by about 359 MB. The call is therefore mixed but explicit: RotaryQuant is worse on the no-offload Gemma-4 path, no change in throughput and worse in memory on the offload path, and better only when the memory budget genuinely forces the full stack [18].

### 5.6 Failure-mode characterisation

The failure-boundary experiment asks how the system fails when the memory cap drops below the working set. On Gemma-4 with `--expert-offload --kv-cache-type isoquant`, caps of 8,192 MB, 6,144 MB, 5,120 MB, and 4,096 MB all succeed with peak usage near 3,641 MB, while 3,072 MB fails with a cap-exceeded OOM. The boundary is therefore between 3,072 MB and 4,096 MB for this configuration [18].

The failure mode is graceful rather than catastrophic. The run reports `memory.fits_cap = False` in the JSON artefact and exits non-zero; no Metal crash or silent corruption is reported. The earlier `--max-kv-size` sweep was removed because the harness does not expose that flag. The call is therefore better than an uncontrolled OOM and useful for capacity planning [18].

### 5.7 Stock-baseline comparison

The stock-baseline comparison falsified the earlier prediction. Upstream `mlx-lm 0.31.2` from `pip` does load the `gemma4-layer-aware` checkpoint without OOM, so the issue is not capacity but correctness. On the test prompt, the stock package emits degenerate repeated text of the form "This is a of a of a of a of ...", while warning that it is "loading checkpoint of type gemma4 into model of type ``". The source diagnosis is incomplete `gemma4` model-type registration upstream [18].

No paired throughput figure is reported, so this remains a correctness comparison rather than a speed claim. The earlier "stock OOMs" story should be discarded. That matters because a load-without-decoding baseline could otherwise be misread as functional parity, when the real issue is that upstream does not decode this checkpoint family correctly. The warning text points to checkpoint interpretation rather than prompt-specific degeneration. The call is therefore better for the fork in correctness: stock loads the checkpoint but decodes nonsense, whereas the fork produces usable text [18].

### 5.8 Concurrent-request scaling

Concurrent load was tested against `mlx-lm.server` with no expert offload and RotaryQuant KV on `gemma4-layer-aware`. Aggregate throughput rises from 15.65 tok/s at one client to 21.73 tok/s at two clients, only 69% scaling efficiency, and then plateaus at 21.21 tok/s at four clients, where efficiency has already fallen to 34%. Mean latency rises from 6.39 s to 7.71 s and then 11.75 s across those same points [18].

The critical result is at eight clients: 4 of 8 HTTP 200 responses returned empty bodies, so the source marks the outcome unreliable. The client now treats zero-token completions as failures, but the diagnosis stands that `mlx-lm.server` uses Python's `ThreadingHTTPServer` and is not built for robust concurrent inference. This is a serving-layer failure, not just a disappointing scaling curve, because the API can report success while returning no completion. The plateau by four clients is therefore diagnostic rather than incidental: the server is already serialising generation before the explicit failures appear at eight. In other words, throughput saturates long before client count does. The call is therefore worse for multi-tenant use: two clients give only a modest gain, four clients plateau, and eight clients fail silently at the HTTP layer [18].

### 5.9 RotaryQuant per-step instrumentation

The first explicit `kv_cache` JSON block resolves a basic observability problem: earlier artefacts exposed MoE expert-cache hits and misses, not KV-cache behaviour. On `gemma4-layer-aware`, profile A, with RotaryQuant plus expert offload, the new counters show `fused_metal_success_rate = 1.0` across 3,612 fused decode attempts, 7,280 `compress_calls`, and 28 `finalize_calls`, one per attention layer at the prefill-to-decode boundary [18].

The zeroes matter too. `decompress_calls = 0`, `read_keys_calls = 0`, and `read_values_calls = 0`, confirming that the fused Metal path is not materialising unpacked KV tensors. `fallback_invocations = 0`, so the run never left the RotaryQuant path. `packed_cache_hit_rate = 0.0` is by design because each write invalidates the packed view and forces a rebuild, which explains why packed-cache reuse is absent even though compression itself is active. The `compress_calls` count is also informative rather than incidental: it is consistent with roughly two compress operations per decode token plus the 28 layer-finalise batches at the prefill-to-decode transition. Taken together, the counters describe a fully fused steady-state decode path rather than a hidden materialise-and-read implementation. They also resolve the earlier ambiguity between the MoE expert cache and the KV cache. The call is therefore better than the prior state of no KV-specific instrumentation, and it directly disproves the earlier mistaken claim that zero `decode_hits` meant RotaryQuant was not engaging [18].

### 5.10 Prior validations

This subsection preserves earlier validation work as context rather than making a new net performance claim. The quantitative result retained in the current source is the kurtosis split used to justify the mixed-precision weight schedule: shared experts have excess kurtosis 10.10, routed experts 0.41, a 24.6× gap. That supports keeping shared experts at the higher-precision tier while quantising routed experts more aggressively [18].

Earlier TurboQuant baselines, the IsoQuant v1-to-v2-to-RotaryQuant evolution, and the codebook precomputation pipeline are also prior validations on which the present work depends; their numerical outputs are documented in the original implementation artefacts rather than restated here. The call is therefore no new performance claim: these are precondition validations, with only the kurtosis evidence quantified in this section [18].

### 5.11 Go / no-go decisions

Table 5.11 gives the go/no-go disposition as of April 2026 [18].

| Component | Decision | Rationale |
|---|---|---|
| RotaryQuant (WHT + SO(4)) | **Go** | Quality parity with default ($\Delta$ PPL $\leq 0.001$ @ 2048 on Qwen3/Nemotron), 64× fewer stored rotation parameters |
| Fused Metal pipeline (MLX) | **Go** | Verified by 9 correctness tests, eliminated materialisation |
| RotaryQuant (llama.cpp) | **Active — open correctness question** | Fused `kernel_turbo_wht_so4` recovers near-`turbo3` throughput but has $O(1)$ numerical divergence vs composed F32 path (RMSE 0.95, 7/10 top-10 overlap) — see §4.6 (open issue 1). |
| Deferred prefill | **Go** | Eliminates compounding error; transient FP16 buffer is small (≈230 MB at $L=28$, $H_{kv}=8$, $d=128$, $T_{\text{prefill}}=2048$ — see §4.3) and manageable within 16-32 GB envelopes |
| Gemma4 pathway | **Go** | All gates pass at 12.85 tok/s within 16 GB budget |
| Nemotron-120B pathway | **Go** | All gates pass at 14.85 tok/s within 32 GB budget |
| Qwen3 pathway | **Blocked** | Quality issues (8/12) — model/checkpoint limitation |
| AttnRes predictor | **No-go** | 10.6–11.2% throughput regression with no hit-rate improvement |
| Task-aware pinning | **No-go** | 0% hit-rate improvement over baseline LRU |
| QES | **Planned** | Background evolution strategies for gate-weight optimisation |

Taken together, the table says the core MLX stack is go, the Gemma-4 and Nemotron-H pathways are go, Qwen-3 remains blocked on quality, and AttnRes plus task-aware pinning are no-go. The call is therefore better/go for the core constrained-memory path, worse/no-go for the two negative-result scheduling ideas, and still open for the llama.cpp correctness track and Qwen-3 quality recovery [18].

---

## 6. Concluding Remarks

### 6.1 Structured rotation matches dense rotation at fractional cost

The clearest net-new result is that structured rotation can match the quality of dense rotation at a fraction of its representation and application cost. Section 4.2 replaces TurboQuant's dense per-head matrix with a WHT pre-mix plus blockwise isoclinic SO(4), reducing stored rotation parameters from 16,384 to 256 entries per head at $d = 128$ and changing the per-vector application cost from $O(d^2)$ to $O(d \log d)$ [6][1]. Section 5.2 shows that this simplification is not paid for in measured quality on the pathways where the KV compressor actually engages. At 2,048 context, Qwen-3 moves by only +0.0009 PPL under RotaryQuant versus +0.0405 under TurboQuant, and Nemotron-H by +0.0012 versus +0.0039 [18]. The Gemma-4 row should be read as a caveat rather than a third confirmation, because only 5 of its 30 layers are materially exposed to RotaryQuant and the +0.0000 delta is therefore close to a no-op measurement [18]. The practical implication is parity at materially lower cost: on the architectures where the structured path is genuinely exercised, RotaryQuant matches the dense baseline within the practical noise band while reducing stored rotation parameters by 64× and per-vector application cost from $O(d^2)$ to $O(d \log d)$.

### 6.2 Three-axis composition works end-to-end on consumer hardware

The second net-new result is that the three-axis composition works end-to-end on consumer Apple Silicon as a system, not merely as a set of individually plausible mechanisms. Section 5.1 reports that Gemma-4-26B-A4B passes the 12-prompt pathway harness at 12.85 tok/s with 5.4 GB peak resident memory inside the simulated 16 GB envelope, while Nemotron-H 120B passes 12/12 at 14.85 tok/s with 17.2 GB peak resident memory inside the simulated 32 GB envelope and remains stable through a two-hour soak [18]. These are pathway proofs, and that is precisely why they matter: mixed-precision weights, KV compression, and expert offload are all active in the same serving path rather than being validated one axis at a time. Qwen-3 remains at 8/12 and therefore cannot yet be counted as a proven pathway, but the paper treats that block as a model or checkpoint quality issue rather than a stack malfunction [18]. The honest conclusion is therefore that the stack has crossed from component plausibility into end-to-end operational proof on the Gemma-4 and Nemotron-H pathways.

### 6.3 Operational ordering and the conditional value of expert offload

The third net-new result is operational rather than purely algorithmic: the value of the stack depends on ordering and on whether memory pressure is real. Section 4.1 argues that weight quantisation, KV compression, and expert offload do not commute operationally; the order weight quantisation $\rightarrow$ KV compression $\rightarrow$ expert offload must therefore be treated as part of the experimental configuration rather than as an incidental implementation detail. Section 5.5 shows why. On Gemma-4 in the no-offload default path, throughput is 109.8 tok/s; enabling `--expert-offload` on the same default KV path collapses decode to 1.01 tok/s, roughly a hundred-fold penalty, while peak memory falls from 10,649 MB to 2,445 MB [18]. RotaryQuant does not rescue the no-offload case: throughput falls from 109.8 tok/s to 20.6 tok/s, a 5.3× slowdown, for negligible memory relief; on the offload path it is essentially throughput-neutral at 1.05 versus 1.01 tok/s, but adds about 359 MB of peak memory [18]. The practical takeaway is therefore conditional. The combined stack is better only when the target budget genuinely forces offload; when the model already fits comfortably in RAM, the same machinery becomes an avoidable systems cost rather than a win.

### 6.4 Per-step instrumentation closes a long-standing observability gap

The fourth net-new result is methodological: the `kv_cache` JSON block closes an observability gap that had previously obscured what the KV path was actually doing. In Section 5.9, the first Gemma-4 RotaryQuant measurement with expert offload records `fused_metal_success_rate = 1.0` over 3,612 fused decode attempts, alongside 7,280 `compress_calls` and 28 `finalize_calls`, one per attention layer at the prefill-to-decode boundary [18]. Equally important are the zeroes. `decompress_calls = 0`, `read_keys_calls = 0`, and `read_values_calls = 0` show that the fused decode path bypasses materialisation as Section 4.4 claims, while `packed_cache_hit_rate = 0.0` is not evidence of inactivity but a by-design consequence of invalidating the packed view after each write, as discussed in Section 4.6 [18]. This instrumentation also resolves an earlier mistaken reading: zero `decode_hits` had been taken to mean that RotaryQuant was inactive, when in fact that field belonged to the MoE expert cache rather than the KV cache [18]. The methodological lesson is that future work in this area needs counters that distinguish KV-cache activity from MoE expert-residency activity, because collapsing those two subsystems into one hit-rate narrative is enough to produce materially wrong conclusions.

### 6.5 Next steps

Next steps should therefore be deliberately conservative. The first is to rerun the proven pathways on native 16 GB hardware, because the current Gemma-4 result is still a `--memory-limit-mb` simulation on a larger host rather than a same-class physical machine [18]. The second is to harden the current counters into multi-process-safe instrumentation suitable for longer-running serving rather than single-process benchmarking. The third is to extend the evaluation to additional model families so that the present conclusions can be separated into architecture-specific and stack-general effects. Beyond the obvious packed-cache amortisation opportunity already noted in Section 4.6, these are validation and hardening tasks rather than a change of direction.

### 6.6 Recommendation

The practical recommendation is therefore selective rather than universal. RotaryQuant should be used when the target memory budget would otherwise rule the model out, because that is the regime in which the three-axis composition earns its keep: the validated reference points are Gemma-4-26B-A4B inside a 16 GB envelope and Nemotron-H 120B inside a 32 GB envelope [18]. It should not be treated as a default acceleration path for models that already fit comfortably in RAM; on Gemma-4, the offload ablation shows that enabling `--expert-offload` can reduce decode throughput from 109.8 tok/s to 1.01 tok/s, and RotaryQuant on the no-offload path itself is a net loss at 20.6 tok/s [18]. Operationally, deployment assumes that the Lloyd-Max codebook directory has been precomputed in advance; the existing TurboQuant codebooks for $d = 128$ and 1/2/3/4-bit settings are reused, while `TURBOQUANT_BITS` and `TURBOQUANT_SKIP_LAYERS` control bit-width and layer skipping, `--kv-cache-type isoquant` selects the cache, and `--expert-offload` enables the residency policy. The present caveats should remain explicit: the `llama.cpp` fused-kernel correctness question is still open, concurrent serving beyond two clients on the current `mlx-lm.server` is unreliable and plateaus by four, and the evidence reported here remains single-process benchmarking rather than multi-tenant production validation. Acronym definitions are given upfront in the Acronyms section.

---

## References

This paper combines three layers of contribution that should not be conflated: (1) prior quantisation/compression literature, (2) implementation frameworks and upstream repositories, and (3) this repository's MLX/Nemotron integration work. The validated 120B mixed-checkpoint path described here is implemented in [TurboQuantNemo](https://github.com/2096955/TurboQuantNemo) [18], built on [MLX](https://github.com/ml-explore/mlx) [15] and an [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) [16] fork, with a parallel [llama.cpp](https://github.com/ggml-org/llama.cpp) [17] validation track.

[1] Zandieh, A., Daliri, M., Han, I., and co-authors.
*TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate.*
International Conference on Learning Representations (ICLR) 2026.
arXiv: [2504.19874](https://arxiv.org/abs/2504.19874).

[2] Zandieh, A., Daliri, M., Han, I.
*QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead.*
AAAI Conference on Artificial Intelligence (AAAI) 2025.
arXiv: [2406.03482](https://arxiv.org/abs/2406.03482).

[3] Liu, Z., Yuan, J., and co-authors.
*KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.*
International Conference on Machine Learning (ICML) 2024.
arXiv: [2402.02750](https://arxiv.org/abs/2402.02750).
Code: https://github.com/jy-yuan/KIVI.

[4] Zhang, H., Liu, J., and co-authors.
*KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization.*
Conference on Neural Information Processing Systems (NeurIPS) 2024.
arXiv: [2401.18079](https://arxiv.org/abs/2401.18079).

[5] Kang, H., Li, Y., and co-authors.
*GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless LLM Inference.*
arXiv preprint, 2024.
arXiv: [2403.05527](https://arxiv.org/abs/2403.05527).

[6] IsoQuant authors.
*IsoQuant: Structured-Rotation KV Cache Compression for LLMs.*
arXiv preprint 2603.28430, 2026.
arXiv: [2603.28430](https://arxiv.org/abs/2603.28430).
*Note:* Author names and final canonical title to be confirmed against the upstream record before publication; the arXiv ID itself is the authoritative identifier.

[7] Ashkboos, C., Mohtashami, S., and co-authors.
*QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs.*
Conference on Neural Information Processing Systems (NeurIPS) 2024.
arXiv: [2404.00456](https://arxiv.org/abs/2404.00456).

[8] Chmiel, B., Gale, T., and co-authors.
*QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks.*
International Conference on Machine Learning (ICML) 2024.
arXiv: [2402.04396](https://arxiv.org/abs/2402.04396).

[9] Eliseev, D., Mazur, M.
*Fast Inference of Mixture-of-Experts Language Models with Offloading.*
arXiv preprint, 2023.
arXiv: [2312.17238](https://arxiv.org/abs/2312.17238).

[10] mudler and contributors.
*APEX-Quant: Layer-Aware Expert Quantization for Mixture-of-Experts Models.*
GitHub repository, 2024.
Code: https://github.com/mudler/apex-quant.

[11] Zhang, Z., Yang, Y., and co-authors.
*MxMoE: Mixed-Precision Quantization for MoE with Accuracy and Efficiency.*
International Conference on Machine Learning (ICML) 2025.
arXiv: [2505.05799](https://arxiv.org/abs/2505.05799).

[12] Chitty-Venkata, A., Patel, V., and co-authors.
*MoPEQ: Mixture of Mixed Precision Quantized Experts.*
ICCV 2025, BiVision Workshop.
arXiv: [2509.02512](https://arxiv.org/abs/2509.02512).

[13] Chen, Y., Narayanan, P., and co-authors.
*Dynamic Expert Quantization for Scalable Mixture-of-Experts Inference (DynaExQ).*
arXiv preprint, 2025.
arXiv: [2511.15015](https://arxiv.org/abs/2511.15015).

[14] Moonshot AI / Kimi Team.
*AttnRes: Block-Attention Residual for Cross-Layer Attention Signals.*
arXiv preprint, 2026.
arXiv: [2603.15031](https://arxiv.org/abs/2603.15031).

[15] MLX Team.
*MLX: Numerical Computing Framework for Apple Silicon.*
GitHub repository.
https://github.com/ml-explore/mlx.

[16] MLX Team.
*mlx-lm: Large Language Model Utilities for MLX.*
GitHub repository.
https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm.

[17] Gerganov, G., and contributors.
*llama.cpp: Inference of LLaMA Models in C/C++.*
GitHub repository.
https://github.com/ggml-org/llama.cpp.

[18] TurboQuantNemo Authors.
*TurboQuantNemo: MLX / Nemotron Integration with TurboQuant-Style KV Cache Compression.*
GitHub repository.
https://github.com/2096955/TurboQuantNemo.

[19] tonbistudio.
*turboquant-pytorch: PyTorch Reference Implementation of TurboQuant.*
GitHub repository.
https://github.com/tonbistudio/turboquant-pytorch.

[20] Lloyd, S.
*Least Squares Quantization in PCM.*
IEEE Transactions on Information Theory, 28(2):129–137, 1982.
https://ieeexplore.ieee.org/document/1056489.

[21] Johnson, W. B., Lindenstrauss, J.
*Extensions of Lipschitz Mappings into a Hilbert Space.*
Contemporary Mathematics, 26:189–206, 1984.

[22] Frantar, E., Ashkboos, S., Hoefler, T., and Alistarh, D.
*GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.*
ICLR, 2023.
arXiv: [2210.17323](https://arxiv.org/abs/2210.17323).

[23] Lin, J., Tang, J., Tang, H., Yang, S., Dang, X., and Han, S.
*AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration.*
MLSys, 2024.
arXiv: [2306.00978](https://arxiv.org/abs/2306.00978).

[24] Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., and Han, S.
*SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.*
ICML, 2023.
arXiv: [2211.10438](https://arxiv.org/abs/2211.10438).

[25] Lepikhin, D., and co-authors.
*GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.*
ICLR, 2021.
arXiv: [2006.16668](https://arxiv.org/abs/2006.16668).

[26] Fedus, W., Zoph, B., and Shazeer, N.
*Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.*
JMLR, 2022.
arXiv: [2101.03961](https://arxiv.org/abs/2101.03961).

[27] Rudelson, M., and Vershynin, R.
*Hanson-Wright Inequality and Sub-Gaussian Concentration.*
Electronic Communications in Probability, 2013.
arXiv: [1306.2872](https://arxiv.org/abs/1306.2872).

[28] Hwang, R., Wei, J., Cao, S., Hwang, C., Tang, X., Cao, T., and Yang, M.
*Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference.*
ISCA, 2024.
arXiv: [2308.12066](https://arxiv.org/abs/2308.12066).

[29] Zhu, X., and co-authors.
*A Survey on Model Compression for Large Language Models.*
arXiv preprint, 2024.
arXiv: [2308.07633](https://arxiv.org/abs/2308.07633).

[30] Xia, H., and co-authors.
*Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding.*
arXiv preprint, 2024.
arXiv: [2401.07851](https://arxiv.org/abs/2401.07851).

[31] Gemma Team, Google DeepMind.
*Gemma 4 Technical Report.*
Google DeepMind, 2026.
(Technical report; full citation pending finalisation.) Official announcement: https://developers.googleblog.com/bring-state-of-the-art-agentic-skills-to-the-edge-with-gemma-4/.

[32] Chandiramani, A., and co-authors.
*Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning.*
arXiv preprint, 2026.
arXiv: [2604.12374](https://arxiv.org/abs/2604.12374).

[33] Qwen Team.
*Qwen3 Technical Report.*
arXiv preprint, 2025.
arXiv: [2505.09388](https://arxiv.org/abs/2505.09388).

[34] Li, P., Yang, J., Islam, M. A., and Ren, S.
*Making AI Less "Thirsty": Uncovering and Addressing the Secret Water Footprint of AI Models.*
arXiv preprint, 2023.
arXiv: [2304.03271](https://arxiv.org/abs/2304.03271).

[35] Schneider, I., and Mattia, T.
*Carbon Accounting in the Cloud: A Methodology for Allocating Emissions Across Data Center Users.*
arXiv preprint, 2024.
arXiv: [2406.09645](https://arxiv.org/abs/2406.09645).

[36] Noor, R., and co-authors.
*US Microelectronics Packaging Ecosystem: Challenges and Opportunities.*
arXiv preprint, 2023.
arXiv: [2310.11651](https://arxiv.org/abs/2310.11651).

[37] Luccioni, A. S., Jernite, Y., and Strubell, E.
*Power Hungry Processing: Watts Driving the Cost of AI Deployment?*
FAccT, 2024.
arXiv: [2311.16863](https://arxiv.org/abs/2311.16863).

[38] Gill, S. S., and co-authors.
*Edge AI: A Taxonomy, Systematic Review and Future Directions.*
arXiv preprint, 2024.
arXiv: [2407.04053](https://arxiv.org/abs/2407.04053).

---

## Appendix A: Mathematical Derivations

### A.1 Hanson-Wright concentration bound for Hadamard-mixed quantisation error

Let $u \in S^{d-1}$ and let $H_d$ denote the orthogonal Walsh-Hadamard matrix, scaled so that $H_d^\top H_d = I$. If $u$ is viewed as a random point on the sphere, orthogonality preserves covariance:

$$
\mathbb{E}[(H_d u)(H_d u)^\top] = H_d \, \mathbb{E}[u u^\top] \, H_d^\top = H_d \left(\frac{1}{d} I \right) H_d^\top = \frac{1}{d} I.
$$

Hence each coordinate $(H_d u)_i$ has variance $1/d$. The WHT preserves total energy while dispersing it across coordinates, removing the concentration of mass in a few unstable dimensions. For any fixed symmetric matrix $A$, define the quadratic form

$$
Z = u^\top H_d^\top A H_d u.
$$

Its expectation is

$$
\mathbb{E}[Z] = \frac{1}{d} \mathrm{tr}(A),
$$

because $H_d u$ is isotropic in expectation. A Hanson-Wright-type bound then gives, for $t > 0$,

$$
\Pr\!\left(\left|Z - \frac{1}{d}\mathrm{tr}(A)\right| > t\right)
\leq 2 \exp\!\left[-c \min\!\left(\frac{d^2 t^2}{\|A\|_F^2}, \frac{d t}{\|A\|_{\mathrm{op}}}\right)\right],
$$

for an absolute constant $c$. Large deviations become exponentially unlikely unless $A$ itself is badly conditioned. Quantisation error after Hadamard mixing therefore behaves like many small, weakly correlated perturbations rather than a few dominant outliers. Aggressive scalar quantisation becomes viable because once the coordinates are approximately equal-variance, the Lloyd-Max codebook sees near-Gaussian marginals, so distortion is spread smoothly across dimensions instead of being concentrated into a handful of catastrophic coordinates.

*The rough mix in the big bowl spreads any clumps of pork or ginger across every dumpling, so when the thin wrapping paper goes on, no single dumpling carries enough excess to tear the wrapper. The Hanson-Wright bound is the formal version of "no clump big enough to crush the wrapper".*

### A.2 Isoclinic SO(4) parameterisation by paired unit quaternions

Identify $\mathbb{R}^4$ with the Hamilton quaternion algebra $\mathbb{H}$ by writing

$$
v = (v_0, v_1, v_2, v_3) \longleftrightarrow v_0 + v_1 \mathbf{i} + v_2 \mathbf{j} + v_3 \mathbf{k}.
$$

For unit quaternions $q_L, q_R \in \mathbb{H}$ with $\|q_L\| = \|q_R\| = 1$, define

$$
T_{q_L,q_R}(v) = q_L \, v \, \bar{q}_R.
$$

Left and right multiplication by unit quaternions preserve the Euclidean norm on $\mathbb{H}$, so $T_{q_L,q_R}$ is orthogonal. The classical double-cover statement is that every element of $\mathrm{SO}(4)$ can be represented this way, with the identification $(q_L, q_R)$ and $(-q_L, -q_R)$ inducing the same rotation.

Why are two quaternions required? If one restricts to single-quaternion conjugation,

$$
v \mapsto q \, v \, \bar{q},
$$

the map collapses to an isoclinic subgroup rather than the full six-parameter family, because one set of coefficients ties left and right actions together and the two independent rotation planes of $\mathbb{R}^4$ cannot be controlled separately. Each quaternion has four real coordinates subject to one unit-norm constraint,

$$
a^2 + b^2 + c^2 + d^2 = 1,
$$

so each contributes three degrees of freedom. The pair $(q_L, q_R)$ therefore carries $3 + 3 = 6$ real parameters, exactly matching $\dim \mathrm{SO}(4) = 6$. This is why the RotaryQuant blockwise rotation remains compact without losing expressivity inside each $4 \times 4$ block.

*With one set of hands you can tilt a jug and mix between two neighbouring bowls. With two independent sets of hands you can redistribute filling between any pair of the four bowls in every possible combination — the same six degrees of freedom that the maths reports formally.*

### A.3 FMA accounting for RotaryQuant vs TurboQuant rotation cost

The decode-time cost difference between TurboQuant and RotaryQuant is easiest to express in FMAs. A dense TurboQuant rotation applies a full matrix $R \in \mathbb{R}^{d \times d}$ to one vector:

$$
y = R x.
$$

That matvec requires $d^2$ FMAs. At $d = 128$, this is

$$
128^2 = 16{,}384
$$

FMAs per vector.

RotaryQuant factors the rotation as

$$
R = R_{\text{block}} \circ H_d,
$$

where $H_d$ is the WHT and $R_{\text{block}}$ applies independent $\mathrm{SO}(4)$ transforms on four-dimensional blocks. The fast Walsh-Hadamard transform has $d/2$ butterflies per stage and $\log_2 d$ stages, so its cost is

$$
d \log_2 d.
$$

For $d = 128$, that is

$$
128 \times 7 = 896
$$

FMAs. The block stage contains $d/4$ separate $4 \times 4$ matvecs. Each costs $16$ FMAs, so the block cost is

$$
\frac{d}{4} \times 16 = 4d.
$$

At $d = 128$, this is $512$ FMAs. The total RotaryQuant rotation cost is therefore

$$
896 + 512 = 1{,}408
$$

FMAs per vector, versus 16,384 for TurboQuant. That is about $11.6\times$ lower here, and the asymptotic gap widens because TurboQuant is $O(d^2)$ while RotaryQuant is $O(d \log d)$.

*TurboQuant has the chef weigh every dumpling against every other dumpling; RotaryQuant has them do a quick global stir, then perfect each batch of four with two-handed work. Same evenness, far fewer arm motions per service.*

## Appendix B: Mathematical Formulation of AttnRes (negative result)

### B.1 Standard residual stream

In a vanilla transformer, each layer adds to a residual stream:

$$h_l = h_{l-1} + f_l(h_{l-1})$$

Every layer contributes equally to the residual state.

### B.2 Block attention residuals

AttnRes [14] replaces the additive residual with learned depth-wise attention. Group $L$ layers into $N \approx 8$ blocks:

$$h_l = \sum_{n=0}^{N} \alpha_{n \to l} \cdot B_n$$

$$\alpha_{n \to l} = \frac{\exp(w_l^\top \, \text{RMSNorm}(B_n))}{\sum_{n'} \exp(w_l^\top \, \text{RMSNorm}(B_{n'}))}$$

Here $w_l \in \mathbb{R}^d$ is a single learned pseudo-query per layer. The softmax is over the *depth* dimension — which block to attend to, not which token.

> **Note: The critical causal property.** The $\alpha$ weights are computed *before* the MoE router fires. We know which blocks matter for this token before deciding which chefs to call in. AttnRes is not just a modelling improvement — it is a runtime control signal that collapses multiple systems problems (prefetch, eviction, precision allocation) into one observable.

## Appendix C: Intuitive Companion — The Yum Cha Kitchen

The supervisor's preferred academic structure pushes intuitive analogies out of the main text. This appendix preserves them for readers who find a single running metaphor more accessible than the formal exposition. All analogies use the same kitchen: *a yum cha kitchen preparing 384 dim sum dishes from a tiny service area*. The AI model is the kitchen. The specialist experts are station chefs. Incoming tokens are customer orders. RAM is counter and steamer-basket space. Disk is the back alley where the off-duty chefs wait. The star dish — siu loong bao (soup dumplings) — stands in for the most demanding operation: KV cache compression.

The companion below maps each main-paper section to the corresponding kitchen scene.

### C.1 The kitchen layout (cf. §1.1, §2.2 — attention and KV cache)

The main prep area holds containers of fillings already prepped earlier in the service. Every time a new order comes in, the head chef checks every prepped container — *how was the pork-and-ginger mix from table 4? Is the prawn paste from the first batch of har gow still fresh?* That sweep across every container is what the maths calls *attention*. The full set of prep containers is the *KV cache*. As service progresses the counter fills with bowls. The recurring problem is that the counter is finite.

### C.2 The 384 station chefs (cf. §2.1 — sparse mixture-of-experts)

The kitchen has 384 specialised station chefs, but any one order needs only 8 of them. The shared expert is the kitchen *si fu* — the master chef who inspects, adjusts, and signs off on every dish before it leaves the pass. Because the si fu's hands touch everything, they receive the best equipment and prime counter space (Q8_0 precision). The other 376 specialists are idle most of the time. The 8 active for the current order stay at their stations; the remainder sit out the back playing cards. The floor manager (the router) yells a name and they return to their station within seconds.

### C.3 Three ways to fit (cf. §2 — the three compression axes)

There are three independent ways to fit 384 dishes into a tiny kitchen. *Shrink the recipe cards*: write shorthand the chefs can still follow, like "B3" instead of a full-page recipe (weight quantisation). *Stack the fillings in small tubs in the fridge*: instead of 128 open bowls covering the entire counter, portion each filling into a small labelled tub and stack them in the walk-in. When the chef needs one, they pull the tub and reconstitute (KV cache compression). *Send most chefs outside*: only the 8 active stay at their stations; the other 376 are out the back, hearing the floor manager and returning when called (expert offloading). These three approaches do not conflict because each targets a different space problem: recipe storage, counter space, and how many bodies are inside the kitchen.

### C.4 Wrapping the dumplings (cf. §2.4 — the four-stage KV pipeline)

The KV compression pipeline has four steps applied to every prepped dumpling. *Weigh* each dumpling and note the weight (normalise). *Even out the portions* — if some dumplings are overstuffed and others are nearly empty, a standard wrapper will not fit them all, so the chef redistributes filling until every dumpling is the same size (rotate). *Wrap*: now that they are uniform, fold each one into a standard skin with a stamped code (quantise). *Stack in the steamer*: uniform dumplings stack tightly (bit-pack). When the chef needs one later, they unwrap and reconstitute. The single point on which TurboQuant and RotaryQuant disagree is *how* you do the evening-out in step 2.

### C.5 The thin paper rule (cf. §4.2 — why isotropy makes 3-bit work)

There are 1,000 prepped dumplings and the steamer is tiny. You cannot throw any away. Instead you wrap each one in much thinner paper (3-bit quantisation). The danger is that the thin paper might tear or crush the filling, ruining the flavour. The mathematics says: if you portion the filling evenly first (the rotation), the pressure from the thin paper is spread across every dumpling. No single part gets crushed. You keep all 1,000 dumplings, and they still taste right, because the even portioning prevents the thin wrappers from failing.

### C.6 Two ways to portion (cf. §2.4 vs §4.2 — TurboQuant vs RotaryQuant)

TurboQuant is the thorough but expensive version of portioning. The chef weighs all 128 dumplings against all the others and redistributes filling until every one is perfectly balanced. The result is excellent portion control, but the chef had to handle the whole batch at once. On Apple Silicon, that is why the dense rotation is mathematically sound but operationally expensive.

RotaryQuant uses two stages instead. First, a global rough mix: pour everything into one big bowl and stir broadly (the Walsh-Hadamard transform). Then a two-handed fine mix in batches of four: with two sets of hands per batch, the chef can redistribute between all four bowls in every possible combination, rather than only between neighbouring pairs (the isoclinic SO(4) rotation). The combination is faster and uses smaller utensils than TurboQuant's single all-at-once weighing, while producing the same evenness.

### C.7 The siu loong bao trials (cf. §4.2 — the v1 → v2 → v3 evolution)

The head chef tried three ways to mix the filling for siu loong bao. The filling needs perfect balance of pork, ginger, scallion, and soup gelatin. *v1: One-handed mixing.* Lumpy, uneven, inedible. *v2: Two hands, but isolated batches.* Some batches right, others wrong. *v3: Global rough mix first, then two-handed batches.* Pour everything into one big bowl and roughly stir (WHT), then split into batches of four and use two hands per batch for the fine work (SO(4)). That is the first version that consistently balanced the filling across the whole service, and it is the version branded RotaryQuant in the main text.

### C.8 Cleaning the trays (cf. §4.4 — inverse rotation at read time)

The inverse rotation is like scrubbing the steamer trays clean between sweet and savoury dishes. The same steamer stack that just held har gow (shrimp dumplings) now steams the coconut tarts. If you do not clean between batches, salt clings to the trays and contaminates the sweet filling. The inverse rotation scrubs the trays clean before the next load — restoring the original purity of each dish.

### C.9 Don't taste every dumpling (cf. §4.3 — deferred prefill)

Both TurboQuant and RotaryQuant introduce a small wrapping error on every insertion. During prefill, that means $T$ successive insertions each carrying a small independent error. Wrap 4,000 dumplings one at a time during the rush, and the head chef tastes the accumulated degradation by the end of service. Deferred prefill changes the protocol: hold the prep in fresh form during the busy lunch sitting, then bulk-wrap the lot in one calmer batch at the changeover into dinner. That single bulk wrap introduces one well-controlled error rather than 4,000 compounding ones.

### C.10 Pre-concentrated stock (cf. MLA discussion in the source paper)

MLA is like the kitchen already serving concentrated stock cubes instead of fresh broth. The filling is already compact. Adding another compression layer on top is like trying to dehydrate a stock cube — diminishing returns, unless the extra squeeze saves meaningful shelf space without ruining the flavour. The honest answer for the architectures inspected here is that the further squeeze does not justify the extra step.

### C.11 The mid-prep taste test (cf. AttnRes negative result)

The head chef is already tasting the dish to decide how to garnish it — that signal could in principle tell the runners which specialist to fetch before the plate reaches the next station. The signal is genuinely informative. But managing the runners costs enough attention to slow the kitchen down by more than the prefetch saves. The experiment was instructive even though the conclusion was negative: a good signal is not the same thing as a useful schedule.

### C.12 The full service (cf. §5.1 — pathway proofs)

By the end of the paper every system runs together in one kitchen on one service. The prep area is organised, the station chefs are on call, the filling is properly portioned, and the si fu has the best equipment. The question is no longer whether each trick works in isolation. The question is whether the kitchen can run a full dinner service without dropping a dish. That is exactly what the benchmark, smoke test, and soak artefacts in Section 5 are meant to demonstrate.

### C.13 Symbol-to-kitchen mapping

For readers who want to translate equations directly into kitchen vocabulary while reading the main text:

| Symbol | Maths role | Kitchen equivalent |
|---|---|---|
| $Q$ | Query matrix | Current **customer order ticket** — what dish is being requested now |
| $K$ | Key matrix | **Labels** on every prepped filling bowl — what each container says it holds |
| $V$ | Value matrix | The **actual fillings** inside the bowls — the substance that gets served |
| $QK^{\top}$ | Attention score | Head chef **checking the match** between order and label |
| $\text{MoE}(x)$ | Expert mixture | **Calling the station chefs** for a dish |
| $G(x)$ | Gating function | **Floor manager** deciding which chefs to call |
| $D$ | Distortion | **Dumpling deformation** (squashed filling) |
| $c_i$ | Lloyd-Max centroids | **Steamer basket sizes** |
| $b_i$ | Decision boundaries | **Sorting rule** for portioning dumplings |
| $H_d$ | WHT rotation | **Global mix** (rough stir in a single big bowl) |
| $q_L, q_R$ | SO(4) quaternion pair | **Two-handed fine mix** (perfecting batches of four) |
| $\Pi$ | Isometric rotation | **Portioning** (evening out the filling) |
| $\sigma_q^2$ | Quantisation error | **Crush factor** (lost filling under thin wrapping paper) |
| $\alpha_{n \to l}$ | AttnRes block weights | **Mid-prep taste test** |
