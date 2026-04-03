# TurboQuantNemo

**Run a 120B coding model on your laptop. No cloud GPU. No $200/month subscriptions.**

100B+ MoE models are unusable on a laptop if every expert stays in BF16 in RAM. But MoE models only *route* a few experts per token — the rest can live on disk. Quantize those routed experts aggressively (2-bit), keep the dense layers at 4-bit, load experts on demand, and a 120B model fits in 17 GB of memory at ~19 tok/s.

This repo is the validated implementation of that idea for **Nemotron-H** on **Apple Silicon**, wired into **Claude Code** so you get a 120B local coding specialist for the price of a coordination API call.

| Profile | RAM | Model | Disk | Decode | Status |
|---------|-----|-------|------|--------|--------|
| **120B** | 32 GB | Nemotron-H 120B (2-bit experts) | ~80 GB | ~18.7 tok/s | Quality-gated |
| **30B** | 16 GB | Nemotron-H 30B (2-bit experts) | ~11 GB | ~25+ tok/s | Quality-gated |

## How it works

```text
You ──> Claude Code (orchestrator)
            │
            ├── Plans the approach (Claude API, small token cost)
            ├── Delegates code generation to local model (0 API tokens)
            │       └── mlx_lm.server + expert offload
            │           ├── 2-bit routed experts loaded on demand from disk
            │           ├── 4-bit dense layers resident in memory
            │           └── TurboQuant KV cache compression (attention layers)
            └── Reviews output + integrates (Claude API, small token cost)
```

Claude Code delegates heavy coding tasks — generation, review, debugging — to your local Nemotron-H through [MCP](https://modelcontextprotocol.io). The local model does the expensive work. Claude handles planning and orchestration. [RTK](https://github.com/rtk-ai/rtk) compacts CLI output for 60-90% token savings on top.

## We want your feedback

This is ready for **real code testing** on Apple Silicon laptops. We're looking for:

- Bug reports (especially on 16 GB machines with the 30B model)
- Quality comparisons: does 2-bit expert quality hold up on *your* codebase?
- Memory/performance reports from different M-series chips
- Workflow feedback: is the Claude Code + local model delegation useful in practice?

**Fork it, try it, file issues.** The [release candidate checklist](docs/RELEASE_CANDIDATE_CHECKLIST.md) has the exact validation commands.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- **32 GB RAM** for the 120B model, or **16 GB RAM** for the 30B model
- Python 3.11+
- Node.js 18+
- [Claude Code CLI](https://claude.ai/code) installed
- Disk space: ~80 GB (120B) or ~15 GB (30B) for the converted checkpoint
- HuggingFace access to `nvidia/Nemotron-3-Super-120B` (120B) or `nvidia/Nemotron-H-47B-Reasoning-128K` (30B base)

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/2096955/TurboQuantNemo.git
cd TurboQuantNemo

# Install the MLX-LM fork with expert offload support
cd mlx-lm
pip install -e .
cd ..

# Build the MCP server that connects Claude Code to the local model
cd qwen-coder-mcp
npm install && npm run build
cd ..

# Optional: session memory for tracking token usage
cd fast-session-memory-mcp
npm install && npm run build
cd ..
```

### 2. Download and convert the model

Choose your profile based on available RAM:

#### Profile A: 120B model (32 GB machines)

```bash
# Download
huggingface-cli download nvidia/Nemotron-3-Super-120B \
  --local-dir ~/Models/nemotron-3-super-120b

# Convert to 2-bit experts / 4-bit dense (~80 GB output)
python3 -m mlx_lm.convert \
  --hf-path ~/Models/nemotron-3-super-120b \
  --mlx-path ~/Models/nemotron-120b-mixed \
  -q --q-bits 4 \
  --mixed-expert-bits 2

# Repack experts for offloading (required)
python3 -m mlx_lm.repack_experts --model ~/Models/nemotron-120b-mixed

# Verify
python3 scripts/checkpoint_integrity.py \
  --model ~/Models/nemotron-120b-mixed \
  --require-config --expect-repack --expect-expert-keys
```

#### Profile B: 30B model (16 GB machines)

```bash
# Download
huggingface-cli download nvidia/Nemotron-H-47B-Reasoning-128K \
  --local-dir ~/Models/nemotron-h-47b

# Convert to 2-bit experts / 4-bit dense (~11 GB output)
python3 -m mlx_lm.convert \
  --hf-path ~/Models/nemotron-h-47b \
  --mlx-path ~/Models/nemotron-30b-mixed \
  -q --q-bits 4 \
  --mixed-expert-bits 2

# Repack experts for offloading (required)
python3 -m mlx_lm.repack_experts --model ~/Models/nemotron-30b-mixed

# Verify
python3 scripts/checkpoint_integrity.py \
  --model ~/Models/nemotron-30b-mixed \
  --require-config --expect-repack --expect-expert-keys
```

### 3. Start the local model server

```bash
# 120B (32 GB machines)
python3 -m mlx_lm.server \
  --model ~/Models/nemotron-120b-mixed \
  --expert-offload \
  --host 127.0.0.1 \
  --port 8080

# OR 30B (16 GB machines)
python3 -m mlx_lm.server \
  --model ~/Models/nemotron-30b-mixed \
  --expert-offload \
  --host 127.0.0.1 \
  --port 8080
```

Verify it's running:

```bash
curl -s http://127.0.0.1:8080/health   # Should return {"status":"ok",...}
curl -s http://127.0.0.1:8080/ready    # Should return {"ready":true,...}
```

### 4. Register the MCP server with Claude Code

From a **normal terminal** (not inside an active Claude Code session):

```bash
claude mcp add qwen-coder \
  --scope user \
  -e QWEN_BASE_URL=http://localhost:8080 \
  -e QWEN_API=openai \
  -- node /path/to/TurboQuantNemo/qwen-coder-mcp/build/index.js
```

Replace `/path/to/TurboQuantNemo` with your actual clone path.

### 5. Install token efficiency tools (recommended)

```bash
bash scripts/setup-token-efficiency.sh
```

This installs:
- **[RTK](https://github.com/rtk-ai/rtk)** — 60-90% token savings on CLI output
- **[Superpowers](https://github.com/obra/superpowers)** — structured coding skills for Claude Code

### 6. Use Claude Code

Start Claude Code normally. For complex coding tasks, Claude will automatically delegate to your local 120B model:

```bash
claude
```

You can also explicitly request delegation:

```
> Use the Qwen specialist to implement a concurrent hash map in Rust
```

Claude Code will:
1. Plan the approach (Claude API - small token cost)
2. Delegate code generation to your local 120B model (0 API tokens)
3. Review the output and integrate it (Claude API - small token cost)

## Performance

Measured on M-series Apple Silicon:

| Metric | 120B (32 GB) | 30B (16 GB) |
|--------|-------------|------------|
| Decode throughput | ~18.7 tok/s | ~25+ tok/s |
| Peak memory | ~17.4 GB | ~8 GB |
| Load time | ~2s | ~1s |
| Prefill (1024 tokens) | ~8.5s | ~3s |

Both profiles use 2-bit quantized routed experts with 4-bit dense layers. Expert weights are loaded on-demand from disk via memory-mapped shards, keeping resident memory well within the machine's limit.

> 30B numbers are estimates pending formal benchmarking on 16 GB hardware.

## Alternative quantization recipes

```bash
# 3-bit experts (better quality, ~100 GB on disk)
python3 -m mlx_lm.convert \
  --hf-path ~/Models/nemotron-3-super-120b \
  --mlx-path ~/Models/nemotron-120b-mixed-3bit \
  -q --q-bits 4 --mixed-expert-bits 3

# Uniform 4-bit (best quality, ~127 GB on disk)
python3 -m mlx_lm.convert \
  --hf-path ~/Models/nemotron-3-super-120b \
  --mlx-path ~/Models/nemotron-120b-4bit \
  -q --q-bits 4

# Always repack after conversion
python3 -m mlx_lm.repack_experts --model ~/Models/<your-checkpoint>
```

All three recipes pass the quality gate on coding, reasoning, and instruction-following tasks.

## Quality validation

Run the quality gate to verify your checkpoint:

```bash
python3 scripts/eval_quality_gate.py \
  --model ~/Models/nemotron-120b-mixed \
  --expert-offload \
  --suite all \
  --strict \
  --seed 42
```

Pass criteria: all prompts produce coherent output, no repetition loops, correct answers for reasoning tasks. Exit code 0 = pass, 1 = fail.

## Server options

For network-exposed or shared use, add auth and backpressure:

```bash
python3 -m mlx_lm.server \
  --model ~/Models/nemotron-120b-mixed \
  --expert-offload \
  --host 127.0.0.1 \
  --port 8080 \
  --api-key YOUR_SECRET_KEY \
  --max-pending-requests 8 \
  --queue-put-timeout-s 0
```

Endpoints:
- `GET /health` — server status (unauthenticated)
- `GET /ready` — model loaded + worker alive (unauthenticated)
- `GET /metrics` — queue depth + expert cache stats (requires auth if `--api-key` set)
- `POST /v1/chat/completions` — OpenAI-compatible inference
- `POST /v1/completions` — text completion

## Project structure

```
TurboQuantNemo/
├── mlx-lm/                    # MLX-LM fork with expert offload + TurboQuant
│   ├── mlx_lm/expert_offload.py    # LRU expert cache + on-demand loading
│   ├── mlx_lm/repack_experts.py    # Checkpoint repacker for per-expert keys
│   ├── mlx_lm/models/nemotron_h.py # Nemotron-H model (Mamba-attention hybrid)
│   └── mlx_lm/server.py            # HTTP server with queue/auth/health
├── qwen-coder-mcp/           # MCP server: connects Claude Code to local model
├── fast-session-memory-mcp/   # MCP server: session tracking (<10ms hot path)
├── turboquant-mlx/            # KV cache compression for attention layers
├── scripts/
│   ├── eval_quality_gate.py   # Quality gate (5+ prompts, automated pass/fail)
│   ├── benchmark_moe_offload.py # Performance benchmark
│   ├── checkpoint_integrity.py # Checkpoint validation
│   └── setup-token-efficiency.sh # RTK + Superpowers installer
└── docs/
    ├── ORIGIN_ATTRIBUTION_AND_MATH.md
    ├── RELEASE_CANDIDATE_CHECKLIST.md
    ├── PRODUCTION_ROADMAP.md
    └── EXECUTION_BOARD.md
```

## What's in the name

**TurboQuantNemo** = **TurboQuant** + **Nemotron**. Two separate techniques bundled in one repo:

| Track | What it does | Source |
|-------|-------------|--------|
| **MoE expert offload** (weights) | Group-affine quantization of expert weights + LRU disk-to-GPU loading + `gather_qmm` | Engineering in this repo on top of [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) |
| **TurboQuant** (KV cache) | Rotation + Lloyd-Max codebooks + QJL 1-bit residual + asymmetric score estimator | [Frantar et al., arXiv:2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026) |

The **validated 120B story** in this repo is **quantized expert offload** — standard weight quantization plus systems engineering. **TurboQuant** is the separate KV cache compression paper; `turboquant-mlx/` bundles an MLX port for attention layers. Don't conflate them.

See [`docs/ORIGIN_ATTRIBUTION_AND_MATH.md`](./docs/ORIGIN_ATTRIBUTION_AND_MATH.md) for full attribution, citations, and equations.

## What this is and isn't

**This is:**
- A validated path to run a 120B parameter model for code generation on a 32 GB laptop
- Quality-gated at 2-bit, 3-bit, and 4-bit on real coding tasks
- Designed for single-user local use with Claude Code

**This is not:**
- A production multi-user serving system
- Validated for long-running unattended operation (soak testing pending)
- A replacement for Claude — it's a specialist that Claude orchestrates

## Troubleshooting

**"unknown model architecture" from llama-server**: This repo uses MLX, not llama.cpp. Start the server with `python3 -m mlx_lm.server`, not `llama-server`.

**Model produces repetitive output**: Ensure you're using the mlx-lm fork from this repo (not upstream). The fixes for `mx.load()` tensor loading and `time_step_limit` are critical.

**Out of memory during conversion**: Conversion needs more RAM than inference. Close other applications, or convert on a machine with more RAM and copy the checkpoint.

**MCP server not connecting**: Register with `claude mcp add` from a normal terminal, not inside an active Claude Code session. Check `QWEN_BASE_URL` points to the running server.

## License

MLX-LM components follow the [MIT License](mlx-lm/LICENSE). Nemotron-H model weights are subject to [NVIDIA's license terms](https://huggingface.co/nvidia/Nemotron-3-Super-120B).
