# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QwenCoderLocal replaces Cursor's paid agent with **Claude (coordinator) + local Qwen3-Coder-Next (specialist)** via MCP, with session memory (<10ms hot path). It contains three TypeScript MCP servers, a full-stack AI agent system (deer-flow), and MLX-based KV cache compression libraries.

## Architecture

```
QwenCoderLocal/
├── qwen-coder-mcp/           # MCP server: local Qwen code tools (TS, single src/index.ts)
├── fast-session-memory-mcp/  # MCP server: hot/cold session tracking (TS, src/index.ts + cold-worker.ts)
├── handover-compression-mcp/ # MCP server: context compression via Claude API (TS, single src/index.ts)
├── deer-flow/                # Full-stack agent (Python backend + Next.js frontend)
│   ├── backend/              # LangGraph orchestration, entry: langgraph.json → make_lead_agent
│   └── frontend/             # Next.js app
├── turboquant-mlx/           # KV cache compression for Apple Silicon (Python)
│   └── codebooks/            # Precomputed codebooks: dim_128_{1,2,3,4}bit.npz
├── mlx-lm/                   # MLX LLM inference fork with TurboQuant + expert offload
├── nemotron-30b-mixed/       # Nemotron-H 120B checkpoint (mixed-quant safetensors, ~11.6GB)
├── skills/                   # Claude Code skill definitions (qwen-coder-specialist/)
├── scripts/                  # Delegation + benchmark + verification scripts
└── docker/                   # Fleet worker Dockerfile (Node.js + Claude Code + Bun)
```

## Build & Run Commands

### MCP Servers (all three follow the same pattern)

```bash
cd <server-dir> && npm install && npm run build
# Runs as stdio MCP server: node build/index.js
```

### qwen-coder-mcp Tests

```bash
cd qwen-coder-mcp
npm test                # Stub tests (no model needed)
npm run test:real       # Against gemma3:latest via Ollama
npm run test:qwen       # Against Qwen on localhost:8080
```

### fast-session-memory-mcp Tests

```bash
cd fast-session-memory-mcp
npm test                # E2E test suite
```

### deer-flow

```bash
cd deer-flow
make config             # Generate config from examples
make install            # Install all deps (backend + frontend)
make dev                # Full stack: LangGraph + Gateway + Next.js via nginx on :2026

# Individual services:
cd backend && make dev      # LangGraph server on :2024
cd backend && make gateway  # Gateway API on :8001
cd frontend && pnpm dev     # Next.js on :3000

# Backend quality:
cd backend && make test     # pytest
cd backend && make lint     # ruff check
cd backend && make format   # ruff format
```

### turboquant-mlx

```bash
cd turboquant-mlx
python codebook_precompute.py           # Phase 1: precompute codebooks
python test_mlx_turboquant.py           # Validation suite
python validate_real_kv.py --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --prompt-tokens 512
```

### mlx-lm

```bash
cd mlx-lm
pip install -e .            # Editable install
pip install -e .[test]      # With test deps (includes pytest-timeout; each test fails after 300s)
pytest tests/               # Full test suite

# Fast quality smoke (small model + short decode; v2 harness):
# python ../scripts/eval_quality_gate.py --model mlx-community/Llama-3.2-1B-Instruct-4bit --quick

# Run with TurboQuant KV cache:
python -m mlx_lm.generate --model <model> --prompt "Hello" --kv-cache-type turboquant
mlx_lm.server --model <model> --kv-cache-type turboquant --port 8080
```

## Environment Variables

| Variable | Used By | Default |
|----------|---------|---------|
| `QWEN_BASE_URL` | qwen-coder-mcp | `http://localhost:8080` |
| `QWEN_MODEL` | qwen-coder-mcp | Qwen3-Coder-Next GGUF filename |
| `QWEN_API` | qwen-coder-mcp | `openai` (alt: `ollama`) |
| `QWEN_REQUEST_TIMEOUT_MS` | qwen-coder-mcp | `120000` |
| `FAST_SESSION_MEMORY_COLD_DB` | fast-session-memory-mcp | `data/sessions-persistent.db` |
| `ANTHROPIC_API_KEY` | handover-compression-mcp | required |
| `HANDOVER_COMPRESSION_MODEL` | handover-compression-mcp | `claude-3-5-sonnet-20241022` |

deer-flow requires additional API keys in `.env` -- see `deer-flow/.env.example`.

### TurboQuant-specific

| Variable | Default | Notes |
|----------|---------|-------|
| `TURBOQUANT_CODEBOOK_DIR` | `turboquant_codebooks/` | Path to precomputed `.npz` codebook files |
| `TURBOQUANT_BITS` | `3` | Quantization bit-width for KV cache (TurboQuant) |
| `ISOQUANT_BITS` | (falls back to `TURBOQUANT_BITS`) | Quantization bit-width for IsoQuant KV cache |
| `TURBOQUANT_SKIP_LAYERS` | (none) | Comma-separated layer indices to skip compression |

**IsoQuant fused decode (head_dim=256, 3-bit path):** set `ISOQUANT_USE_NPT8_FUSED=1` to enable NPT=8 fused attention kernels (D=256 only). For `seq_len >= 512` the implementation uses T-tiled + FA2 merge (Phase 3b); shorter sequences use the v1 single-pass kernel. Override tile width with `ISOQUANT_NPT8_TILE_SIZE` (default `256`; invalid values fall back to 256). `ISOQUANT_USE_METAL` is a separate flag that only controls the SO(4) rotation runtime (Metal vs Python); it does not gate fused attention dispatch.

| Variable | Default | Notes |
|----------|---------|-------|
| `ISOQUANT_USE_METAL` | `0` | `1` uses Metal kernels for SO(4) inverse rotation (not fused attention — rotation only) |
| `ISOQUANT_CACHE_MODE` | `concat_append` | `prealloc` for O(1) buffer-slice decode; `concat_append` for default concat path |
| `ISOQUANT_USE_NPT8_FUSED` | `0` | `1` enables NPT=8 fused QK+softmax+V for D=256 (v1 single-pass or T-tiled+FA2 merge) |
| `ISOQUANT_NPT8_TILE_SIZE` | `256` | T tokens per tile for the tiled NPT=8 path (long contexts); must be a positive int |

## MCP Tool Surface

**qwen-coder-mcp**: `code_generation`, `code_review`, `debug_assistance`

**fast-session-memory-mcp**: `start_task`, `quick_strategy_check`, `log_step_fast`, `finish_task`, `get_quick_stats`, `store_handover_summary`

**handover-compression-mcp**: `compress_handover`

## Key Design Decisions

- **Hot/cold memory split**: fast-session-memory uses in-memory SQLite for <1ms reads, file-backed SQLite for persistence via background worker thread.
- **TypeScript MCP servers** all compile to `build/` with ES2022 target, Node16 module resolution, strict mode.
- **deer-flow backend** uses LangGraph for agent orchestration; the gateway is FastAPI. Security hardening covers 22 gateway tests.
- **Qwen inference**: llama.cpp `/v1/completions` (default) or Ollama `/api/generate` -- toggled via `QWEN_API` env var.
- **TurboQuant**: Codebook-based KV cache compression for MLX; codebook `.npz` files live in `mlx-lm/mlx_lm/models/turboquant_codebooks/`. **Must not** be applied to Mamba/SSM `ArrayCache` — only full attention layers.
- **IsoQuant NPT=8 (D=256)**: With `ISOQUANT_USE_NPT8_FUSED=1`, decode uses fused Metal attention; for `seq_len` at or above 512, the T-tiled kernel + Python FA2 merge is selected (see `ISOQUANT_NPT8_TILE_SIZE`); shorter contexts use the v1 single-pass kernel.
- **Expert offload** (mlx-lm): `mlx_lm/expert_offload.py` with LRU caching. Use `--expert-offload` flag. `repack_experts.py` writes to `repacked-*.safetensors` with atomic index update.
- **No symlinks**: `mlx-lm/mlx_lm/models/mlx_turboquant.py` and `mlx_isoquant.py` are independent regular files (the historical symlink from `turboquant-mlx/` was replaced). Edit them in place under `mlx-lm/mlx_lm/models/`. The `turboquant-mlx/` directory now holds only codebook generation tooling.

## Delegation Pattern

For complex coding tasks, use the Qwen specialist workflow: `start_task` -> `quick_strategy_check` -> delegate to `code_generation`/`code_review`/`debug_assistance` -> `log_step_fast` -> `finish_task`. See `skills/qwen-coder-specialist/SKILL.md` for full details.

For large/noisy context, call `compress_handover` first, then pass the compressed result as context to Qwen tools.

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `scripts/delegate-to-ollama.sh` | Claude → Ollama delegation with `--context` file injection |
| `scripts/delegate-to-gemini.sh` | Claude → Gemini CLI delegation with JSON output |
| `scripts/benchmark_moe_offload.py` | MoE expert offload + TurboQuant benchmark (profiles A/B) |
| `scripts/eval_quality_gate.py` | Quality gate: 5-prompt automated pass/fail with repetition detection |
| `scripts/verify_turboquant_wiring.py` | End-to-end DeerFlow tool-calling verification |
| `scripts/setup-token-efficiency.sh` | Install RTK + Superpowers for token-efficient Claude Code usage |
| `scripts/run_variance_study.sh` | Inter-run variance: N benchmark repeats with mean ± stdev |
| `scripts/run_ablation_study.sh` | Ablation: 2×2 matrix (offload × KV) with quality + throughput |
| `scripts/run_failure_boundary.sh` | Failure modes: memory cap sweep + KV overflow boundary |
| `scripts/run_stock_comparison.sh` | Stock vs fork quality comparison on identical prompts |
| `scripts/load_test_concurrent.py` | Concurrent load test for mlx-lm server (1/2/4/8 clients) |

## Token Efficiency (for 32GB machines)

When using Claude Code as the orchestrator with local Nemotron/Qwen inference, install the token efficiency stack to minimize API token burn:

```bash
bash scripts/setup-token-efficiency.sh
```

This installs two external tools:

- **[RTK](https://github.com/rtk-ai/rtk)** (v0.23+) — CLI proxy that rewrites Bash tool output for 60-90% token savings. Installed as a Claude Code PreToolUse hook; all `git`, `npm`, `cargo`, etc. output is automatically compacted. Use `rtk gain` to see savings.
- **[Superpowers](https://github.com/obra/superpowers)** (v5+) — Claude Code plugin adding `/tdd`, `/plan`, `/review`, `/debug` skills that structure work into token-efficient patterns (e.g., plan before implementing, test-first development).

Combined with the MCP delegation pattern (heavy codegen offloaded to local model, only orchestration on Claude), this keeps Claude API usage focused on coordination rather than raw code generation.

### Token flow architecture

```text
User request
  -> Claude Code (orchestrator, API tokens)
       -> RTK hook compacts CLI output (60-90% savings)
       -> Superpowers structures work (fewer wasted rounds)
       -> MCP: code_generation/code_review (local Qwen/Nemotron, 0 API tokens)
       -> MCP: session memory tracks efficiency (local SQLite, 0 API tokens)
       -> MCP: handover compression (one Claude call to compress, saves many downstream)
  <- Claude Code returns result
```

## Operational Reference

- `DEMO_RUNBOOK.md` — full operational procedures for llama.cpp (Track A) and MLX (Track B) server setup, Nemotron-H preparation, expert repacking, and end-to-end wiring tests.
- `docs/cursor-mcp-config.md` — MCP registration instructions for both Cursor and Claude Code.

## Workspace Notes

- Primary app code lives under `deer-flow/`; other top-level folders are MCP servers and tooling.
- `deer-flow/backend` imports: `ToolRuntime` from `langgraph.prebuilt`, `InjectedToolCallId` from `langchain_core.tools.base`.
- Local GGUF serving with `llama-server` can fail with "unknown model architecture" if the llama.cpp build is too old for newer Qwen architectures.
- MCP servers are registered via `.cursor/mcp.json` (Cursor) or `~/.claude.json` (Claude Code). Register with `claude mcp add` from a normal terminal, not inside an active Claude Code session.
- `nemotron-30b-mixed/` contains a locally stored Nemotron-H 120B model checkpoint with custom `modeling_nemotron_h.py` — not a submodule.
- `.gemini/GEMINI.md` defines Gemini CLI as the implementation agent in a two-agent workflow (Claude reviews, Gemini implements).
