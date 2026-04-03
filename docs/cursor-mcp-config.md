# MCP Configuration for Cursor and Claude Code

Register **qwen-coder-mcp**, **fast-session-memory-mcp**, and **handover-compression-mcp** so Cursor and Claude Code can use their tools.

## Prerequisites

- **Qwen3-Coder-Next** running:
  - **llama.cpp** at `http://localhost:8080` (recommended; e.g. [Qwen/Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next) GGUF via llama-server). Set `QWEN_BASE_URL=http://localhost:8080`, `QWEN_MODEL=<model id from /v1/models>`, `QWEN_API=openai`. This repo’s `.cursor/mcp.json` defaults to llama.cpp.
  - **Ollama** at `http://localhost:11434` with Qwen pulled (e.g. `ollama pull qwen3-coder-next:124b-q5`). Set `QWEN_BASE_URL=http://localhost:11434`, `QWEN_MODEL=qwen3-coder-next:124b-q5`, `QWEN_API=ollama`.
- All MCP packages built: `npm run build` in `qwen-coder-mcp/`, `fast-session-memory-mcp/`, and `handover-compression-mcp/`.
- **Phase 4 (optional)**: For handover compression, set `ANTHROPIC_API_KEY` in the handover-compression server env so it can call Claude to compress context before Qwen.

## Cursor (project)

This repo includes `.cursor/mcp.json` so Cursor loads both servers when you open QwenCoderLocal.

If you use a different workspace root, copy the `mcpServers` block into your Cursor MCP config (project `.cursor/mcp.json` or Cursor Settings → MCP).

## Claude Code

Add to `~/.claude.json` (merge with existing `mcpServers` if present):

```json
{
  "mcpServers": {
    "qwen-coder": {
      "command": "node",
      "args": ["/Users/anthonylui/QwenCoderLocal/qwen-coder-mcp/build/index.js"],
      "env": {
        "QWEN_BASE_URL": "http://localhost:8080",
        "QWEN_MODEL": "Qwen3-Coder-Next-Q5_K_M-00001-of-00002.gguf",
        "QWEN_API": "openai"
      }
    },
    "session-memory": {
      "command": "node",
      "args": ["/Users/anthonylui/QwenCoderLocal/fast-session-memory-mcp/build/index.js"]
    },
    "handover-compression": {
      "command": "node",
      "args": ["/Users/anthonylui/QwenCoderLocal/handover-compression-mcp/build/index.js"],
      "env": { "ANTHROPIC_API_KEY": "<your-key>" }
    }
  }
}
```

Adjust the `args` paths if your repo lives elsewhere.

## Environment (optional)

- **qwen-coder-mcp**
  - `QWEN_BASE_URL` — default `http://localhost:8080` (llama.cpp). Use `http://localhost:11434` for Ollama.
  - `QWEN_MODEL` — default from `/v1/models` for llama.cpp (e.g. `Qwen3-Coder-Next-Q5_K_M-00001-of-00002.gguf`). For Ollama use e.g. `qwen3-coder-next:124b-q5`.
  - `QWEN_API` — `openai` for llama.cpp (`/v1/completions`), `ollama` for Ollama (`/api/generate`). Default: `openai`.
- **fast-session-memory-mcp**
  - `FAST_SESSION_MEMORY_COLD_DB` — path for cold DB file; default `data/sessions-persistent.db` (under cwd; `data/` is created if missing).
- **handover-compression-mcp** (Phase 4)
  - `ANTHROPIC_API_KEY` — required for `compress_handover` (Claude API).
  - `HANDOVER_COMPRESSION_MODEL` — optional; default `claude-3-5-sonnet-20241022`.

## Verify

1. **llama.cpp**: `curl http://localhost:8080/v1/models` to list models; ensure Qwen3-Coder-Next is loaded. For Ollama: `curl http://localhost:11434/api/tags`.
2. **Cursor**: Open this project; in chat, confirm the MCP servers and their tools appear.
3. **Claude Code**: Restart Claude Code after editing `~/.claude.json`; confirm tools are available.

---

## IDE Workflow (Phase 2)

### Option A – Continue.dev (recommended first)

1. Install **Continue** in Cursor (Extensions or [continue.dev](https://continue.dev)).
2. Configure models:
   - **Primary**: Claude (e.g. Sonnet) for orchestration and when to delegate.
   - **Ollama**: Add `qwen3-coder-next:124b-q5` (and optionally `66b-q4` for tab completion).
3. Enable custom commands that “use Qwen for implementation” where the [Qwen Coder skill](../skills/qwen-coder-specialist/SKILL.md) applies.
4. Use Cmd+L (or equivalent) to chat; Claude + skill will decide when to call Qwen MCP tools (`code_generation`, `code_review`, `debug_assistance`).

### Option B – Aider (multi-file agentic edits)

- Use Claude as “architect” and Qwen as “editor” via Aider’s config.
- Run Aider from the terminal for features that need git-aware, multi-file edits.
- Configure Aider to use Claude for planning and Ollama/Qwen for implementation where appropriate.

---

## Phase 4: Semantic Compression for Handover

Use when context passed to Qwen is too large or noisy.

### Workflow

1. **Before** calling Qwen MCP: call **`compress_handover`** (handover-compression-mcp) with `conversation` (summary or transcript) and `task` (current coding goal).
2. Pass the returned compressed handoff as the **`context`** (or `task`) when calling **`code_generation`** (qwen-coder-mcp).
3. **Optionally**: after compressing, call **`store_handover_summary`** (session-memory) with `task_type` and `summary_text` to persist the summary in cold memory for analytics—still no vector search on the hot path.

### Tools

- **compress_handover** (handover-compression-mcp): Uses Claude to produce a minimal structured handoff (facts, constraints, decisions, code patterns, dependencies). Requires `ANTHROPIC_API_KEY`.
- **store_handover_summary** (session-memory): Enqueues a background write of the compressed summary to cold DB; returns immediately.

### Note

Semantic search (e.g. "similar past errors") or GraphRAG, if added later, must run in a **separate process** and must **not** block the main coding flow.
