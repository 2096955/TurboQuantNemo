# End-to-End Test

## Automated tests (npm test)

Both MCP packages have a `test` script that runs Node’s built-in test runner.

**You can verify end-to-end without pulling or running Qwen (or any real model):**

```bash
cd qwen-coder-mcp && npm run build && npm test
```

- The **stub test** (`code-generation-stub.test.js`) starts a fake “Ollama” HTTP server and drives the MCP server through `code_generation`, `code_review`, and `debug_assistance`. It proves the full path: tools/call → prompt build → fetch → parse response → return. No Ollama, no Qwen, no network beyond localhost.
- The **real-run test** (`code-generation-real.test.js`) runs only if Ollama is reachable; it’s skipped otherwise. Use it when you want to hit a real model (e.g. `npm run test:real` with Gemma, or after `ollama pull qwen3-coder-next:124b-q5` for Qwen).

```bash
# Qwen Coder MCP – list tools + stub e2e (no model) + optional real Ollama
cd qwen-coder-mcp && npm run build && npm test

# Fast Session Memory MCP – tools/list + hot path
cd fast-session-memory-mcp && npm run build && npm test
```

From repo root, run both:

```bash
cd qwen-coder-mcp && npm run build && npm test && cd ../fast-session-memory-mcp && npm run build && npm test
```

- **qwen-coder-mcp**:
  - **End-to-end without any model**: `test/code-generation-stub.test.js` – starts a stub HTTP server that pretends to be Ollama; MCP server calls it and returns the stub reply. Proves full pipeline (tools/call → prompt build → fetch → parse → return) with **no Ollama or Qwen**.
  - `test/list-tools.test.js` – tools/list and schemas.
  - **Optional real run**: `test/code-generation-real.test.js` – calls real Ollama; skip if Ollama not running. Use `QWEN_MODEL=gemma3:latest npm test` or `npm run test:real` without Qwen; or `ollama pull qwen3-coder-next:124b-q5` then `npm test` for real Qwen.
- **fast-session-memory-mcp**: `test/e2e.test.js` – spawns server with temp cold DB, tests `tools/list` and full hot path (start_task, quick_strategy_check, log_step_fast, finish_task, get_quick_stats). Self-contained; no external services.

## 1. Verify MCP servers list tools (manual)

From repo root:

```bash
# Session Memory MCP (no backend required)
cd fast-session-memory-mcp
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | node build/index.js
# Expect JSON with tools: start_task, quick_strategy_check, log_step_fast, finish_task, get_quick_stats

# Qwen Coder MCP (Ollama optional for tools/list)
cd ../qwen-coder-mcp
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | node build/index.js
# Expect JSON with tools: code_generation, code_review, debug_assistance
```

## 2. Verify model (Phase 1 – manual)

```bash
# llama.cpp
curl -s http://localhost:8080/v1/models

# Ollama
curl -s http://localhost:11434/api/tags
```

## 3. Run session-memory E2E (hot path &lt;10ms)

**Preferred**: use the package’s own tests (self-contained):

```bash
cd fast-session-memory-mcp && npm test
```

**Alternative**: from repo root, run the standalone script in `docs/`:

```bash
node docs/e2e-session-memory.js
```

Both drive the Fast Session Memory MCP over stdio: `start_task` → `log_step_fast` → `finish_task` (and the package test also covers `quick_strategy_check` and `get_quick_stats`). Hot path target: &lt;10ms per operation.

## 4. Full E2E in Cursor / Claude Code

1. Start llama.cpp server on port 8080 with Qwen3-Coder-Next (or set `QWEN_BASE_URL` for Ollama).
2. Open this project in Cursor (or use Claude Code with `~/.claude.json` configured).
3. In chat: ask for a non-trivial coding task (e.g. “Implement a function that does X”).
4. Confirm the agent:
   - Calls `start_task` and optionally `quick_strategy_check`
   - Delegates to `code_generation` (Qwen Coder MCP)
   - Calls `log_step_fast` and `finish_task`
5. Optionally call `get_quick_stats` to see the session recorded.

## 5. Performance

- Memory operations (start_task, quick_strategy_check, log_step_fast, finish_task) should add **&lt;10ms** total.
- Background learning runs after `finish_task` and does not block the response.

---

## Test review (critical summary)

| Area | Covered | Not covered |
|------|--------|-------------|
| **qwen-coder-mcp** | `tools/list` (3 tools), **stub e2e** (code_generation, code_review, debug_assistance via fake Ollama – no real model) | Real Ollama/Qwen run is optional (`code-generation-real.test.js` skips if Ollama down) |
| **fast-session-memory-mcp** | `tools/list` (5 tools), full hot path (start_task → quick_strategy_check → log_step_fast → finish_task → get_quick_stats) | Background learning persistence across restarts; hot-path timing assertion (&lt;10ms); cleanup interval |
| **Build** | `npm run build` in both packages | Root-level `npm test` that runs both packages (run each package’s `npm test` manually or from a root script) |

**Recommendations**: (1) Run `npm test` in both packages after any change. (2) For qwen-coder `tools/call`, either add a test that mocks `fetch` and asserts prompt/response shape, or document that full tool calls require a running Ollama. (3) Optionally add a root `package.json` with a `test` script that runs both packages’ tests.
