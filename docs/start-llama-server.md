# Start llama.cpp server (Phase 1)

Model and server are already set up. To start the server:

```bash
/Users/anthonylui/llama.cpp/build/bin/llama-server \
  -m /Users/anthonylui/Models/qwen3-coder-next/Q5_K_M/Qwen3-Coder-Next-Q5_K_M-00001-of-00002.gguf \
  -c 8192 \
  -ngl 999 \
  --port 8080
```

- **Model**: Qwen3-Coder-Next 124B Q5_K_M (two GGUF parts in `~/Models/qwen3-coder-next/Q5_K_M/`).
- **Port**: 8080 (qwen-coder-mcp uses `QWEN_BASE_URL=http://localhost:8080`).
- **-ngl 999**: Offload all layers to Metal (GPU).

Verify once loaded:

```bash
curl -s http://localhost:8080/v1/models
```

The server is currently running in the background (started during Phase 1 setup). To stop it, find the process (`pgrep -f llama-server`) and kill it. Restart with the command above when you need Qwen for the MCP tools.
