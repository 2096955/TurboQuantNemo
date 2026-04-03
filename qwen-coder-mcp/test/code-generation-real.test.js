/**
 * Integration test: call code_generation against a real Ollama instance.
 * Requires Ollama running at QWEN_BASE_URL (default localhost:11434).
 * Use QWEN_MODEL to choose model (e.g. gemma3:latest or qwen3-coder-next:124b-q5).
 *
 * Run: QWEN_MODEL=gemma3:latest npm test   (or omit if Qwen is pulled)
 */
import { spawn } from "child_process";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { test } from "node:test";
import assert from "node:assert";

const __dirname = dirname(fileURLToPath(import.meta.url));
const SERVER_PATH = join(__dirname, "..", "build", "index.js");

const QWEN_BASE_URL = process.env.QWEN_BASE_URL ?? "http://localhost:11434";
const QWEN_MODEL = process.env.QWEN_MODEL ?? "qwen3-coder-next:124b-q5";
/** "openai" = llama.cpp /v1/completions; "ollama" = Ollama /api/generate */
const QWEN_API = process.env.QWEN_API ?? "ollama";

function sendRequest(proc, method, params = {}, id = null) {
  const reqId = id ?? Date.now();
  return new Promise((resolve, reject) => {
    const msg =
      JSON.stringify({
        jsonrpc: "2.0",
        id: reqId,
        method,
        params: Object.keys(params).length ? params : undefined,
      }) + "\n";

    const onData = (raw) => {
      const lines = raw.toString().split("\n").filter(Boolean);
      for (const line of lines) {
        try {
          const res = JSON.parse(line);
          if (res.id === reqId) {
            proc.stdout.off("data", onData);
            if (res.error) reject(new Error(res.error.message || JSON.stringify(res.error)));
            else resolve(res.result);
            return;
          }
        } catch (_) {}
      }
    };

    proc.stdout.on("data", onData);
    proc.stdin.write(msg, (err) => {
      if (err) reject(err);
    });
  });
}

/** Returns true if backend is reachable and QWEN_MODEL is available (Ollama /api/tags or OpenAI /v1/models). */
async function modelAvailable() {
  const base = QWEN_BASE_URL.replace(/\/$/, "");
  try {
    if (QWEN_API === "openai") {
      const r = await fetch(`${base}/v1/models`);
      if (!r.ok) return false;
      const j = await r.json();
      const models = j.data ?? j.models ?? [];
      const names = models.map((m) => m.id ?? m.name ?? m.model).filter(Boolean);
      return names.some((n) => n === QWEN_MODEL || n.includes("Qwen") || n.includes("qwen"));
    }
    const r = await fetch(`${base}/api/tags`);
    if (!r.ok) return false;
    const j = await r.json();
    const names = (j.models ?? []).map((m) => m.name ?? m.model).filter(Boolean);
    return names.includes(QWEN_MODEL);
  } catch (_) {
    return false;
  }
}

test("code_generation calls real Qwen (Ollama or llama.cpp) and returns non-empty text", async () => {
  const available = await modelAvailable();
  if (!available) {
    console.log("Skipping real-run test: backend not reachable or model not found. For Qwen3-Coder-Next on llama.cpp: QWEN_BASE_URL=http://localhost:8080 QWEN_MODEL=Qwen3-Coder-Next-Q5_K_M-00001-of-00002.gguf QWEN_API=openai npm test. For Ollama: ollama pull <model> then QWEN_MODEL=<name> npm test.");
    return;
  }

  const proc = spawn("node", [SERVER_PATH], {
    cwd: join(__dirname, ".."),
    stdio: ["pipe", "pipe", "pipe"],
    env: {
      ...process.env,
      QWEN_BASE_URL,
      QWEN_MODEL,
      QWEN_API,
    },
  });

  try {
    const result = await sendRequest(proc, "tools/call", {
      name: "code_generation",
      arguments: {
        task: "Reply with exactly the word OK and nothing else. No code, no explanation.",
        language: "text",
      },
    });

    assert.ok(result, "result should be defined");
    assert.ok(Array.isArray(result.content), "result.content should be an array");
    assert.ok(result.content.length >= 1, "at least one content item");
    const text = result.content[0]?.text;
    assert.ok(typeof text === "string" && text.length > 0, "content[0].text should be non-empty string (real model reply)");
    assert.ok(text.length >= 1 && text.length <= 5000, "reply should be short but non-empty");
  } finally {
    proc.kill("SIGTERM");
  }
});
