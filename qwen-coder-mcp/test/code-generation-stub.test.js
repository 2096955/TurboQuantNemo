/**
 * End-to-end test without any real Ollama or Qwen: a stub HTTP server
 * pretends to be Ollama. Verifies: MCP tools/call → prompt build → fetch
 * → parse response → return to client.
 *
 * Run: npm test (no Ollama required)
 */
import { createServer } from "http";
import { spawn } from "child_process";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { test } from "node:test";
import assert from "node:assert";

const __dirname = dirname(fileURLToPath(import.meta.url));
const SERVER_PATH = join(__dirname, "..", "build", "index.js");

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

/** Start a stub server that responds to POST /api/generate with { response: stubText } */
function startStubOllama(stubText) {
  return new Promise((resolve, reject) => {
    const server = createServer((req, res) => {
      if (req.method === "POST" && req.url === "/api/generate") {
        let body = "";
        req.on("data", (chunk) => (body += chunk));
        req.on("end", () => {
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ response: stubText }));
        });
      } else {
        res.statusCode = 404;
        res.end();
      }
    });
    server.listen(0, "127.0.0.1", () => {
      const port = server.address().port;
      resolve({ server, baseUrl: `http://127.0.0.1:${port}` });
    });
    server.on("error", reject);
  });
}

test("code_generation end-to-end with stub Ollama (no real model)", async () => {
  const stubReply = "stub reply from fake Ollama";
  const { server, baseUrl } = await startStubOllama(stubReply);

  const proc = spawn("node", [SERVER_PATH], {
    cwd: join(__dirname, ".."),
    stdio: ["pipe", "pipe", "pipe"],
    env: {
      ...process.env,
      QWEN_BASE_URL: baseUrl,
      QWEN_MODEL: "stub-model",
    },
  });

  try {
    const result = await sendRequest(proc, "tools/call", {
      name: "code_generation",
      arguments: {
        task: "Say hello",
        language: "text",
      },
    });

    assert.ok(result, "result should be defined");
    assert.ok(Array.isArray(result.content), "result.content should be an array");
    assert.ok(result.content.length >= 1, "at least one content item");
    const text = result.content[0]?.text;
    assert.strictEqual(text, stubReply, "MCP should return stub reply (full pipeline: prompt → fetch → parse → return)");
  } finally {
    proc.kill("SIGTERM");
    server.close();
  }
});

test("code_review and debug_assistance use stub (same pipeline)", async () => {
  const stubReply = "stub review";
  const { server, baseUrl } = await startStubOllama(stubReply);

  const proc = spawn("node", [SERVER_PATH], {
    cwd: join(__dirname, ".."),
    stdio: ["pipe", "pipe", "pipe"],
    env: {
      ...process.env,
      QWEN_BASE_URL: baseUrl,
      QWEN_MODEL: "stub-model",
    },
  });

  try {
    const reviewResult = await sendRequest(proc, "tools/call", {
      name: "code_review",
      arguments: { code: "const x = 1;", focus: "bugs" },
    });
    assert.strictEqual(reviewResult?.content?.[0]?.text, stubReply, "code_review returns stub");

    const debugResult = await sendRequest(proc, "tools/call", {
      name: "debug_assistance",
      arguments: { code: "fn()", error: "undefined" },
    });
    assert.strictEqual(debugResult?.content?.[0]?.text, stubReply, "debug_assistance returns stub");
  } finally {
    proc.kill("SIGTERM");
    server.close();
  }
});
