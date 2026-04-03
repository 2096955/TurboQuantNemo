/**
 * Test that qwen-coder-mcp lists the three MCP tools (code_generation, code_review, debug_assistance).
 * Spawns the built server and sends tools/list over stdio.
 */
import { spawn } from "child_process";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { test } from "node:test";
import assert from "node:assert";

const __dirname = dirname(fileURLToPath(import.meta.url));
const SERVER_PATH = join(__dirname, "..", "build", "index.js");

function sendRequest(proc, method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = 1;
    const msg =
      JSON.stringify({
        jsonrpc: "2.0",
        id,
        method,
        params: Object.keys(params).length ? params : undefined,
      }) + "\n";

    const onData = (raw) => {
      const lines = raw.toString().split("\n").filter(Boolean);
      for (const line of lines) {
        try {
          const res = JSON.parse(line);
          if (res.id === id) {
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

test("tools/list returns code_generation, code_review, debug_assistance", async () => {
  const proc = spawn("node", [SERVER_PATH], {
    cwd: join(__dirname, ".."),
    stdio: ["pipe", "pipe", "pipe"],
  });

  try {
    const result = await sendRequest(proc, "tools/list");
    assert.ok(result, "result should be defined");
    assert.ok(Array.isArray(result.tools), "result.tools should be an array");
    const names = result.tools.map((t) => t.name);
    assert.deepStrictEqual(
      names.sort(),
      ["code_generation", "code_review", "debug_assistance"].sort(),
      "expected three tools"
    );
    for (const t of result.tools) {
      assert.ok(t.description, `tool ${t.name} should have description`);
      assert.ok(t.inputSchema?.type === "object", `tool ${t.name} should have inputSchema`);
    }
  } finally {
    proc.kill("SIGTERM");
  }
});
