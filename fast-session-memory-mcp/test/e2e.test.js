/**
 * E2E test for Fast Session Memory MCP: hot path (start_task, log_step_fast, finish_task)
 * and quick_strategy_check, get_quick_stats. Uses temp cold DB via env.
 */
import { spawn } from "child_process";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { test } from "node:test";
import assert from "node:assert";

const __dirname = dirname(fileURLToPath(import.meta.url));
const SERVER_PATH = join(__dirname, "..", "build", "index.js");

function sendRequest(proc, method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = Date.now();
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

test("tools/list returns all five tools", async () => {
  const tmpDir = mkdtempSync(join(tmpdir(), "fast-session-memory-test-"));
  const coldDb = join(tmpDir, "cold.db");
  const proc = spawn("node", [SERVER_PATH], {
    cwd: join(__dirname, ".."),
    stdio: ["pipe", "pipe", "pipe"],
    env: { ...process.env, FAST_SESSION_MEMORY_COLD_DB: coldDb },
  });

  try {
    const result = await sendRequest(proc, "tools/list");
    assert.ok(result?.tools?.length === 5, "expected 5 tools");
    const names = result.tools.map((t) => t.name).sort();
    assert.deepStrictEqual(names, [
      "finish_task",
      "get_quick_stats",
      "log_step_fast",
      "quick_strategy_check",
      "start_task",
    ]);
  } finally {
    proc.kill("SIGTERM");
    rmSync(tmpDir, { recursive: true, force: true });
  }
});

test("hot path: start_task → log_step_fast → finish_task and get_quick_stats", async () => {
  const tmpDir = mkdtempSync(join(tmpdir(), "fast-session-memory-test-"));
  const coldDb = join(tmpDir, "cold.db");
  const proc = spawn("node", [SERVER_PATH], {
    cwd: join(__dirname, ".."),
    stdio: ["pipe", "pipe", "pipe"],
    env: { ...process.env, FAST_SESSION_MEMORY_COLD_DB: coldDb },
  });

  try {
    const list = await sendRequest(proc, "tools/list");
    assert.ok(list?.tools?.length === 5, "expected 5 tools");

    const startResult = await sendRequest(proc, "tools/call", {
      name: "start_task",
      arguments: { task_type: "e2e_test", task_description: "E2E run" },
    });
    const text = startResult?.content?.[0]?.text;
    assert.ok(typeof text === "string", "start_task should return text content");
    const { session_id } = JSON.parse(text);
    assert.ok(session_id, "session_id should be present");

    const strategyBefore = await sendRequest(proc, "tools/call", {
      name: "quick_strategy_check",
      arguments: { task_type: "e2e_test" },
    });
    const strategyText = strategyBefore?.content?.[0]?.text;
    assert.ok(typeof strategyText === "string");
    const strategy = JSON.parse(strategyText);
    assert.strictEqual(strategy.has_strategy, false, "no strategy yet for e2e_test");

    await sendRequest(proc, "tools/call", {
      name: "log_step_fast",
      arguments: { session_id, tokens: 100, latency_ms: 50 },
    });

    const finishResult = await sendRequest(proc, "tools/call", {
      name: "finish_task",
      arguments: { session_id, outcome: "success" },
    });
    const finishText = finishResult?.content?.[0]?.text;
    assert.ok(typeof finishText === "string");
    const finish = JSON.parse(finishText);
    assert.strictEqual(finish.outcome, "success");
    assert.strictEqual(finish.total_tokens, 100);
    assert.strictEqual(finish.total_latency_ms, 50);
    assert.ok(finish.message?.includes("background"), "message should mention background");

    const statsResult = await sendRequest(proc, "tools/call", {
      name: "get_quick_stats",
      arguments: { task_type: "e2e_test" },
    });
    const statsText = statsResult?.content?.[0]?.text;
    assert.ok(typeof statsText === "string");
    const stats = JSON.parse(statsText);
    assert.ok(Array.isArray(stats.recent_outcomes), "recent_outcomes should be array");
    assert.ok(typeof stats.avg_tokens === "number", "avg_tokens should be number");
  } finally {
    proc.kill("SIGTERM");
    rmSync(tmpDir, { recursive: true, force: true });
  }
});
