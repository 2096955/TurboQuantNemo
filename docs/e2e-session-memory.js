#!/usr/bin/env node
/**
 * E2E test for Fast Session Memory MCP: hot path (start_task, log_step_fast, finish_task).
 * Run from repo root: node docs/e2e-session-memory.js
 * Expect: hot path < 10ms, no errors.
 */
const { spawn } = require("child_process");
const path = require("path");

const SERVER_PATH = path.join(
  __dirname,
  "..",
  "fast-session-memory-mcp",
  "build",
  "index.js"
);

function sendRequest(proc, method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = Date.now();
    const msg = JSON.stringify({
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
            if (res.error) reject(new Error(res.error.message));
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

async function main() {
  const proc = spawn("node", [SERVER_PATH], {
    cwd: path.join(__dirname, ".."),
    stdio: ["pipe", "pipe", "pipe"],
    env: { ...process.env, FAST_SESSION_MEMORY_COLD_DB: ":memory:" },
  });

  const results = [];
  proc.stdout.on("data", (d) => results.push(d.toString()));
  proc.stderr.on("data", (d) => process.stderr.write(d));

  const t0 = Date.now();
  try {
    // MCP may require initialize first; if tools/list works without it, we skip init
    const list = await sendRequest(proc, "tools/list");
    if (!list?.tools?.length) throw new Error("No tools in list");
    const toolNames = list.tools.map((t) => t.name);
    if (
      !["start_task", "log_step_fast", "finish_task"].every((n) =>
        toolNames.includes(n)
      )
    ) {
      throw new Error("Missing expected tools: " + JSON.stringify(toolNames));
    }

    const startResult = await sendRequest(proc, "tools/call", {
      name: "start_task",
      arguments: { task_type: "e2e_test", task_description: "E2E run" },
    });
    const sessionId =
      typeof startResult?.content?.[0]?.text === "string"
        ? JSON.parse(startResult.content[0].text).session_id
        : null;
    if (!sessionId) throw new Error("No session_id: " + JSON.stringify(startResult));

    await sendRequest(proc, "tools/call", {
      name: "log_step_fast",
      arguments: { session_id: sessionId, tokens: 100, latency_ms: 50 },
    });

    await sendRequest(proc, "tools/call", {
      name: "finish_task",
      arguments: { session_id: sessionId, outcome: "success" },
    });

    const elapsed = Date.now() - t0;
    if (elapsed > 50) {
      console.warn(`Warning: total time ${elapsed}ms (target hot path <10ms per op)`);
    }
    console.log("OK: session_memory E2E passed (start_task → log_step_fast → finish_task)");
    console.log(`  Tools: ${list.tools.length}, elapsed: ${elapsed}ms`);
  } catch (err) {
    console.error("E2E failed:", err.message);
    process.exit(1);
  } finally {
    proc.kill("SIGTERM");
  }
}

main();
