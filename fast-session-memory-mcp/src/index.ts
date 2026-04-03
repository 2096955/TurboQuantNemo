// fast-session-memory-mcp - Hot/cold session memory, <10ms hot path, no vectors
import { Worker } from "worker_threads";
import * as path from "path";
import * as fs from "fs";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import Database from "better-sqlite3";
import * as crypto from "crypto";

// In-memory DB for hot path
const hotDB = new Database(":memory:");

// Persistent DB for cold storage; default to data/sessions-persistent.db
const defaultColdPath = path.join(process.cwd(), "data", "sessions-persistent.db");
const coldDbPath =
  process.env.FAST_SESSION_MEMORY_COLD_DB ?? defaultColdPath;
const coldDir = path.dirname(coldDbPath);
if (!fs.existsSync(coldDir)) {
  fs.mkdirSync(coldDir, { recursive: true });
}
const coldDB = new Database(coldDbPath);

// Simple schema - no vectors, no embeddings
hotDB.exec(`
  CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    task_type TEXT,
    task_description TEXT,
    started_at INTEGER,
    step_count INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    total_latency_ms INTEGER DEFAULT 0
  );

  CREATE TABLE strategies_cache (
    task_type TEXT PRIMARY KEY,
    approach TEXT,
    success_rate REAL,
    avg_tokens INTEGER,
    last_updated INTEGER
  );

  CREATE TABLE recent_outcomes (
    session_id TEXT,
    task_type TEXT,
    outcome TEXT,
    tokens INTEGER,
    latency_ms INTEGER,
    timestamp INTEGER
  );

  CREATE INDEX idx_outcomes_type ON recent_outcomes(task_type, outcome);
`);

coldDB.exec(`
  CREATE TABLE IF NOT EXISTS sessions_full (
    session_id TEXT PRIMARY KEY,
    task_type TEXT,
    task_description TEXT,
    execution_trace TEXT,
    outcome TEXT,
    reflection TEXT,
    started_at INTEGER,
    completed_at INTEGER,
    total_tokens INTEGER,
    total_latency_ms INTEGER
  );

  CREATE TABLE IF NOT EXISTS learned_patterns (
    pattern_id TEXT PRIMARY KEY,
    task_type TEXT UNIQUE NOT NULL,
    pattern_description TEXT,
    examples TEXT,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    avg_tokens INTEGER,
    last_updated INTEGER
  );

  CREATE TABLE IF NOT EXISTS handover_summaries (
    summary_id TEXT PRIMARY KEY,
    task_type TEXT,
    summary_text TEXT,
    created_at INTEGER
  );
`);

function loadStrategiesCache(): void {
  try {
    const strategies = coldDB
      .prepare(
        `
      SELECT 
        task_type,
        pattern_description as approach,
        CAST(success_count AS REAL) / (success_count + failure_count) as success_rate,
        avg_tokens
      FROM learned_patterns
      WHERE success_count > 2 AND (success_count + failure_count) > 0
      `
      )
      .all() as Array<{
      task_type: string;
      approach: string;
      success_rate: number;
      avg_tokens: number | null;
    }>;

    const insert = hotDB.prepare(`
      INSERT OR REPLACE INTO strategies_cache 
      (task_type, approach, success_rate, avg_tokens, last_updated)
      VALUES (?, ?, ?, ?, ?)
    `);

    for (const s of strategies) {
      insert.run(
        s.task_type,
        s.approach,
        s.success_rate,
        s.avg_tokens ?? 0,
        Date.now()
      );
    }
  } catch {
    // learned_patterns may be empty or not yet populated
  }
}

loadStrategiesCache();

const server = new Server(
  { name: "fast-session-memory", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "start_task",
      description: "Start tracking a task (instant, <1ms)",
      inputSchema: {
        type: "object",
        properties: {
          task_type: { type: "string" },
          task_description: { type: "string" },
        },
        required: ["task_type"],
      },
    },
    {
      name: "quick_strategy_check",
      description:
        "Get learned strategy if available (instant, cache lookup)",
      inputSchema: {
        type: "object",
        properties: { task_type: { type: "string" } },
        required: ["task_type"],
      },
    },
    {
      name: "log_step_fast",
      description: "Log execution step (instant, in-memory)",
      inputSchema: {
        type: "object",
        properties: {
          session_id: { type: "string" },
          tokens: { type: "number" },
          latency_ms: { type: "number" },
        },
        required: ["session_id", "tokens", "latency_ms"],
      },
    },
    {
      name: "finish_task",
      description: "Finish task and trigger background learning",
      inputSchema: {
        type: "object",
        properties: {
          session_id: { type: "string" },
          outcome: {
            type: "string",
            enum: ["success", "failure", "partial"],
          },
        },
        required: ["session_id", "outcome"],
      },
    },
    {
      name: "get_quick_stats",
      description: "Instant performance stats (in-memory only)",
      inputSchema: {
        type: "object",
        properties: { task_type: { type: "string" } },
      },
    },
    {
      name: "store_handover_summary",
      description:
        "Store a compressed handover summary in cold memory for analytics (background write, non-blocking)",
      inputSchema: {
        type: "object",
        properties: {
          task_type: { type: "string" },
          summary_text: { type: "string" },
          session_id: { type: "string" },
        },
        required: ["summary_text"],
      },
    },
  ],
}));

interface SessionRow {
  session_id: string;
  task_type: string;
  task_description: string | null;
  started_at: number;
  step_count: number;
  total_tokens: number;
  total_latency_ms: number;
}

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  const a = args as Record<string, unknown>;

  switch (name) {
    case "start_task": {
      const task_type = typeof a.task_type === "string" ? a.task_type : "";
      if (!task_type.trim()) {
        throw new Error("start_task requires non-empty task_type");
      }
      const session_id = crypto.randomUUID();
      const task_description =
        typeof a.task_description === "string" ? a.task_description : null;
      hotDB
        .prepare(
          "INSERT INTO sessions (session_id, task_type, task_description, started_at) VALUES (?, ?, ?, ?)"
        )
        .run(session_id, task_type, task_description, Date.now());
      return {
        content: [{ type: "text", text: JSON.stringify({ session_id }) }],
      };
    }
    case "quick_strategy_check": {
      const strategy = hotDB
        .prepare("SELECT * FROM strategies_cache WHERE task_type = ?")
        .get(a.task_type as string) as
        | { approach: string; success_rate: number; avg_tokens: number }
        | undefined;
      if (!strategy) {
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                has_strategy: false,
                message:
                  "No learned strategy, will learn from this attempt",
              }),
            },
          ],
        };
      }
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify({
              has_strategy: true,
              approach: strategy.approach,
              success_rate: strategy.success_rate,
              expected_tokens: strategy.avg_tokens,
            }),
          },
        ],
      };
    }
    case "log_step_fast": {
      const session_id = typeof a.session_id === "string" ? a.session_id : "";
      if (!session_id) {
        throw new Error("log_step_fast requires session_id");
      }
      const tokens = Number(a.tokens);
      const latency_ms = Number(a.latency_ms);
      if (Number.isNaN(tokens) || Number.isNaN(latency_ms)) {
        throw new Error("log_step_fast requires numeric tokens and latency_ms");
      }
      const update = hotDB.prepare(
        "UPDATE sessions SET step_count = step_count + 1, total_tokens = total_tokens + ?, total_latency_ms = total_latency_ms + ? WHERE session_id = ?"
      );
      const result = update.run(tokens, latency_ms, session_id);
      if (result.changes === 0) {
        throw new Error(`log_step_fast: session_id not found: ${session_id}`);
      }
      return { content: [{ type: "text", text: "logged" }] };
    }
    case "finish_task": {
      const session = hotDB
        .prepare("SELECT * FROM sessions WHERE session_id = ?")
        .get(a.session_id as string) as SessionRow;
      if (!session) {
        throw new Error(`Unknown session_id: ${a.session_id}`);
      }
      hotDB
        .prepare(
          "INSERT INTO recent_outcomes (session_id, task_type, outcome, tokens, latency_ms, timestamp) VALUES (?, ?, ?, ?, ?, ?)"
        )
        .run(
          a.session_id,
          session.task_type,
          a.outcome,
          session.total_tokens,
          session.total_latency_ms,
          Date.now()
        );
      const outcomeStr = a.outcome as string;
      const workerUrl = new URL("./cold-worker.js", import.meta.url);
      const worker = new Worker(workerUrl, {
        workerData: { coldDbPath, session, outcome: outcomeStr },
      });
      worker.on("error", (err) =>
        console.error("Background learning worker error:", err)
      );
      worker.on("exit", (code) => {
        if (code === 0) loadStrategiesCache();
      });
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify({
              session_id: a.session_id,
              outcome: a.outcome,
              total_tokens: session.total_tokens,
              total_latency_ms: session.total_latency_ms,
              message: "Learning triggered in background",
            }),
          },
        ],
      };
    }
    case "get_quick_stats": {
      const taskType = typeof a.task_type === "string" ? a.task_type.trim() : "";
      const cutoff = Date.now() - 7 * 24 * 60 * 60 * 1000;
      const byType = taskType.length > 0;
      const recent = (byType
        ? hotDB
            .prepare(
              "SELECT outcome, COUNT(*) as count FROM recent_outcomes WHERE task_type = ? AND timestamp > ? GROUP BY outcome"
            )
            .all(taskType, cutoff)
        : hotDB
            .prepare(
              "SELECT outcome, COUNT(*) as count FROM recent_outcomes WHERE timestamp > ? GROUP BY outcome"
            )
            .all(cutoff)) as Array<{ outcome: string; count: number }>;
      const avgMetrics = (byType
        ? hotDB
            .prepare(
              "SELECT AVG(tokens) as avg_tokens, AVG(latency_ms) as avg_latency FROM recent_outcomes WHERE task_type = ? AND outcome = 'success' AND timestamp > ?"
            )
            .get(taskType, cutoff)
        : hotDB
            .prepare(
              "SELECT AVG(tokens) as avg_tokens, AVG(latency_ms) as avg_latency FROM recent_outcomes WHERE outcome = 'success' AND timestamp > ?"
            )
            .get(cutoff)) as
        | { avg_tokens: number | null; avg_latency: number | null }
        | undefined;
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                task_type: byType ? taskType : null,
                recent_outcomes: recent,
                avg_tokens: Math.round(avgMetrics?.avg_tokens ?? 0),
                avg_latency_ms: Math.round(avgMetrics?.avg_latency ?? 0),
              },
              null,
              2
            ),
          },
        ],
      };
    }
    case "store_handover_summary": {
      const summary_id = crypto.randomUUID();
      const task_type = (a.task_type as string) ?? null;
      const summary_text = (a.summary_text as string) ?? "";
      setImmediate(() => {
        try {
          coldDB
            .prepare(
              "INSERT INTO handover_summaries (summary_id, task_type, summary_text, created_at) VALUES (?, ?, ?, ?)"
            )
            .run(summary_id, task_type, summary_text, Date.now());
        } catch (err) {
          console.error("store_handover_summary background error:", err);
        }
      });
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify({
              summary_id,
              message: "Handover summary enqueued for cold storage",
            }),
          },
        ],
      };
    }
    default:
      throw new Error(`Unknown tool: ${name}`);
  }
});

// Cleanup old hot memory periodically (every hour)
setInterval(() => {
  const cutoff = Date.now() - 24 * 60 * 60 * 1000;
  hotDB.prepare("DELETE FROM recent_outcomes WHERE timestamp < ?").run(cutoff);
  hotDB.prepare("DELETE FROM sessions WHERE started_at < ?").run(cutoff);
}, 60 * 60 * 1000);

const transport = new StdioServerTransport();
await server.connect(transport);
