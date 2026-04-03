/**
 * Worker: runs cold DB writes off the event loop so sync better-sqlite3 does not block the main thread.
 */
import { workerData } from "worker_threads";
import Database from "better-sqlite3";

interface SessionPayload {
  session_id: string;
  task_type: string;
  task_description: string | null;
  started_at: number;
  step_count: number;
  total_tokens: number;
  total_latency_ms: number;
}

const { coldDbPath, session, outcome } = workerData as {
  coldDbPath: string;
  session: SessionPayload;
  outcome: string;
};

const coldDB = new Database(coldDbPath);

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
`);

const completed_at = Date.now();

coldDB
  .prepare(
    `INSERT OR REPLACE INTO sessions_full (session_id, task_type, task_description, outcome, started_at, completed_at, total_tokens, total_latency_ms)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
  )
  .run(
    session.session_id,
    session.task_type,
    session.task_description ?? null,
    outcome,
    session.started_at,
    completed_at,
    session.total_tokens,
    session.total_latency_ms
  );

const row = coldDB
  .prepare("SELECT COUNT(*) as count FROM sessions_full WHERE task_type = ?")
  .get(session.task_type) as { count: number };

if (row.count >= 5 && row.count % 10 === 0) {
  const recentSuccesses = coldDB
    .prepare(
      "SELECT * FROM sessions_full WHERE task_type = ? AND outcome = 'success' ORDER BY completed_at DESC LIMIT 5"
    )
    .all(session.task_type) as Array<{
    total_tokens?: number | null;
    total_latency_ms?: number | null;
  }>;

  if (recentSuccesses.length >= 3) {
    const avgTokens =
      recentSuccesses.reduce((sum, s) => sum + (s.total_tokens ?? 0), 0) /
      recentSuccesses.length;
    const counts = coldDB
      .prepare(
        "SELECT SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) as sc, SUM(CASE WHEN outcome != 'success' THEN 1 ELSE 0 END) as fc FROM sessions_full WHERE task_type = ?"
      )
      .get(session.task_type) as { sc: number; fc: number };
    const patternId = `pattern-${session.task_type}`;
    coldDB
      .prepare(
        `INSERT OR REPLACE INTO learned_patterns (pattern_id, task_type, pattern_description, success_count, failure_count, avg_tokens, last_updated)
         VALUES (?, ?, 'standard_approach', ?, ?, ?, ?)`
      )
      .run(
        patternId,
        session.task_type,
        counts.sc ?? 0,
        counts.fc ?? 0,
        Math.round(avgTokens),
        Date.now()
      );
  }
}

coldDB.close();
