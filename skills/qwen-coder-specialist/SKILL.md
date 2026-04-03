# Qwen3-Coder Specialist Skill

## Purpose

Delegate complex coding tasks to a local Qwen3-Coder-Next model (124B) running via MCP. This specialist handles deep code generation, refactoring, and technical implementation that benefits from a model trained specifically on code.

## When to Use

- Complex algorithm implementation (graph algorithms, optimization, data structures)
- Large-scale code refactoring (multi-file, architectural changes)
- Performance-critical code generation
- Security-focused code review
- Converting between languages/frameworks
- Generating boilerplate with complex patterns
- Lock-free concurrency, memory-safe Rust, low-level systems code

## When NOT to Use

- Simple code snippets (handle directly)
- Explaining code concepts (handle directly)
- Quick syntax fixes (handle directly)
- Code needing immediate conversational context

## Available MCP Tools

### code_generation

Generate production-ready code for complex tasks.

**Input**:
- `task` (string, required): Detailed description of what to build
- `language` (string, optional): Target language/framework
- `context` (string, optional): Existing code, interfaces, constraints

**Usage**:
```
Call the code_generation tool with:
- task: "Implement a B+ tree with configurable fan-out supporting range queries"
- language: "Rust"
- context: "Must implement the Index trait from src/traits.rs"
```

### code_review

Deep code review with specific focus areas.

**Input**:
- `code` (string, required): Code to review
- `focus` (string, optional): security | performance | bugs | concurrency | memory

**Usage**:
```
Call the code_review tool with:
- code: [paste the code]
- focus: "concurrency"
```

### debug_assistance

Help debug complex issues.

**Input**:
- `code` (string, required): Problematic code
- `error` (string, required): Error message or unexpected behaviour

**Usage**:
```
Call the debug_assistance tool with:
- code: [paste the code]
- error: "Deadlock occurs when two threads call update() simultaneously"
```

## Phase 4: When Context Is Large or Noisy

If the conversation or task context would be too large or noisy to pass directly to Qwen:

1. Call **compress_handover** (handover-compression MCP) with a short `conversation` summary and the current `task`.
2. Use the returned compressed handoff as the **context** (or task) when calling **code_generation**.
3. Optionally call **store_handover_summary** (session-memory) with the compressed text to persist it in cold memory for analytics.

This keeps the hot path fast; compression uses Claude API and is only invoked when handover size is a problem.

## Session Memory Integration

Track performance to enable self-improvement:

### Before Starting Work
```
1. Call start_task with task_type (e.g., "code_generation", "refactoring", "debugging")
2. Call quick_strategy_check to see if there's a learned approach for this task_type
3. If strategy exists, follow the suggested approach
```

### During Work
```
4. Delegate to Qwen tools as needed
5. After each Qwen call, call log_step_fast with tokens used and latency
```

### After Completing
```
6. Call finish_task with outcome: "success" | "failure" | "partial"
7. Background learning will update strategy cache automatically
```

## Workflow Pattern

1. **Assess**: Is this complex enough to warrant Qwen, or can I handle it directly?
2. **Track**: Call `start_task` and `quick_strategy_check`
3. **Delegate**: "I'll use the Qwen specialist for this implementation..."
4. **Call**: Use the appropriate tool with detailed context
5. **Integrate**: Review output, verify correctness, add explanatory context
6. **Complete**: Call `finish_task` with honest outcome assessment

## Example Session

**User**: "I need a high-performance concurrent hash map in Rust"

**Response**:
```
This is complex enough for the Qwen specialist. Let me track this session and delegate.

[Call start_task with task_type: "code_generation"]
[Call quick_strategy_check with task_type: "code_generation"]

Strategy check shows 87% success rate for code_generation tasks with average 2,400 tokens.

[Call code_generation with:
  task: "Implement a concurrent hash map in Rust using lock-free techniques.
         Support get, insert, remove operations. Use atomic operations for
         thread safety. Optimize for read-heavy workloads."
  language: "Rust"
  context: "Target Rust 1.75+, use std::sync::atomic, no external crates"]

[Call log_step_fast with tokens: 3200, latency_ms: 18000]

Here's what the specialist generated:
[Show code with explanation]

I've reviewed this - it uses proper memory ordering (Acquire/Release),
the resize logic handles concurrent access correctly.

[Call finish_task with outcome: "success"]

Want me to add benchmarks or explain any part of the implementation?
```

## Quality Control

After receiving Qwen's output:
- Verify it matches the request
- Check for obvious issues (missing error handling, unsafe patterns)
- Add explanatory context for the user
- Suggest testing approach
- Offer to iterate if needed

## Fallback Behaviour

If MCP server unavailable or Qwen times out:
1. Inform user: "The Qwen specialist is currently unavailable"
2. Offer alternative: "I can handle this directly, though for complex implementations the specialist usually produces better results"
3. Suggest check: "Verify llama.cpp server is running on port 8080 (or Ollama on 11434) and that Qwen3-Coder-Next is loaded"

## Technical Notes

- **Model**: Qwen3-Coder-Next (e.g. Q5_K_M GGUF or 124B/66B via Ollama)
- **Source**: https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF, [Unsloth docs](https://unsloth.ai/docs/models/qwen3-coder-next)
- **Inference**: llama.cpp at localhost:8080 (default; `/v1/completions`) or Ollama at localhost:11434 (`/api/generate`)
- **Response time**: 10-30s for complex generations
- **Temperature**: 0.2 (deterministic coding)
- **Memory overhead**: <10ms for session tracking (hot path only)

## Self-Improvement Metrics

The session memory tracks:
- Success/failure rates by task_type
- Token efficiency (tokens used per successful task)
- Latency patterns
- Strategy refinement over time

After ~10 sessions per task_type, the system learns optimal approaches and can suggest them via `quick_strategy_check`.
