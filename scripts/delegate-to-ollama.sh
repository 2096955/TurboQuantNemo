#!/bin/bash
# delegate-to-ollama.sh — Claude Code → Ollama delegation wrapper
# Usage: ./scripts/delegate-to-ollama.sh "<task description>" [--model MODEL] [--context FILE...]
#
# Claude Code delegates implementation/analysis tasks to local Ollama models.
# Unlike Gemini CLI, Ollama doesn't have filesystem tools, so we inject
# file contents directly into the prompt.
#
# Models (set via --model or OLLAMA_MODEL env var):
#   nemotron-3-super:120b-a12b  — best reasoning (default, needs ~86GB)
#   qwen2.5-coder:32b-instruct-q4_K_M — best for code tasks
#   Qwen3.5:35b-a3b            — good general + code
#   mistral-small:22b           — fast, decent quality
#   qwen2.5-coder:14b           — fastest code model

set -eo pipefail

# Parse arguments
MODEL="${OLLAMA_MODEL:-qwen2.5-coder:32b-instruct-q4_K_M}"
TASK=""
CONTEXT_FILES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --context)
            shift
            while [[ $# -gt 0 && ! "$1" == --* ]]; do
                CONTEXT_FILES+=("$1")
                shift
            done
            ;;
        *)
            if [ -z "$TASK" ]; then
                TASK="$1"
            fi
            shift
            ;;
    esac
done

if [ -z "$TASK" ]; then
    echo "Usage: $0 \"<task>\" [--model MODEL] [--context FILE1 FILE2 ...]"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/ollama-delegations"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TASK_SLUG=$(echo "$TASK" | tr ' ' '-' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9-]//g' | cut -c1-50)
LOG_FILE="${LOG_DIR}/${TIMESTAMP}_${TASK_SLUG}.md"

echo ">>> Delegating to Ollama (${MODEL})..."
echo ">>> Task: ${TASK}"

# Build the prompt with injected file context
PROMPT="You are an expert code reviewer and implementer. You are being delegated a task by a senior AI orchestrator (Claude Code).

## Your Role
- Execute the task thoroughly
- Be precise with file paths and line numbers
- Show your reasoning
- End with a clear Summary of Changes or Findings section

"

# Inject file contents if provided
if [ ${#CONTEXT_FILES[@]} -gt 0 ]; then
    PROMPT+="## Context Files
"
    for f in "${CONTEXT_FILES[@]}"; do
        if [ -f "$f" ]; then
            echo ">>> Context: ${f}"
            PROMPT+="
### File: ${f}
\`\`\`
$(cat "$f")
\`\`\`
"
        else
            echo ">>> Warning: ${f} not found, skipping"
        fi
    done
fi

PROMPT+="
## Task
${TASK}
"

echo ">>> Log: ${LOG_FILE}"
echo ""

# Run Ollama and capture output
START_TIME=$(date +%s)

RESPONSE=$(ollama run "$MODEL" "$PROMPT" 2>/dev/null)
EXIT_CODE=$?

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Write log
cat > "$LOG_FILE" <<EOF
# Ollama Delegation Log
- **Timestamp**: $(date -Iseconds)
- **Model**: ${MODEL}
- **Task**: ${TASK}
- **Context Files**: ${CONTEXT_FILES[*]:-"(none)"}
- **Duration**: ${DURATION}s
- **Exit Code**: ${EXIT_CODE}

---

## Response

${RESPONSE}

---

## Review Status
- **Awaiting Review**: YES (Claude Code must review and sign off)
EOF

# Print to stdout
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "OLLAMA RESPONSE (${MODEL}, ${DURATION}s):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "$RESPONSE"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ">>> Duration: ${DURATION}s | Model: ${MODEL}"
echo ">>> Log: ${LOG_FILE}"

if [ $EXIT_CODE -eq 0 ]; then
    echo ">>> Ollama completed. Awaiting Claude Code review."
else
    echo ">>> Ollama exited with code ${EXIT_CODE}."
fi

exit $EXIT_CODE
