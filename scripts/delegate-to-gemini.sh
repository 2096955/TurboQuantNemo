#!/bin/bash
# delegate-to-gemini.sh — Claude Code → Gemini CLI delegation wrapper
# Usage: ./scripts/delegate-to-gemini.sh "<task description>" [context_paths...]
#
# Claude Code uses this to dispatch implementation tasks to Gemini CLI,
# then reviews the output before signing off.
#
# Modes (set via GEMINI_MODE env var):
#   plan     — read-only, no file edits (default)
#   auto_edit — auto-approve file edits
#   yolo     — auto-approve everything (use with caution)

set -eo pipefail

TASK="$1"
shift || true
CONTEXT_PATHS=("${@}")

MODE="${GEMINI_MODE:-auto_edit}"

# Build context string from paths
CONTEXT=""
if [ ${#CONTEXT_PATHS[@]} -gt 0 ]; then
    for path in "${CONTEXT_PATHS[@]}"; do
        CONTEXT+="@${path} "
    done
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/gemini-delegations"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TASK_SLUG=$(echo "$TASK" | tr ' ' '-' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9-]//g' | cut -c1-50)
LOG_FILE="${LOG_DIR}/${TIMESTAMP}_${TASK_SLUG}.md"
JSON_FILE="${LOG_DIR}/${TIMESTAMP}_${TASK_SLUG}.json"

echo ">>> Delegating to Gemini CLI..."
echo ">>> Task: ${TASK}"
echo ">>> Context: ${CONTEXT:-@./}"
echo ">>> Mode: ${MODE}"
echo ""

# Run Gemini in non-interactive mode, capture JSON output
cd "$REPO_ROOT"
PROMPT="${TASK}"

RAW_FILE="${JSON_FILE}.raw"
gemini -p "$PROMPT" -o json --approval-mode "$MODE" 2>/dev/null > "$RAW_FILE"
EXIT_CODE=${PIPESTATUS[0]}

# Clean: strip any text before the first '{' (MCP warnings leak into stdout)
python3 -c "
raw = open('$RAW_FILE').read()
idx = raw.find('{')
if idx >= 0:
    print(raw[idx:], end='')
" > "$JSON_FILE"

# Extract response from JSON
if [ -f "$JSON_FILE" ] && [ -s "$JSON_FILE" ] && command -v python3 &>/dev/null; then
    RESPONSE=$(python3 -c "
import json, sys
try:
    data = json.load(open('$JSON_FILE'))
    print(data.get('response', '(no response)'))
except Exception as e:
    print(f'(failed to parse JSON: {e})')
")
    STATS=$(python3 -c "
import json, sys
try:
    data = json.load(open('$JSON_FILE'))
    stats = data.get('stats', {})
    tools = stats.get('tools', {})
    files = stats.get('files', {})
    models = stats.get('models', {})
    # Get first model's token info
    for m, info in models.items():
        tokens = info.get('tokens', {})
        print(f'Model: {m}')
        print(f'Tokens: {tokens.get(\"input\",0)} in / {tokens.get(\"candidates\",0)} out / {tokens.get(\"thoughts\",0)} thinking')
        break
    print(f'Tool calls: {tools.get(\"totalCalls\",0)} ({tools.get(\"totalSuccess\",0)} ok, {tools.get(\"totalFail\",0)} fail)')
    print(f'Files: +{files.get(\"totalLinesAdded\",0)} -{files.get(\"totalLinesRemoved\",0)} lines')
except:
    print('(no stats)')
")
else
    RESPONSE="(could not extract response — check $JSON_FILE)"
    STATS="(no stats)"
fi

# Write readable log
cat > "$LOG_FILE" <<EOF
# Gemini Delegation Log
- **Timestamp**: $(date -Iseconds)
- **Task**: ${TASK}
- **Context**: ${CONTEXT_PATHS[*]:-"(repo root)"}
- **Mode**: ${MODE}
- **Exit Code**: ${EXIT_CODE}

---

## Gemini Response

${RESPONSE}

---

## Stats
${STATS}

---

## Review Status
- **Awaiting Review**: YES (Claude Code must review and sign off)
EOF

# Print summary to stdout for Claude Code
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "GEMINI RESPONSE:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "$RESPONSE"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STATS: ${STATS}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo ">>> Log: ${LOG_FILE}"
echo ">>> JSON: ${JSON_FILE}"

if [ $EXIT_CODE -eq 0 ]; then
    echo ">>> Gemini completed. Awaiting Claude Code review."
else
    echo ">>> Gemini exited with code ${EXIT_CODE}."
fi

exit $EXIT_CODE
