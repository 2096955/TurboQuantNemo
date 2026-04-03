#!/usr/bin/env bash
# setup-token-efficiency.sh
#
# Installs RTK (token optimization proxy) and Superpowers (Claude Code plugin)
# for efficient token usage when running QwenCoderLocal on a 32GB machine.
#
# What this does:
#   1. Installs RTK via Homebrew (macOS) or Cargo
#   2. Initializes RTK global hooks for Claude Code
#   3. Clones and installs the Superpowers plugin for Claude Code
#   4. Registers the RTK PreToolUse hook in Claude Code settings
#
# Usage: bash scripts/setup-token-efficiency.sh
#
# Prerequisites:
#   - Claude Code CLI installed (claude.ai/code)
#   - macOS with Homebrew, or Rust/Cargo installed
#   - git

set -euo pipefail

CLAUDE_DIR="${HOME}/.claude"
HOOKS_DIR="${CLAUDE_DIR}/hooks"
PLUGINS_DIR="${CLAUDE_DIR}/plugins/marketplaces"

info()  { printf "\033[1;34m[info]\033[0m  %s\n" "$1"; }
ok()    { printf "\033[1;32m[ok]\033[0m    %s\n" "$1"; }
warn()  { printf "\033[1;33m[warn]\033[0m  %s\n" "$1"; }
error() { printf "\033[1;31m[error]\033[0m %s\n" "$1"; exit 1; }

# ---------- 1. Install RTK ----------

install_rtk() {
    if command -v rtk &>/dev/null; then
        local ver
        ver=$(rtk --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
        ok "RTK already installed: v${ver}"
        return
    fi

    info "Installing RTK (https://github.com/rtk-ai/rtk)..."

    if [[ "$(uname)" == "Darwin" ]] && command -v brew &>/dev/null; then
        brew install rtk-ai/tap/rtk
    elif command -v cargo &>/dev/null; then
        cargo install rtk
    else
        error "Cannot install RTK: need Homebrew (macOS) or Cargo (Rust). See https://github.com/rtk-ai/rtk#installation"
    fi

    if command -v rtk &>/dev/null; then
        ok "RTK installed: $(rtk --version 2>/dev/null)"
    else
        error "RTK installation failed"
    fi
}

# ---------- 2. Initialize RTK hooks ----------

init_rtk_hooks() {
    if [[ ! -d "${CLAUDE_DIR}" ]]; then
        error "Claude Code config directory not found at ${CLAUDE_DIR}. Is Claude Code installed?"
    fi

    info "Initializing RTK global hooks for Claude Code..."
    rtk init -g 2>/dev/null || true

    if [[ -f "${HOOKS_DIR}/rtk-rewrite.sh" ]]; then
        ok "RTK hook installed at ${HOOKS_DIR}/rtk-rewrite.sh"
    else
        warn "rtk init -g did not create hook file. Creating minimal hook..."
        mkdir -p "${HOOKS_DIR}"
        cat > "${HOOKS_DIR}/rtk-rewrite.sh" << 'HOOKEOF'
#!/usr/bin/env bash
# RTK token-saving hook for Claude Code PreToolUse (Bash commands)
# Rewrites commands through rtk for compact output (60-90% token savings)

if ! command -v jq &>/dev/null; then
  exit 0
fi
if ! command -v rtk &>/dev/null; then
  exit 0
fi

INPUT=$(cat)
CMD=$(echo "$INPUT" | jq -r '.tool_input.command // empty')
if [ -z "$CMD" ]; then
  exit 0
fi

REWRITTEN=$(rtk rewrite "$CMD" 2>/dev/null) || REWRITTEN=""
if [ -z "$REWRITTEN" ] || [ "$CMD" = "$REWRITTEN" ]; then
  exit 0
fi

ORIGINAL_INPUT=$(echo "$INPUT" | jq -c '.tool_input')
UPDATED_INPUT=$(echo "$ORIGINAL_INPUT" | jq --arg cmd "$REWRITTEN" '.command = $cmd')

jq -n \
  --argjson updated "$UPDATED_INPUT" \
  '{
    "hookSpecificOutput": {
      "hookEventName": "PreToolUse",
      "permissionDecision": "allow",
      "permissionDecisionReason": "RTK auto-rewrite",
      "updatedInput": $updated
    }
  }'
HOOKEOF
        chmod +x "${HOOKS_DIR}/rtk-rewrite.sh"
        ok "Minimal RTK hook created"
    fi
}

# ---------- 3. Register RTK hook in Claude Code settings ----------

register_rtk_hook() {
    local settings_file="${CLAUDE_DIR}/settings.json"

    if [[ ! -f "$settings_file" ]]; then
        warn "No settings.json found. Creating minimal config with RTK hook..."
        cat > "$settings_file" << SETTINGSEOF
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "${HOOKS_DIR}/rtk-rewrite.sh"
          }
        ]
      }
    ]
  }
}
SETTINGSEOF
        ok "Created settings.json with RTK hook"
        return
    fi

    # Check if RTK hook is already registered
    if grep -q "rtk-rewrite" "$settings_file" 2>/dev/null; then
        ok "RTK hook already registered in settings.json"
        return
    fi

    warn "RTK hook not found in existing settings.json."
    warn "Add this to your settings.json under hooks.PreToolUse:"
    echo ""
    echo '  {'
    echo '    "matcher": "Bash",'
    echo '    "hooks": ['
    echo '      {'
    echo '        "type": "command",'
    echo "        \"command\": \"${HOOKS_DIR}/rtk-rewrite.sh\""
    echo '      }'
    echo '    ]'
    echo '  }'
    echo ""
}

# ---------- 4. Install Superpowers plugin ----------

install_superpowers() {
    local sp_dir="${PLUGINS_DIR}/superpowers"

    if [[ -d "$sp_dir" ]] && [[ -f "$sp_dir/.claude-plugin/plugin.json" ]]; then
        local ver
        ver=$(python3 -c "import json; print(json.load(open('${sp_dir}/.claude-plugin/plugin.json')).get('version','?'))" 2>/dev/null || echo "?")
        ok "Superpowers already installed: v${ver}"
        return
    fi

    info "Installing Superpowers plugin (https://github.com/obra/superpowers)..."
    mkdir -p "${PLUGINS_DIR}"

    if command -v git &>/dev/null; then
        git clone --depth 1 https://github.com/obra/superpowers.git "$sp_dir" 2>/dev/null
    else
        error "git is required to install Superpowers"
    fi

    if [[ -d "$sp_dir" ]]; then
        ok "Superpowers installed at ${sp_dir}"
    else
        error "Superpowers installation failed"
    fi
}

# ---------- 5. Verify MCP servers ----------

verify_mcp_servers() {
    info "Checking QwenCoderLocal MCP servers..."
    local script_dir
    script_dir="$(cd "$(dirname "$0")" && pwd)"
    local repo_dir
    repo_dir="$(dirname "$script_dir")"

    local all_ok=true
    for server in qwen-coder-mcp fast-session-memory-mcp handover-compression-mcp; do
        local server_dir="${repo_dir}/${server}"
        if [[ -d "$server_dir/build" ]]; then
            ok "${server}: built"
        elif [[ -d "$server_dir" ]]; then
            warn "${server}: not built. Run: cd ${server_dir} && npm install && npm run build"
            all_ok=false
        else
            warn "${server}: directory not found at ${server_dir}"
            all_ok=false
        fi
    done

    if $all_ok; then
        ok "All MCP servers built"
    fi
}

# ---------- Main ----------

main() {
    echo ""
    echo "QwenCoderLocal Token Efficiency Setup"
    echo "======================================"
    echo ""
    echo "This script installs:"
    echo "  - RTK (https://github.com/rtk-ai/rtk) — 60-90% token savings on CLI output"
    echo "  - Superpowers (https://github.com/obra/superpowers) — TDD, planning, review skills"
    echo ""

    install_rtk
    echo ""
    init_rtk_hooks
    echo ""
    register_rtk_hook
    echo ""
    install_superpowers
    echo ""
    verify_mcp_servers

    echo ""
    echo "======================================"
    echo ""
    ok "Setup complete. Token efficiency stack:"
    echo ""
    echo "  RTK hook:     Rewrites Bash commands for compact output (automatic)"
    echo "  Superpowers:  /tdd, /plan, /review, /debug skills in Claude Code"
    echo "  MCP servers:  Delegate heavy codegen to local Qwen/Nemotron"
    echo "  Session mem:  Track token usage per task type for self-optimization"
    echo ""
    echo "  Verify: rtk gain         — show token savings analytics"
    echo "  Verify: claude /skills   — list available skills"
    echo ""
}

main "$@"
