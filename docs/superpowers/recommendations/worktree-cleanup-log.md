# Worktree Cleanup Log — Phase 0 of Research Reality Program

Captured: 2026-04-20T21:23:01Z
Spec: docs/superpowers/specs/2026-04-20-research-reality-program-design.md §0.2–0.3
Plan: docs/superpowers/plans/2026-04-20-phase-0-cleanup-and-invariance.md (Task 6)

This file is the audit trail for every destructive worktree action.
Each branch deleted is recorded here with its HEAD SHA so it can be
recovered if needed via `git update-ref refs/heads/<branch> <sha>`.

## Inventory (pre-cleanup)

```
---
Path: /Users/anthonylui/QwenCoderLocal/.claude/worktrees/mojo-vs-mlx-benchmark
Branch: worktree-mojo-vs-mlx-benchmark
SHA: bcae8ca9638bc85f269e9733c3eab38d7ec25231
Ahead/Behind main: 23/46
Working tree: DIRTY
Last commit: 2026-04-14 15:56:18 +0100 data: add benchmark results from initial run
---
Path: /Users/anthonylui/QwenCoderLocal/.worktrees/attnres-expert-management
Branch: feature/attnres-expert-management
SHA: a74f64bfdf777282fe9ba6ea6efd66bead84149e
Ahead/Behind main: 0/41
Working tree: clean
Last commit: 2026-04-04 23:16:24 +0100 feat: wire AttnResExpertPredictor into Kimi forward pass
---
Path: /Users/anthonylui/QwenCoderLocal/.worktrees/gemma-nemotron-val
Branch: feature/gemma-nemotron-val
SHA: a8dd2ed2ae9f8fec42eb203fd7d49ee941a513cc
Ahead/Behind main: 0/34
Working tree: clean
Last commit: 2026-04-06 23:28:27 +0100 fix: prompt cache trimming on exact and shorter matches
---
Path: /Users/anthonylui/QwenCoderLocal/.worktrees/nemotron-gemma-eval
Branch: feature/nemotron-gemma-eval
SHA: 9e10a91da524d24f31f52451a7b51966d00a8dcd
Ahead/Behind main: 4/34
Working tree: clean
Last commit: 2026-04-09 21:24:44 +0100 chore: add Nemotron 120B validation pathway script
---
Path: /Users/anthonylui/QwenCoderLocal/.worktrees/phase-0-codegen
Branch: feature/phase-0-codegen
SHA: 0811b99861dc1f0c1c04ed55635ffec52b04dc47
Ahead/Behind main: 0/0
Working tree: clean
Last commit: 2026-04-20 22:05:02 +0100 docs: add Phase 0 implementation plan (cleanup + invariance harness)
---
Path: /Users/anthonylui/QwenCoderLocal/.worktrees/qwen3-deferred-dedekimi
Branch: feature/qwen3-deferred-dedekimi
SHA: a8dd2ed2ae9f8fec42eb203fd7d49ee941a513cc
Ahead/Behind main: 0/34
Working tree: clean
Last commit: 2026-04-06 23:28:27 +0100 fix: prompt cache trimming on exact and shorter matches
---
Path: /Users/anthonylui/QwenCoderLocal/.worktrees/qwen3-deferred-dedekimi-impl
Branch: feature/qwen3-deferred-dedekimi-impl
SHA: c7c5b57c6419e5764978312f20b4c5a90ddc9fd9
Ahead/Behind main: 1/34
Working tree: DIRTY
Last commit: 2026-04-08 09:10:15 +0100 feat: implement Qwen3 offload plumbing and wiring
---
Path: /Users/anthonylui/QwenCoderLocal/.worktrees/qwen3-wiring
Branch: feature/qwen3-wiring
SHA: a8dd2ed2ae9f8fec42eb203fd7d49ee941a513cc
Ahead/Behind main: 0/34
Working tree: DIRTY
Last commit: 2026-04-06 23:28:27 +0100 fix: prompt cache trimming on exact and shorter matches
---
Path: /Users/anthonylui/QwenCoderLocal/.worktrees/rotorquant
Branch: feature/rotorquant
SHA: a8dd2ed2ae9f8fec42eb203fd7d49ee941a513cc
Ahead/Behind main: 0/34
Working tree: clean
Last commit: 2026-04-06 23:28:27 +0100 fix: prompt cache trimming on exact and shorter matches
```

## Decisions

### USER GATE 7 — 2026-04-21

**ABANDONED candidates (4 worktrees, all clean, 0 commits ahead of main):**
- `feature/attnres-expert-management` — SHA `a74f64bfdf777282fe9ba6ea6efd66bead84149e`
- `feature/gemma-nemotron-val` — SHA `a8dd2ed2ae9f8fec42eb203fd7d49ee941a513cc`
- `feature/qwen3-deferred-dedekimi` — SHA `a8dd2ed2ae9f8fec42eb203fd7d49ee941a513cc`
- `feature/rotorquant` — SHA `a8dd2ed2ae9f8fec42eb203fd7d49ee941a513cc`

**Verdict:** **DEFERRED — keep for now.** User chose to retain all four pending later review. Phase 0 exit gate impact: Task 8 deletion step is intentionally skipped; revisit before Phase 0 sign-off.

**Recovery (when revisited):** if deletion is later approved, the SHAs above are sufficient to recreate any branch via `git update-ref refs/heads/<branch> <sha>`. No data is at risk while the worktrees remain on disk.

### feature/nemotron-gemma-eval — 2026-04-21

**State:** clean working tree, 4 commits ahead of main, 11 days stale, 12k LoC behind main.

**Unique commits:**
- `7f263b5` test: skip non-deterministic tests and fix glm4 config
- `59a59a1` test: revise quality gate harness token limits
- `62c871b` chore: add Gemma 4 validation pathway script
- `9e10a91` chore: add Nemotron 120B validation pathway script

**Verdict:** **CHERRY-PICK then DELETE.** The two `chore:` commits add useful smoke-wrappers (`scripts/validate_gemma4_pathway.sh`, `scripts/validate_nemotron_pathway.sh`) that did not exist on main. The two `test:` commits target a stale tree state and were dropped.

**Action taken:**
- Cherry-picked `62c871b` → main as `1c78ef4`
- Cherry-picked `9e10a91` → main as `13b98e1`
- Removed worktree at `.worktrees/nemotron-gemma-eval`
- Deleted branch `feature/nemotron-gemma-eval` (was at `9e10a91da524d24f31f52451a7b51966d00a8dcd`)

**Recovery:** `git update-ref refs/heads/feature/nemotron-gemma-eval 9e10a91da524d24f31f52451a7b51966d00a8dcd` would restore the original branch tip if needed.
