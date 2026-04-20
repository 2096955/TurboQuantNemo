# Phase 0 — Cleanup + Invariance Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pay down accumulated debt (Mojo benchmark merge, worktree DRY violations) and ship a numerical invariance harness that gates Phase 1 work. Land a clean foundation for the rest of the program.

**Architecture:** Mostly git operations + small Python edits. The one new code artifact is `scripts/invariance_check.py` — a bit-exactness gate that runs the same prompt twice through `mlx_lm.generate` (greedy decoding, fixed seed) and asserts identical output. It catches nondeterminism that async expert prefetch could introduce in Phase 1. Worktree triage uses a "document → user-ratify → execute" pattern so destructive actions never run without explicit approval.

**Tech Stack:** Python 3.12, pytest, `mlx_lm` (existing in repo), git worktree commands, pixi (for Mojo bench re-run).

**Spec reference:** `docs/superpowers/specs/2026-04-20-research-reality-program-design.md` Phase 0 (sections 0.1–0.6).

---

## File Structure

| Path | Action | Responsibility |
|------|--------|----------------|
| `scripts/invariance_check.py` | Create | CLI: run prompt twice, hash outputs, exit 0 if identical |
| `tests/test_invariance_check.py` | Create | Unit tests for hash + diff helpers |
| `docs/superpowers/notes/2026-04-21-spec-decode-rejection-audit.md` | Create | δ audit memo with verdict |
| `docs/superpowers/recommendations/qwen3-wiring-triage.md` | Create | Per-hunk classification of dirty worktree |
| `docs/superpowers/recommendations/qwen3-deferred-dedekimi-impl-triage.md` | Create | Per-hunk classification of dirty worktree |
| `docs/superpowers/recommendations/worktree-cleanup-log.md` | Create | Audit log of all worktree decisions + SHAs (recoverable if needed) |
| `docs/superpowers/specs/2026-04-13-mojo-vs-mlx-kernel-benchmark-design.md` | Modify | Mark Status DONE — superseded |
| `docs/superpowers/specs/2026-04-16-qwen36-mixed-precision-pathway.md` | Modify | Mark Status DONE — superseded |
| `mojo-bench/` (top-level) | Create on main | Imported via merge of mojo-vs-mlx-benchmark worktree |
| `mojo-bench/results/*.json` | Replace | Re-run sweep with DW-fixed code overwrites pre-fix JSONs |

---

## Worktree Inventory (current state, captured 2026-04-20 from `git worktree list`)

| Worktree path | Branch | SHA | Initial classification (verify in Task 7) |
|---------------|--------|-----|-------------------------------------------|
| `~/QwenCoderLocal` | main | 018ba46 | Active |
| `.claude/worktrees/mojo-vs-mlx-benchmark` | worktree-mojo-vs-mlx-benchmark | bcae8ca | **MERGEABLE** (Task 1–5) |
| `.worktrees/attnres-expert-management` | feature/attnres-expert-management | a74f64b | Verify in Task 7 |
| `.worktrees/gemma-nemotron-val` | feature/gemma-nemotron-val | a8dd2ed | Suspected ABANDONED (shares SHA with 3 others) |
| `.worktrees/nemotron-gemma-eval` | feature/nemotron-gemma-eval | 9e10a91 | Verify in Task 7 |
| `.worktrees/qwen3-deferred-dedekimi` | feature/qwen3-deferred-dedekimi | a8dd2ed | Suspected ABANDONED (shares SHA) |
| `.worktrees/qwen3-deferred-dedekimi-impl` | feature/qwen3-deferred-dedekimi-impl | c7c5b57 | **DIRTY** (Task 12–14) |
| `.worktrees/qwen3-wiring` | feature/qwen3-wiring | a8dd2ed | **DIRTY** (Task 9–11) |
| `.worktrees/rotorquant` | feature/rotorquant | a8dd2ed | Suspected ABANDONED (shares SHA) |

Plus orphan branch `feature/rotorquant-dev` (no worktree).

Task 6 catalogues the canonical state. Task 7 verifies the abandoned classifications before Task 8 deletes anything.

---

## Task Sequence (25 tasks)

Tasks are ordered to minimize risk: Mojo merge (independent) → audit (read-only) → deletes (with verification) → user-ratified triage → spec status updates → δ audit → invariance harness (TDD) → exit verification.

Run all tasks from `/Users/anthonylui/QwenCoderLocal` (the main worktree) unless explicitly stated otherwise.

---

### Task 1: Re-run Mojo benchmark sweep with DW-fixed code

**Why:** The 21 result JSONs in `mojo-bench/results/` were generated *before* the Durbin-Watson code fix (DW called before sort, fixed at `bench_matmul.mojo:211/214`, `bench_rope.mojo:221/224`, `bench_softmax.mojo:184/187`). Need fresh JSONs for the merge.

**Files:**
- Modify: `.claude/worktrees/mojo-vs-mlx-benchmark/mojo-bench/results/*.json` (regenerated)

- [ ] **Step 1: Verify pixi environment intact**

```bash
cd /Users/anthonylui/QwenCoderLocal/.claude/worktrees/mojo-vs-mlx-benchmark/mojo-bench
pixi info | head -20
```

Expected: shows pixi 0.x and pinned mojo `0.26.3.0.dev2026040805`.

- [ ] **Step 2: Stash old JSONs for diff later**

```bash
cd /Users/anthonylui/QwenCoderLocal/.claude/worktrees/mojo-vs-mlx-benchmark/mojo-bench
mkdir -p results-pre-dw-fix
cp results/*.json results-pre-dw-fix/
ls results-pre-dw-fix/ | wc -l
```

Expected: `21`

- [ ] **Step 3: Run full benchmark sweep**

```bash
cd /Users/anthonylui/QwenCoderLocal/.claude/worktrees/mojo-vs-mlx-benchmark/mojo-bench
pixi run bench-all 2>&1 | tee /tmp/mojo-bench-rerun.log
```

Expected: completes in ~3-4 hours wall-clock; new JSONs in `results/` with current timestamps. Watch for non-zero exit; if any kernel fails, capture the error and stop here for human review (do not proceed to merge).

- [ ] **Step 4: Verify all 21 JSONs regenerated**

```bash
cd /Users/anthonylui/QwenCoderLocal/.claude/worktrees/mojo-vs-mlx-benchmark/mojo-bench
ls results/*.json | wc -l
```

Expected: `21`

- [ ] **Step 5: Spot-check one JSON for DW field presence**

```bash
python3 -c "import json; d=json.load(open('/Users/anthonylui/QwenCoderLocal/.claude/worktrees/mojo-vs-mlx-benchmark/mojo-bench/results/matmul_4096_4096_4096_fp16.json')); print('dw_statistic:', d.get('dw_statistic'))"
```

Expected: prints a `dw_statistic` value. If still ~0.0157 (the pre-fix value), the rerun didn't pick up fixed code — STOP and investigate.

---

### Task 2: Diff old vs new JSONs, capture material changes

**Why:** If DW fix changed conclusions materially, surface a note for the eventual blog post.

**Files:**
- Create: `.claude/worktrees/mojo-vs-mlx-benchmark/mojo-bench/results/DW_FIX_DIFF.md`

- [ ] **Step 1: Generate per-kernel comparison**

```bash
cd /Users/anthonylui/QwenCoderLocal/.claude/worktrees/mojo-vs-mlx-benchmark/mojo-bench
python3 - <<'PY'
import json, glob, os
diff = []
for new_path in sorted(glob.glob("results/*.json")):
    name = os.path.basename(new_path)
    old_path = f"results-pre-dw-fix/{name}"
    if not os.path.exists(old_path):
        continue
    n = json.load(open(new_path)); o = json.load(open(old_path))
    n_dw = n.get("dw_statistic"); o_dw = o.get("dw_statistic")
    n_med = n.get("median_us"); o_med = o.get("median_us")
    diff.append({"kernel": name, "dw_old": o_dw, "dw_new": n_dw,
                 "med_us_old": o_med, "med_us_new": n_med,
                 "med_pct_change": ((n_med-o_med)/o_med*100) if (o_med and n_med) else None})
import json as j; print(j.dumps(diff, indent=2))
PY
```

Expected: prints JSON with per-kernel old vs new. Median timing should be ≤±5% (DW fix doesn't change timing logic, only post-processing).

- [ ] **Step 2: Write diff summary**

Capture the output above into a markdown summary (handwritten, not a script — this is an analyst note).

```bash
cat > /Users/anthonylui/QwenCoderLocal/.claude/worktrees/mojo-vs-mlx-benchmark/mojo-bench/results/DW_FIX_DIFF.md <<'EOF'
# DW Fix Diff — pre-fix JSONs vs re-run JSONs

Pre-fix JSONs preserved in `results-pre-dw-fix/`.
Re-run JSONs are now in `results/`.

## Median timing
[Paste comparison table from Task 2 Step 1 output here]

## DW statistic
Pre-fix: dw_statistic ~0.0157 across kernels (sort-then-DW bug).
Post-fix: dw_statistic varies per kernel based on actual sample autocorrelation.

## Conclusion
[One sentence: "no material change to ranking" or "ranking changed for kernel X — flag for blog"]
EOF
```

- [ ] **Step 3: Commit re-run results + diff**

```bash
cd /Users/anthonylui/QwenCoderLocal/.claude/worktrees/mojo-vs-mlx-benchmark/mojo-bench
git add results/ results-pre-dw-fix/
git commit -m "$(cat <<'EOF'
data: re-run benchmark sweep with DW-fixed code

Pre-fix JSONs preserved in results-pre-dw-fix/ for audit trail.
Per-kernel diff captured in results/DW_FIX_DIFF.md.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Rebase mojo-vs-mlx-benchmark worktree onto current main

**Why:** Worktree branch may be behind main. Rebase first, resolve any conflicts (none expected — `mojo-bench/` is a new top-level dir).

**Files:** No file edits — git operation.

- [ ] **Step 1: Fetch + check distance from main**

```bash
cd /Users/anthonylui/QwenCoderLocal/.claude/worktrees/mojo-vs-mlx-benchmark
git fetch origin
git log --oneline main..HEAD | head -5
git log --oneline HEAD..main | head -5
```

Expected: shows commits unique to each branch. If HEAD..main is non-empty, rebase needed.

- [ ] **Step 2: Rebase onto main**

```bash
cd /Users/anthonylui/QwenCoderLocal/.claude/worktrees/mojo-vs-mlx-benchmark
git rebase main
```

Expected: clean rebase. If conflicts: STOP, do not use `git rebase --skip`, surface the conflict for human review. Resolve manually then `git rebase --continue`.

- [ ] **Step 3: Confirm clean state**

```bash
cd /Users/anthonylui/QwenCoderLocal/.claude/worktrees/mojo-vs-mlx-benchmark
git status
```

Expected: `nothing to commit, working tree clean`.

---

### Task 4: Smoke test in rebased worktree

**Why:** Confirm pixi environment + Mojo benchmarks still work after rebase before merging.

- [ ] **Step 1: Run cheapest benchmark as smoke**

```bash
cd /Users/anthonylui/QwenCoderLocal/.claude/worktrees/mojo-vs-mlx-benchmark/mojo-bench
pixi run bench-vec-add 2>&1 | tail -20
```

Expected: completes in <30s, prints throughput numbers, exit 0. If fails: STOP — rebase may have broken something.

- [ ] **Step 2: Verify result JSON written**

```bash
ls -la /Users/anthonylui/QwenCoderLocal/.claude/worktrees/mojo-vs-mlx-benchmark/mojo-bench/results/vec_add*.json
```

Expected: file exists with current timestamp.

---

### Task 5: Merge mojo-vs-mlx-benchmark to main + delete worktree

**Files:** No new files; merges existing worktree changes into main.

- [ ] **Step 1: Switch to main and merge**

```bash
cd /Users/anthonylui/QwenCoderLocal
git checkout main
git merge --no-ff worktree-mojo-vs-mlx-benchmark -m "$(cat <<'EOF'
Merge worktree-mojo-vs-mlx-benchmark: Mojo kernel benchmark suite

21 kernel benchmarks across 7 kernel families (matmul, softmax, rope,
isoquant_rotate, kv_compress, fused_attention, vec_add).
Includes Durbin-Watson fix re-run; pre-fix JSONs preserved for audit.

Closes the kernel-level Mojo work; end-to-end Mojo RotaryQuant continues
in Phase 2 of the research-reality program.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

Expected: clean merge commit on main.

- [ ] **Step 2: Verify mojo-bench/ now on main**

```bash
cd /Users/anthonylui/QwenCoderLocal
ls mojo-bench/ | head
git log --oneline -3
```

Expected: `mojo-bench/` directory present with `kernels/`, `harness/`, `results/`, `pixi.toml`. Most recent commit is the merge.

- [ ] **Step 3: Smoke from main worktree**

```bash
cd /Users/anthonylui/QwenCoderLocal/mojo-bench
pixi run bench-vec-add 2>&1 | tail -10
```

Expected: succeeds — confirms merge produced reproducible state.

- [ ] **Step 4: Remove the worktree**

```bash
cd /Users/anthonylui/QwenCoderLocal
git worktree remove .claude/worktrees/mojo-vs-mlx-benchmark
git branch -D worktree-mojo-vs-mlx-benchmark
git worktree list
```

Expected: `worktree-mojo-vs-mlx-benchmark` gone from `git worktree list`.

---

### Task 6: Catalog all remaining worktrees with last-touched + classification

**Why:** Task 7 needs a canonical record of what we're about to act on.

**Files:**
- Create: `docs/superpowers/recommendations/worktree-cleanup-log.md`

- [ ] **Step 1: Generate inventory**

```bash
cd /Users/anthonylui/QwenCoderLocal
for wt in $(git worktree list --porcelain | awk '/worktree/ && !/QwenCoderLocal$/ {print $2}'); do
  branch=$(cd "$wt" && git branch --show-current)
  sha=$(cd "$wt" && git rev-parse HEAD)
  last_commit=$(cd "$wt" && git log -1 --format='%ai %s')
  ahead=$(cd "$wt" && git rev-list --count main..HEAD)
  behind=$(cd "$wt" && git rev-list --count HEAD..main)
  dirty=$(cd "$wt" && [ -n "$(git status --porcelain)" ] && echo "DIRTY" || echo "clean")
  echo "---"
  echo "Path: $wt"
  echo "Branch: $branch"
  echo "SHA: $sha"
  echo "Ahead/Behind main: $ahead/$behind"
  echo "Working tree: $dirty"
  echo "Last commit: $last_commit"
done > /tmp/worktree-inventory.txt
cat /tmp/worktree-inventory.txt
```

Expected: text dump of every worktree's current state.

- [ ] **Step 2: Write canonical log**

```bash
cat > /Users/anthonylui/QwenCoderLocal/docs/superpowers/recommendations/worktree-cleanup-log.md <<'EOF'
# Worktree Cleanup Log — Phase 0 of Research Reality Program

Captured: __CAPTURED_DATE__
Spec: docs/superpowers/specs/2026-04-20-research-reality-program-design.md §0.2–0.3

This file is the audit trail for every destructive worktree action.
Each branch deleted is recorded here with its HEAD SHA so it can be
recovered if needed via `git update-ref refs/heads/<branch> <sha>`.

## Inventory (pre-cleanup)

[Paste content of /tmp/worktree-inventory.txt above this line]

## Decisions

(Filled in as Tasks 7–14 execute)

EOF
# Then substitute the date placeholder in-place (works on macOS BSD sed)
sed -i '' "s/__CAPTURED_DATE__/$(date -u +%Y-%m-%dT%H:%M:%SZ)/" \
  /Users/anthonylui/QwenCoderLocal/docs/superpowers/recommendations/worktree-cleanup-log.md
# Then paste inventory into the file using Edit tool (NOT shell — preserves formatting)
```

- [ ] **Step 3: Commit the log skeleton**

```bash
cd /Users/anthonylui/QwenCoderLocal
git add docs/superpowers/recommendations/worktree-cleanup-log.md
git commit -m "$(cat <<'EOF'
docs: open worktree cleanup audit log (Phase 0)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Verify the 4 suspected-abandoned worktrees actually have no work

**Why:** Three worktrees share SHA `a8dd2ed` (`gemma-nemotron-val`, `qwen3-deferred-dedekimi`, `rotorquant`); plus `feature/qwen3-wiring` also at `a8dd2ed` but that's classified DIRTY because it has uncommitted edits. Plus `nemotron-gemma-eval` (9e10a91) and `attnres-expert-management` (a74f64b) need verification.

**Files:**
- Modify: `docs/superpowers/recommendations/worktree-cleanup-log.md` (append decisions)

- [ ] **Step 1: For each candidate worktree, diff against main**

```bash
cd /Users/anthonylui/QwenCoderLocal
for wt_branch in feature/gemma-nemotron-val feature/qwen3-deferred-dedekimi feature/rotorquant feature/nemotron-gemma-eval feature/attnres-expert-management; do
  echo "=== $wt_branch ==="
  echo "Commits ahead of main:"
  git log --oneline main..$wt_branch | head -10
  echo ""
  echo "Files changed vs main:"
  git diff --stat main...$wt_branch | tail -5
  echo ""
done > /tmp/worktree-diffs.txt
cat /tmp/worktree-diffs.txt
```

Expected: text dump of per-branch divergence.

- [ ] **Step 2: Classify each one**

For each branch, decide:
- `ABANDONED — DELETE`: zero commits ahead, no working-tree changes, branch can be deleted
- `KEEP — REVIEW LATER`: has commits ahead OR has working-tree changes; not deleting now

Update `worktree-cleanup-log.md` with classification + reasoning per branch using the Edit tool.

Hard rule: if ANY branch shows commits ahead of main that aren't already merged in, mark `KEEP — REVIEW LATER` and flag for user. **Do not delete.**

- [ ] **Step 3: Get user confirmation before any deletion**

Print the list of `ABANDONED — DELETE` candidates with their HEAD SHAs and ask user explicitly:

> "Phase 0 Task 7: Confirmed ABANDONED candidates for deletion: [list]. Each branch's HEAD SHA recorded in worktree-cleanup-log.md. Proceed with deletion (Task 8)? (yes/no)"

**Wait for "yes" before Task 8.** If user says no or asks for more info, stop and address.

---

### Task 8: Delete the user-ratified abandoned worktrees

**Files:** No file edits — git operations.

- [ ] **Step 1: For each ABANDONED — DELETE branch, capture SHA then delete**

For each branch from Task 7 (example shown for one — repeat per branch):

```bash
cd /Users/anthonylui/QwenCoderLocal
WORKTREE_PATH=".worktrees/<name>"
BRANCH="feature/<name>"
SHA=$(git rev-parse "$BRANCH")
echo "Deleting $BRANCH at SHA $SHA (recoverable via 'git update-ref refs/heads/$BRANCH $SHA')"
git worktree remove "$WORKTREE_PATH"
git branch -D "$BRANCH"
```

Expected: each command succeeds; `git worktree list` shows reduced set.

- [ ] **Step 2: Verify final worktree count**

```bash
cd /Users/anthonylui/QwenCoderLocal
git worktree list
```

Expected: only `main` worktree + the two DIRTY worktrees (`qwen3-wiring`, `qwen3-deferred-dedekimi-impl`) plus any KEEP — REVIEW LATER from Task 7.

- [ ] **Step 3: Update + commit log**

Edit `docs/superpowers/recommendations/worktree-cleanup-log.md` to add the "Executed" section listing every deletion with SHA. Then commit:

```bash
cd /Users/anthonylui/QwenCoderLocal
git add docs/superpowers/recommendations/worktree-cleanup-log.md
git commit -m "$(cat <<'EOF'
docs: log abandoned worktree deletions with recovery SHAs

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Generate triage document for qwen3-wiring (DIRTY worktree #1)

**Why:** Branch is 32 commits behind main with dirty edits; some hunks may duplicate work already on main. Per spec §0.3, classify each hunk before any destructive action.

**Files:**
- Create: `docs/superpowers/recommendations/qwen3-wiring-triage.md`

- [ ] **Step 1: Generate diff vs main**

```bash
cd /Users/anthonylui/QwenCoderLocal/.worktrees/qwen3-wiring
git fetch origin
git log --oneline main..HEAD > /tmp/qwen3-wiring-commits.txt
git status --short > /tmp/qwen3-wiring-status.txt
git diff main...HEAD > /tmp/qwen3-wiring-committed-diff.txt
git diff > /tmp/qwen3-wiring-working-diff.txt
wc -l /tmp/qwen3-wiring-*.txt
```

Expected: four files written, sizes printed.

- [ ] **Step 2: Per-hunk classification**

For each hunk in the diffs, classify:
- `duplicates_main` — same change already exists on main
- `genuinely_new` — change not on main; needs decision: keep or discard
- `ambiguous` — needs case-by-case review

Write the triage document at `docs/superpowers/recommendations/qwen3-wiring-triage.md` with format:

```markdown
# qwen3-wiring Triage

**Branch:** feature/qwen3-wiring
**HEAD SHA:** a8dd2ed
**Last commit date:** [from `git log -1 --format='%ai'`]
**Commits ahead of main:** [N]
**Working-tree dirty files:** [count]

## Hunks Classification

### Committed changes (main..HEAD)

| File | Hunk summary | Classification | Recommended action |
|------|--------------|----------------|-------------------|
| path/to/file.py:LL-LL | One-line description | duplicates_main / genuinely_new / ambiguous | keep / discard / split |

### Working-tree changes (uncommitted)

[Same table format]

## Recommendation

[Paragraph: rebase what onto what, abandon what, escalate what]

## Recovery info

If approved deletions go wrong, restore branch via:
`git update-ref refs/heads/feature/qwen3-wiring a8dd2ed`
And working-tree changes are NOT recoverable — surface this risk to user.
```

- [ ] **Step 3: Commit triage doc**

```bash
cd /Users/anthonylui/QwenCoderLocal
git add docs/superpowers/recommendations/qwen3-wiring-triage.md
git commit -m "$(cat <<'EOF'
docs: triage recommendation for qwen3-wiring worktree

Awaits user ratification before any destructive action (Phase 0 §0.3).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: User ratification gate for qwen3-wiring

**This is a non-code step. Wait for user response before Task 11.**

- [ ] **Step 1: Present triage doc to user**

> "Phase 0 Task 10: qwen3-wiring triage written at `docs/superpowers/recommendations/qwen3-wiring-triage.md`. Please review. Reply 'approved' to proceed with the recommended actions, 'changes' with specific instructions, or 'hold' to defer this worktree."

- [ ] **Step 2: Record ratification in the triage doc**

When user responds, append to the triage doc:

```markdown
## User Ratification

**Date:** YYYY-MM-DD
**Decision:** approved / approved-with-changes / hold
**Notes:** [user's verbatim response]
```

Commit the ratification record.

---

### Task 11: Execute approved actions for qwen3-wiring

**Files:** Per Task 9 recommendation — varies.

- [ ] **Step 1: Execute only the actions ratified in Task 10**

If `approved`: execute the recommendation from Task 9 verbatim (rebase salvageable hunks onto a clean branch, abandon the rest).
If `approved-with-changes`: execute per user's modifications.
If `hold`: skip Task 11, leave worktree intact, note in cleanup log.

- [ ] **Step 2: Update worktree-cleanup-log.md with actions taken**

Append a "qwen3-wiring — Executed" section listing every action with SHA pre/post. Commit.

---

### Task 12: Generate triage document for qwen3-deferred-dedekimi-impl (DIRTY worktree #2)

**Files:**
- Create: `docs/superpowers/recommendations/qwen3-deferred-dedekimi-impl-triage.md`

- [ ] **Step 1: Generate diff vs main** — same shell pattern as Task 9 Step 1, substituting `qwen3-deferred-dedekimi-impl` and SHA `c7c5b57`.

- [ ] **Step 2: Per-hunk classification** — same format as Task 9 Step 2.

- [ ] **Step 3: Commit triage doc** — same pattern as Task 9 Step 3.

---

### Task 13: User ratification gate for qwen3-deferred-dedekimi-impl

Same pattern as Task 10. Wait for user before Task 14.

---

### Task 14: Execute approved actions for qwen3-deferred-dedekimi-impl

Same pattern as Task 11.

---

### Task 15: Mark prior specs DONE

**Files:**
- Modify: `docs/superpowers/specs/2026-04-13-mojo-vs-mlx-kernel-benchmark-design.md` (status block, lines ~1-6)
- Modify: `docs/superpowers/specs/2026-04-16-qwen36-mixed-precision-pathway.md` (status block, lines ~1-6)

- [ ] **Step 1: Update mojo-vs-mlx-benchmark spec status**

In `docs/superpowers/specs/2026-04-13-mojo-vs-mlx-kernel-benchmark-design.md`, find the Date line and replace the surrounding header to add a Status field:

Old (after the title `# Mojo vs MLX Kernel Benchmark — Design Spec`):

```markdown
**Date**: 2026-04-13
**Goal**: Publication-quality kernel-level benchmark...
```

New:

```markdown
**Date**: 2026-04-13
**Status**: DONE — superseded by `2026-04-20-research-reality-program-design.md`. Kernel-level work merged to main 2026-04-2X (see Task 5 of `2026-04-20-phase-0-cleanup-and-invariance.md`).
**Goal**: Publication-quality kernel-level benchmark...
```

- [ ] **Step 2: Update qwen36 spec status**

In `docs/superpowers/specs/2026-04-16-qwen36-mixed-precision-pathway.md`, replace:

Old:

```markdown
**Date:** 2026-04-16
**Status:** Design approved, pending implementation
```

New:

```markdown
**Date:** 2026-04-16
**Status:** DONE — superseded by `2026-04-20-research-reality-program-design.md`. Mixed-precision Qwen3.6-35B-A3B converted and validated; serves as B-γ baseline in the new program.
```

- [ ] **Step 3: Commit both updates**

```bash
cd /Users/anthonylui/QwenCoderLocal
git add docs/superpowers/specs/2026-04-13-mojo-vs-mlx-kernel-benchmark-design.md docs/superpowers/specs/2026-04-16-qwen36-mixed-precision-pathway.md
git commit -m "$(cat <<'EOF'
docs: mark prior specs DONE — superseded by research-reality program

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 16: δ note audit — read rejection notes + cross-reference history

**Why:** Per spec §0.4. Determine if spec-decode rejection was global or narrow.

**Files:** Read-only research; produces input for Task 17.

- [ ] **Step 1: Read the rejection text in full context**

```bash
cd /Users/anthonylui/QwenCoderLocal
sed -n '500,540p' docs/FROM_ATTENTION_TO_CONSUMER_HARDWARE.md
```

Expected: the surrounding paragraphs around lines 515-520. Note the model size and config under which the rejection was claimed.

- [ ] **Step 2: Find when rejection text was added**

```bash
cd /Users/anthonylui/QwenCoderLocal
git log --all -p --follow -- docs/FROM_ATTENTION_TO_CONSUMER_HARDWARE.md \
  | grep -B 2 -A 30 "speculative" | head -100
```

Expected: shows commit SHA + author + date for the introduction of the rejection rationale.

- [ ] **Step 3: Search for any prior measurement evidence**

```bash
cd /Users/anthonylui/QwenCoderLocal
grep -r -l "speculative" docs/ scripts/ mlx-lm/ 2>/dev/null | head -20
```

Expected: list of files mentioning speculative decoding. Read each briefly to find: (a) actual measurements, (b) model+config under test, (c) hit-rate threshold derivations.

- [ ] **Step 4: Capture findings for Task 17**

Take freeform notes (in your head or a scratch file) to feed Task 17. Specifically answer:

1. Was the <70% hit-rate finding measured, or projected?
2. What model + offload config was the rejection scoped to?
3. Are smaller (resident) models even subject to the same routing-chaos argument?
4. Is there a narrow case (e.g., draft model that uses only shared experts) the rejection didn't address?

---

### Task 17: Write δ verdict memo

**Files:**
- Create: `docs/superpowers/notes/2026-04-21-spec-decode-rejection-audit.md` (date matches actual execution day)

- [ ] **Step 1: Write the memo**

```bash
cat > /Users/anthonylui/QwenCoderLocal/docs/superpowers/notes/2026-04-21-spec-decode-rejection-audit.md <<'EOF'
# Spec-Decode Rejection Audit (δ memo)

**Date:** [actual execution date]
**Audit scope:** `docs/FROM_ATTENTION_TO_CONSUMER_HARDWARE.md:515-520` and related history.
**Triggered by:** Phase 0 §0.4 of `2026-04-20-research-reality-program-design.md`.

## Original rejection text

> [Verbatim quote of the rejection paragraph from the doc]

## Provenance

- Introduced in commit [SHA] on [date] by [author]
- Surrounding context at time of introduction: [one-line summary]

## Evidence basis

- Measured: [yes/no, with model + config]
- Projected: [if measurement absent, what argument]
- 70% threshold derivation: [cite source or note "asserted, not derived"]

## Scope analysis

- Rejection covers: [model classes, offload state, draft model assumptions]
- Rejection does NOT address: [enumerate gaps]

## Verdict

One of:
- `kill_permanently` — rejection holds for all cases of interest
- `reopen_for_narrow_case_X` — describe the case and why it's worth a future sprint
- `inconclusive_need_measurement` — rejection rests on assertion; future sprint should measure

[Selected verdict with one paragraph rationale]

## Future-work implication

If `reopen_for_narrow_case_X`: rough scope estimate (engineering days, success criteria, decision authority).
If `kill_permanently`: this memo closes B-δ permanently.
If `inconclusive_need_measurement`: scope a measurement-only sprint (no implementation).
EOF
# Edit the file with actual content from Task 16
```

- [ ] **Step 2: Commit memo**

```bash
cd /Users/anthonylui/QwenCoderLocal
git add docs/superpowers/notes/2026-04-21-spec-decode-rejection-audit.md
git commit -m "$(cat <<'EOF'
docs: add δ memo — audit of spec-decode rejection for offload pathway

Closes Phase 0 §0.4. Verdict feeds future-sprint scoping.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 18: Invariance harness — write failing unit tests for hash + diff helpers

**Why:** TDD. The harness has two pure helpers (`output_hash`, `_diff_first_chars`) and a CLI orchestrator. Test the helpers first.

**Files:**
- Create: `tests/test_invariance_check.py`

- [ ] **Step 1: Decide test location**

The repo has tests in `mlx-lm/tests/` for mlx-lm-internal code, and `scripts/` lacks a tests directory. New top-level `tests/` is fine for `scripts/`-level tests.

```bash
cd /Users/anthonylui/QwenCoderLocal
mkdir -p tests
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_invariance_check.py`:

```python
"""Unit tests for scripts/invariance_check.py helpers."""
import sys
from pathlib import Path

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import invariance_check as ic


class TestOutputHash:
    def test_identical_strings_hash_equal(self):
        assert ic.output_hash("hello world") == ic.output_hash("hello world")

    def test_different_strings_hash_differ(self):
        assert ic.output_hash("hello") != ic.output_hash("world")

    def test_empty_string_hashes(self):
        assert isinstance(ic.output_hash(""), str)
        assert len(ic.output_hash("")) == 64  # sha256 hex


class TestDiffFirstChars:
    def test_identical_inputs_returns_empty_dict(self):
        assert ic._diff_first_chars("abc", "abc") == {}

    def test_returns_first_diff_index(self):
        result = ic._diff_first_chars("abcXdef", "abcYdef")
        assert result["first_diff_index"] == 3

    def test_includes_context_around_diff(self):
        result = ic._diff_first_chars("0123456789Xabcdefghij", "0123456789Yabcdefghij")
        assert "0123456789X" in result["context_a"]
        assert "0123456789Y" in result["context_b"]

    def test_length_mismatch_when_one_is_prefix(self):
        result = ic._diff_first_chars("abc", "abcdef")
        assert result["length_mismatch"] == [3, 6]
```

- [ ] **Step 3: Run tests; verify they fail with ImportError**

```bash
cd /Users/anthonylui/QwenCoderLocal
python -m pytest tests/test_invariance_check.py -v
```

Expected: `ImportError: No module named 'invariance_check'` (because the module doesn't exist yet).

---

### Task 19: Implement invariance_check.py helpers

**Files:**
- Create: `scripts/invariance_check.py`

- [ ] **Step 1: Write the module skeleton with helpers**

Create `scripts/invariance_check.py`:

```python
#!/usr/bin/env python3
"""Bit-exactness invariance check for mlx_lm.generate.

Runs the same prompt twice through `mlx_lm.generate` with greedy decoding and
a fixed seed, then asserts the two outputs are identical. Catches
nondeterminism that async expert prefetch (Phase 1) could introduce.

Usage:
    # Single config
    python scripts/invariance_check.py --model <path-or-hf-id> [--prompt TEXT] [--max-tokens N]

    # Standard three-config suite (per Phase 0.6 spec)
    python scripts/invariance_check.py --config-suite

Exits 0 if all configs produce identical output across both runs; 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


CANONICAL_PROMPT = "The capital of France is"
DEFAULT_MAX_TOKENS = 50

# Three standard configs covered by the harness (per spec §0.6)
STANDARD_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "dense_llama_3_2_3b",
        "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "extra_args": [],
    },
    {
        "name": "moe_qwen36_35b_a3b",
        # Update this path when the local mixed-precision checkpoint is finalized
        "model": str(Path.home() / "Models" / "qwen3.6-35b-a3b-mixed"),
        "extra_args": [],
    },
    {
        "name": "moe_qwen36_35b_a3b_offload",
        "model": str(Path.home() / "Models" / "qwen3.6-35b-a3b-mixed"),
        "extra_args": ["--expert-offload", "--max-resident-experts", "2048"],
    },
]


def output_hash(text: str) -> str:
    """Return sha256 hex digest of text. Used for fast equality check."""
    return hashlib.sha256(text.encode()).hexdigest()


def _diff_first_chars(a: str, b: str) -> dict[str, Any]:
    """Find first character index where two strings differ.

    Returns:
        Empty dict if strings identical.
        Dict with first_diff_index + context if they differ at some position.
        Dict with length_mismatch if one is a prefix of the other.
    """
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            return {
                "first_diff_index": i,
                "context_a": a[max(0, i - 20) : i + 20],
                "context_b": b[max(0, i - 20) : i + 20],
            }
    if len(a) != len(b):
        return {"length_mismatch": [len(a), len(b)]}
    return {}


def run_generation(
    model: str, prompt: str, max_tokens: int, extra_args: list[str]
) -> str:
    """Run `mlx_lm.generate` with deterministic settings; return stdout."""
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.generate",
        "--model", model,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temp", "0.0",
        "--seed", "42",
        *extra_args,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return proc.stdout


def check_invariance(
    model: str,
    prompt: str,
    max_tokens: int,
    extra_args: list[str],
    config_name: str,
) -> tuple[bool, dict[str, Any]]:
    """Run generation twice; return (passed, diagnostics)."""
    out1 = run_generation(model, prompt, max_tokens, extra_args)
    out2 = run_generation(model, prompt, max_tokens, extra_args)
    h1 = output_hash(out1)
    h2 = output_hash(out2)
    passed = h1 == h2
    diag: dict[str, Any] = {
        "config": config_name,
        "passed": passed,
        "hash_run_1": h1,
        "hash_run_2": h2,
        "len_run_1": len(out1),
        "len_run_2": len(out2),
    }
    if not passed:
        diag["diff_chars"] = _diff_first_chars(out1, out2)
    return passed, diag


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", help="Model path (HF id or local path)")
    parser.add_argument("--prompt", default=CANONICAL_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--extra-args",
        nargs="*",
        default=[],
        help="Extra args passed through to mlx_lm.generate",
    )
    parser.add_argument(
        "--config-suite",
        action="store_true",
        help="Run all three standard configs from STANDARD_CONFIGS",
    )
    parser.add_argument("--json", help="Path to write JSON results")
    args = parser.parse_args()

    if args.config_suite:
        results = []
        all_passed = True
        for config in STANDARD_CONFIGS:
            print(f"=== {config['name']} ===")
            passed, diag = check_invariance(
                config["model"],
                args.prompt,
                args.max_tokens,
                config["extra_args"],
                config["name"],
            )
            results.append(diag)
            all_passed = all_passed and passed
            print(f"  {'PASS' if passed else 'FAIL'}")
            if not passed:
                print(f"  diff: {diag.get('diff_chars')}")
        if args.json:
            Path(args.json).write_text(json.dumps(results, indent=2))
        sys.exit(0 if all_passed else 1)
    else:
        if not args.model:
            parser.error("--model required when not using --config-suite")
        passed, diag = check_invariance(
            args.model,
            args.prompt,
            args.max_tokens,
            args.extra_args,
            args.model,
        )
        print(json.dumps(diag, indent=2))
        if args.json:
            Path(args.json).write_text(json.dumps(diag, indent=2))
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make script executable**

```bash
chmod +x /Users/anthonylui/QwenCoderLocal/scripts/invariance_check.py
```

- [ ] **Step 3: Run tests; verify they pass**

```bash
cd /Users/anthonylui/QwenCoderLocal
python -m pytest tests/test_invariance_check.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 4: Commit harness + tests**

```bash
cd /Users/anthonylui/QwenCoderLocal
git add scripts/invariance_check.py tests/test_invariance_check.py
git commit -m "$(cat <<'EOF'
feat: add invariance_check.py — bit-exactness gate for mlx_lm.generate

Runs the same prompt twice through mlx_lm.generate with greedy decoding +
fixed seed; asserts identical output. Catches nondeterminism that async
expert prefetch (Phase 1) could introduce.

Six unit tests cover the pure-function helpers (output_hash, _diff_first_chars).
Standard three-config suite covers dense (Llama 3.2 3B), MoE (Qwen3.6-35B-A3B),
and MoE+offload per Phase 0 §0.6 of the research-reality program spec.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 20: Smoke-run invariance harness on dense model

**Why:** Validate the harness end-to-end with the smallest config before the full suite.

- [ ] **Step 1: Run dense config in single-config mode**

```bash
cd /Users/anthonylui/QwenCoderLocal
python scripts/invariance_check.py \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --max-tokens 20 \
  2>&1 | tee /tmp/invariance-dense-smoke.log
```

Expected: prints JSON diagnostics with `"passed": true` and identical hashes. Exit 0.

If `passed: false` on a vanilla model with greedy decoding: STOP — there's a deeper reproducibility issue in mlx_lm.generate that needs investigation before continuing.

- [ ] **Step 2: Spot-check the JSON shape**

```bash
cd /Users/anthonylui/QwenCoderLocal
python scripts/invariance_check.py \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --max-tokens 10 \
  --json /tmp/invariance-test.json
cat /tmp/invariance-test.json
```

Expected: file contains `passed`, `hash_run_1`, `hash_run_2`, `len_run_1`, `len_run_2`.

---

### Task 21: Run invariance harness on MoE config (Qwen3.6-35B-A3B, no offload)

- [ ] **Step 1: Verify Qwen3.6-35B-A3B mixed checkpoint exists**

```bash
ls -la ~/Models/qwen3.6-35b-a3b-mixed/ 2>&1 | head -10
```

Expected: directory exists with `config.json`, `model.safetensors.index.json`, and weight shards.

If checkpoint missing: pause Phase 0 — this checkpoint is from the qwen36 spec marked DONE in Task 15. If it's not on disk, the qwen36 work is not actually DONE. Surface to user before proceeding.

- [ ] **Step 2: Run invariance check on the MoE config**

```bash
cd /Users/anthonylui/QwenCoderLocal
python scripts/invariance_check.py \
  --model ~/Models/qwen3.6-35b-a3b-mixed \
  --max-tokens 30 \
  2>&1 | tee /tmp/invariance-moe-noffload.log
```

Expected: passes, exit 0.

If fails on no-offload MoE: there's nondeterminism in MoE routing or KV cache handling. STOP and investigate — this is exactly the bug class the harness is designed to catch.

---

### Task 22: Run invariance harness on MoE+offload config

- [ ] **Step 1: Run the offload config**

```bash
cd /Users/anthonylui/QwenCoderLocal
python scripts/invariance_check.py \
  --model ~/Models/qwen3.6-35b-a3b-mixed \
  --max-tokens 30 \
  --extra-args --expert-offload --max-resident-experts 2048 \
  2>&1 | tee /tmp/invariance-moe-offload.log
```

Expected: passes, exit 0. (Offload itself shouldn't introduce nondeterminism — async prefetch is what would, and that doesn't exist yet.)

If fails: there's pre-existing nondeterminism in expert offload that Phase 1 needs to address before adding async prefetch on top.

---

### Task 23: Run the full --config-suite and capture baseline

- [ ] **Step 1: Run the full suite with JSON output**

```bash
cd /Users/anthonylui/QwenCoderLocal
mkdir -p artifacts/phase-0
python scripts/invariance_check.py \
  --config-suite \
  --json artifacts/phase-0/invariance-baseline.json \
  2>&1 | tee artifacts/phase-0/invariance-baseline.log
```

Expected: prints PASS for each of three configs, exits 0, writes JSON.

- [ ] **Step 2: Commit baseline artifact**

```bash
cd /Users/anthonylui/QwenCoderLocal
git add artifacts/phase-0/invariance-baseline.json artifacts/phase-0/invariance-baseline.log
git commit -m "$(cat <<'EOF'
data: capture Phase 0 invariance harness baseline (all 3 configs PASS)

Establishes pre-Phase-1 reference. Any future failure on these three
configs indicates a regression that must be addressed before the
offending change ships.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 24: Wire invariance check into CI (optional smoke target)

**Why:** Phase 1 work needs the gate enforced automatically, not just runnable manually.

**Files:**
- Modify or create: `Makefile` or equivalent CI config

- [ ] **Step 1: Identify existing test/CI entry point**

```bash
cd /Users/anthonylui/QwenCoderLocal
ls Makefile mlx-lm/Makefile deer-flow/Makefile 2>/dev/null
cat mlx-lm/Makefile 2>/dev/null | head -20
```

Identify which Makefile (if any) is the canonical entry point. If none exists at repo root, **skip this task** and document the manual invocation in the next task's exit criteria.

- [ ] **Step 2 (only if a top-level Makefile exists or you create one): Add invariance target**

Add to Makefile:

```makefile
.PHONY: invariance-check
invariance-check:
	python scripts/invariance_check.py --config-suite --json artifacts/phase-0/invariance-latest.json
```

- [ ] **Step 3: Document how to run it in `scripts/invariance_check.py` docstring or a README note**

If you skipped Step 2, instead add a short note to `docs/superpowers/notes/`:

```markdown
# Invariance Check — Manual Run Convention

Until CI integration is added (deferred), Phase 1 PRs that touch expert offload
or scheduling MUST manually run:

    python scripts/invariance_check.py --config-suite

…and confirm exit 0 before merge. Failure blocks the PR.
```

Commit whichever you did:

```bash
cd /Users/anthonylui/QwenCoderLocal
git add [Makefile or note]
git commit -m "$(cat <<'EOF'
chore: wire invariance harness into Phase 1 PR gate

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 25: Phase 0 exit verification

**Why:** Confirm every spec exit criterion is met before declaring Phase 0 done.

- [ ] **Step 1: Verify Mojo bench on main**

```bash
cd /Users/anthonylui/QwenCoderLocal
ls mojo-bench/results/*.json | wc -l
test -f mojo-bench/pixi.toml && echo "pixi.toml present" || echo "MISSING"
```

Expected: `21` JSONs and pixi.toml present.

- [ ] **Step 2: Verify worktrees clean**

```bash
cd /Users/anthonylui/QwenCoderLocal
git worktree list
```

Expected: only main + any KEEP — REVIEW LATER worktrees from Task 7. No abandoned, no worktree-mojo-vs-mlx-benchmark, no DIRTY ones (unless held).

- [ ] **Step 3: Verify δ memo committed**

```bash
cd /Users/anthonylui/QwenCoderLocal
ls docs/superpowers/notes/2026-04-2*-spec-decode-rejection-audit.md
git log --oneline -- docs/superpowers/notes/ | head -3
```

Expected: file exists; appears in git log.

- [ ] **Step 4: Verify prior specs marked DONE**

```bash
cd /Users/anthonylui/QwenCoderLocal
grep -h "^\*\*Status" docs/superpowers/specs/2026-04-13-*.md docs/superpowers/specs/2026-04-16-*.md
```

Expected: both lines start with `**Status**: DONE — superseded`.

- [ ] **Step 5: Verify invariance harness passes**

```bash
cd /Users/anthonylui/QwenCoderLocal
python -m pytest tests/test_invariance_check.py -v
python scripts/invariance_check.py --config-suite
echo "exit code: $?"
```

Expected: pytest all pass; harness exits 0.

- [ ] **Step 6: Write Phase 0 completion summary**

Create `docs/superpowers/notes/phase-0-completion.md`:

```markdown
# Phase 0 — Completion Summary

**Completed:** YYYY-MM-DD
**Spec:** docs/superpowers/specs/2026-04-20-research-reality-program-design.md §Phase 0
**Plan:** docs/superpowers/plans/2026-04-20-phase-0-cleanup-and-invariance.md

## Deliverables
- [x] mojo-bench/ on main, 21 fresh JSONs, reproducible from clean checkout
- [x] Worktrees clean: [N] deleted, [M] held for later, log at docs/superpowers/recommendations/worktree-cleanup-log.md
- [x] δ memo verdict: [verdict from Task 17]
- [x] Prior specs marked DONE
- [x] Invariance harness passes on three standard configs (baseline at artifacts/phase-0/invariance-baseline.json)

## Findings worth carrying into Phase 1
- [Anything Tasks 1-24 surfaced that affects 1-α design]

## Phase 1 unblocked.
```

Commit:

```bash
cd /Users/anthonylui/QwenCoderLocal
git add docs/superpowers/notes/phase-0-completion.md
git commit -m "$(cat <<'EOF'
docs: Phase 0 complete — clean foundation for Phase 1

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 7: Notify user**

> "Phase 0 complete. Summary at `docs/superpowers/notes/phase-0-completion.md`. Ready to proceed with Plan 2 (Phase 1 — speed) when you are."

---

## Self-Review Checklist (run before declaring plan ready)

Skim each spec section and verify task coverage:

| Spec section | Implementing task(s) |
|--------------|---------------------|
| §0.1 Land Mojo bench | Tasks 1-5 |
| §0.2 Delete abandoned worktrees | Tasks 6-8 |
| §0.3 Triage 2 dirty worktrees | Tasks 9-14 |
| §0.4 δ note audit | Tasks 16-17 |
| §0.5 Mark prior specs DONE | Task 15 |
| §0.6 Invariance harness | Tasks 18-23 |
| Phase 0 exit criteria | Task 25 |

Plus:
- Task 24 (CI wiring) is a nice-to-have not strictly in spec; included to operationalize §0.6 enforcement.

Type/name consistency check:
- `output_hash`, `_diff_first_chars`, `run_generation`, `check_invariance`, `STANDARD_CONFIGS`, `CANONICAL_PROMPT`, `DEFAULT_MAX_TOKENS` — used consistently across Tasks 18-19 ✓
- File paths are absolute and consistent ✓
- Branch names match `git worktree list` output ✓
