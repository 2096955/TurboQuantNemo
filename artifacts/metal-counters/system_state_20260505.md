# System State Check — 2026-05-05

## Context

Machine fully rebooted before this check. Phase 0 sanity gate from
`docs/superpowers/plans/2026-05-02-wrap-loose-ends-and-bandwidth-roadmap.md`.

This is the follow-up to `system_state_20260502.md`, which reproduced an MLX
import abort during pytest collection on a clean reboot.

## Step 0.1: Process / swap scan

- `sysctl vm.swapusage`: `total = 0.00M  used = 0.00M  free = 0.00M  (encrypted)`.
- `ps -axo pid,stat,command | grep -E '(mlx|pytest|benchmark_nvfp4|mlx_lm.generate)'`: no matches.
- No stuck `UEs` processes from prior session.

## Step 0.2: Minimal MLX import smoke

Command:

```bash
PYTHONPATH=mlx-lm python -c "import mlx.core as mx; print(mx.default_device())"
```

Result: `Device(gpu, 0)` — clean exit, no leftover process.

**PASS.** This is the regression that failed on 2026-05-02; it now succeeds.

## Step 0.3: Focused NPT8 + tiled tests

Command:

```bash
PYTHONPATH=mlx-lm /opt/homebrew/opt/python@3.11/libexec/bin/python -m pytest \
  mlx-lm/tests/test_fused_npt8.py \
  mlx-lm/tests/test_fused_npt8_tiled.py -q
```

Result: `19 passed, 2 warnings in 3.32s`.

Process scan after run: no stuck processes.

**PASS.** This is the regression that aborted on 2026-05-02; it now succeeds.

### Note on python interpreter selection

A first attempt via `python -m pytest …` from the orchestrator's Bash tool
returned only `Pytest: No tests collected` (no error, no test list). Routing
through the absolute interpreter path
(`/opt/homebrew/opt/python@3.11/libexec/bin/python`) restored normal collection
and 19 passing tests. In a separate verification shell, `command -v python` and
`type python` both already resolve to the same absolute path, so this is
specific to the orchestrator's Bash environment, not a global system condition.
Recommendation: always invoke pytest via the absolute interpreter path until
the underlying shim is identified.

## Step 0.4: Git / artifact hygiene

- HEAD: `c2fc20dc4e156d3d3a828d5c699682536c3b2b2f`
- Branch: `main`, in sync with `origin/main`
- `git status --short` **before this artifact was written**: clean (no
  modifications, no untracked).
- After this artifact was written, `git status --short` shows
  `?? artifacts/metal-counters/system_state_20260505.md` — expected (the file
  is the artifact itself; flagging here for honesty).

## Independent verification (Codex, 2026-05-05)

Codex was asked to re-run Phase 0 from a separate session for adversarial
review. Result: **environment-limited PARTIAL**.

- Codex's sandbox could not enumerate Metal devices: the same import command
  raised `NSRangeException: index 0 beyond bounds for empty array` from
  `libmlx.dylib` during Metal device init — i.e., Metal returned an empty
  device list to that process. This blocks the import smoke from succeeding
  inside Codex.
- Codex's sandbox also blocks `ps` and `pgrep` (`sysmond service not found`),
  so Codex could not independently verify the "no stuck process" claim.
- Re-run of the import + pytest from the orchestrator host (same shell that
  produced the original artifact) reproduced cleanly: `Device(gpu, 0)`,
  `19 passed in 3.31s`, no stuck Python in `ps`.
- Codex correctly flagged the two documentation imprecisions above (git status
  framing and python shim being environment-specific). Both are now corrected.

Conclusion: the Metal/MLX failure mode Codex saw is a property of the Codex
sandbox, not a falsification of host Phase 0 status. Anyone re-verifying must
run from a shell that has Metal device access.

### Second pass: Codex static verification (2026-05-05)

Re-dispatched to Codex with a verification angle that does not require Metal.
Evidence bundle: `artifacts/metal-counters/phase0_verification_20260505/`
(captured `pytest_full.log`, `pytest_exit.txt`, `env.txt`, `git_log_npt8_files.txt`,
`source_hashes.txt`, `test_function_counts.txt`, `skip_markers.txt`,
`pytest.ini.copy`).

Codex confirmed all 7 static checks PASS:

1. `pytest_full.log` shows `collected 19 items`, all `PASSED`,
   `19 passed, 2 warnings in 2.59s`, exit 0; no `SKIPPED`/`FAILED`/`XFAIL`.
2. 19 `def test_*` functions confirmed (8 in `test_fused_npt8.py` +
   11 in `test_fused_npt8_tiled.py`); no top-level `skip`/`skipif`/`xfail`
   decorators; the 3 in-body `pytest.skip(...)` guards (codebook fallback) did
   not fire in this run.
3. Tests genuinely invoke NPT8 / tiled NPT8 kernel functions
   (`fused_attention_npt8`, `fused_attention_npt8_tiled`,
   `_fused_attention_npt8_tiled` spy with `call_count == 1` assertion).
4. `mlx-lm/pytest.ini` only sets `pythonpath = .` and `testpaths = tests` —
   no `addopts`, marker filters, or `--collect-only` defaults that would
   silently exclude tests.
5. Both test files and both NPT8 kernel files have NOT been modified between
   2026-05-02 (when MLX import aborted) and 2026-05-05 (now). Last commit
   touching them is `980985f` from 2026-04-27. **The same source that aborted
   on 2026-05-02 is what passes now — change is environmental (boot state),
   not code.**
6. `system_state_20260505.md` (this file) accurately describes the prior
   Codex sandbox-limited FAIL.
7. `env.txt` is normal: Python 3.11.13, pytest 8.4.1, mlx 0.31.1,
   `Device(gpu, 0)`.

Codex log: `logs/codex-delegations/phase0-static-verify-*.log`.

## Conclusion

Phase 0 sanity gate **PASSES** on the orchestrator host at HEAD `c2fc20d`.
MLX/Metal import is healthy and the focused NPT8 / tiled NPT8 tests run to
completion without abort or stuck processes. The pass is **host-specific**;
external verifiers in restricted Metal sandboxes will not reproduce it without
Metal device access.

Cleared to proceed with one of the three lanes (A: Nemotron RC, B: write-path
§3.4 ablation, C: Kimi default-cache residency sweep) per the plan's
"Recommended Next Action".
