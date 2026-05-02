# System State Check — 2026-05-02

## Context

Machine was fully rebooted before this check.

## Checks

- `sysctl vm.swapusage`: `total = 0.00M  used = 0.00M  free = 0.00M`.
- Pre-test process scan found no stuck MLX/pytest jobs; only the `ps | rg` command itself matched.

## Focused NPT8 Test

Command:

```bash
PYTHONPATH=mlx-lm python -m pytest \
  mlx-lm/tests/test_fused_npt8.py \
  mlx-lm/tests/test_fused_npt8_tiled.py -q
```

Result:

- Python aborted during pytest collection while importing `mlx.core` from `mlx-lm/tests/test_fused_npt8.py:17`.
- The process remained in `UEs` state after abort:

```text
Python -m pytest mlx-lm/tests/test_fused_npt8.py mlx-lm/tests/test_fused_npt8_tiled.py -q
```

- `kill -9` did not clear the stuck process.

## Conclusion

This is not stale state from the prior boot. The focused NPT8 test currently reproduces a Metal/MLX abort on a clean reboot and leaves a stuck process. Do not collect new NPT8/tiled benchmark artifacts from this boot. Open a focused stability/import task before continuing decode-kernel measurement.
