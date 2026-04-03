# Historical: TurboQuant mlx changes (repo root)

The following bullets were previously the entire contents of the repository root `README.md`. They are preserved here for traceability.

## Summary of Changes

- Fixed P0 issues where `TurboQuantKVCache` wasn't imported properly in `qwen3_next.py`, leading to immediate crashes.
- Fixed P0 issues where `__init__.py` exported nonexistent classes, resolving immediate ImportErrors.
- Fixed P0 issues where the test suite (`test_mlx_turboquant.py`) was broken by aligning it with the latest API changes.
- Fixed P1 issues regarding `codebook_dir` hardcoding by allowing it to fall back to an environment variable, `TURBOQUANT_CODEBOOK_DIR`, and dynamically resolving its location.
- Fixed P1 issues where `validate_real_kv.py` depended on `mlx_lm.models.mlx_turboquant` instead of the local `mlx_turboquant`.
- Fixed P1 issues concerning `TurboQuantKVCache.from_state()` being a stub by providing a legitimate reconstruction method for prefix caching.
- Fixed P1 issues where `TurboQuantKVCache.nbytes` consistently returning `0` by supplying a method to accurately tally cache memory usage.
- Resolved P2 issues by aligning `README.md` to reflect the updated metrics (accounting for associative optimization), eliminating discrepancies with the actual code structure.
- Removed P2 divergent `capture_kv.py` versus `validate_real_kv.py` divergence, standardizing on a single implementation format.
- Addressed P2 fragility inside `generate.py`'s `_eval_prompt_cache_state` by switching to `mlx.utils.tree_flatten`, allowing it to cleanly ignore array depth structure variations.
- Resolved P3 concerns relating to the hardcoded `bit_width=3` inside `cache.py:57` by introducing an environment variable `TURBOQUANT_BITS`.
- Fixed P3 `meta_state` tracking by allowing `TurboQuantKVCache` to preserve its structural state across prompt cache loading.
- Fixed P3 discrepancies by keeping both `turboquant-mlx/mlx_turboquant.py` and `mlx-lm/mlx_lm/models/mlx_turboquant.py` files physically linked via `ln -s` during development, eliminating dual maintenance problems.
