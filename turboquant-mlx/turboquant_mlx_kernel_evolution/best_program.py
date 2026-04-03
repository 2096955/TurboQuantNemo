"""
Best-program slot — after a run, replace this file with the winning candidate or
keep delegating to `initial_program.py` (default).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_p = Path(__file__).resolve().parent / "initial_program.py"
_spec = importlib.util.spec_from_file_location("tq_initial_program", _p)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

asymmetric_attention_scores = _mod.asymmetric_attention_scores
