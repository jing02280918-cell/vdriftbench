"""Ensures the repo root (and therefore the `vdriftbench` package) is
importable when running `pytest` without `pip install -e .` first."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
