"""Shared sys.path bootstrap so scripts run correctly whether or not the
package was installed with `pip install -e .`. Every script under
`scripts/` starts with `from _bootstrap import ensure_repo_on_path` and
calls it before importing `vdriftbench`.
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
