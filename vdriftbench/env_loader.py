"""Minimal `.env` file loader for API keys/endpoints (e.g. a DeepSeek target
model's key), so they don't have to be `export`ed by hand every session.

Deliberately dependency-free (no `python-dotenv`) — this server has had
enough pip/mirror friction already (see HANDOFF.md 部署史) that a tiny
hand-rolled parser is more robust than adding another package. Only sets a
variable if it is not already present in `os.environ`, so real shell
`export`s (or a CI secret manager) always take priority over the file.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _parse_env_file(path: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.lower().startswith("export "):
                line = line[len("export "):].strip()
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            if key:
                pairs[key] = value
    return pairs


def load_dotenv_if_present(*paths: str) -> list[str]:
    """Load KEY=VALUE pairs from the first existing file among `paths`
    (defaults to `.env` in the current working directory, then the
    repository root's `.env`). Returns the list of variable names that were
    actually applied (never the values — this return value is safe to log).
    Variables already present in `os.environ` are left untouched."""

    candidates = list(paths) or [".env", str(_REPO_ROOT / ".env")]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        applied = []
        for key, value in _parse_env_file(path).items():
            if key not in os.environ:
                os.environ[key] = value
                applied.append(key)
        return applied
    return []
