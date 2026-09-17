"""Shared path helpers for locating data and result directories.

Default data and result directories are resolved relative to ``$THREE_DBENCH_HOME`` when it is
set, otherwise relative to the current working directory (not relative to the installed package).
"""

from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> Path:
    """Return ``$THREE_DBENCH_HOME`` if set, else the current working directory."""
    env = os.environ.get("THREE_DBENCH_HOME")
    return Path(env).expanduser().resolve() if env else Path.cwd()


def get_data_root() -> Path:
    return get_project_root() / "data"


def get_results_root() -> Path:
    return get_project_root() / "results"


# Module-level constants, evaluated at import time (kept for backwards compatibility).
PROJECT_ROOT = get_project_root()
SRC_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"
HF_DATA_ROOT = DATA_ROOT / "hf"
EMBEDDINGS_ROOT = DATA_ROOT / "embeddings"


def ensure_dir(path: Path) -> Path:
    """Create *path* if it does not yet exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
