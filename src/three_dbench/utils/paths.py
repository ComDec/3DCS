"""Shared path helpers for locating data and result directories."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"
HF_DATA_ROOT = DATA_ROOT / "hf"


def ensure_dir(path: Path) -> Path:
    """Create *path* if it does not yet exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
