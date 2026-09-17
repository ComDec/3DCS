"""Shared fixtures for 3DBench tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# `tests/test_demo.py` imports `examples.demo`, which lives in the checkout, not in the installed
# package. Make it importable regardless of the working directory pytest is started from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_embeddings(rng):
    """20 embeddings of dimension 8."""
    return rng.standard_normal((20, 8)).astype(np.float32)


@pytest.fixture
def chirality_key_counts():
    """Minimal chirality key-count mapping: 2 molecules, 2 enantiomers each."""
    return {
        "MOL_A::en0": 5,
        "MOL_A::en1": 5,
        "MOL_B::en0": 4,
        "MOL_B::en1": 6,
    }
