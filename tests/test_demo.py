"""Tests for the demo script using bundled fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "examples" / "fixtures"


@pytest.mark.slow
def test_demo_chirality():
    """Test chirality demo end-to-end (requires HF download)."""
    from examples.demo import demo_chirality

    demo_chirality(output_dir=None)


@pytest.mark.slow
def test_demo_trajectory():
    """Test trajectory demo end-to-end (requires HF download)."""
    from examples.demo import demo_trajectory

    demo_trajectory(output_dir=None)


def test_fixtures_exist():
    """Verify that all expected fixture files are present."""
    assert (FIXTURES / "chirality_gemnet_demo.npz").exists()
    assert (FIXTURES / "traj_gemnet_rmd17_aspirin.npz").exists()
    assert (FIXTURES / "traj_gemnet_rmd17_ethanol.npz").exists()


def test_fixture_shapes():
    """Verify fixture embedding shapes."""
    import numpy as np

    with np.load(FIXTURES / "chirality_gemnet_demo.npz") as d:
        assert d["gemnet"].shape == (500, 128)

    with np.load(FIXTURES / "traj_gemnet_rmd17_aspirin.npz") as d:
        assert d["gemnet"].shape == (5000, 128)

    with np.load(FIXTURES / "traj_gemnet_rmd17_ethanol.npz") as d:
        assert d["gemnet"].shape == (5000, 128)
