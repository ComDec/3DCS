"""Window schemes and molecule selection for the trajectory benchmark."""

from __future__ import annotations

import numpy as np
import pytest

from three_dbench.traj.evaluation import DEFAULT_MOL_TYPES
from three_dbench.traj.protocol import (
    LEGACY_TRAJ_LEN,
    draw_window_starts,
    order_molecules,
    resolve_molecules,
)

RMD17_FRAMES = {
    "rmd17_aspirin": 100_000,
    "rmd17_azobenzene": 99_988,
    "rmd17_benzene": 100_000,
    "rmd17_ethanol": 100_000,
    "rmd17_malonaldehyde": 100_000,
    "rmd17_naphthalene": 100_000,
    "rmd17_paracetamol": 100_000,
    "rmd17_salicylic": 100_000,
    "rmd17_toluene": 100_000,
    "rmd17_uracil": 100_000,
}


def test_legacy_starts_are_published_windows():
    starts = draw_window_starts(RMD17_FRAMES, window_scheme="legacy", n_samples=100, window=2000, random_seed=2025)
    reference = np.random.default_rng(2025).integers(0, LEGACY_TRAJ_LEN - 2000, size=100, endpoint=False)
    assert list(reference[:3]) == [43852, 97456, 97273]  # backed-up energy_metrics_out windows
    for mol, s in starts.items():
        np.testing.assert_array_equal(s, reference, err_msg=mol)
    # max start 97,636: no window is truncated for azobenzene (99,988 frames)
    assert int(reference.max()) + 2000 <= RMD17_FRAMES["rmd17_azobenzene"]


def test_legacy_starts_are_deterministic_and_subset_independent():
    full = draw_window_starts(RMD17_FRAMES, window_scheme="legacy", n_samples=7, window=50, random_seed=11)
    again = draw_window_starts(RMD17_FRAMES, window_scheme="legacy", n_samples=7, window=50, random_seed=11)
    subset = draw_window_starts(
        {"rmd17_uracil": 100_000}, window_scheme="legacy", n_samples=7, window=50, random_seed=11
    )
    for mol in full:
        np.testing.assert_array_equal(full[mol], again[mol])
    np.testing.assert_array_equal(full["rmd17_uracil"], subset["rmd17_uracil"])


def test_shared_scheme_matches_0_1_0_cli():
    n_samples, window, seed = 5, 2000, 2025
    starts = draw_window_starts(
        RMD17_FRAMES, window_scheme="shared", n_samples=n_samples, window=window, random_seed=seed
    )
    rng = np.random.default_rng(seed)  # the 0.1.0 CLI loop (benchmarks/trajectory.py)
    for mol, n_frames in RMD17_FRAMES.items():
        max_start = n_frames - window
        expected = rng.integers(0, max_start, size=min(n_samples, max_start), endpoint=False)
        np.testing.assert_array_equal(starts[mol], expected)
    assert not np.array_equal(starts["rmd17_aspirin"], starts["rmd17_azobenzene"])


def test_legacy_window_beyond_trajectory_raises():
    with pytest.raises(ValueError, match="exceeds"):
        draw_window_starts({"rmd17_x": 500}, window_scheme="legacy", n_samples=10, window=100, random_seed=0)
    starts = draw_window_starts(
        {"rmd17_x": 500}, window_scheme="legacy", n_samples=10, window=100, random_seed=0, legacy_traj_len=500
    )
    assert int(starts["rmd17_x"].max()) + 100 <= 500


def test_unknown_scheme_raises():
    with pytest.raises(ValueError):
        draw_window_starts(RMD17_FRAMES, window_scheme="random")


def test_order_and_resolve_molecules():
    dataset_order = sorted(DEFAULT_MOL_TYPES)
    assert order_molecules(dataset_order, "legacy") == list(DEFAULT_MOL_TYPES)
    assert order_molecules(dataset_order, "shared") == dataset_order
    assert resolve_molecules(["aspirin", "rmd17_uracil", "aspirin"], dataset_order) == [
        "rmd17_aspirin",
        "rmd17_uracil",
    ]
    assert resolve_molecules(None, dataset_order) == dataset_order
    with pytest.raises(KeyError):
        resolve_molecules(["caffeine"], dataset_order)
