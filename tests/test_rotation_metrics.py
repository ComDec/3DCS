"""Geometry metric definitions (paper / v2 / legacy presets)."""

from __future__ import annotations

import numpy as np
import pytest

from three_dbench.common.metrics import cka_rbf, isotonic_r2, local_isometry_error
from three_dbench.rotation.metrics import (
    PRESETS,
    angular_smoothness_variant,
    cka_rbf_paper_run,
    compute_geometry_metrics,
    isotonic_r2_reference_on_model,
    local_isometry_error_knn,
    resolve_metric_spec,
)


def _dist(X):
    X = np.asarray(X, dtype=float)
    return np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))


@pytest.fixture
def toy(rng):
    Z = rng.standard_normal((9, 4))
    D = _dist(rng.standard_normal((9, 3)))
    return D, _dist(Z), Z


def test_lie_include_self_matches_released(toy):
    D, Delta, _ = toy
    assert local_isometry_error_knn(D, Delta, k=10, include_self=True) == pytest.approx(
        local_isometry_error(D, Delta, k=10)
    )


def test_lie_exclude_self_uses_other_conformers():
    # 1D points: nearest other neighbours are well defined
    x = np.array([[0.0], [1.0], [3.0], [6.0]])
    D = _dist(x)
    # identical geometry -> zero error for any k when self is excluded
    assert local_isometry_error_knn(D, D.copy(), k=2, include_self=False) == pytest.approx(0.0, abs=1e-9)
    # a distortion of one far pair changes the exclude-self score
    Delta = D.copy()
    Delta[0, 1] = Delta[1, 0] = 2.0
    assert local_isometry_error_knn(D, Delta, k=2, include_self=False) > 0.0


def test_as_variants():
    deg = np.array([170.0, -170.0, 0.0, 90.0])  # sorted: -170, 0, 90, 170 ; wrap step 170 -> -170 = 20 deg
    Z = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [6.0, 0.0]])
    Delta = _dist(Z)
    lin = angular_smoothness_variant(deg, variant="median_dz_linear", Z=Z, Delta=Delta)
    order = np.argsort(np.deg2rad(deg))
    steps = [
        np.linalg.norm(Z[order[i + 1]] - Z[order[i]]) / abs(np.deg2rad(deg[order[i + 1]] - deg[order[i]]))
        for i in range(3)
    ]
    assert lin == pytest.approx(np.median(steps))
    circ = angular_smoothness_variant(deg, variant="median_delta_circular", Delta=Delta)
    half = angular_smoothness_variant(deg, variant="median_halfdelta_circular", Delta=Delta)
    assert half == pytest.approx(circ / 2)
    mean_circ = angular_smoothness_variant(deg, variant="mean_delta_circular", Delta=Delta)
    assert np.isfinite(mean_circ)
    with pytest.raises(ValueError):
        angular_smoothness_variant(deg, variant="nope", Delta=Delta)


def test_isotonic_direction():
    D = _dist(np.arange(5, dtype=float)[:, None])
    const = np.ones_like(D)
    np.fill_diagonal(const, 0.0)
    # released direction (Delta = f(D)) rewards a constant representation
    assert isotonic_r2(D, const) == pytest.approx(1.0)
    # appendix direction (D = f(Delta)) does not
    assert np.isnan(isotonic_r2_reference_on_model(D, const, guard=True))
    assert isotonic_r2_reference_on_model(D, const, guard=False) == pytest.approx(0.0, abs=1e-6)
    two = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert isotonic_r2_reference_on_model(two, two, guard=False) == pytest.approx(1.0)


def test_cka_variants_identical_inputs(toy):
    D, _, _ = toy
    assert cka_rbf_paper_run(D, D) == pytest.approx(1.0)
    assert cka_rbf(D, D) == pytest.approx(1.0)


def test_presets_and_overrides():
    assert PRESETS["v2"].lie_k == 3 and PRESETS["v2"].lie_include_self is False
    assert PRESETS["paper"].kendall_min_conformers == 11
    spec = resolve_metric_spec("paper", lie_k=3, lie_include_self=False, as_variant="median_dz_linear")
    assert (spec.lie_k, spec.lie_include_self, spec.as_variant) == (3, False, "median_dz_linear")
    assert spec.name == "paper+custom"
    with pytest.raises(ValueError):
        resolve_metric_spec("v3")


def test_compute_geometry_metrics_keys(toy, rng):
    D, Delta, Z = toy
    deg = rng.uniform(-180, 180, size=D.shape[0])
    paper = compute_geometry_metrics(D, Delta, torsion_deg=deg, spec=PRESETS["paper"], Z=Z)
    assert set(paper) == {"A1_spearman", "A2_kendall", "G_cka_rbf", "H_LIE@k", "J_isotonic_R2", "torsion_sp", "AS"}
    assert np.isnan(paper["A2_kendall"])  # paper runs: Kendall only for >= 11 conformers
    v2 = compute_geometry_metrics(D, Delta, torsion_deg=deg, spec=PRESETS["v2"], Z=Z)
    assert np.isfinite(v2["A2_kendall"])
    legacy = compute_geometry_metrics(D, Delta, torsion_deg=deg, spec=PRESETS["legacy"], Z=Z)
    assert "C_mantel_r" in legacy and "E_triplet_OP" in legacy
