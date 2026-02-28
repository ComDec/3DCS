"""Tests for the common metrics module."""

from __future__ import annotations

import numpy as np
import pytest

from three_dbench.common.metrics import (
    cka_rbf,
    condensed_length,
    distance_correlation,
    isotonic_r2,
    kendall_correlation,
    n_from_condensed_length,
    spearman_correlation,
    to_condensed,
    to_square,
)


def test_condensed_length_values():
    assert condensed_length(0) == 0
    assert condensed_length(1) == 0
    assert condensed_length(5) == 10
    assert condensed_length(10) == 45


def test_condensed_length_negative():
    with pytest.raises(ValueError):
        condensed_length(-1)


def test_n_from_condensed_roundtrip():
    for n in [3, 5, 10, 50]:
        m = condensed_length(n)
        assert n_from_condensed_length(m) == n


def test_n_from_condensed_invalid():
    with pytest.raises(ValueError):
        n_from_condensed_length(7)  # not a valid condensed length


def test_to_square_and_back(rng):
    D = rng.random((5, 5))
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    v = to_condensed(D)
    D2 = to_square(v)
    np.testing.assert_allclose(D, D2, atol=1e-10)


def test_spearman_perfect():
    D = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
    rho = spearman_correlation(D, D)
    assert abs(rho - 1.0) < 1e-10


def test_kendall_perfect():
    D = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
    tau = kendall_correlation(D, D)
    assert abs(tau - 1.0) < 1e-10


def test_isotonic_r2_perfect():
    D = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
    r2 = isotonic_r2(D, D)
    assert r2 > 0.99


def test_distance_correlation_identical():
    D = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
    dc = distance_correlation(D, D)
    assert dc > 0.99


def test_cka_rbf_identical():
    D = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
    cka = cka_rbf(D, D)
    assert cka > 0.99


def test_spearman_anti_correlated():
    D1 = np.array([[0, 1, 3], [1, 0, 2], [3, 2, 0]], dtype=float)
    D2 = np.array([[0, 3, 1], [3, 0, 2], [1, 2, 0]], dtype=float)
    rho = spearman_correlation(D1, D2)
    assert rho < 0
