"""Chirality: ``metric_version`` paper vs v2 definitions (synthetic, fast)."""

from __future__ import annotations

import numpy as np
import pytest

from three_dbench.chirality.evaluation import (
    ChiralitySettings,
    davies_bouldin_centroid,
    davies_bouldin_from_D,
    evaluate_en_separation_from_counts,
    evaluate_molecule,
    nn1_leave_one_out_from_D,
    nn1_leave_one_out_v2,
)


def test_invalid_metric_version_raises(rng):
    with pytest.raises(ValueError, match="metric_version"):
        evaluate_en_separation_from_counts({"M::en0": 3, "M::en1": 3}, rng.standard_normal((6, 3)), metric_version="v3")


def test_default_metric_version_is_paper(chirality_key_counts, rng):
    X = rng.standard_normal((sum(chirality_key_counts.values()), 5)).astype(np.float32)
    r_default, s_default = evaluate_en_separation_from_counts(chirality_key_counts, X)
    r_paper, s_paper = evaluate_en_separation_from_counts(chirality_key_counts, X, metric_version="paper")
    assert s_default == s_paper


def test_nn1_ties_paper_vs_v2():
    # Point 0 has two exactly tied nearest neighbours: point 1 (same class) and point 2 (other class).
    D = np.array(
        [
            [0.0, 1.0, 1.0, 5.0],
            [1.0, 0.0, 3.0, 5.0],
            [1.0, 3.0, 0.0, 2.0],
            [5.0, 5.0, 2.0, 0.0],
        ]
    )
    y = np.array(["a", "a", "b", "b"], dtype=object)
    # paper: np.argmin breaks the tie towards the lowest index (point 1, correct) -> 3/4
    assert nn1_leave_one_out_from_D(D, y) == pytest.approx(0.75)
    # v2: point 0 scores 1/2 (one of its two tied neighbours shares its label)
    assert nn1_leave_one_out_v2(D, y) == pytest.approx((0.5 + 1.0 + 0.0 + 1.0) / 4)
    # swapping rows 1 and 2 flips the paper tie-break (-> 2/4) but leaves v2 unchanged
    p = np.array([0, 2, 1, 3])
    Dp, yp = D[np.ix_(p, p)], y[p]
    assert nn1_leave_one_out_from_D(Dp, yp) == pytest.approx(0.5)
    assert nn1_leave_one_out_v2(Dp, yp) == pytest.approx(nn1_leave_one_out_v2(D, y))


def test_nn1_v2_excludes_points_without_same_class_partner():
    D = np.array([[0.0, 1.0, 4.0], [1.0, 0.0, 4.0], [4.0, 4.0, 0.0]])
    y = np.array(["a", "a", "b"], dtype=object)
    assert nn1_leave_one_out_from_D(D, y) == pytest.approx(2 / 3)  # point 2 can never be right
    assert nn1_leave_one_out_v2(D, y) == pytest.approx(1.0)
    y_all_single = np.array(["a", "b", "c"], dtype=object)
    assert nn1_leave_one_out_from_D(D, y_all_single) == 0.0
    assert np.isnan(nn1_leave_one_out_v2(D, y_all_single))


def test_dbi_centroid_matches_sklearn(rng):
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    X = np.vstack([rng.standard_normal((6, 3)) + 4, rng.standard_normal((5, 3)) - 4, rng.standard_normal((7, 3))])
    y = np.array(["0"] * 6 + ["1"] * 5 + ["2"] * 7, dtype=object)
    ref = sklearn_metrics.davies_bouldin_score(X, y)
    assert davies_bouldin_centroid(X, y) == pytest.approx(ref, rel=1e-9)
    # the paper (medoid) approximation differs in general
    from three_dbench.chirality.evaluation import euclidean_distances

    D = euclidean_distances(X)
    assert davies_bouldin_from_D(D, y, "continuous") != pytest.approx(ref, rel=1e-6)


def test_v2_all_singleton_molecule_is_nan_not_zero(rng):
    counts = {f"M::en{i}": 1 for i in range(4)}  # four stereoisomers, one conformer each
    X = rng.standard_normal((4, 3))
    rows_p, _ = evaluate_en_separation_from_counts(counts, X, metric_version="paper")
    rows_v, _ = evaluate_en_separation_from_counts(counts, X, metric_version="v2")
    assert rows_p[0]["NN1_acc"] == 0.0
    assert np.isnan(rows_v[0]["NN1_acc"])
    assert np.isnan(rows_v[0]["DBI"])
    assert np.isnan(rows_p[0]["ESA_AUC"]) and np.isnan(rows_v[0]["ESA_AUC"])


def test_hopkins_population_single_stereoisomer(rng):
    X = rng.standard_normal((12, 4))
    y = np.array(["0"] * 12, dtype=object)
    paper = evaluate_molecule("M", X, y, "continuous", ChiralitySettings(metric_version="paper"))
    v2 = evaluate_molecule("M", X, y, "continuous", ChiralitySettings(metric_version="v2"))
    assert paper["mode"] == v2["mode"] == "skip_single_en"
    assert np.isfinite(paper["hopkins"])
    assert np.isnan(v2["hopkins"])
    small = evaluate_molecule(
        "S", X[:9], np.array(["0"] * 5 + ["1"] * 4, dtype=object), "continuous", ChiralitySettings()
    )
    assert np.isnan(small["hopkins"])  # n < 10 in both versions


def test_v2_euclidean_matches_paper_on_shared_definitions(rng):
    counts = {"A::en0": 7, "A::en1": 6, "B::en0": 5, "B::en1": 8}
    X = rng.standard_normal((26, 5)).astype(np.float32)
    rp, _ = evaluate_en_separation_from_counts(counts, X, metric_version="paper")
    rv, _ = evaluate_en_separation_from_counts(counts, X, metric_version="v2")
    for a, b in zip(rp, rv):
        for k in ("ESA_AUC", "sil_sup", "hopkins", "clarity"):
            assert a[k] == b[k]
        # best-k silhouette on precomputed Euclidean D equals the silhouette on X up to rounding
        assert b["sil_unsup"] == pytest.approx(a["sil_unsup"], abs=1e-5)
        assert b["k_unsup"] == a["k_unsup"]


def test_v2_cosine_uses_unit_sphere_geometry(rng):
    counts = {"A::en0": 12, "A::en1": 12}
    X = rng.standard_normal((24, 5)).astype(np.float32)
    scales = rng.uniform(0.5, 20.0, size=(24, 1)).astype(np.float32)
    # v2 + cosine is invariant to per-row rescaling; paper + cosine is not (Hopkins/KMeans on raw X)
    r1, s1 = evaluate_en_separation_from_counts(counts, X, distance="cosine", metric_version="v2")
    r2, s2 = evaluate_en_separation_from_counts(counts, X * scales, distance="cosine", metric_version="v2")
    for k in ("ESA_AUC_mean", "NN1_acc_mean", "sil_sup_mean", "hopkins_mean", "DBI_mean", "sil_unsup_mean"):
        assert s1[k] == pytest.approx(s2[k], abs=1e-4)
    _, p1 = evaluate_en_separation_from_counts(counts, X, distance="cosine", metric_version="paper")
    _, p2 = evaluate_en_separation_from_counts(counts, X * scales, distance="cosine", metric_version="paper")
    assert p1["hopkins_mean"] != pytest.approx(p2["hopkins_mean"], abs=1e-4)
