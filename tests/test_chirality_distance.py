"""Chirality: distance switch, best-k silhouette kmax, CLI options and dataset/embedding validation."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys

import numpy as np
import pytest

from three_dbench.chirality.evaluation import (
    distance_matrix_for_subset,
    evaluate_en_separation_from_counts,
    resolve_unsup_kmax,
)


def _scale_only_embeddings(rng, n_per_class=20, dim=6):
    """Both classes share one direction and differ only in norm (x1): separable under Euclidean only."""
    u = rng.standard_normal(dim)
    u /= np.linalg.norm(u)
    x0 = u + 0.05 * rng.standard_normal((n_per_class, dim))
    x1 = 10.0 * (u + 0.05 * rng.standard_normal((n_per_class, dim)))
    return np.vstack([x0, x1]).astype(np.float32)


def _tiny_dataset(tmp_path, counts):
    datasets = pytest.importorskip("datasets")
    rows, offset = [], 0
    for key, n in counts.items():
        mol_id, en = key.split("::en")
        rows.append(
            {"key": key, "mol_id": mol_id, "en_id": en, "n_conformers": n, "offset": offset, "mol_blocks": [""] * n}
        )
        offset += n
    ds = datasets.Dataset.from_list(rows)
    path = tmp_path / "hf_chirality"
    ds.save_to_disk(str(path))
    return path


# ----------------------------- distance switch -----------------------------
def test_distance_matrices(rng):
    X = rng.standard_normal((5, 4)).astype(np.float32)
    idx = np.arange(5)
    De = distance_matrix_for_subset(X, idx, "continuous", distance="euclidean")
    ref = np.linalg.norm(X[:, None, :].astype(np.float64) - X[None, :, :], axis=2)
    np.testing.assert_allclose(De, ref, atol=1e-5)

    Dc = distance_matrix_for_subset(X, idx, "continuous", distance="cosine")
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    ref_c = 1.0 - Xn @ Xn.T
    np.fill_diagonal(ref_c, 0.0)
    np.testing.assert_allclose(Dc, ref_c, atol=1e-6)
    assert np.all(np.diag(Dc) == 0.0)


def test_default_distance_is_euclidean(rng):
    counts = {"MOL::en0": 20, "MOL::en1": 20}
    emb = _scale_only_embeddings(rng)
    _, s_default = evaluate_en_separation_from_counts(counts, emb, unsup_kmax=4)
    _, s_euc = evaluate_en_separation_from_counts(counts, emb, unsup_kmax=4, distance="euclidean")
    assert s_default == s_euc


def test_distance_changes_supervised_metrics(rng):
    counts = {"MOL::en0": 20, "MOL::en1": 20}
    emb = _scale_only_embeddings(rng)
    _, s_euc = evaluate_en_separation_from_counts(counts, emb, unsup_kmax=4, distance="euclidean")
    _, s_cos = evaluate_en_separation_from_counts(counts, emb, unsup_kmax=4, distance="cosine")
    assert s_euc["NN1_acc_mean"] > 0.95
    assert s_euc["ESA_AUC_mean"] > 0.95
    # Directions are identically distributed in both classes, so cosine cannot separate them.
    assert s_cos["NN1_acc_mean"] < 0.75
    assert s_cos["ESA_AUC_mean"] < 0.65


def test_invalid_distance_raises(rng):
    with pytest.raises(ValueError, match="distance"):
        evaluate_en_separation_from_counts({"M::en0": 3, "M::en1": 3}, rng.standard_normal((6, 3)), distance="l1")


def test_fingerprints_ignore_distance():
    DataStructs = pytest.importorskip("rdkit.DataStructs")
    rng = np.random.default_rng(1)
    fps = []
    for cls in range(2):
        for _ in range(6):
            bv = DataStructs.ExplicitBitVect(64)
            on = rng.choice(32, size=20, replace=False) + 32 * cls
            for b in on:
                bv.SetBit(int(b))
            fps.append(bv)
    counts = {"FP::en0": 6, "FP::en1": 6}
    r_e, s_e = evaluate_en_separation_from_counts(counts, fps, distance="euclidean")
    r_c, s_c = evaluate_en_separation_from_counts(counts, fps, distance="cosine")
    assert r_e[0]["embedding_mode"] == "fingerprint"
    assert s_e == s_c
    assert s_e["ESA_AUC_mean"] == pytest.approx(1.0)


# ----------------------------- best-k silhouette kmax -----------------------------
def test_resolve_unsup_kmax():
    assert resolve_unsup_kmax(40, None) == 39
    assert resolve_unsup_kmax(40, 10) == 10
    assert resolve_unsup_kmax(5, 10) == 4
    with pytest.raises(ValueError):
        resolve_unsup_kmax(5, 1)


@pytest.mark.filterwarnings("ignore:Number of distinct clusters")
def test_unsup_kmax_default_is_unbounded():
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((12, 5)) * 50.0
    X = np.vstack([c + 0.01 * rng.standard_normal((3, 5)) for c in centers]).astype(np.float32)
    counts = {"MOL::en0": 18, "MOL::en1": 18}  # 12 tight clusters, labels unrelated to them
    rows_default, _ = evaluate_en_separation_from_counts(counts, X)
    rows_capped, _ = evaluate_en_separation_from_counts(counts, X, unsup_kmax=10)
    assert rows_default[0]["k_unsup"] == 12
    assert rows_capped[0]["k_unsup"] <= 10
    assert rows_default[0]["sil_unsup"] > rows_capped[0]["sil_unsup"]


def test_n_jobs_does_not_change_results(rng):
    counts = {}
    for m in range(5):
        counts[f"M{m}::en0"] = 4 + m
        counts[f"M{m}::en1"] = 3 + m
    total = sum(counts.values())
    X = rng.standard_normal((total, 6)).astype(np.float32)
    r1, s1 = evaluate_en_separation_from_counts(counts, X, n_jobs=1)
    r2, s2 = evaluate_en_separation_from_counts(counts, X, n_jobs=2)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    assert json.dumps(s1, sort_keys=True) == json.dumps(s2, sort_keys=True)


# ----------------------------- validation -----------------------------
def test_row_count_mismatch_message(rng):
    counts = {"M::en0": 3, "M::en1": 3}
    with pytest.raises(ValueError, match=r"Embedding rows \(5\) != total conformers in the dataset \(6"):
        evaluate_en_separation_from_counts(counts, rng.standard_normal((5, 4)))


def test_non_finite_and_bad_shape_raise(rng):
    counts = {"M::en0": 3, "M::en1": 3}
    X = rng.standard_normal((6, 4))
    X[2, 1] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        evaluate_en_separation_from_counts(counts, X)
    with pytest.raises(ValueError, match="2-D"):
        evaluate_en_separation_from_counts(counts, np.zeros(6))
    with pytest.raises(ValueError, match="Unrecognised embeddings"):
        evaluate_en_separation_from_counts(counts, [[0.0]] * 6)


def test_dataset_offset_and_duplicate_checks():
    from three_dbench.benchmarks.chirality import _key_counts_from_dataset

    ok = [
        {"key": "A::en0", "n_conformers": 2, "offset": 0},
        {"key": "A::en1", "n_conformers": 3, "offset": 2},
    ]
    assert dict(_key_counts_from_dataset(ok)) == {"A::en0": 2, "A::en1": 3}
    bad_offset = [dict(ok[0]), dict(ok[1], offset=0)]
    with pytest.raises(ValueError, match="offset"):
        _key_counts_from_dataset(bad_offset)
    dup = [dict(ok[0]), dict(ok[0], offset=2)]
    with pytest.raises(ValueError, match="Duplicate key"):
        _key_counts_from_dataset(dup)


def test_load_chirality_embeddings_pickle_dict(tmp_path, rng):
    from three_dbench.benchmarks.chirality import load_chirality_embeddings

    path = tmp_path / "fp.pkl"
    with path.open("wb") as f:
        pickle.dump({"fp_a": [1, 2, 3], "fp_b": [4, 5, 6]}, f)
    with pytest.raises(ValueError, match="fp_b"):
        load_chirality_embeddings(path)
    assert load_chirality_embeddings(path, key="fp_a").array == [1, 2, 3]

    npz = tmp_path / "e.npz"
    np.savez(npz, gemnet=rng.standard_normal((3, 2)))
    with pytest.raises(KeyError, match="gemnet"):
        load_chirality_embeddings(npz, key="arr_0")


# ----------------------------- benchmark entry point + CLI -----------------------------
def test_benchmark_writes_config(tmp_path, rng):
    from three_dbench.benchmarks.chirality import evaluate_chirality_embeddings

    counts = {"MOLA::en0": 5, "MOLA::en1": 5, "MOLB::en0": 4, "MOLB::en1": 3}
    ds_dir = _tiny_dataset(tmp_path, counts)
    X = rng.standard_normal((17, 4)).astype(np.float32)
    out = tmp_path / "out"
    _, summary = evaluate_chirality_embeddings(
        dataset_dir=ds_dir,
        embeddings=X,
        output_dir=out,
        model_name="toy",
        distance="cosine",
        metric_version="v2",
        unsup_kmax=3,
    )
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["distance"] == "cosine"
    assert cfg["metric_version"] == "v2"
    assert cfg["unsup_kmax"] == 3
    assert cfg["n_conformers"] == 17
    assert cfg["coverage"]["n_molecules"] == 2
    assert (out / "toy_per_molecule.json").exists()
    assert (out / "summary.csv").exists()
    assert summary["n_molecules"] == 2


def test_cli_chirality_options(tmp_path, rng):
    counts = {"MOLA::en0": 5, "MOLA::en1": 5}
    ds_dir = _tiny_dataset(tmp_path, counts)
    emb = tmp_path / "emb.npz"
    np.savez(emb, arr_0=rng.standard_normal((10, 4)).astype(np.float32))
    out = tmp_path / "cli_out"
    cmd = [
        sys.executable,
        "-m",
        "three_dbench",
        "evaluate",
        "chirality",
        "--dataset-dir",
        str(ds_dir),
        "--embeddings",
        str(emb),
        "--model-name",
        "toy",
        "--output-dir",
        str(out),
        "--distance",
        "cosine",
        "--metric-version",
        "v2",
        "--unsup-kmax",
        "n-1",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    cfg = json.loads((out / "config.json").read_text())
    assert (cfg["distance"], cfg["metric_version"], cfg["unsup_kmax"]) == ("cosine", "v2", "n-1")

    bad = subprocess.run(cmd[:-1] + ["1"], capture_output=True, text=True)
    assert bad.returncode != 0
    assert "--unsup-kmax" in bad.stderr


def test_cli_help_lists_chirality_options():
    res = subprocess.run(
        [sys.executable, "-m", "three_dbench", "evaluate", "--help"], capture_output=True, text=True, check=True
    )
    for flag in ("--distance", "--metric-version", "--unsup-kmax", "--n-jobs"):
        assert flag in res.stdout


def test_import_has_no_side_effects():
    code = (
        "import sys; import three_dbench.chirality.evaluation as e; "
        "print(int('three_dbench.utils.paths' in sys.modules), int('sklearn_extra' in sys.modules))"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert res.stdout.split() == ["0", "0"]
