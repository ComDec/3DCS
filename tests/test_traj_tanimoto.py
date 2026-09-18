"""Fingerprint (E3FP) inputs with Tanimoto distance in the trajectory benchmark."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from traj_helpers import random_fingerprints, save_energy_dataset, synthetic_embeddings, synthetic_energies

pytest.importorskip("rdkit")

from three_dbench.traj.evaluation import pairwise_tanimoto_fps_large  # noqa: E402
from three_dbench.traj.io import load_traj_embeddings, resolve_embedding  # noqa: E402
from three_dbench.traj.protocol import resolve_embed_metric, tanimoto_distances_dense, window_distances  # noqa: E402


def _dense(fps, n_bits):
    X = np.zeros((len(fps), n_bits), dtype=np.uint8)
    for i, fp in enumerate(fps):
        X[i, list(fp.GetOnBits())] = 1
    return X


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_dense_tanimoto_matches_rdkit(dtype):
    fps = random_fingerprints(60, 128, seed=0)
    ref = pairwise_tanimoto_fps_large(fps, block_size=16, out_mode="condensed", dtype_out=dtype, progress=False)
    dense = tanimoto_distances_dense(_dense(fps, 128), dtype_out=dtype)
    assert dense.dtype == ref.dtype
    # Pairs of two all-zero fingerprints are excluded: RDKit's similarity for them changed from 1.0
    # (<= 2025.09) to 0.0 (>= 2026.03). The published fingerprints contain no all-zero vector, so the
    # reproduction is unaffected; see test_dense_tanimoto_empty_pairs for the dense convention.
    on_bits = np.asarray([fp.GetNumOnBits() for fp in fps])
    i, j = np.triu_indices(len(fps), 1)
    comparable = (on_bits[i] > 0) | (on_bits[j] > 0)
    np.testing.assert_array_equal(dense[comparable], ref[comparable])


def test_dense_tanimoto_empty_pairs():
    """Two all-zero fingerprints have similarity 0 (distance 1) in the dense implementation."""
    X = np.zeros((3, 8), dtype=np.uint8)
    X[2, [1, 4]] = 1
    D = tanimoto_distances_dense(X, dtype_out=np.float32)
    assert D[0] == pytest.approx(1.0)  # (0, 1): both empty
    assert D[1] == pytest.approx(1.0)  # (0, 2): empty vs non-empty


def test_fingerprints_default_to_tanimoto():
    fps = random_fingerprints(5, 32, seed=1)
    assert resolve_embed_metric(None, fps) == "tanimoto"
    assert resolve_embed_metric(None, np.zeros((5, 4))) == "cosine"
    with pytest.raises(ValueError):
        resolve_embed_metric("cosine", fps)
    D = window_distances(fps, metric="tanimoto", metric_version="paper")
    assert D.dtype == np.float16 and D.size == 10


def test_trajectory_directory_with_fingerprint_pickles(tmp_path):
    from three_dbench.benchmarks import evaluate_trajectory_embeddings

    n = 150
    energies = {"rmd17_a": synthetic_energies(n, seed=2), "rmd17_b": synthetic_energies(n, seed=3)}
    dataset_dir = save_energy_dataset(tmp_path / "ds", energies)
    emb_dir = tmp_path / "e3fp"
    emb_dir.mkdir()
    for k, mol in enumerate(energies):
        with (emb_dir / f"{mol}.pkl").open("wb") as f:
            pickle.dump(random_fingerprints(n, 256, seed=10 + k, density=0.3), f)

    loaders = load_traj_embeddings(emb_dir)
    assert sorted(loaders) == ["rmd17_a", "rmd17_b"]
    assert isinstance(resolve_embedding(loaders["rmd17_a"]), list)

    details, summary = evaluate_trajectory_embeddings(
        dataset_dir=dataset_dir,
        embeddings_by_mol=loaders,
        output_dir=tmp_path / "out",
        model_name="E3FP",
        n_samples=2,
        window=50,
        legacy_traj_len=n,
        verbose=False,
    )
    assert set(details["mol_type"]) == {"rmd17_a", "rmd17_b"}
    ks = summary.set_index("metric").loc["KS", "mean"]
    assert np.isfinite(ks)
    import json

    config = json.loads((tmp_path / "out" / "config.json").read_text())
    assert config["metric_embed_resolved"] == {"rmd17_a": "tanimoto", "rmd17_b": "tanimoto"}


def test_directory_loader_handles_npz_keys(tmp_path):
    e = synthetic_energies(30, seed=4)
    np.savez(tmp_path / "rmd17_a.npz", embeddings=synthetic_embeddings(e, 4, seed=5), smiles=np.array(["C"] * 30))
    np.savez(tmp_path / "rmd17_b.npz", gemnet=synthetic_embeddings(e, 4, seed=6), other=np.zeros((30, 2)))
    loaders = load_traj_embeddings(tmp_path)
    assert resolve_embedding(loaders["rmd17_a"]).shape == (30, 4)  # only numeric 2-D array
    with pytest.raises(ValueError, match="embedding-key"):
        resolve_embedding(loaders["rmd17_b"])
    assert resolve_embedding(load_traj_embeddings(tmp_path, key="gemnet")["rmd17_b"]).shape == (30, 4)
