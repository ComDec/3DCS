"""Tests for embedding loading utilities."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from three_dbench.embeddings import EmbeddingArray, load_embeddings, load_embeddings_dict, load_embeddings_dir


def test_load_npz(tmp_path, rng):
    arr = rng.standard_normal((10, 4)).astype(np.float32)
    path = tmp_path / "test.npz"
    np.savez(path, arr_0=arr)
    emb = load_embeddings(path)
    assert isinstance(emb, EmbeddingArray)
    assert emb.kind == "vector"
    np.testing.assert_array_equal(emb.array, arr)


def test_load_npz_with_key(tmp_path, rng):
    arr = rng.standard_normal((10, 4)).astype(np.float32)
    path = tmp_path / "test.npz"
    np.savez(path, my_key=arr)
    emb = load_embeddings(path, key="my_key")
    np.testing.assert_array_equal(emb.array, arr)


def test_load_npy(tmp_path, rng):
    arr = rng.standard_normal((10, 4)).astype(np.float32)
    path = tmp_path / "test.npy"
    np.save(path, arr)
    emb = load_embeddings(path)
    np.testing.assert_array_equal(emb.array, arr)


def test_load_pkl(tmp_path, rng):
    arr = rng.standard_normal((10, 4)).astype(np.float32)
    path = tmp_path / "test.pkl"
    with path.open("wb") as f:
        pickle.dump({"my_key": arr}, f)
    emb = load_embeddings(path, key="my_key")
    np.testing.assert_array_equal(emb.array, arr)


def test_load_embeddings_dict_npz(tmp_path, rng):
    a = rng.standard_normal((5, 4)).astype(np.float32)
    b = rng.standard_normal((3, 4)).astype(np.float32)
    path = tmp_path / "test.npz"
    np.savez(path, mol_a=a, mol_b=b)
    d = load_embeddings_dict(path)
    assert set(d.keys()) == {"mol_a", "mol_b"}
    np.testing.assert_array_equal(d["mol_a"], a)


def test_load_embeddings_dir(tmp_path, rng):
    for name in ["rmd17_aspirin", "rmd17_ethanol"]:
        arr = rng.standard_normal((20, 4)).astype(np.float32)
        np.savez(tmp_path / f"{name}.npz", arr_0=arr)
    d = load_embeddings_dir(tmp_path, file_glob="rmd17_*.npz", key="arr_0")
    assert set(d.keys()) == {"rmd17_aspirin", "rmd17_ethanol"}
    assert d["rmd17_aspirin"].shape == (20, 4)


def test_unsupported_format(tmp_path):
    path = tmp_path / "test.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="Unsupported"):
        load_embeddings(path)
