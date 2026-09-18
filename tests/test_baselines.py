"""Offline tests for the extraction scripts in ``baselines/``.

Only the parts that do not need a model are exercised: that every script has a working
``--help`` with the shared flags, and that the helpers in ``baselines/common.py`` behave.
Running a model is out of scope here; each one needs its own environment and a GPU.
"""

from __future__ import annotations

import importlib.util
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINES_DIR = REPO_ROOT / "baselines"
MODELS = ("e3fp", "fmg", "gemnet", "mace", "molae", "molspectra", "unimol")


def _load_common():
    """Import ``baselines/common.py`` the way the extraction scripts do."""
    spec = importlib.util.spec_from_file_location("baselines_common", BASELINES_DIR / "common.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = _load_common()


# --------------------------------------------------------------------------- #
# the scripts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model", MODELS)
def test_extract_script_help(model):
    """``--help`` works without any model dependency installed."""
    script = BASELINES_DIR / model / "extract_chirality.py"
    assert script.exists()
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    for flag in ("--dataset", "--out", "--verify"):
        assert flag in result.stdout, f"{model}: {flag} missing from --help"


@pytest.mark.parametrize("model", MODELS)
def test_model_directory_is_documented(model):
    assert (BASELINES_DIR / model / "ENVIRONMENT.md").is_file()
    assert f"]({model}/)" in (BASELINES_DIR / "README.md").read_text(encoding="utf-8")


def test_published_table_matches_the_package():
    """``common.PUBLISHED_CHIRALITY`` mirrors ``three_dbench.embeddings``."""
    from three_dbench.embeddings import published_embedding

    assert set(common.PUBLISHED_CHIRALITY) == set(MODELS)
    for model, (path, key) in common.PUBLISHED_CHIRALITY.items():
        entry = published_embedding("chirality", model)
        assert entry.path == path
        assert entry.key == key


# --------------------------------------------------------------------------- #
# files
# --------------------------------------------------------------------------- #
def test_sha256_file(tmp_path):
    import hashlib

    path = tmp_path / "blob.bin"
    payload = b"3dcs" * 1000
    path.write_bytes(payload)
    assert common.sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_write_npz_roundtrip(tmp_path):
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    for key, compress in (("arr_0", False), ("gemnet", False), ("embeddings", True)):
        path = tmp_path / f"{key}.npz"
        digest = common.write_npz(path, arr, key=key, compress=compress)
        assert digest == common.sha256_file(path)
        with np.load(path) as data:
            assert data.files == [key]
        np.testing.assert_array_equal(common.load_embedding(path, key), arr)
        # a single numeric array is found without naming its key
        np.testing.assert_array_equal(common.load_embedding(path), arr)


def test_load_embedding_key_errors(tmp_path):
    path = tmp_path / "two.npz"
    np.savez(path, a=np.zeros(3), b=np.ones(3))
    with pytest.raises(KeyError):
        common.load_embedding(path)
    with pytest.raises(KeyError):
        common.load_embedding(path, "c")
    np.testing.assert_array_equal(common.load_embedding(path, "b"), np.ones(3))


def test_load_embedding_pickle(tmp_path):
    path = tmp_path / "fp.pkl"
    with path.open("wb") as handle:
        pickle.dump({"e3fp": [1, 2, 3], "morgan": [4]}, handle)
    assert common.load_embedding(path, "e3fp") == [1, 2, 3]
    with pytest.raises(KeyError):
        common.load_embedding(path)


def test_load_embedding_unsupported(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("nope")
    with pytest.raises(ValueError):
        common.load_embedding(path)


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #
def test_compare_vectors_identical(rng):
    arr = rng.standard_normal((16, 8)).astype(np.float32)
    result = common.compare_vectors(arr, arr.copy())
    assert result["shape_match"] and result["array_equal"]
    assert result["max_abs_diff"] == 0.0
    assert result["cosine_min"] == pytest.approx(1.0)
    assert result["rows_cosine_below_1-1e-6"] == 0
    assert result["allclose_1e-5"]


def test_compare_vectors_perturbed(rng):
    arr = rng.standard_normal((16, 8)).astype(np.float32)
    other = arr + np.float32(1e-4)
    result = common.compare_vectors(arr, other)
    assert not result["array_equal"]
    assert result["max_abs_diff"] == pytest.approx(1e-4, rel=0.05)
    assert not result["allclose_1e-5"]
    assert result["allclose_1e-3"]
    assert result["cosine_min"] == pytest.approx(1.0, abs=1e-6)


def test_compare_vectors_shape_mismatch():
    result = common.compare_vectors(np.zeros((4, 3)), np.zeros((5, 3)))
    assert result["shape_match"] is False
    assert "cosine_min" not in result


def test_compare_vectors_zero_rows():
    arr = np.zeros((3, 4))
    arr[1] = 1.0
    result = common.compare_vectors(arr, arr.copy())
    assert result["zero_norm_rows"] == 2
    assert result["cosine_min"] == pytest.approx(1.0)


def test_compare_fingerprints():
    from rdkit.DataStructs import ExplicitBitVect

    def bits(indices):
        vector = ExplicitBitVect(16)
        for index in indices:
            vector.SetBit(index)
        return vector

    left = [bits([0, 1]), bits([2, 3]), bits([4])]
    right = [bits([0, 1]), bits([2, 3]), bits([5])]
    result = common.compare_fingerprints(left, right)
    assert result["shape_match"]
    assert result["identical"] == 2
    assert result["identical_fraction"] == pytest.approx(2 / 3)
    assert result["tanimoto_min"] == pytest.approx(0.0)
    assert result["first_mismatched_rows"] == [2]

    assert common.compare_fingerprints(left, right[:2])["shape_match"] is False


def test_format_report_covers_every_field():
    text = common.format_report({"a": 1.5, "b": "x", "c": True})
    assert "a" in text and "b" in text and "c" in text
    assert len(text.splitlines()) == 3


# --------------------------------------------------------------------------- #
# references and verification
# --------------------------------------------------------------------------- #
def test_resolve_reference_local_path():
    path, key = common.resolve_reference("some/where/file.npz")
    assert path == Path("some/where/file.npz")
    assert key is None


def test_resolve_reference_unknown_model():
    with pytest.raises(ValueError):
        common.resolve_reference("published", model="not-a-model")


def test_verify_against_a_local_file(tmp_path, rng, capsys):
    arr = rng.standard_normal((10, 6)).astype(np.float32)
    produced = tmp_path / "produced.npz"
    reference = tmp_path / "reference.npz"
    common.write_npz(produced, arr, key="arr_0")
    common.write_npz(reference, arr + np.float32(1e-6), key="arr_0")

    result = common.verify(
        produced,
        model="mace",
        reference=str(reference),
        produced_key="arr_0",
        reference_key="arr_0",
    )
    assert result["produced_sha256"] == common.sha256_file(produced)
    assert result["reference_sha256"] == common.sha256_file(reference)
    assert result["sha256_identical"] is False
    assert result["shape_match"] and not result["array_equal"]
    assert result["cosine_min"] > 0.999999
    assert "[verify]" in capsys.readouterr().out


def test_verify_detects_an_identical_file(tmp_path, rng):
    arr = rng.standard_normal((5, 4)).astype(np.float32)
    produced = tmp_path / "produced.npz"
    common.write_npz(produced, arr, key="arr_0")
    result = common.verify(produced, model="mace", reference=str(produced), reference_key="arr_0")
    assert result["sha256_identical"] is True
    assert result["array_equal"] is True


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def _ethanol():
    from rdkit import Chem

    return Chem.AddHs(Chem.MolFromSmiles("CCO"))


def test_load_conformers_from_a_list_pickle(tmp_path):
    path = tmp_path / "mols.pkl"
    with path.open("wb") as handle:
        pickle.dump([_ethanol(), _ethanol(), _ethanol()], handle)
    assert len(common.load_conformers(str(path))) == 3
    assert len(common.load_conformers(str(path), limit=2)) == 2


def test_load_conformers_from_a_dict_pickle(tmp_path):
    path = tmp_path / "mols.pkl"
    with path.open("wb") as handle:
        pickle.dump({"a": [_ethanol(), _ethanol()], "b": [_ethanol()]}, handle)
    assert len(common.load_conformers(str(path))) == 3


def test_load_conformers_rejects_other_payloads(tmp_path):
    path = tmp_path / "mols.pkl"
    with path.open("wb") as handle:
        pickle.dump(42, handle)
    with pytest.raises(TypeError):
        common.load_conformers(str(path))


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def test_print_versions(capsys):
    found = common.print_versions(("numpy", "definitely_not_a_module"))
    assert found["python"] == sys.version.split()[0]
    assert found["numpy"] == np.__version__
    assert "not importable" in found["definitely_not_a_module"]
    assert "[versions]" in capsys.readouterr().out
