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


# --------------------------------------------------------------------------- #
# the one --dataset syntax (baselines/README.md "Common to all of them")
# --------------------------------------------------------------------------- #
def test_parse_dataset_spec_hub_forms():
    assert common.parse_dataset_spec("hf:EscheWang/3dcs:chirality") == ("hub", "EscheWang/3dcs", "chirality")
    assert common.parse_dataset_spec("hf:EscheWang/3dcs") == ("hub", "EscheWang/3dcs", "chirality")
    assert common.parse_dataset_spec("EscheWang/3dcs") == ("hub", "EscheWang/3dcs", "chirality")
    assert common.parse_dataset_spec("hf:EscheWang/3dcs:rotation") == ("hub", "EscheWang/3dcs", "rotation")
    assert common.parse_dataset_spec("hf:EscheWang/3dcs", hf_config="traj")[2] == "traj"


def test_parse_dataset_spec_local_forms(tmp_path):
    directory = tmp_path / "chirality"
    directory.mkdir()
    pickle_path = tmp_path / "mols.pkl"
    pickle_path.write_bytes(b"")
    lmdb_path = tmp_path / "shard.lmdb"
    lmdb_path.write_bytes(b"")

    assert common.parse_dataset_spec(f"hfdisk:{directory}") == ("disk", str(directory), None)
    assert common.parse_dataset_spec(str(directory)) == ("disk", str(directory), None)
    assert common.parse_dataset_spec(str(pickle_path)) == ("pickle", str(pickle_path), None)
    assert common.parse_dataset_spec(f"lmdb:{lmdb_path}") == ("lmdb", str(lmdb_path), None)
    assert common.parse_dataset_spec(str(lmdb_path)) == ("lmdb", str(lmdb_path), None)


def test_parse_dataset_spec_rejects_unusable_values(tmp_path):
    with pytest.raises(FileNotFoundError):
        common.parse_dataset_spec(str(tmp_path / "missing.pkl"))
    with pytest.raises(FileNotFoundError):
        common.parse_dataset_spec("./not/here")
    with pytest.raises(ValueError):
        common.parse_dataset_spec("")
    with pytest.raises(ValueError):
        common.parse_dataset_spec("hf:")


# --------------------------------------------------------------------------- #
# reading the published dataset layout
# --------------------------------------------------------------------------- #
def _mol_block(smiles: str, seed: int) -> str:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    return Chem.MolToMolBlock(mol)


@pytest.fixture(scope="module")
def tiny_chirality_dir(tmp_path_factory):
    """A `save_to_disk` directory shaped like the `chirality` config, rows out of order.

    Three rows holding 2, 1 and 3 conformers: 6 in total, whose published row order is the
    order of ascending `offset`, not the order the rows happen to be stored in.
    """
    datasets = pytest.importorskip("datasets")
    pytest.importorskip("rdkit")
    rows = [
        {"key": "c", "offset": 3, "mol_blocks": [_mol_block(s, i) for i, s in enumerate(("CCO", "CCC", "CCN"), 30)]},
        {"key": "a", "offset": 0, "mol_blocks": [_mol_block(s, i) for i, s in enumerate(("C[C@H](N)O", "CO"), 10)]},
        {"key": "b", "offset": 2, "mol_blocks": [_mol_block("CCCl", 20)]},
    ]
    directory = tmp_path_factory.mktemp("hf") / "chirality"
    datasets.Dataset.from_list(rows).save_to_disk(str(directory))
    return directory


@pytest.fixture(scope="module")
def tiny_chirality_pickle(tiny_chirality_dir, tmp_path_factory):
    """The same six conformers as a pickle, in the same row order."""
    path = tmp_path_factory.mktemp("pkl") / "mols.pkl"
    with path.open("wb") as handle:
        pickle.dump(common.load_conformers(f"hfdisk:{tiny_chirality_dir}"), handle)
    return path


def _atom_counts(mols):
    return [mol.GetNumAtoms() for mol in mols]


#: Atom counts of the six conformers of ``tiny_chirality_dir``, in ascending ``offset``:
#: row "a" (offset 0) C[C@H](N)O, CO; row "b" (offset 2) CCCl; row "c" (offset 3) CCO, CCC, CCN.
TINY_ATOM_COUNTS = [11, 6, 8, 9, 11, 10]


def test_load_conformers_orders_rows_by_offset(tiny_chirality_dir):
    """The rows are stored c, a, b; reading them gives the conformers in offset order."""
    mols = common.load_conformers(f"hfdisk:{tiny_chirality_dir}")
    assert len(mols) == 6
    assert _atom_counts(mols) == TINY_ATOM_COUNTS
    plain = common.load_conformers(str(tiny_chirality_dir))
    assert _atom_counts(plain) == TINY_ATOM_COUNTS


def test_load_conformers_limit_and_start(tiny_chirality_dir):
    all_mols = common.load_conformers(f"hfdisk:{tiny_chirality_dir}")
    assert _atom_counts(common.load_conformers(f"hfdisk:{tiny_chirality_dir}", limit=2)) == _atom_counts(all_mols[:2])
    assert _atom_counts(common.load_conformers(f"hfdisk:{tiny_chirality_dir}", start=4)) == _atom_counts(all_mols[4:])
    sliced = common.load_conformers(f"hfdisk:{tiny_chirality_dir}", start=2, limit=2)
    assert _atom_counts(sliced) == _atom_counts(all_mols[2:4])


def test_load_conformers_rejects_a_dataset_without_mol_blocks(tmp_path):
    datasets = pytest.importorskip("datasets")
    directory = tmp_path / "wrong"
    datasets.Dataset.from_list([{"key": "a", "offset": 0, "smiles": "CCO"}]).save_to_disk(str(directory))
    with pytest.raises(KeyError, match="mol_blocks"):
        common.load_conformers(f"hfdisk:{directory}")


def test_load_conformers_rejects_rows_that_are_not_contiguous(tmp_path):
    datasets = pytest.importorskip("datasets")
    directory = tmp_path / "gap"
    rows = [
        {"key": "a", "offset": 0, "mol_blocks": [_mol_block("CCO", 1)]},
        {"key": "b", "offset": 7, "mol_blocks": [_mol_block("CCC", 2)]},
    ]
    datasets.Dataset.from_list(rows).save_to_disk(str(directory))
    with pytest.raises(ValueError, match="contiguous"):
        common.load_conformers(f"hfdisk:{directory}")


# --------------------------------------------------------------------------- #
# every script takes the same --dataset values (baselines/README.md)
# --------------------------------------------------------------------------- #
def _script_module(model):
    """Import ``baselines/<model>/extract_chirality.py`` without its model dependencies."""
    name = f"baselines_{model}_extract"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, BASELINES_DIR / model / "extract_chirality.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # a module-level @dataclass looks itself up in sys.modules
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[name]
        raise
    return module


@pytest.mark.parametrize("model", MODELS)
def test_every_script_reads_every_input_form(model, tiny_chirality_dir, tiny_chirality_pickle):
    """`hfdisk:`, a plain directory and a pickle give the same conformers in every script."""
    load = _script_module(model).load_molecules
    reference = _atom_counts(common.load_conformers(f"hfdisk:{tiny_chirality_dir}"))
    for spec in (f"hfdisk:{tiny_chirality_dir}", str(tiny_chirality_dir), str(tiny_chirality_pickle)):
        assert _atom_counts(list(load(spec))) == reference, f"{model} disagrees on {spec}"
    assert _atom_counts(list(load(f"hfdisk:{tiny_chirality_dir}", limit=3))) == reference[:3]
    assert _atom_counts(list(load(f"hfdisk:{tiny_chirality_dir}", start=3))) == reference[3:]


@pytest.mark.parametrize("model", MODELS)
def test_every_script_resolves_the_hub_form_without_downloading(model, monkeypatch):
    """`--dataset hf:EscheWang/3dcs:chirality` reaches ``datasets.load_dataset`` in every script."""
    pytest.importorskip("datasets")
    seen = {}

    def fake_load_dataset(repo, name=None, split=None, revision=None):
        seen.update(repo=repo, name=name, split=split)
        raise RuntimeError("stop before the download")

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    with pytest.raises(RuntimeError, match="stop before the download"):
        list(_script_module(model).load_molecules("hf:EscheWang/3dcs:chirality"))
    assert seen == {"repo": "EscheWang/3dcs", "name": "chirality", "split": "train"}


def test_molae_reads_one_row_per_conformer(tiny_chirality_dir):
    """Regression: the Mol-AE HF path reads the `mol_blocks` list of every row, not one block."""
    molae = _script_module("molae")
    source = (BASELINES_DIR / "molae" / "extract_chirality.py").read_text(encoding="utf-8")
    assert "mol_block" not in source.replace("mol_blocks", "")
    mols = molae.load_molecules(f"hfdisk:{tiny_chirality_dir}")
    assert len(mols) == 6  # three rows, 2 + 1 + 3 conformers
    assert molae.load_molecules(str(tiny_chirality_dir), "train", "chirality", limit=2) is not None


# --------------------------------------------------------------------------- #
# --verify over part of the conformers
# --------------------------------------------------------------------------- #
def test_parse_row_selection_defaults():
    assert common.parse_row_selection(None, produced_rows=10, reference_rows=10) == (None, "full")
    indices, mode = common.parse_row_selection(None, produced_rows=3, reference_rows=10)
    assert mode == "prefix"
    np.testing.assert_array_equal(indices, [0, 1, 2])


def test_parse_row_selection_slices_and_files(tmp_path):
    indices, mode = common.parse_row_selection("2000:2003", produced_rows=3, reference_rows=52391)
    assert mode == "slice"
    np.testing.assert_array_equal(indices, [2000, 2001, 2002])
    indices, mode = common.parse_row_selection("5+", produced_rows=2, reference_rows=10)
    np.testing.assert_array_equal(indices, [5, 6])

    path = tmp_path / "rows.npy"
    np.save(path, np.array([7, 1, 4]))
    indices, mode = common.parse_row_selection(f"@{path}", produced_rows=3, reference_rows=10)
    assert mode == "file"
    np.testing.assert_array_equal(indices, [7, 1, 4])

    text = tmp_path / "rows.txt"
    text.write_text("0 2 4\n")
    indices, _ = common.parse_row_selection(f"@{text}", produced_rows=3, reference_rows=10)
    np.testing.assert_array_equal(indices, [0, 2, 4])


def test_parse_row_selection_errors(tmp_path):
    with pytest.raises(ValueError, match="full"):
        common.parse_row_selection("full", produced_rows=3, reference_rows=10)
    with pytest.raises(ValueError, match="more than"):
        common.parse_row_selection("prefix", produced_rows=11, reference_rows=10)
    with pytest.raises(ValueError, match="selects"):
        common.parse_row_selection("0:5", produced_rows=3, reference_rows=10)
    with pytest.raises(ValueError, match="outside"):
        common.parse_row_selection("8:11", produced_rows=3, reference_rows=10)
    with pytest.raises(ValueError, match="expected"):
        common.parse_row_selection("every second row", produced_rows=3, reference_rows=10)


def test_compare_vectors_over_a_row_subset(rng):
    reference = rng.standard_normal((20, 6)).astype(np.float32)
    produced = reference[:5] + np.float32(1e-6)
    result = common.compare_vectors(produced, reference, rows=np.arange(5), row_mode="prefix")
    assert result["shape_match"] is False  # the two files do not have the same number of rows
    assert result["compared_shape_match"] is True
    assert result["compared_rows"] == 5
    assert result["row_selection"] == "prefix"
    assert result["cosine_min"] > 0.999999
    assert result["max_abs_diff"] == pytest.approx(1e-6, rel=0.05)

    scattered = common.compare_vectors(reference[[7, 1, 4]], reference, rows=np.array([7, 1, 4]), row_mode="file")
    assert scattered["array_equal"] is True
    assert scattered["compared_rows"] == 3


def test_verify_compares_a_partial_output_with_the_prefix(tmp_path, rng, capsys):
    reference = rng.standard_normal((50, 6)).astype(np.float32)
    ref_path = tmp_path / "reference.npz"
    common.write_npz(ref_path, reference, key="arr_0")
    produced = tmp_path / "produced.npz"
    common.write_npz(produced, reference[:8] + np.float32(1e-6), key="arr_0")

    result = common.verify(produced, model="mace", reference=str(ref_path), produced_key="arr_0", reference_key="arr_0")
    assert result["compared_rows"] == 8
    assert result["row_selection"] == "prefix"
    assert result["compared_shape_match"] is True
    assert result["cosine_min"] > 0.999999
    assert "8 of the 50 rows" in capsys.readouterr().out


def test_verify_compares_a_slice_of_the_reference(tmp_path, rng):
    reference = rng.standard_normal((50, 6)).astype(np.float32)
    ref_path = tmp_path / "reference.npz"
    common.write_npz(ref_path, reference, key="arr_0")
    produced = tmp_path / "produced.npz"
    common.write_npz(produced, reference[20:30], key="arr_0")

    result = common.verify(
        produced,
        model="mace",
        reference=str(ref_path),
        produced_key="arr_0",
        reference_key="arr_0",
        rows="20:30",
    )
    assert result["row_selection"] == "slice"
    assert result["array_equal"] is True
    assert result["compared_rows"] == 10

    wrong = common.verify(
        produced, model="mace", reference=str(ref_path), produced_key="arr_0", reference_key="arr_0", rows="0:10"
    )
    assert wrong["array_equal"] is False


def test_compare_fingerprints_over_a_subset():
    from rdkit.DataStructs import ExplicitBitVect

    def bits(indices):
        vector = ExplicitBitVect(16)
        for index in indices:
            vector.SetBit(index)
        return vector

    reference = [bits([0]), bits([1]), bits([2]), bits([3])]
    result = common.compare_fingerprints(reference[:2], reference, rows=np.arange(2), row_mode="prefix")
    assert result["compared_shape_match"] is True
    assert result["identical"] == 2
    assert result["compared_rows"] == 2


# --------------------------------------------------------------------------- #
# the documented commands and environments
# --------------------------------------------------------------------------- #
def _readme_commands():
    """The `baselines/README.md` "One command per model" block, one command per model."""
    text = (BASELINES_DIR / "README.md").read_text(encoding="utf-8")
    block = text.split("## One command per model", 1)[1].split("```bash", 1)[1].split("```", 1)[0]
    commands = {}
    for raw in block.replace("\\\n", " ").splitlines():
        line = raw.strip()
        if not line.startswith("python baselines/"):
            continue
        tokens = line.split()
        model = tokens[1].split("/")[1]
        commands[model] = tokens
    return commands


@pytest.mark.parametrize("model", MODELS)
def test_readme_command_is_runnable(model):
    """Every flag of the README command exists in that script, and the input spec is the shared one."""
    commands = _readme_commands()
    assert model in commands, f"{model} has no command in baselines/README.md"
    tokens = commands[model]
    assert (REPO_ROOT / tokens[1]).is_file()
    assert "--dataset" in tokens
    assert tokens[tokens.index("--dataset") + 1] == "hf:EscheWang/3dcs:chirality"

    help_text = subprocess.run(
        [sys.executable, str(REPO_ROOT / tokens[1]), "--help"], capture_output=True, text=True, timeout=180
    ).stdout
    for token in tokens:
        if token.startswith("--"):
            assert token in help_text, f"{model}: {token} is not a flag of {tokens[1]}"


@pytest.mark.parametrize("model", MODELS)
def test_help_documents_the_shared_flags(model):
    script = BASELINES_DIR / model / "extract_chirality.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    for flag in ("--dataset", "--out", "--verify", "--verify-rows", "--limit", "--start"):
        assert flag in result.stdout, f"{model}: {flag} missing from --help"
    assert "hf:EscheWang/3dcs:chirality" in result.stdout


def test_fmg_requirements_are_installable_together():
    """`pip install -r` needs a tqdm floor that `datasets==4.0.0` accepts."""
    packaging_specifiers = pytest.importorskip("packaging.specifiers")
    text = (BASELINES_DIR / "fmg" / "requirements.txt").read_text(encoding="utf-8")
    assert "reproduce/embeddings" not in text  # the file lives at baselines/fmg/
    assert "baselines/fmg/extract_chirality.py" in text
    pins = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name = line.split("==")[0].split(">=")[0].split("<")[0].strip()
        pins[name] = line[len(name) :]
    assert pins["datasets"] == "==4.0.0"
    # datasets 4.0.0 requires tqdm>=4.66.3; the pin here has to allow it
    assert packaging_specifiers.SpecifierSet(pins["tqdm"]).contains("4.66.3")


def test_weight_retrieval_is_documented():
    fmg = (BASELINES_DIR / "fmg" / "ENVIRONMENT.md").read_text(encoding="utf-8")
    assert "drive.google.com/drive/folders/1XpOfCPRvPu22dSgbWgfGRF0Lul7ygdC7" in fmg
    assert "gdown" in fmg
    assert "f55ec38f2b6c20ad3a2e4e6287efb77af3901d46548449543bdbab33357d2afa" in fmg

    molae = (BASELINES_DIR / "molae" / "ENVIRONMENT.md").read_text(encoding="utf-8")
    assert "ZIP" in molae and "unzip" in molae
    assert "1NKObZCfE80GCLS9yJ7hqMGzjfGol4LLo" in molae
    assert "b4ca21a63799976fbf435a1c7275d5ef6e93a854cc7d90955dbaec50ef89b8c0" in molae


@pytest.mark.parametrize("model", MODELS)
def test_environment_states_the_input_precision(model):
    text = (BASELINES_DIR / model / "ENVIRONMENT.md").read_text(encoding="utf-8")
    assert "## Input precision" in text, f"{model}: no Input precision section"
    section = text.split("## Input precision", 1)[1].split("\n## ", 1)[0]
    assert "full float precision" in section
    assert "four decimals" in section
