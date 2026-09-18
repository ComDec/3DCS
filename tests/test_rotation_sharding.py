"""Shard-aware slicing for the rotation benchmark (synthetic data in the published HF layout)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("rdkit")
datasets = pytest.importorskip("datasets")

from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem, rdMolTransforms  # noqa: E402

from three_dbench.benchmarks.rotation import (  # noqa: E402
    ShardedEmbeddings,
    build_rotation_index,
    evaluate_rotation_embeddings,
    find_shard_files,
    select_rows,
    write_flat_embeddings,
)
from three_dbench.embeddings import EmbeddingArray  # noqa: E402

# Shards appear in string order, as in EscheWang/3dcs (rotation): 0, 1, 10, 2.
SHARD_ROWS = {0: [3, 4], 1: [5, 1, 3], 10: [4, 3], 2: [3, 2]}
SMILES = ["CCCC", "CCCO", "CCCN", "OCCO"]


def _conformer_blocks(smiles: str, n: int, seed: int) -> tuple[list[str], list[float]]:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    rng = np.random.default_rng(seed)
    angles = sorted(rng.uniform(-180.0, 180.0, size=n).tolist())
    blocks = []
    for ang in angles:
        m = Chem.Mol(mol)
        rdMolTransforms.SetDihedralDeg(m.GetConformer(), 0, 1, 2, 3, float(ang))
        blocks.append(Chem.MolToMolBlock(m))
    return blocks, angles


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(0)
    rows = []
    shard_arrays = {}
    truth = {}
    seed = 0
    for shard in (0, 1, 10, 2):
        local = 0
        chunks = []
        for i, n in enumerate(SHARD_ROWS[shard]):
            key = f"s{shard}_m{i}"
            blocks, degs = _conformer_blocks(SMILES[seed % len(SMILES)], n, seed)
            seed += 1
            Z = rng.standard_normal((n, 8)).astype(np.float32)
            chunks.append(Z)
            truth[key] = Z
            rows.append(
                {
                    "key": key,
                    "shard": shard,
                    "n_conformers": n,
                    "offset": local,
                    "mol_blocks": blocks,
                    "torsion_deg": degs,
                }
            )
            local += n
        shard_arrays[shard] = np.concatenate(chunks)
    ds = datasets.Dataset.from_list(rows)
    return ds, shard_arrays, truth


def _per_key(result):
    return result["per_key"].sort_values(["key", "space"]).reset_index(drop=True)


def test_index_detects_per_shard_offsets(synthetic):
    ds, _, _ = synthetic
    idx = build_rotation_index(ds)
    assert idx.attrs["offset_mode"] == "per-shard"
    assert idx["flat_offset"].tolist() == [0, 3, 7, 12, 13, 16, 20, 23, 26]
    assert idx["local_offset"].tolist() == [0, 3, 0, 5, 6, 0, 4, 0, 3]


def test_by_shard_matches_by_key(synthetic, tmp_path):
    ds, shard_arrays, truth = synthetic
    for shard, arr in shard_arrays.items():
        np.savez(tmp_path / f"rotation_conformers_{shard}.npz", gemnet=arr)
    assert sorted(find_shard_files(tmp_path)) == [0, 1, 2, 10]
    sharded = ShardedEmbeddings.from_directory(tmp_path)
    kw = dict(dataset=ds, metrics=["cosine", "euclidean"], metric_version="paper", min_conformers=2)
    by_shard = evaluate_rotation_embeddings(embeddings_by_shard=sharded, **kw)
    by_key = evaluate_rotation_embeddings(embeddings_by_key=truth, **kw)
    pd.testing.assert_frame_equal(_per_key(by_shard), _per_key(by_key))
    # the 1-conformer molecule is skipped
    assert "s1_m1" not in set(by_shard["per_key"]["key"])
    assert by_shard["config"]["selection"]["skipped_min_conformers"] == 1


def test_flat_dataset_order_matches_by_key(synthetic, tmp_path):
    ds, shard_arrays, truth = synthetic
    idx = build_rotation_index(ds)
    flat = write_flat_embeddings(idx, ShardedEmbeddings.from_arrays(shard_arrays), tmp_path / "flat.npy")
    manual = np.concatenate([truth[k] for k in ds["key"]])
    np.testing.assert_array_equal(np.asarray(flat), manual)
    kw = dict(dataset=ds, metrics=["cosine"], metric_version="v2")
    res_flat = evaluate_rotation_embeddings(embeddings=EmbeddingArray(array=flat, kind="vector"), **kw)
    res_key = evaluate_rotation_embeddings(embeddings_by_key=truth, **kw)
    pd.testing.assert_frame_equal(_per_key(res_flat), _per_key(res_key))


def test_naive_global_offsets_would_misalign(synthetic):
    """The released code sliced a flat array with the stored per-shard offset; that picks wrong rows."""
    ds, shard_arrays, truth = synthetic
    flat = np.concatenate([shard_arrays[s] for s in sorted(shard_arrays)])  # numeric shard order
    wrong = 0
    for row in ds:
        if not np.array_equal(flat[row["offset"] : row["offset"] + row["n_conformers"]], truth[row["key"]]):
            wrong += 1
    assert wrong > 0


def test_global_offsets_with_by_shard_embeddings(synthetic):
    """Datasets written by the converter store global offsets in numeric shard order."""
    ds, shard_arrays, truth = synthetic
    rows = sorted(ds.to_list(), key=lambda r: (r["shard"], r["offset"]))
    cursor = 0
    for r in rows:
        r["offset"] = cursor
        cursor += r["n_conformers"]
    ds_global = datasets.Dataset.from_list(rows)
    idx = build_rotation_index(ds_global)
    assert idx.attrs["offset_mode"] == "global"
    kw = dict(dataset=ds_global, metrics=["cosine"], metric_version="paper")
    res_shard = evaluate_rotation_embeddings(embeddings_by_shard=shard_arrays, **kw)
    res_key = evaluate_rotation_embeddings(embeddings_by_key=truth, **kw)
    pd.testing.assert_frame_equal(_per_key(res_shard), _per_key(res_key))


def test_selection_options(synthetic, tmp_path):
    ds, _, _ = synthetic
    idx = build_rotation_index(ds)
    keys_file = tmp_path / "keys.txt"
    keys_file.write_text("# comment\ns10_m1\ns2_m0\nmissing_key\n")
    sel, stats = select_rows(idx, molecule_list=keys_file)
    assert sel["key"].tolist() == ["s10_m1", "s2_m0"]
    assert stats["molecule_list_missing"] == 1
    sel, _ = select_rows(idx, shards=[10])
    assert set(sel["shard"]) == {10}
    a, _ = select_rows(idx, sample_ratio=0.5, sample_seed=7)
    b, _ = select_rows(idx, sample_ratio=0.5, sample_seed=7)
    assert a["key"].tolist() == b["key"].tolist()
    sel, _ = select_rows(idx, max_keys=3)
    assert len(sel) == 3


def test_missing_shard_file_raises(synthetic):
    ds, shard_arrays, _ = synthetic
    partial = {0: shard_arrays[0]}
    with pytest.raises(ValueError, match="No embedding file for shards"):
        evaluate_rotation_embeddings(dataset=ds, embeddings_by_shard=partial, metrics=["cosine"])
    res = evaluate_rotation_embeddings(dataset=ds, embeddings_by_shard=partial, metrics=["cosine"], shards=[0])
    assert set(res["per_key"]["shard"]) == {0}


def test_outputs_written(synthetic, tmp_path):
    ds, shard_arrays, _ = synthetic
    out = tmp_path / "out"
    res = evaluate_rotation_embeddings(
        dataset=ds, embeddings_by_shard=shard_arrays, metrics=["cosine"], output_dir=out, model_name="toy"
    )
    assert (out / "toy_per_key.parquet").exists()
    assert (out / "config.json").exists()
    summary = pd.read_csv(out / "summary.csv")
    assert {"model", "metric_version", "space", "metric", "mean", "median", "n", "n_finite"} <= set(summary.columns)
    assert res["config"]["layout"] == "by-shard"


def test_replicate_offset_drift(synthetic):
    """A failing molecule shifts later embeddings of its shard only in the __offset_drift columns."""
    ds, shard_arrays, truth = synthetic
    rows = ds.to_list()
    # make s1_m0 (shard 1, local offset 0, 5 conformers) fail: mix molecules with different heavy-atom counts
    bad = next(r for r in rows if r["key"] == "s1_m0")
    other, _ = _conformer_blocks("CCCCC", 1, 99)
    bad["mol_blocks"] = bad["mol_blocks"][:-1] + other
    ds_bad = datasets.Dataset.from_list(rows)
    res = evaluate_rotation_embeddings(
        dataset=ds_bad, embeddings_by_shard=shard_arrays, metrics=["cosine"], replicate_offset_drift=True
    )
    cfg = res["config"]["selection"]
    assert list(cfg["failed_keys"]) == ["s1_m0"]
    assert cfg["offset_drift"]["1"]["n_affected_molecules"] == 1  # s1_m2 (s1_m1 has 1 conformer, skipped)
    pk = res["per_key"].set_index("key")
    # regular columns: aligned embeddings
    aligned = evaluate_rotation_embeddings(dataset=ds_bad, embeddings_by_key=truth, metrics=["cosine"])
    al = aligned["per_key"].set_index("key")
    assert np.allclose(pk.loc["s1_m2", "H_LIE@k"], al.loc["s1_m2", "H_LIE@k"], equal_nan=True)
    # drift columns: s1_m2 (local offset 6) uses rows 1..3 of shard 1; other shards unchanged
    shifted = {"s1_m2": shard_arrays[1][6 - 5 : 6 - 5 + 3]}
    ds_one = datasets.Dataset.from_list([r for r in rows if r["key"] == "s1_m2"])
    drift = evaluate_rotation_embeddings(dataset=ds_one, embeddings_by_key=shifted, metrics=["cosine"])["per_key"]
    assert pk.loc["s1_m2", "AS__offset_drift"] == pytest.approx(float(drift["AS"].iloc[0]))
    assert pk.loc["s1_m2", "H_LIE@k__offset_drift"] == pytest.approx(float(drift["H_LIE@k"].iloc[0]))
    assert pk.loc["s10_m0", "AS__offset_drift"] == pytest.approx(pk.loc["s10_m0", "AS"])
    with pytest.raises(ValueError, match="by-shard"):
        evaluate_rotation_embeddings(dataset=ds_bad, embeddings_by_key=truth, replicate_offset_drift=True)


def test_subset_of_published_layout_uses_per_shard_offsets(synthetic):
    """A filtered subset (first rows of shards removed) still slices with the per-shard offsets."""
    ds, shard_arrays, truth = synthetic
    subset = ds.filter(lambda k: k not in {"s0_m0", "s1_m0", "s10_m0", "s2_m0"}, input_columns=["key"])
    idx = build_rotation_index(subset)
    assert idx.attrs["offset_mode"] == "per-shard"
    kw = dict(dataset=subset, metrics=["cosine"], metric_version="v2")
    res_shard = evaluate_rotation_embeddings(embeddings_by_shard=shard_arrays, **kw)
    res_key = evaluate_rotation_embeddings(embeddings_by_key=truth, **kw)
    pd.testing.assert_frame_equal(_per_key(res_shard), _per_key(res_key))
