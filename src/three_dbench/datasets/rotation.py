"""Hugging Face dataset conversion for the rotation benchmark."""

from __future__ import annotations

import pickle
from collections.abc import Iterable
from pathlib import Path

import datasets
import lmdb

from .serialization import mols_to_blocks


def convert_rotation_lmdb_to_hf(
    lmdb_root: Path,
    output_dir: Path,
    *,
    shards: Iterable[int] | None = None,
    include_mol_blocks: bool = True,
) -> datasets.Dataset:
    """Convert LMDB rotation sources into a Hugging Face dataset.

    The resulting dataset stores one row per key with conformer MolBlocks,
    torsion angles, and embedding offsets. This makes it possible to align
    flattened embedding arrays with the dataset order.
    """
    fp_dir = lmdb_root / "fingerprint"
    deg_dir = lmdb_root / "sources"
    shard_ids = list(shards) if shards is not None else list(range(16))

    def _iter_rows():
        offset = 0
        for shard in shard_ids:
            fp_path = fp_dir / f"rot_{shard}.lmdb"
            deg_path = deg_dir / f"rot_conf_deg_{shard}.lmdb"
            has_fp = fp_path.exists()
            if not deg_path.exists():
                raise FileNotFoundError(f"Missing sources LMDB: {deg_path}")
            env_fp = None
            if has_fp:
                env_fp = lmdb.open(str(fp_path), subdir=False, readonly=True, lock=False, readahead=True)
            env_deg = lmdb.open(str(deg_path), subdir=False, readonly=True, lock=False, readahead=True)
            with env_deg.begin() as txn_deg:
                if has_fp:
                    with env_fp.begin() as txn_fp:
                        cursor = txn_fp.cursor()
                        for k, v in cursor:
                            data_fp = pickle.loads(v)
                            data_deg = pickle.loads(txn_deg.get(k))
                            mol_list = [x[0] for x in data_fp["base"]]
                            torsion_deg = [float(x[2]) for x in data_deg]
                            n_conf = len(mol_list)
                            blocks = mols_to_blocks(mol_list) if include_mol_blocks else []
                            yield {
                                "key": k.decode("utf-8"),
                                "shard": int(shard),
                                "n_conformers": int(n_conf),
                                "offset": int(offset),
                                "mol_blocks": blocks,
                                "torsion_deg": torsion_deg,
                            }
                            offset += n_conf
                else:
                    cursor = txn_deg.cursor()
                    for k, v in cursor:
                        data_deg = pickle.loads(v)
                        mol_list = [x[0] for x in data_deg]
                        torsion_deg = [float(x[2]) for x in data_deg]
                        n_conf = len(mol_list)
                        blocks = mols_to_blocks(mol_list) if include_mol_blocks else []
                        yield {
                            "key": k.decode("utf-8"),
                            "shard": int(shard),
                            "n_conformers": int(n_conf),
                            "offset": int(offset),
                            "mol_blocks": blocks,
                            "torsion_deg": torsion_deg,
                        }
                        offset += n_conf

    features = datasets.Features(
        {
            "key": datasets.Value("string"),
            "shard": datasets.Value("int32"),
            "n_conformers": datasets.Value("int32"),
            "offset": datasets.Value("int64"),
            "mol_blocks": datasets.Sequence(datasets.Value("string")),
            "torsion_deg": datasets.Sequence(datasets.Value("float32")),
        }
    )

    ds = datasets.Dataset.from_generator(_iter_rows, features=features)
    ds.save_to_disk(str(output_dir))
    return ds


def load_rotation_dataset(dataset_dir: Path) -> datasets.Dataset:
    """Load the rotation dataset from disk."""
    return datasets.load_from_disk(str(dataset_dir))
