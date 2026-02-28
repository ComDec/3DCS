"""Hugging Face dataset conversion for the chirality benchmark."""

from __future__ import annotations

import pickle
from pathlib import Path

import datasets

from three_dbench.chirality.evaluation import parse_key_en

from .serialization import mols_to_blocks


def convert_chirality_pkl_to_hf(
    input_pkl: Path,
    output_dir: Path,
    *,
    include_mol_blocks: bool = True,
) -> datasets.Dataset:
    """Convert the chirality pickle file into a Hugging Face dataset.

    Each row corresponds to a composite key (mol_id + enantiomer id) and stores
    the conformer MolBlocks plus the flattened offset for embedding alignment.
    """
    with input_pkl.open("rb") as f:
        key_to_mols: dict[str, list] = pickle.load(f)

    rows = []
    offset = 0
    for key, mols in key_to_mols.items():
        mol_id, en_id = parse_key_en(key)
        n_conf = len(mols) if mols is not None else 0
        blocks = mols_to_blocks(mols) if include_mol_blocks else []
        rows.append(
            {
                "key": key,
                "mol_id": mol_id,
                "en_id": en_id,
                "n_conformers": int(n_conf),
                "offset": int(offset),
                "mol_blocks": blocks,
            }
        )
        offset += n_conf

    ds = datasets.Dataset.from_list(rows)
    ds.save_to_disk(str(output_dir))
    return ds


def load_chirality_dataset(dataset_dir: Path) -> datasets.Dataset:
    """Load the chirality dataset from disk."""
    return datasets.load_from_disk(str(dataset_dir))
