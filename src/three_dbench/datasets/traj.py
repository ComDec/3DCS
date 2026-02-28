"""Hugging Face dataset conversion for the trajectory benchmark."""

from __future__ import annotations

import pickle
from pathlib import Path

import datasets
import numpy as np

from .serialization import mol_to_block


def convert_traj_frames_pkl_to_hf(
    mol_pkl_dir: Path,
    output_dir: Path,
    *,
    include_mol_blocks: bool = True,
) -> datasets.Dataset:
    """Convert trajectory conformer pickles into a Hugging Face dataset."""
    pkl_paths = sorted(mol_pkl_dir.glob("rmd17_*.pkl"))

    def _iter_rows():
        for pkl_path in pkl_paths:
            mol_type = pkl_path.stem
            with pkl_path.open("rb") as f:
                mol_list = pickle.load(f)
            for idx, mol in enumerate(mol_list):
                yield {
                    "mol_type": mol_type,
                    "frame_idx": int(idx),
                    "mol_block": mol_to_block(mol) if include_mol_blocks else "",
                }

    features = datasets.Features(
        {
            "mol_type": datasets.Value("string"),
            "frame_idx": datasets.Value("int32"),
            "mol_block": datasets.Value("string"),
        }
    )
    ds = datasets.Dataset.from_generator(_iter_rows, features=features)
    ds.save_to_disk(str(output_dir))
    return ds


def convert_traj_energy_npz_to_hf(energy_dir: Path, output_dir: Path) -> datasets.Dataset:
    """Convert trajectory energies into a Hugging Face dataset."""
    rows = []
    for npz_path in sorted(energy_dir.glob("rmd17_*.npz")):
        mol_type = npz_path.stem
        energies = np.load(npz_path, mmap_mode="r")["energies"].astype(np.float32)
        rows.append(
            {
                "mol_type": mol_type,
                "n_frames": int(energies.shape[0]),
                "energies": energies.tolist(),
            }
        )
    ds = datasets.Dataset.from_list(rows)
    ds.save_to_disk(str(output_dir))
    return ds


def load_traj_frames_dataset(dataset_dir: Path) -> datasets.Dataset:
    """Load the trajectory frames dataset from disk."""
    return datasets.load_from_disk(str(dataset_dir))


def load_traj_energy_dataset(dataset_dir: Path) -> datasets.Dataset:
    """Load the trajectory energies dataset from disk."""
    return datasets.load_from_disk(str(dataset_dir))
