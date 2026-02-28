"""Hugging Face dataset builders and loaders for 3DBench."""

__all__ = [
    "convert_chirality_pkl_to_hf",
    "load_chirality_dataset",
    "convert_rotation_lmdb_to_hf",
    "load_rotation_dataset",
    "convert_traj_energy_npz_to_hf",
    "convert_traj_frames_pkl_to_hf",
    "load_traj_energy_dataset",
    "load_traj_frames_dataset",
]


def __getattr__(name: str):
    if name in {"convert_chirality_pkl_to_hf", "load_chirality_dataset"}:
        from .chirality import convert_chirality_pkl_to_hf, load_chirality_dataset

        return convert_chirality_pkl_to_hf if name == "convert_chirality_pkl_to_hf" else load_chirality_dataset
    if name in {"convert_rotation_lmdb_to_hf", "load_rotation_dataset"}:
        from .rotation import convert_rotation_lmdb_to_hf, load_rotation_dataset

        return convert_rotation_lmdb_to_hf if name == "convert_rotation_lmdb_to_hf" else load_rotation_dataset
    if name in {
        "convert_traj_energy_npz_to_hf",
        "convert_traj_frames_pkl_to_hf",
        "load_traj_energy_dataset",
        "load_traj_frames_dataset",
    }:
        from .traj import (
            convert_traj_energy_npz_to_hf,
            convert_traj_frames_pkl_to_hf,
            load_traj_energy_dataset,
            load_traj_frames_dataset,
        )

        lookup = {
            "convert_traj_energy_npz_to_hf": convert_traj_energy_npz_to_hf,
            "convert_traj_frames_pkl_to_hf": convert_traj_frames_pkl_to_hf,
            "load_traj_energy_dataset": load_traj_energy_dataset,
            "load_traj_frames_dataset": load_traj_frames_dataset,
        }
        return lookup[name]
    raise AttributeError(f"module 'three_dbench.datasets' has no attribute {name!r}")
