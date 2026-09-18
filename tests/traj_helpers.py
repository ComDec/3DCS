"""Synthetic trajectory data shared by the trajectory tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def synthetic_energies(n_frames: int, seed: int, offset: float = -4.06e5) -> np.ndarray:
    """Absolute float64 energies with a few kcal/mol spread (not float32-representable)."""
    rng = np.random.default_rng(seed)
    return offset + 4.0 * rng.standard_normal(n_frames) + rng.random(n_frames) * 1e-6


def synthetic_embeddings(energies: np.ndarray, dim: int, seed: int) -> np.ndarray:
    """float32 vectors weakly coupled to the energies."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((energies.size, dim))
    z[:, 0] += 0.3 * (energies - energies.mean())
    return z.astype(np.float32)


def save_energy_dataset(path: Path, energies_by_mol: dict[str, np.ndarray]) -> Path:
    import datasets

    rows = [
        {"mol_type": mol, "n_frames": int(e.size), "energies": np.asarray(e, dtype=np.float64).tolist()}
        for mol, e in sorted(energies_by_mol.items())
    ]
    datasets.Dataset.from_list(rows).save_to_disk(str(path))
    return path


def random_fingerprints(n: int, n_bits: int, seed: int, density: float = 0.1):
    from rdkit.DataStructs import ExplicitBitVect

    rng = np.random.default_rng(seed)
    fps = []
    for i in range(n):
        bv = ExplicitBitVect(n_bits)
        if i % 17 != 3:  # keep a few empty fingerprints
            on = np.flatnonzero(rng.random(n_bits) < density)
            bv.SetBitsFromList([int(b) for b in on])
        fps.append(bv)
    return fps
