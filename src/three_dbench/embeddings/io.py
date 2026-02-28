"""Embedding loading utilities."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EmbeddingArray:
    """Container for a single embedding array."""

    array: np.ndarray | list
    kind: str  # "vector" or "fingerprint"


def _select_npz_key(data: np.lib.npyio.NpzFile, key: str | None) -> np.ndarray:
    if key is not None:
        return data[key]
    if "arr_0" in data.files:
        return data["arr_0"]
    return data[data.files[0]]


def load_embeddings(path: Path, *, key: str | None = None) -> EmbeddingArray:
    """Load embeddings from NPZ/NPY/PKL containers.

    Returns an EmbeddingArray with kind inferred from the object type.
    """
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path) as data:
            arr = _select_npz_key(data, key)
    elif suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".pkl":
        with path.open("rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict):
            if key is not None:
                arr = payload[key]
            elif len(payload) == 1:
                arr = next(iter(payload.values()))
            else:
                raise ValueError("Pickle contains multiple entries; specify --embedding-key.")
        else:
            arr = payload
    else:
        raise ValueError(f"Unsupported embedding file type: {suffix}")

    kind = "fingerprint" if isinstance(arr, (list, tuple)) else "vector"
    return EmbeddingArray(array=arr, kind=kind)


def load_embeddings_dict(path: Path, *, key: str | None = None) -> dict[str, np.ndarray]:
    """Load a dict of embeddings keyed by molecule or shard IDs."""
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path) as data:
            if key is not None:
                return {key: np.asarray(data[key])}
            return {k: np.asarray(data[k]) for k in data.files}
    if suffix == ".pkl":
        with path.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            raise ValueError("Expected a dict in the pickle file.")
        if key is not None:
            return {key: np.asarray(payload[key])}
        return {k: np.asarray(v) for k, v in payload.items()}
    raise ValueError(f"Unsupported embedding dict file type: {suffix}")


def load_embeddings_dir(
    embeddings_dir: Path,
    *,
    file_glob: str,
    key: str | None = None,
) -> dict[str, np.ndarray]:
    """Load embeddings from a directory of NPZ files.

    The returned dict keys are derived from the file stem.
    """
    out = {}
    for npz_path in sorted(embeddings_dir.glob(file_glob)):
        with np.load(npz_path) as data:
            arr = _select_npz_key(data, key)
        out[npz_path.stem] = np.asarray(arr)
    return out
