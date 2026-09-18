"""Embedding loading utilities."""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Array names tried, in order, when no key is given. These cover the files published in
#: ``EscheWang/3dcs-embeddings`` (``arr_0``, ``embeddings`` for FMG, ``gemnet``, ``mol_feature``).
DEFAULT_ARRAY_KEYS = ("arr_0", "embeddings", "gemnet", "mol_feature")
#: Dict entries tried, in order, for pickled dicts when no key is given (E3FP pickles hold
#: ``{"e3fp": [...], "morgan": [...]}``).
DEFAULT_PICKLE_KEYS = ("e3fp", "embeddings", "arr_0")
EMBEDDING_SUFFIXES = (".npz", ".npy", ".pkl")


@dataclass
class EmbeddingArray:
    """Container for a single embedding array."""

    array: np.ndarray | list
    kind: str  # "vector" or "fingerprint"


def select_array_key(data: np.lib.npyio.NpzFile, key: str | None) -> str:
    """Return the name of the array to read from an NPZ archive.

    With ``key=None`` the first name in :data:`DEFAULT_ARRAY_KEYS` that is present is used;
    otherwise the first numeric array (so that string side-arrays such as FMG's ``smiles``
    are never selected).
    """
    if key is not None:
        if key not in data.files:
            raise KeyError(f"Key {key!r} not found in NPZ; available: {list(data.files)}")
        return key
    for name in DEFAULT_ARRAY_KEYS:
        if name in data.files:
            return name
    for name in data.files:
        arr = data[name]
        if np.issubdtype(arr.dtype, np.number) or arr.dtype == bool:
            return name
    return data.files[0]


def _select_npz_key(data: np.lib.npyio.NpzFile, key: str | None) -> np.ndarray:
    return data[select_array_key(data, key)]


def resolve_embedding_file(path: Path) -> Path:
    """Resolve a directory holding exactly one embedding file to that file."""
    path = Path(path)
    if not path.is_dir():
        return path
    candidates = sorted(p for p in path.iterdir() if p.suffix.lower() in EMBEDDING_SUFFIXES)
    if len(candidates) != 1:
        raise ValueError(
            f"{path} is a directory with {len(candidates)} embedding files; pass a file path "
            "(or use the by-shard / per-molecule directory options of the benchmark)."
        )
    return candidates[0]


def load_embeddings(path: Path, *, key: str | None = None) -> EmbeddingArray:
    """Load embeddings from NPZ/NPY/PKL containers.

    ``path`` may also be a directory that contains exactly one embedding file (e.g.
    ``data/embeddings/chirality/gemnet``). Returns an EmbeddingArray with kind inferred from
    the object type.
    """
    path = resolve_embedding_file(Path(path))
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
                chosen = next((k for k in DEFAULT_PICKLE_KEYS if k in payload), None)
                if chosen is None:
                    raise ValueError("Pickle contains multiple entries; specify --embedding-key.")
                warnings.warn(
                    f"{path.name}: no embedding key given; using entry {chosen!r} of {sorted(payload)}",
                    stacklevel=2,
                )
                arr = payload[chosen]
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
