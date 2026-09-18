"""Loading per-molecule trajectory embeddings (continuous vectors or RDKit fingerprints).

Supported layouts
-----------------
* A directory with one file per molecule, ``rmd17_<mol>.{npz,npy,pkl}`` (the layout of
  ``EscheWang/3dcs-embeddings`` under ``traj/<model>/``). NPZ files are read with ``key`` (default
  ``arr_0``, or the only numeric array). Pickles may hold a list of RDKit bit vectors (E3FP), an
  array, or a dict (read with ``key``).
* A single ``.npz`` or ``.pkl`` whose keys/entries are molecule names (``rmd17_<mol>``).

Values are loaded lazily: the returned mapping holds zero-argument loaders, so only one molecule's
embeddings need to be in memory at a time.
"""

from __future__ import annotations

import pickle
from functools import partial
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np

TrajEmbedding = Union[np.ndarray, list]
TrajEmbeddingLoader = Callable[[], TrajEmbedding]

_FILE_SUFFIXES = (".npz", ".npy", ".pkl")


def is_fingerprint_sequence(obj: Any) -> bool:
    """True for a list/tuple of RDKit bit vectors (objects exposing ``GetNumBits``)."""
    if not isinstance(obj, (list, tuple)):
        return False
    return len(obj) > 0 and hasattr(obj[0], "GetNumBits")


def _as_embedding(obj: Any, *, source: str) -> TrajEmbedding:
    if is_fingerprint_sequence(obj):
        return list(obj)
    arr = np.asarray(obj)
    if arr.ndim != 2 or not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"Expected a 2-D numeric array or a list of RDKit bit vectors in {source}, got {arr.shape}")
    return arr


def _select_npz_array(data: Any, key: str | None, *, source: str) -> np.ndarray:
    files = list(data.files)
    if key is not None:
        if key not in files:
            raise KeyError(f"Key {key!r} not found in {source} (available: {files})")
        return np.asarray(data[key])
    if "arr_0" in files:
        return np.asarray(data["arr_0"])
    numeric = [k for k in files if np.issubdtype(data[k].dtype, np.number) and data[k].ndim == 2]
    if len(numeric) == 1:
        return np.asarray(data[numeric[0]])
    raise ValueError(f"Cannot choose an array in {source} (keys {files}); pass --embedding-key")


def load_traj_embedding_file(path: Path, key: str | None = None) -> TrajEmbedding:
    """Load one molecule's embeddings from ``.npz``, ``.npy`` or ``.pkl``."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return _as_embedding(_select_npz_array(data, key, source=str(path)), source=str(path))
    if suffix == ".npy":
        return _as_embedding(np.load(path, allow_pickle=False), source=str(path))
    if suffix == ".pkl":
        with path.open("rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict):
            if key is not None:
                payload = payload[key]
            elif len(payload) == 1:
                payload = next(iter(payload.values()))
            else:
                raise ValueError(f"Pickle {path} holds several entries {list(payload)[:5]}; pass --embedding-key")
        return _as_embedding(payload, source=str(path))
    raise ValueError(f"Unsupported embedding file type: {path}")


def _load_npz_member(path: Path, member: str) -> TrajEmbedding:
    with np.load(path, allow_pickle=False) as data:
        return _as_embedding(np.asarray(data[member]), source=f"{path}[{member}]")


def _load_pickle_member(path: Path, member: str) -> TrajEmbedding:
    with path.open("rb") as f:
        payload = pickle.load(f)
    return _as_embedding(payload[member], source=f"{path}[{member}]")


def load_traj_embeddings(
    path: Path,
    *,
    key: str | None = None,
    file_prefix: str = "rmd17_",
) -> dict[str, TrajEmbeddingLoader]:
    """Return ``{mol_type: loader}`` for a directory or a single multi-molecule file."""
    path = Path(path)
    if path.is_dir():
        found: dict[str, Path] = {}
        for candidate in sorted(path.iterdir()):
            if not candidate.is_file() or candidate.suffix.lower() not in _FILE_SUFFIXES:
                continue
            if not candidate.name.startswith(file_prefix):
                continue
            stem = candidate.stem
            if stem in found:
                raise ValueError(f"Several embedding files for {stem} in {path}: {found[stem].name}, {candidate.name}")
            found[stem] = candidate
        if not found:
            raise FileNotFoundError(f"No {file_prefix}*{{.npz,.npy,.pkl}} files in {path}")
        return {stem: partial(load_traj_embedding_file, p, key) for stem, p in found.items()}

    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            members = list(data.files)
        if key is not None:
            members = [key]
        return {m: partial(_load_npz_member, path, m) for m in members}
    if suffix == ".pkl":
        with path.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a dict of molecule -> embeddings in {path}")
        members = [key] if key is not None else list(payload)
        return {m: partial(_load_pickle_member, path, m) for m in members}
    raise ValueError(f"Unsupported embeddings path: {path}")


def resolve_embedding(value: TrajEmbedding | TrajEmbeddingLoader) -> TrajEmbedding:
    """Materialise a loader, or pass arrays / fingerprint lists through."""
    if callable(value):
        value = value()
    if isinstance(value, (list, tuple)):
        return list(value)
    return np.asarray(value)


def embedding_length(value: TrajEmbedding) -> int:
    return len(value) if isinstance(value, (list, tuple)) else int(np.asarray(value).shape[0])
