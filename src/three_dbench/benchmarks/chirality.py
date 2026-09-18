"""Chirality benchmark evaluation for user-provided embeddings.

Defaults reproduce the published Table 2 protocol: Euclidean distance for continuous embeddings,
Tanimoto distance for RDKit fingerprints, ``metric_version="paper"`` and an unbounded best-k
silhouette scan (``unsup_kmax=None`` -> ``k <= n - 1``). See ``docs/metrics/chirality.md``.
"""

from __future__ import annotations

import json
import pickle
import platform
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from three_dbench.chirality.evaluation import (
    DISTANCES,
    METRIC_VERSIONS,
    coverage_from_rows,
    evaluate_en_separation_from_counts,
)
from three_dbench.datasets.chirality import load_chirality_dataset
from three_dbench.embeddings import EmbeddingArray, load_embeddings


def _key_counts_from_dataset(dataset) -> dict[str, int]:
    """Ordered ``key -> n_conformers`` mapping, validated against the dataset ``offset`` column."""
    columns = set(getattr(dataset, "column_names", []) or [])
    if {"key", "n_conformers"} <= columns:
        keys = list(dataset["key"])
        counts = [int(c) for c in dataset["n_conformers"]]
        offsets = list(dataset["offset"]) if "offset" in columns else None
    else:  # generic iterable of row dicts
        rows = list(dataset)
        keys = [r["key"] for r in rows]
        counts = [int(r["n_conformers"]) for r in rows]
        offsets = [r["offset"] for r in rows] if rows and "offset" in rows[0] else None

    out: dict[str, int] = OrderedDict()
    for k, c in zip(keys, counts):
        if k in out:
            raise ValueError(f"Duplicate key {k!r} in the chirality dataset; keys must be unique.")
        out[k] = c

    if offsets is not None:
        expected = np.concatenate([[0], np.cumsum(counts)[:-1]]) if counts else np.zeros(0, dtype=int)
        bad = np.flatnonzero(np.asarray(offsets, dtype=np.int64) != expected)
        if bad.size:
            i = int(bad[0])
            raise ValueError(
                f"Dataset 'offset' column is not the running sum of 'n_conformers' (first mismatch at row {i}: "
                f"offset={offsets[i]}, expected {int(expected[i])}). Was the dataset re-ordered or filtered? "
                "Embeddings are aligned by dataset row order."
            )
    return out


def _available_keys(path: Path) -> list[str] | None:
    try:
        if path.suffix.lower() == ".npz":
            with np.load(path) as data:
                return list(data.files)
        if path.suffix.lower() == ".pkl":
            with path.open("rb") as f:
                payload = pickle.load(f)
            return sorted(map(str, payload)) if isinstance(payload, dict) else None
    except Exception:
        return None
    return None


def load_chirality_embeddings(path: Path, key: str | None = None) -> EmbeddingArray:
    """Load chirality embeddings (NPZ/NPY array or pickled list of RDKit fingerprints).

    Thin wrapper around :func:`three_dbench.embeddings.load_embeddings` that lists the available keys
    when the key is missing or ambiguous. The published E3FP file (``chirality/e3fp/sampled_chi.pkl``)
    is a dict with ``'e3fp'`` (Table 2, 1024-bit) and ``'morgan'`` fingerprint lists: pass
    ``--embedding-key e3fp``.
    """
    path = Path(path)
    try:
        return load_embeddings(path, key=key)
    except (ValueError, KeyError) as exc:
        keys = _available_keys(path)
        if keys is None:
            raise
        raise type(exc)(
            f"{exc} [{path}: available keys {keys}; pass --embedding-key, e.g. 'e3fp' for the published E3FP file]"
        ) from exc


def _embedding_description(arr: Any) -> dict:
    if isinstance(arr, np.ndarray):
        return {"type": "ndarray", "shape": list(arr.shape), "dtype": str(arr.dtype)}
    first = arr[0] if len(arr) else None
    desc = {"type": type(arr).__name__, "length": len(arr), "element": type(first).__name__}
    if first is not None and hasattr(first, "GetNumBits"):
        desc["n_bits"] = int(first.GetNumBits())
    return desc


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("numpy", "scipy", "sklearn", "rdkit", "three_dbench"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            out[mod] = None
    return out


def evaluate_chirality_embeddings(
    *,
    dataset_dir: Path,
    embeddings: EmbeddingArray | np.ndarray | list,
    output_dir: Path | None = None,
    model_name: str = "custom",
    per_mol_min_n: int = 2,
    do_unsup_when_single_en: bool = False,
    unsup_kmax: int | None = None,
    max_molecules: int | None = None,
    distance: str = "euclidean",
    metric_version: str = "paper",
    n_jobs: int = 1,
    progress: bool = False,
) -> tuple[dict, dict]:
    """Evaluate chirality embeddings against the HF dataset.

    Args:
        dataset_dir: ``save_to_disk`` directory of the chirality dataset
            (``load_dataset("EscheWang/3dcs", name="chirality", split="train")``).
        embeddings: one row per conformer in dataset order (array) or a list of RDKit fingerprints.
        output_dir: if given, writes ``{model_name}_per_molecule.json``, ``summary.csv`` and ``config.json``.
        distance: ``"euclidean"`` (default, published Table 2) or ``"cosine"``; fingerprints always
            use Tanimoto distance.
        metric_version: ``"paper"`` (default, published definitions) or ``"v2"``.
        unsup_kmax: best-k silhouette upper bound; ``None`` = ``n - 1`` (published run).
        n_jobs: worker processes over molecules (``-1`` = all CPUs); results do not depend on it.

    Returns:
        ``({"rows": per_molecule_rows}, summary)``.
    """
    if distance not in DISTANCES:
        raise ValueError(f"distance must be one of {DISTANCES}, got {distance!r}")
    if metric_version not in METRIC_VERSIONS:
        raise ValueError(f"metric_version must be one of {METRIC_VERSIONS}, got {metric_version!r}")

    array = embeddings.array if isinstance(embeddings, EmbeddingArray) else embeddings
    dataset = load_chirality_dataset(dataset_dir)
    key_to_counts = _key_counts_from_dataset(dataset)

    rows, summary = evaluate_en_separation_from_counts(
        key_to_counts,
        array,
        per_mol_min_n=per_mol_min_n,
        do_unsup_when_single_en=do_unsup_when_single_en,
        unsup_kmax=unsup_kmax,
        max_molecules=max_molecules,
        distance=distance,
        metric_version=metric_version,
        n_jobs=n_jobs,
        progress=progress,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        per_mol_path = output_dir / f"{model_name}_per_molecule.json"
        with per_mol_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)

        summary_df = pd.DataFrame([summary])
        summary_df.insert(0, "model", model_name)
        summary_path = output_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False)

        is_fp = not isinstance(array, np.ndarray)
        config = {
            "model_name": model_name,
            "dataset_dir": str(dataset_dir),
            "n_keys": len(key_to_counts),
            "n_conformers": int(sum(key_to_counts.values())),
            "embeddings": _embedding_description(array),
            "distance": "tanimoto (fingerprints)" if is_fp else distance,
            "distance_requested": distance,
            "metric_version": metric_version,
            "unsup_kmax": "n-1" if unsup_kmax is None else int(unsup_kmax),
            "per_mol_min_n": per_mol_min_n,
            "do_unsup_when_single_en": do_unsup_when_single_en,
            "max_molecules": max_molecules,
            "coverage": coverage_from_rows(rows),
            "versions": _versions(),
        }
        with (output_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    return {"rows": rows}, summary
