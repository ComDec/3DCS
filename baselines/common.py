"""Shared helpers for the baseline embedding-extraction scripts in ``baselines/``.

The per-model scripts under ``baselines/<model>/`` are standalone: each one keeps the
input handling that its model needs and depends only on that model's own stack. This
module holds what they have in common and what is useful after a run:

* :func:`load_conformers` -- read the chirality conformers from the Hugging Face dataset,
  from a ``save_to_disk`` directory or from a pickle of RDKit molecules, in benchmark row
  order;
* :func:`write_npz` / :func:`sha256_file` -- write the output array under a given key and
  report its checksum;
* :func:`compare_vectors`, :func:`compare_fingerprints`, :func:`verify` -- compare a file
  that was just written against a reference file, usually the published embedding of the
  same model in ``EscheWang/3dcs-embeddings``;
* :func:`print_versions` -- print the versions of the libraries that were used.

Nothing here imports torch or any model package, so it can be used from every one of the
model environments.
"""

from __future__ import annotations

import hashlib
import pickle
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

EMBEDDINGS_REPO_ID = "EscheWang/3dcs-embeddings"
DATASET_REPO_ID = "EscheWang/3dcs"

#: Published chirality embedding of each baseline: ``model -> (path in the repo, array key)``.
#: Mirrors ``three_dbench.embeddings.published.PUBLISHED_EMBEDDINGS`` for the chirality task,
#: so that ``--verify`` works in a model environment where ``three_dbench`` is not installed.
PUBLISHED_CHIRALITY: dict[str, tuple[str, str]] = {
    "e3fp": ("chirality/e3fp/sampled_chi.pkl", "e3fp"),
    "gemnet": ("chirality/gemnet/sampled_feature.npz", "gemnet"),
    "molae": ("chirality/molae/1.npz", "arr_0"),
    "molspectra": ("chirality/molspectra/sampled_mol_feature.npz", "arr_0"),
    "unimol": ("chirality/unimol/1.npz", "arr_0"),
    "fmg": ("chirality/fmg/chirality_bench_conformers_noised_only_aslist_embed.npz", "embeddings"),
    "mace": ("chirality/mace/chirality.npz", "arr_0"),
}

#: Number of conformers in the chirality set.
N_CHIRALITY_CONFORMERS = 52391


# --------------------------------------------------------------------------- #
# files
# --------------------------------------------------------------------------- #
def sha256_file(path: str | Path, chunk_size: int = 1 << 22) -> str:
    """SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_npz(path: str | Path, array: np.ndarray, key: str = "arr_0", compress: bool = False) -> str:
    """Write ``array`` to ``path`` under ``key`` and return the SHA-256 of the file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = np.savez_compressed if compress else np.savez
    writer(path, **{key: array})
    return sha256_file(path)


def load_embedding(path: str | Path, key: str | None = None) -> Any:
    """Load an embedding container written by one of these scripts or published.

    Returns a NumPy array for ``.npz`` / ``.npy``, and whatever the pickle holds for
    ``.pkl`` (the E3FP file is a dict of lists of RDKit bit vectors).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if key is not None:
                if key not in data.files:
                    raise KeyError(f"{path} has keys {data.files}, not {key!r}")
                return np.asarray(data[key])
            numeric = [k for k in data.files if np.issubdtype(np.asarray(data[k]).dtype, np.number)]
            if len(numeric) != 1:
                raise KeyError(f"{path} holds {data.files}; pass an explicit key")
            return np.asarray(data[numeric[0]])
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".pkl":
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        if isinstance(payload, dict):
            if key is None:
                if len(payload) != 1:
                    raise KeyError(f"{path} holds {sorted(payload)}; pass an explicit key")
                return next(iter(payload.values()))
            return payload[key]
        return payload
    raise ValueError(f"unsupported embedding file type: {path.suffix}")


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #
def compare_vectors(produced: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    """Elementwise and per-row agreement between two embedding matrices.

    Returns a dict with the shapes, the largest and mean absolute difference, whether the
    arrays are equal, the per-row cosine similarity statistics and the number of rows
    outside a few thresholds. Rows of zero norm are excluded from the cosine statistics.
    """
    produced = np.asarray(produced)
    reference = np.asarray(reference)
    out: dict[str, Any] = {
        "produced_shape": tuple(produced.shape),
        "reference_shape": tuple(reference.shape),
        "shape_match": produced.shape == reference.shape,
    }
    if not out["shape_match"]:
        return out
    a = produced.astype(np.float64, copy=False)
    b = reference.astype(np.float64, copy=False)
    diff = np.abs(a - b)
    out["array_equal"] = bool(np.array_equal(produced, reference))
    out["max_abs_diff"] = float(diff.max()) if diff.size else 0.0
    out["mean_abs_diff"] = float(diff.mean()) if diff.size else 0.0
    out["allclose_1e-5"] = bool(np.allclose(a, b, atol=1e-5, rtol=1e-4))
    out["allclose_1e-4"] = bool(np.allclose(a, b, atol=1e-4, rtol=1e-3))
    out["allclose_1e-3"] = bool(np.allclose(a, b, atol=1e-3, rtol=1e-3))
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    ok = (na > 0) & (nb > 0)
    cos = np.full(a.shape[0], np.nan)
    cos[ok] = np.einsum("ij,ij->i", a[ok], b[ok]) / (na[ok] * nb[ok])
    out["zero_norm_rows"] = int((~ok).sum())
    if ok.any():
        out["cosine_mean"] = float(cos[ok].mean())
        out["cosine_min"] = float(cos[ok].min())
        out["rows_cosine_below_1-1e-6"] = int((cos[ok] < 1 - 1e-6).sum())
        out["rows_cosine_below_0.999"] = int((cos[ok] < 0.999).sum())
        out["worst_row"] = int(np.flatnonzero(ok)[int(np.argmin(cos[ok]))])
    return out


def compare_fingerprints(produced: Sequence[Any], reference: Sequence[Any]) -> dict[str, Any]:
    """Agreement between two lists of RDKit bit vectors (E3FP / Morgan fingerprints)."""
    out: dict[str, Any] = {
        "produced_len": len(produced),
        "reference_len": len(reference),
        "shape_match": len(produced) == len(reference),
    }
    if not out["shape_match"]:
        return out
    from rdkit import DataStructs

    identical = 0
    tanimoto = np.empty(len(produced), dtype=np.float64)
    mismatched: list[int] = []
    for i, (p, r) in enumerate(zip(produced, reference)):
        tanimoto[i] = DataStructs.TanimotoSimilarity(p, r)
        if list(p.GetOnBits()) == list(r.GetOnBits()):
            identical += 1
        elif len(mismatched) < 20:
            mismatched.append(i)
    out["identical"] = identical
    out["identical_fraction"] = identical / len(produced) if produced else 0.0
    out["tanimoto_mean"] = float(tanimoto.mean()) if len(tanimoto) else 0.0
    out["tanimoto_min"] = float(tanimoto.min()) if len(tanimoto) else 0.0
    out["first_mismatched_rows"] = mismatched
    return out


def format_report(result: dict[str, Any]) -> str:
    """Render the dict of :func:`compare_vectors` / :func:`compare_fingerprints`."""
    width = max((len(k) for k in result), default=0)
    lines = []
    for key, value in result.items():
        if isinstance(value, float):
            text = f"{value:.10g}"
        else:
            text = str(value)
        lines.append(f"  {key:<{width}}  {text}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# the published reference files
# --------------------------------------------------------------------------- #
def resolve_reference(
    spec: str,
    *,
    model: str | None = None,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
) -> tuple[Path, str | None]:
    """Resolve ``--verify`` into a local file and the array key to read from it.

    ``spec`` is one of

    * ``"published"`` -- the published chirality embedding of ``model``, downloaded from
      ``EscheWang/3dcs-embeddings`` (or taken from the cache of ``huggingface_hub``);
    * ``"hub:<path>"`` -- any file of that dataset repository, e.g.
      ``hub:chirality/mace/chirality.npz``;
    * a local path, in which case the key is inferred by :func:`load_embedding`.
    """
    if spec == "published":
        if model is None or model not in PUBLISHED_CHIRALITY:
            raise ValueError(f"'published' needs a known model, got {model!r}")
        path, key = PUBLISHED_CHIRALITY[model]
        return _download_published(path, cache_dir=cache_dir, revision=revision), key
    if spec.startswith("hub:"):
        path = spec[len("hub:") :]
        key = next((k for p, k in PUBLISHED_CHIRALITY.values() if p == path), None)
        return _download_published(path, cache_dir=cache_dir, revision=revision), key
    return Path(spec), None


def _download_published(path: str, *, cache_dir: str | Path | None, revision: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        repo_id=EMBEDDINGS_REPO_ID,
        filename=path,
        repo_type="dataset",
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return Path(local)


def verify(
    produced_path: str | Path,
    *,
    model: str | None = None,
    reference: str = "published",
    produced_key: str | None = None,
    reference_key: str | None = None,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    stream=None,
) -> dict[str, Any]:
    """Compare a file written by one of these scripts against a reference file.

    Prints the SHA-256 of both files and the agreement statistics, and returns them. The
    caller decides what to do with the result; the scripts print it and exit 0 either way,
    because agreement to the last bit is not expected across machines and library versions.
    """
    stream = stream or sys.stdout
    produced_path = Path(produced_path)
    ref_path, ref_key = resolve_reference(reference, model=model, cache_dir=cache_dir, revision=revision)
    if reference_key is not None:
        ref_key = reference_key

    result: dict[str, Any] = {
        "produced": str(produced_path),
        "reference": str(ref_path),
        "produced_sha256": sha256_file(produced_path),
        "reference_sha256": sha256_file(ref_path),
    }
    result["sha256_identical"] = result["produced_sha256"] == result["reference_sha256"]

    left = load_embedding(produced_path, produced_key)
    right = load_embedding(ref_path, ref_key)
    if isinstance(right, (list, tuple)) or isinstance(left, (list, tuple)):
        result.update(compare_fingerprints(list(left), list(right)))
    else:
        result.update(compare_vectors(left, right))

    print("[verify] comparison with the reference file", file=stream)
    print(format_report(result), file=stream)
    return result


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def load_conformers(
    spec: str,
    *,
    hf_config: str = "chirality",
    hf_split: str = "train",
    revision: str | None = None,
    sanitize: bool = True,
    remove_hs: bool = False,
    limit: int | None = None,
) -> list:
    """Return the conformers as RDKit molecules, in benchmark row order.

    ``spec`` is one of

    * ``hf:<repo_id>`` or ``hf:<repo_id>:<config>`` -- the Hugging Face dataset;
    * ``hfdisk:<dir>`` or a directory holding ``dataset_info.json`` -- a ``save_to_disk`` copy;
    * a path to a pickle holding a list of RDKit molecules, or a dict whose values are
      lists of molecules (flattened in insertion order).

    Rows of the Hugging Face config carry ``mol_blocks`` and the ``offset`` of their first
    conformer, so reading ``mol_blocks`` in ascending ``offset`` gives the row order of the
    published embedding files. MOL blocks store coordinates with four decimals, so
    geometries read that way differ from the source pickle by up to 5e-5 A.
    """
    from rdkit import Chem

    if spec.startswith(("hf:", "hfdisk:")) or (Path(spec).is_dir() and (Path(spec) / "dataset_info.json").exists()):
        rows = _load_hf_rows(spec, hf_config=hf_config, hf_split=hf_split, revision=revision)
        mols = []
        for block in rows:
            mol = Chem.MolFromMolBlock(block, removeHs=remove_hs, sanitize=sanitize)
            if mol is None:
                raise ValueError(f"RDKit could not parse the MOL block of conformer {len(mols)}")
            mols.append(mol)
            if limit is not None and len(mols) >= limit:
                break
        return mols

    with open(spec, "rb") as handle:
        payload = pickle.load(handle)
    mols = _flatten_molecules(payload)
    return mols[:limit] if limit is not None else mols


def _load_hf_rows(spec: str, *, hf_config: str, hf_split: str, revision: str | None) -> Iterable[str]:
    if spec.startswith("hfdisk:") or not spec.startswith("hf:"):
        from datasets import load_from_disk

        ds = load_from_disk(spec[len("hfdisk:") :] if spec.startswith("hfdisk:") else spec)
    else:
        from datasets import load_dataset

        parts = spec[len("hf:") :].split(":")
        repo = parts[0]
        config = parts[1] if len(parts) > 1 else hf_config
        ds = load_dataset(repo, name=config, split=hf_split, revision=revision)
    if "mol_blocks" not in ds.column_names:
        raise KeyError(f"expected a 'mol_blocks' column, got {ds.column_names}")
    order = np.argsort(np.asarray(ds["offset"], dtype=np.int64), kind="stable")
    blocks: list[str] = []
    for row_idx in order:
        row = ds[int(row_idx)]
        if int(row["offset"]) != len(blocks):
            raise ValueError(f"row {int(row_idx)} has offset {row['offset']}, expected {len(blocks)}")
        blocks.extend(row["mol_blocks"])
    return blocks


def _flatten_molecules(payload: Any) -> list:
    if isinstance(payload, dict):
        out: list = []
        for value in payload.values():
            out.extend(value if isinstance(value, (list, tuple)) else [value])
        return out
    if isinstance(payload, (list, tuple)):
        return list(payload)
    raise TypeError(f"expected a list or dict of RDKit molecules, got {type(payload).__name__}")


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def print_versions(modules: Sequence[str] = (), stream=None) -> dict[str, str]:
    """Print the version of Python and of each importable module name in ``modules``."""
    import importlib

    stream = stream or sys.stdout
    found = {"python": sys.version.split()[0], "numpy": np.__version__}
    for name in modules:
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - depends on the environment
            found[name] = f"not importable ({exc.__class__.__name__})"
            continue
        found[name] = str(getattr(module, "__version__", "unknown"))
    for name, version in found.items():
        print(f"[versions] {name:<14} {version}", file=stream)
    return found
