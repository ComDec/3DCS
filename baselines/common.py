"""Shared helpers for the baseline embedding-extraction scripts in ``baselines/``.

Every per-model script under ``baselines/<model>/`` runs in its own model environment and
depends only on that model's own stack. This module holds what they have in common:

* :func:`parse_dataset_spec` / :func:`load_conformers` / :func:`iter_conformers` -- the one
  ``--dataset`` syntax that every script accepts, read into RDKit molecules in benchmark
  row order;
* :func:`write_npz` / :func:`sha256_file` -- write the output array under a given key and
  report its checksum;
* :func:`compare_vectors`, :func:`compare_fingerprints`, :func:`verify` -- compare a file
  that was just written against a reference file, usually the published embedding of the
  same model in ``EscheWang/3dcs-embeddings``, over all rows or over a subset;
* :func:`parse_row_selection` -- how the rows of a partial run line up with the rows of
  the reference file;
* :func:`print_versions` -- print the versions of the libraries that were used.

Nothing here imports torch or any model package, so it can be used from every one of the
model environments.
"""

from __future__ import annotations

import hashlib
import pickle
import sys
from collections.abc import Iterable, Iterator, Sequence
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
# which rows of the reference a partial run covers
# --------------------------------------------------------------------------- #
#: ``--verify-rows`` help text, identical in every script.
ROW_SELECTION_HELP = (
    "which rows of the --verify reference the output covers, for a run that produced only "
    "part of the file: 'prefix' (the default: output row i is reference row i), 'full' "
    "(require the two files to have the same number of rows), '<start>:<stop>' for a slice "
    "of the reference (e.g. '2000:4000', open ends allowed), '<start>+' for the rows from "
    "<start> on, or '@<file>' for a .npy / text file of 0-based reference row indices"
)


def parse_row_selection(
    spec: str | None,
    *,
    produced_rows: int,
    reference_rows: int,
) -> tuple[np.ndarray | None, str]:
    """Resolve ``--verify-rows`` into indices into the reference file.

    Returns ``(indices, mode)``. ``indices`` is ``None`` when every row of the reference is
    compared, i.e. the two files line up one to one; ``mode`` names the resolution
    (``"full"``, ``"prefix"``, ``"slice"`` or ``"file"``) for the report.
    """
    if spec in (None, "", "auto"):
        spec = "full" if produced_rows == reference_rows else "prefix"
    spec = str(spec)

    if spec == "full":
        if produced_rows != reference_rows:
            raise ValueError(
                f"--verify-rows full: the output has {produced_rows} rows and the reference "
                f"{reference_rows}; pass 'prefix', a '<start>:<stop>' slice or '@<file>'"
            )
        return None, "full"

    if spec == "prefix":
        if produced_rows > reference_rows:
            raise ValueError(
                f"--verify-rows prefix: the output has {produced_rows} rows, more than the "
                f"{reference_rows} of the reference"
            )
        if produced_rows == reference_rows:
            return None, "full"
        return np.arange(produced_rows, dtype=np.int64), "prefix"

    if spec.startswith("@"):
        path = Path(spec[1:])
        if path.suffix.lower() == ".npy":
            indices = np.asarray(np.load(path), dtype=np.int64).reshape(-1)
        else:
            text = path.read_text(encoding="utf-8").replace(",", " ").split()
            indices = np.asarray([int(value) for value in text], dtype=np.int64)
        mode = "file"
    elif ":" in spec or spec.endswith("+"):
        if spec.endswith("+"):
            start_text, stop_text = spec[:-1], ""
        else:
            start_text, _, stop_text = spec.partition(":")
        start = int(start_text) if start_text.strip() else 0
        stop = int(stop_text) if stop_text.strip() else start + produced_rows
        indices = np.arange(start, stop, dtype=np.int64)
        mode = "slice"
    else:
        raise ValueError(
            f"--verify-rows: expected 'full', 'prefix', '<start>:<stop>', '<start>+' or '@<file>', got {spec!r}"
        )

    if indices.size != produced_rows:
        raise ValueError(
            f"--verify-rows {spec}: selects {indices.size} reference rows but the output has {produced_rows}"
        )
    if indices.size and (indices.min() < 0 or indices.max() >= reference_rows):
        raise ValueError(
            f"--verify-rows {spec}: row indices {int(indices.min())}..{int(indices.max())} fall outside "
            f"the {reference_rows} rows of the reference"
        )
    if mode == "slice" and indices.size and int(indices[0]) == 0 and indices.size == reference_rows:
        return None, "full"
    return indices, mode


def _take_rows(reference: Any, indices: np.ndarray | None) -> Any:
    if indices is None:
        return reference
    if isinstance(reference, np.ndarray):
        return reference[indices]
    return [reference[int(i)] for i in indices]


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #
def compare_vectors(
    produced: np.ndarray,
    reference: np.ndarray,
    *,
    rows: np.ndarray | None = None,
    row_mode: str | None = None,
) -> dict[str, Any]:
    """Elementwise and per-row agreement between two embedding matrices.

    Returns a dict with the shapes, the largest and mean absolute difference, whether the
    arrays are equal, the per-row cosine similarity statistics and the number of rows
    outside a few thresholds. Rows of zero norm are excluded from the cosine statistics.

    ``rows`` compares row ``i`` of ``produced`` with row ``rows[i]`` of ``reference``, which
    is what a run over part of the conformers needs: the statistics are then computed over
    those rows only, and ``compared_rows`` / ``row_selection`` say so.
    """
    produced = np.asarray(produced)
    reference = np.asarray(reference)
    out: dict[str, Any] = {
        "produced_shape": tuple(produced.shape),
        "reference_shape": tuple(reference.shape),
        "shape_match": produced.shape == reference.shape,
    }
    if rows is not None:
        rows = np.asarray(rows, dtype=np.int64)
        out["row_selection"] = row_mode or "rows"
        out["compared_rows"] = int(rows.size)
        out["reference_rows_used"] = (
            f"{int(rows[0])}..{int(rows[-1])}"
            if rows.size and np.array_equal(rows, np.arange(rows[0], rows[0] + rows.size))
            else "listed"
        )
        reference = reference[rows]
    else:
        out["row_selection"] = row_mode or "full"
        out["compared_rows"] = int(produced.shape[0]) if produced.ndim else 0
    out["compared_shape_match"] = produced.shape == reference.shape
    if not out["compared_shape_match"]:
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
    per_row_close = np.isclose(a, b, atol=1e-5, rtol=1e-4).all(axis=1) if a.ndim == 2 else np.array([])
    out["rows_allclose_1e-5"] = int(per_row_close.sum())
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
        present = np.flatnonzero(ok)
        order = present[np.argsort(cos[present])]
        out["worst_row"] = int(order[0])
        out["worst_rows"] = [int(i) for i in order[:10]]
    return out


def compare_fingerprints(
    produced: Sequence[Any],
    reference: Sequence[Any],
    *,
    rows: np.ndarray | None = None,
    row_mode: str | None = None,
) -> dict[str, Any]:
    """Agreement between two lists of RDKit bit vectors (E3FP / Morgan fingerprints).

    ``rows`` has the same meaning as in :func:`compare_vectors`: fingerprint ``i`` of
    ``produced`` is compared with fingerprint ``rows[i]`` of ``reference``.
    """
    out: dict[str, Any] = {
        "produced_len": len(produced),
        "reference_len": len(reference),
        "shape_match": len(produced) == len(reference),
    }
    if rows is not None:
        rows = np.asarray(rows, dtype=np.int64)
        out["row_selection"] = row_mode or "rows"
        out["compared_rows"] = int(rows.size)
        reference = _take_rows(list(reference), rows)
    else:
        out["row_selection"] = row_mode or "full"
        out["compared_rows"] = len(produced)
    out["compared_shape_match"] = len(produced) == len(reference)
    if not out["compared_shape_match"]:
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
    rows: str | None = None,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    stream=None,
) -> dict[str, Any]:
    """Compare a file written by one of these scripts against a reference file.

    Prints the SHA-256 of both files and the agreement statistics, and returns them. The
    caller decides what to do with the result; the scripts print it and exit 0 either way,
    because agreement to the last bit is not expected across machines and library versions.

    ``rows`` is the ``--verify-rows`` selection (see :func:`parse_row_selection`). A file
    written by a run over part of the conformers -- ``--limit``, ``--start`` -- holds fewer
    rows than the published reference; with the default selection its rows are compared with
    the first rows of the reference and every statistic is computed over those rows.
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
    as_fingerprints = isinstance(right, (list, tuple)) or isinstance(left, (list, tuple))
    indices, mode = parse_row_selection(rows, produced_rows=len(left), reference_rows=len(right))
    if mode != "full":
        print(
            f"[verify] the output holds {len(left)} of the {len(right)} rows of the reference; "
            f"comparing row selection '{mode}'",
            file=stream,
        )
    if as_fingerprints:
        result.update(compare_fingerprints(list(left), list(right), rows=indices, row_mode=mode))
    else:
        result.update(compare_vectors(left, right, rows=indices, row_mode=mode))

    print("[verify] comparison with the reference file", file=stream)
    print(format_report(result), file=stream)
    return result


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
#: ``--dataset`` help text, identical in every script.
DATASET_SPEC_HELP = (
    "conformer source, read in benchmark row order. One of: "
    "'hf:EscheWang/3dcs:chirality' (a Hub dataset and its config), "
    "'hfdisk:<dir>' or a plain path to a 'save_to_disk' directory of that config, "
    "a bare Hub dataset id (config 'chirality'), "
    "a pickle holding a list of RDKit molecules or a dict of such lists, or "
    "'lmdb:<file>' for a rotation shard"
)

#: The kinds :func:`parse_dataset_spec` returns.
DATASET_KINDS = ("hub", "disk", "pickle", "lmdb")

_PICKLE_SUFFIXES = {".pkl", ".pickle", ".pkl.gz"}
_LMDB_SUFFIXES = {".lmdb", ".mdb"}


def parse_dataset_spec(spec: str, *, hf_config: str = "chirality") -> tuple[str, str, str | None]:
    """Resolve a ``--dataset`` value into ``(kind, target, config)``.

    ``kind`` is one of :data:`DATASET_KINDS`. This is pure string and filesystem
    inspection: nothing is downloaded or opened here.

    ========================================  =========  ==========================
    spec                                      kind       target
    ========================================  =========  ==========================
    ``hf:EscheWang/3dcs:chirality``           ``hub``    ``EscheWang/3dcs``
    ``hf:EscheWang/3dcs``                     ``hub``    ``EscheWang/3dcs``
    ``EscheWang/3dcs``                        ``hub``    ``EscheWang/3dcs``
    ``hfdisk:data/hf/chirality``              ``disk``   ``data/hf/chirality``
    ``data/hf/chirality`` (an existing dir)   ``disk``   ``data/hf/chirality``
    ``conformers.pkl`` (an existing file)     ``pickle`` ``conformers.pkl``
    ``lmdb:rotation_0.lmdb``                  ``lmdb``   ``rotation_0.lmdb``
    ========================================  =========  ==========================
    """
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError(f"--dataset: expected a non-empty string. {DATASET_SPEC_HELP}")
    spec = spec.strip()

    if spec.startswith("lmdb:"):
        return "lmdb", spec[len("lmdb:") :], None
    if spec.startswith("hfdisk:"):
        return "disk", spec[len("hfdisk:") :], None
    if spec.startswith("hf:"):
        rest = spec[len("hf:") :]
        repo, sep, config = rest.partition(":")
        if not repo:
            raise ValueError(f"--dataset {spec!r}: 'hf:' needs a repository id, e.g. hf:EscheWang/3dcs:chirality")
        return "hub", repo, (config or hf_config) if sep else hf_config

    path = Path(spec)
    if path.is_dir():
        return "disk", spec, None
    if path.is_file():
        return ("lmdb", spec, None) if path.suffix.lower() in _LMDB_SUFFIXES else ("pickle", spec, None)

    # Not on disk: a Hub id is "<owner>/<name>", anything else is a path that does not exist.
    parts = spec.split("/")
    if len(parts) == 2 and all(parts) and path.suffix.lower() not in _PICKLE_SUFFIXES:
        return "hub", spec, hf_config
    raise FileNotFoundError(f"--dataset {spec!r}: no such file or directory. {DATASET_SPEC_HELP}")


def load_conformers(
    spec: str,
    *,
    hf_config: str = "chirality",
    hf_split: str = "train",
    revision: str | None = None,
    sanitize: bool = True,
    sanitize_fallback: bool = True,
    remove_hs: bool = False,
    limit: int | None = None,
    start: int = 0,
    stream=None,
) -> list:
    """Return the conformers as RDKit molecules, in benchmark row order.

    ``spec`` takes every form of :func:`parse_dataset_spec`, which is the ``--dataset``
    syntax of every script in ``baselines/``.

    Rows of the Hugging Face config carry ``mol_blocks`` -- one MOL block per conformer of
    that stereoisomer -- and the ``offset`` of the first of them, so reading ``mol_blocks``
    in ascending ``offset`` gives the row order of the published embedding files. MOL blocks
    store coordinates with four decimals, so geometries read that way differ from the source
    pickle by up to 5e-5 A.

    ``sanitize`` is passed to ``Chem.MolFromMolBlock``. With ``sanitize_fallback`` a block
    that RDKit refuses to sanitise is re-read unsanitised rather than dropped, so the row
    order never shifts; the number of such rows is printed. Set it to ``False`` where
    sanitisation changes the result (E3FP reads bond orders and stereochemistry).
    """
    return list(
        iter_conformers(
            spec,
            hf_config=hf_config,
            hf_split=hf_split,
            revision=revision,
            sanitize=sanitize,
            sanitize_fallback=sanitize_fallback,
            remove_hs=remove_hs,
            limit=limit,
            start=start,
            stream=stream,
        )
    )


def iter_conformers(
    spec: str,
    *,
    hf_config: str = "chirality",
    hf_split: str = "train",
    revision: str | None = None,
    sanitize: bool = True,
    sanitize_fallback: bool = True,
    remove_hs: bool = False,
    limit: int | None = None,
    start: int = 0,
    stream=None,
) -> Iterator[Any]:
    """:func:`load_conformers` as a generator, for a script that does not want the list."""
    stream = stream or sys.stdout
    kind, target, config = parse_dataset_spec(spec, hf_config=hf_config)
    if kind == "pickle":
        source: Iterable[Any] = _iter_pickle_molecules(target)
    elif kind == "lmdb":
        source = _iter_lmdb_molecules(target)
    else:
        source = _iter_hf_molecules(
            kind,
            target,
            config=config or hf_config,
            hf_split=hf_split,
            revision=revision,
            sanitize=sanitize,
            sanitize_fallback=sanitize_fallback,
            remove_hs=remove_hs,
            stream=stream,
        )

    for index, mol in enumerate(source):
        if index < start:
            continue
        if limit is not None and index - start >= limit:
            return
        yield mol


def _open_hf_dataset(kind: str, target: str, *, config: str, hf_split: str, revision: str | None):
    if kind == "disk":
        from datasets import load_from_disk

        return load_from_disk(target)
    from datasets import load_dataset

    return load_dataset(target, name=config, split=hf_split, revision=revision)


def _iter_hf_molecules(
    kind: str,
    target: str,
    *,
    config: str,
    hf_split: str,
    revision: str | None,
    sanitize: bool,
    sanitize_fallback: bool,
    remove_hs: bool,
    stream,
) -> Iterator[Any]:
    from rdkit import Chem

    ds = _open_hf_dataset(kind, target, config=config, hf_split=hf_split, revision=revision)
    if "mol_blocks" not in ds.column_names:
        raise KeyError(
            f"{target}: expected a 'mol_blocks' column (the 'chirality' config of EscheWang/3dcs "
            f"stores one list of MDL MOL blocks per stereoisomer), got {ds.column_names}"
        )
    order: Sequence[int]
    if "offset" in ds.column_names:
        offsets = np.asarray(ds["offset"], dtype=np.int64)  # one column read, not len(ds) row reads
        order = np.argsort(offsets, kind="stable")
    else:
        offsets = None
        order = np.arange(len(ds), dtype=np.int64)
    blocks_column = ds["mol_blocks"]

    seen = 0
    first_offset: int | None = None
    unsanitised = 0
    for row_idx in order:
        row_idx = int(row_idx)
        if offsets is not None:
            offset = int(offsets[row_idx])
            if first_offset is None:
                first_offset = offset
                if offset:
                    print(f"[data] the rows of {target} start at conformer offset {offset}", file=stream)
            if offset != first_offset + seen:
                raise ValueError(
                    f"{target}: row {row_idx} has offset {offset}, expected {first_offset + seen}; "
                    "the rows do not form one contiguous block of the published row order"
                )
        for block in blocks_column[row_idx]:
            mol = Chem.MolFromMolBlock(block, removeHs=remove_hs, sanitize=sanitize)
            if mol is None and sanitize and sanitize_fallback:
                mol = Chem.MolFromMolBlock(block, removeHs=remove_hs, sanitize=False)
                if mol is not None:
                    unsanitised += 1
            if mol is None:
                raise ValueError(f"RDKit could not parse the MOL block of conformer {seen} (dataset row {row_idx})")
            seen += 1
            yield mol
    if unsanitised:
        print(f"[data] {unsanitised} of {seen} conformers were read without sanitisation", file=stream)


def _iter_pickle_molecules(path: str) -> Iterator[Any]:
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    yield from _flatten_molecules(payload)


def _iter_lmdb_molecules(path: str) -> Iterator[Any]:
    """Rotation shards: each key holds a list of ``(Mol, energy, torsion_deg)``.

    Row order is the LMDB cursor order of the keys, then the position inside each list,
    which is the row order of the published ``rotation/<model>/rotation_conformers_<k>.npz``.
    """
    import lmdb

    env = lmdb.open(path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
    try:
        with env.begin() as txn:
            for _key, value in txn.cursor():
                for item in pickle.loads(value):
                    yield item[0] if isinstance(item, tuple) else item
    finally:
        env.close()


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
