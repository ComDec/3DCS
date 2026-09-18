"""Rotation (geometry) benchmark evaluation for user-provided embeddings.

Embedding layouts
-----------------
``flat``
    One array (or fingerprint list) with one row per conformer, in the **row order of the
    dataset**. The start of each molecule is the cumulative sum of ``n_conformers`` over the
    dataset rows; the stored ``offset`` column is not used for flat arrays.
``by-shard``
    A directory with one file per shard (e.g. ``rotation_conformers_{shard}.npz``, the layout of
    ``EscheWang/3dcs-embeddings``). Rows of shard ``s`` are sliced from the file of shard ``s``
    using the **per-shard** offset. The published HF dataset stores per-shard offsets (they
    restart at 0 in every shard, and shards appear in string order 0, 1, 10, ..., 15, 2, ..., 9);
    datasets written by the original converter store global offsets, which are converted to
    per-shard offsets automatically.
``by-key``
    A dict ``{key: array of shape (n_conformers, dim)}``.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import re
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from three_dbench.embeddings import EmbeddingArray
from three_dbench.embeddings.io import select_array_key

LAYOUTS = ("flat", "by-shard", "by-key")
OFFSET_MODES = ("auto", "per-shard", "global")
SHARD_FILE_PATTERNS = (
    "rotation_conformers_{shard}.npz",
    "rot_mol_list_{shard}_embed.npz",
    "rot{shard}.npz",
    "rot_{shard}.npz",
    "{shard}.npz",
    "rotation_conformers_{shard}.npy",
    "rot_{shard}.pkl",
)
TABLE1_METRICS = ("A1_spearman", "A2_kendall", "G_cka_rbf", "J_isotonic_R2", "H_LIE@k", "torsion_sp", "AS")


# ---------------------------------------------------------------------------
# Index / offsets
# ---------------------------------------------------------------------------


def build_rotation_index(dataset: Any, *, offset_mode: str = "auto") -> pd.DataFrame:
    """Return a per-row index with flat and per-shard offsets.

    Columns: ``row`` (dataset row), ``key``, ``shard``, ``n_conformers``, ``offset`` (as stored),
    ``flat_offset`` (cumulative sum of ``n_conformers`` in dataset row order) and ``local_offset``
    (offset within the row's shard). ``offset_mode`` selects how the stored offsets are
    interpreted. ``auto`` treats them as global only when they equal the cumulative conformer count
    of a multi-shard dataset (the layout written by ``convert rotation``); otherwise they are taken
    as per-shard offsets (the layout of ``EscheWang/3dcs``). Pass ``global`` explicitly for a subset
    of a dataset with global offsets.
    """
    if offset_mode not in OFFSET_MODES:
        raise ValueError(f"offset_mode must be one of {OFFSET_MODES}, got {offset_mode!r}")
    cols = [c for c in ("key", "shard", "n_conformers", "offset") if c in dataset.column_names]
    meta = dataset.select_columns(cols).to_pandas()
    n_conf = meta["n_conformers"].to_numpy(dtype=np.int64)
    idx = pd.DataFrame(
        {
            "row": np.arange(len(n_conf), dtype=np.int64),
            "key": meta["key"].astype(str).to_numpy(),
            "n_conformers": n_conf,
        }
    )
    idx["shard"] = meta["shard"].to_numpy(dtype=np.int64) if "shard" in meta else 0
    stored = meta["offset"].to_numpy(dtype=np.int64) if "offset" in meta else None
    flat = np.concatenate([[0], np.cumsum(n_conf)[:-1]]).astype(np.int64) if len(n_conf) else np.zeros(0, np.int64)
    idx["flat_offset"] = flat
    idx["offset"] = stored if stored is not None else flat

    mode = offset_mode
    if mode == "auto":
        if stored is None or (np.array_equal(stored, flat) and idx["shard"].nunique() > 1):
            mode = "global"
        else:
            mode = "per-shard"
    if mode == "per-shard":
        idx["local_offset"] = idx["offset"]
    else:
        shard_start = idx.groupby("shard", sort=False)["offset"].transform("min")
        idx["local_offset"] = idx["offset"] - shard_start
    idx.attrs["offset_mode"] = mode
    return idx


def validate_per_shard_offsets(index: pd.DataFrame) -> None:
    """Check that the conformer ranges of different rows do not overlap within a shard (raises ValueError)."""
    for shard, grp in index.groupby("shard", sort=False):
        g = grp.sort_values("local_offset")
        start = g["local_offset"].to_numpy()
        end = start + g["n_conformers"].to_numpy()
        if (start < 0).any() or (start[1:] < end[:-1]).any():
            raise ValueError(
                f"Conformer ranges of shard {shard} overlap or are negative; check --offset-mode "
                "(per-shard for EscheWang/3dcs, global for datasets written by 'convert rotation')."
            )


# ---------------------------------------------------------------------------
# Molecule selection
# ---------------------------------------------------------------------------


def read_molecule_list(path: str | Path) -> list[str]:
    """Read one key per line (``#`` comments and blank lines ignored; ``.gz`` supported)."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    keys = []
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                keys.append(line)
    return keys


def select_rows(
    index: pd.DataFrame,
    *,
    shards: Iterable[int] | None = None,
    molecule_list: str | Path | Iterable[str] | None = None,
    sample_ratio: float | None = None,
    sample_seed: int = 2027,
    min_conformers: int = 2,
    max_keys: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Select dataset rows to evaluate; returns the selected index and selection statistics.

    ``sample_ratio`` draws ``ceil(ratio * n_eligible)`` molecules per shard with
    ``numpy.random.default_rng(sample_seed + shard)``. This is a documented, reproducible
    sampler; it does **not** regenerate the 10 % sample of the published run, which is
    distributed as an explicit list (``reproduce/table1_geometry/sampled_molecules_seed2027.txt``)
    for use with ``molecule_list``.
    """
    stats: dict[str, Any] = {"n_rows_dataset": int(len(index))}
    sel = index
    if shards is not None:
        shard_set = {int(s) for s in shards}
        sel = sel[sel["shard"].isin(shard_set)]
        stats["shards"] = sorted(shard_set)
    if molecule_list is not None:
        keys = read_molecule_list(molecule_list) if isinstance(molecule_list, (str, Path)) else list(molecule_list)
        key_set = set(keys)
        present = set(index["key"])
        stats["molecule_list_size"] = len(key_set)
        stats["molecule_list_missing"] = len(key_set - present)
        sel = sel[sel["key"].isin(key_set)]
    n_before = len(sel)
    sel = sel[sel["n_conformers"] >= int(min_conformers)]
    stats["skipped_min_conformers"] = int(n_before - len(sel))
    if sample_ratio is not None:
        if not (0.0 < sample_ratio <= 1.0):
            raise ValueError("sample_ratio must be in (0, 1].")
        parts = []
        for shard, grp in sel.groupby("shard", sort=False):
            n_take = max(1, min(len(grp), int(math.ceil(len(grp) * sample_ratio))))
            rng = np.random.default_rng(int(sample_seed) + int(shard))
            take = np.sort(rng.choice(len(grp), size=n_take, replace=False))
            parts.append(grp.iloc[take])
        sel = pd.concat(parts) if parts else sel.iloc[:0]
        sel = sel.sort_values("row")
        stats["sample_ratio"] = sample_ratio
        stats["sample_seed"] = sample_seed
    if max_keys is not None:
        sel = sel.iloc[: int(max_keys)]
    stats["n_selected"] = int(len(sel))
    return sel, stats


# ---------------------------------------------------------------------------
# Embedding sources
# ---------------------------------------------------------------------------


def _load_array_file(path: Path, key: str | None) -> np.ndarray | list:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return np.asarray(data[select_array_key(data, key)])
    if suffix == ".npy":
        return np.load(path, mmap_mode="r")
    if suffix == ".pkl":
        import pickle

        with path.open("rb") as fh:
            payload = pickle.load(fh)
        if isinstance(payload, dict):
            if key is None:
                raise ValueError(f"{path} holds a dict; pass an embedding key.")
            payload = payload[key]
        return payload
    raise ValueError(f"Unsupported embedding file type: {path}")


def find_shard_files(directory: Path, pattern: str | None = None) -> dict[int, Path]:
    """Map shard id -> file for a by-shard embedding directory."""
    directory = Path(directory)
    patterns = [pattern] if pattern else list(SHARD_FILE_PATTERNS)
    for pat in patterns:
        regex = re.compile("^" + re.escape(pat).replace(re.escape("{shard}"), r"(\d+)") + "$")
        found = {}
        for p in directory.iterdir():
            m = regex.match(p.name)
            if m:
                found[int(m.group(1))] = p
        if found:
            return dict(sorted(found.items()))
    raise FileNotFoundError(f"No per-shard embedding files found in {directory} (patterns: {patterns})")


class ShardedEmbeddings:
    """Lazily loaded per-shard embedding arrays (small LRU cache)."""

    def __init__(
        self,
        files: Mapping[int, str | Path],
        *,
        key: str | None = None,
        cache_size: int = 2,
        arrays: Mapping[int, Any] | None = None,
    ) -> None:
        self.files = {int(k): Path(v) for k, v in files.items()}
        self.key = key
        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[int, Any] = OrderedDict()
        self._fixed = dict(arrays) if arrays is not None else {}

    @classmethod
    def from_directory(cls, directory: str | Path, *, key: str | None = None, pattern: str | None = None):
        return cls(find_shard_files(Path(directory), pattern), key=key)

    @classmethod
    def from_arrays(cls, arrays: Mapping[int, Any]):
        return cls({}, arrays=arrays)

    @property
    def shards(self) -> list[int]:
        return sorted(set(self.files) | set(self._fixed))

    def get(self, shard: int):
        shard = int(shard)
        if shard in self._fixed:
            return self._fixed[shard]
        if shard in self._cache:
            self._cache.move_to_end(shard)
            return self._cache[shard]
        if shard not in self.files:
            raise KeyError(f"No embedding file for shard {shard}; available shards: {self.shards}")
        arr = _load_array_file(self.files[shard], self.key)
        self._cache[shard] = arr
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return arr


def write_flat_embeddings(index: pd.DataFrame, sharded: ShardedEmbeddings, path: str | Path) -> np.ndarray:
    """Write per-shard embeddings as one ``.npy`` array in **dataset row order**.

    ``index`` is the output of :func:`build_rotation_index` for the dataset that will be
    evaluated with the flat layout (so its ``flat_offset`` column defines the target rows).
    The file is created with :func:`numpy.lib.format.open_memmap` (a valid ``.npy`` header), and
    the returned array is a read-only memory map of it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    order = index.sort_values("row")
    total = int(order["n_conformers"].sum())
    first = sharded.get(int(order["shard"].iloc[0]))
    if _is_fingerprint(first):
        raise ValueError("write_flat_embeddings supports numeric arrays only.")
    out = np.lib.format.open_memmap(path, mode="w+", dtype=first.dtype, shape=(total, first.shape[1]))
    shard_col = order["shard"].to_numpy()
    starts = np.flatnonzero(np.r_[True, shard_col[1:] != shard_col[:-1]])
    ends = np.r_[starts[1:], len(order)]
    local = order["local_offset"].to_numpy()
    counts = order["n_conformers"].to_numpy()
    flat = order["flat_offset"].to_numpy()
    for a, b in zip(starts, ends):
        arr = sharded.get(int(shard_col[a]))
        lo = int(local[a])
        hi = int(local[b - 1] + counts[b - 1])
        if not np.array_equal(local[a + 1 : b], (local[a:b] + counts[a:b])[:-1]):
            raise ValueError(f"Rows of shard {shard_col[a]} are not contiguous in dataset order.")
        pos = int(flat[a])
        out[pos : pos + (hi - lo)] = arr[lo:hi]
    out.flush()
    del out
    return np.load(path, mmap_mode="r")


@dataclass
class _Source:
    layout: str
    flat: EmbeddingArray | None = None
    sharded: ShardedEmbeddings | None = None
    by_key: Mapping[str, Any] | None = None


def _is_fingerprint(obj: Any) -> bool:
    return isinstance(obj, (list, tuple))


def _slice(source: _Source, row: dict):
    n = int(row["n_conformers"])
    if source.layout == "flat":
        start = int(row["flat_offset"])
        arr = source.flat.array
        return arr[start : start + n], source.flat.kind == "fingerprint" or _is_fingerprint(arr)
    if source.layout == "by-shard":
        arr = source.sharded.get(int(row["shard"]))
        start = int(row["local_offset"])
        if start + n > len(arr):
            raise ValueError(
                f"Shard {row['shard']} embeddings have {len(arr)} rows but key {row['key']} needs rows "
                f"[{start}, {start + n}). Check the offset layout (--offset-mode)."
            )
        return arr[start : start + n], _is_fingerprint(arr)
    Z = source.by_key[row["key"]]
    if not _is_fingerprint(Z) and len(Z) != n:
        raise ValueError(f"Key {row['key']}: {len(Z)} embeddings for {n} conformers")
    return Z, _is_fingerprint(Z)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

_STATE: dict[str, Any] = {}


def _pin_threads() -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    try:  # pragma: no cover - optional dependency
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except Exception:
        pass


def _evaluate_rows(rows: list[dict]) -> list[dict]:
    from three_dbench.datasets.serialization import blocks_to_mols
    from three_dbench.rotation.evaluation import (
        pairwise_distances_from_embeddings,
        pairwise_distances_from_fingerprint,
        rmsd_matrix_from_one_mol,
    )
    from three_dbench.rotation.metrics import compute_geometry_metrics

    dataset = _STATE["dataset"]
    source: _Source = _STATE["source"]
    spec = _STATE["spec"]
    metrics: Sequence[str] = _STATE["metrics"]
    out = []
    for row in rows:
        rec = dataset[int(row["row"])]
        if not rec["mol_blocks"]:
            raise ValueError("Rotation dataset is missing MolBlocks. Rebuild without --no-mol-blocks.")
        Z, is_fp = _slice(source, row)
        base = {"key": row["key"], "shard": int(row["shard"]), "n_conformers": int(row["n_conformers"])}
        try:
            mols = blocks_to_mols(rec["mol_blocks"], sanitize=False)
            D = rmsd_matrix_from_one_mol(mols)
        except (ValueError, RuntimeError) as exc:
            # e.g. conformers whose heavy-atom counts differ after RemoveHs; reported, not fatal
            out.append({**base, "space": None, "error": f"{type(exc).__name__}: {exc}"})
            continue
        deg = rec.get("torsion_deg")
        if is_fp:
            Delta = pairwise_distances_from_fingerprint(list(Z))
            vals = compute_geometry_metrics(D, Delta, torsion_deg=deg, spec=spec, Z=None, delta_metric="tanimoto")
            out.append({**base, "space": "tanimoto", **vals})
            continue
        Zd = np.asarray(Z, dtype=np.float64)
        for metric in metrics:
            Delta = pairwise_distances_from_embeddings(Zd, metric=metric)
            vals = compute_geometry_metrics(D, Delta, torsion_deg=deg, spec=spec, Z=Zd, delta_metric=metric)
            out.append({**base, "space": metric, **vals})
    return out


def _run_rows(records: list[dict], *, n_jobs: int, chunk_size: int, progress: bool, t0: float) -> list[dict]:
    chunks = [records[i : i + chunk_size] for i in range(0, len(records), chunk_size)]
    results: list[dict] = []
    if n_jobs <= 1 or len(chunks) <= 1:
        for i, chunk in enumerate(chunks):
            results.extend(_evaluate_rows(chunk))
            if progress and (i + 1) % 20 == 0:
                print(f"[rotation] {len(results)} rows done, {time.time() - t0:.0f}s", flush=True)
        return results
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    try:
        ctx = mp.get_context("fork")
    except ValueError as exc:  # pragma: no cover - platforms without fork
        raise RuntimeError("n_jobs > 1 requires the 'fork' start method (Linux).") from exc
    with ProcessPoolExecutor(max_workers=int(n_jobs), mp_context=ctx, initializer=_pin_threads) as ex:
        for i, part in enumerate(ex.map(_evaluate_rows, chunks)):
            results.extend(part)
            if progress and (i + 1) % 50 == 0:
                print(f"[rotation] {i + 1}/{len(chunks)} chunks, {time.time() - t0:.0f}s", flush=True)
    return results


def _offset_drift_records(selected: pd.DataFrame, failed_keys: set[str]) -> tuple[list[dict], dict]:
    """Rows after a skipped molecule of the same shard, with the shifted per-shard offset."""
    records: list[dict] = []
    info: dict[str, Any] = {}
    for shard, grp in selected.groupby("shard", sort=False):
        g = grp.sort_values("local_offset")
        failed = g["key"].isin(failed_keys).to_numpy()
        if not failed.any():
            continue
        shift = np.cumsum(np.where(failed, g["n_conformers"].to_numpy(), 0))
        shift_before = np.concatenate([[0], shift[:-1]])
        affected = (shift_before > 0) & ~failed
        sub = g[affected].copy()
        sub["local_offset"] = sub["local_offset"].to_numpy() - shift_before[affected]
        records.extend(sub[["row", "key", "shard", "n_conformers", "flat_offset", "local_offset"]].to_dict("records"))
        info[str(int(shard))] = {
            "failed_keys": g.loc[failed, "key"].tolist(),
            "first_failed_local_offset": int(g.loc[failed, "local_offset"].iloc[0]),
            "n_affected_molecules": int(affected.sum()),
            "max_shift_rows": int(shift[-1]),
        }
    return records, info


def _merge_offset_drift(per_key: pd.DataFrame, drift_rows: list[dict]) -> pd.DataFrame:
    metric_cols = [c for c in per_key.columns if c not in {"key", "shard", "n_conformers", "space"}]
    out = per_key.copy()
    for c in metric_cols:
        out[f"{c}__offset_drift"] = out[c]
    if not drift_rows:
        return out
    drift = pd.DataFrame(drift_rows).set_index(["key", "space"])
    pos = out.set_index(["key", "space"]).index
    hit = pos.isin(drift.index)
    aligned = drift.reindex(pos[hit])
    for c in metric_cols:
        out.loc[hit, f"{c}__offset_drift"] = aligned[c].to_numpy()
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def summarize_rotation_results(per_key: pd.DataFrame, *, model_name: str, metric_version: str) -> pd.DataFrame:
    """Mean / median over molecules of every metric (non-finite values excluded)."""
    rows = []
    if per_key.empty:
        return pd.DataFrame(columns=["model", "metric_version", "space", "metric", "mean", "median", "n", "n_finite"])
    metric_cols = [c for c in per_key.columns if c not in {"key", "shard", "n_conformers", "space"}]
    for space, grp in per_key.groupby("space", sort=False):
        for col in metric_cols:
            vals = grp[col].to_numpy(dtype=np.float64)
            fin = vals[np.isfinite(vals)]
            rows.append(
                {
                    "model": model_name,
                    "metric_version": metric_version,
                    "space": space,
                    "metric": col,
                    "mean": float(fin.mean()) if fin.size else float("nan"),
                    "median": float(np.median(fin)) if fin.size else float("nan"),
                    "n": int(vals.size),
                    "n_finite": int(fin.size),
                }
            )
    return pd.DataFrame(rows)


def evaluate_rotation_embeddings(
    *,
    dataset_dir: Path | None = None,
    dataset: Any = None,
    embeddings: EmbeddingArray | None = None,
    embeddings_by_shard: ShardedEmbeddings | Mapping[int, Any] | None = None,
    embeddings_by_key: Mapping[str, Any] | None = None,
    output_dir: Path | None = None,
    model_name: str = "custom",
    metrics: Iterable[str] = ("cosine", "euclidean"),
    metric_version: str = "paper",
    lie_k: int | None = None,
    lie_include_self: bool | None = None,
    as_variant: str | None = None,
    extra_metrics: bool | None = None,
    offset_mode: str = "auto",
    shards: Iterable[int] | None = None,
    molecule_list: str | Path | Iterable[str] | None = None,
    sample_ratio: float | None = None,
    sample_seed: int = 2027,
    min_conformers: int = 2,
    max_keys: int | None = None,
    n_jobs: int = 1,
    chunk_size: int = 256,
    progress: bool = False,
    replicate_offset_drift: bool = False,
) -> dict[str, Any]:
    """Evaluate rotation embeddings against the HF rotation dataset.

    Exactly one of ``embeddings`` (flat, dataset row order), ``embeddings_by_shard`` (per-shard
    arrays aligned with per-shard offsets) or ``embeddings_by_key`` must be given.

    Molecules whose RMSD matrix cannot be computed (e.g. conformers with different heavy-atom
    counts after ``RemoveHs``) are skipped and listed in ``config["selection"]["failed_keys"]``.

    ``replicate_offset_drift`` (by-shard layout only) additionally recomputes every metric with the
    embedding indexing of the original full geometry run (Table 1 LIE@k / AS rows): there, the rows
    of every molecule after a skipped one of the same shard are shifted back by the skipped
    molecule's conformer count. These values are stored in extra columns ``<metric>__offset_drift``;
    the regular columns index the embeddings by the dataset ``offset``. Use it with complete shards,
    since only skipped molecules among the evaluated ones are detected.

    Returns a dict with ``per_key`` (DataFrame, one row per key and distance space),
    ``summary`` (DataFrame) and ``config`` (dict). When ``output_dir`` is given, writes
    ``{model_name}_per_key.parquet``, ``summary.csv`` and ``config.json``.
    """
    from three_dbench.rotation.metrics import resolve_metric_spec

    given = [x is not None for x in (embeddings, embeddings_by_shard, embeddings_by_key)]
    if sum(given) != 1:
        raise ValueError("Provide exactly one of embeddings, embeddings_by_shard or embeddings_by_key.")
    if dataset is None:
        if dataset_dir is None:
            raise ValueError("Provide dataset_dir or dataset.")
        from three_dbench.datasets.rotation import load_rotation_dataset

        dataset = load_rotation_dataset(dataset_dir)
    spec = resolve_metric_spec(
        metric_version,
        lie_k=lie_k,
        lie_include_self=lie_include_self,
        as_variant=as_variant,
        extra_metrics=extra_metrics,
    )
    metrics = list(metrics)

    t0 = time.time()
    index = build_rotation_index(dataset, offset_mode=offset_mode)
    if embeddings is not None:
        source = _Source("flat", flat=embeddings)
        total = int(index["n_conformers"].sum())
        if len(embeddings.array) < total:
            raise ValueError(f"Flat embeddings have {len(embeddings.array)} rows < {total} conformers in the dataset.")
    elif embeddings_by_shard is not None:
        sharded = (
            embeddings_by_shard
            if isinstance(embeddings_by_shard, ShardedEmbeddings)
            else ShardedEmbeddings.from_arrays(embeddings_by_shard)
        )
        validate_per_shard_offsets(index)
        source = _Source("by-shard", sharded=sharded)
    else:
        source = _Source("by-key", by_key=embeddings_by_key)

    selected, stats = select_rows(
        index,
        shards=shards,
        molecule_list=molecule_list,
        sample_ratio=sample_ratio,
        sample_seed=sample_seed,
        min_conformers=min_conformers,
        max_keys=max_keys,
    )
    if source.layout == "by-shard":
        missing = sorted(set(selected["shard"].unique()) - set(source.sharded.shards))
        if missing:
            raise ValueError(
                f"No embedding file for shards {missing}. Restrict the evaluation with shards=... "
                f"(available: {source.sharded.shards})."
            )
        selected = selected.sort_values(["shard", "local_offset"], kind="stable")

    if replicate_offset_drift and source.layout != "by-shard":
        raise ValueError("replicate_offset_drift requires the by-shard layout.")
    records = selected[["row", "key", "shard", "n_conformers", "flat_offset", "local_offset"]].to_dict("records")
    _STATE.update(dataset=dataset, source=source, spec=spec, metrics=metrics)
    try:
        results = _run_rows(records, n_jobs=n_jobs, chunk_size=chunk_size, progress=progress, t0=t0)
        failures = [r for r in results if r.get("space") is None]
        drift_results: list[dict] = []
        drift_info: dict[str, Any] = {}
        if replicate_offset_drift and failures:
            drift_records, drift_info = _offset_drift_records(selected, {r["key"] for r in failures})
            if progress:
                print(f"[rotation] offset-drift pass over {len(drift_records)} molecules", flush=True)
            drift_results = _run_rows(drift_records, n_jobs=n_jobs, chunk_size=chunk_size, progress=progress, t0=t0)
    finally:
        _STATE.clear()

    per_key = pd.DataFrame([r for r in results if r.get("space") is not None])
    stats["n_failed"] = len(failures)
    stats["failed_keys"] = {r["key"]: r["error"] for r in failures}
    if failures:
        print(f"[rotation] {len(failures)} molecule(s) could not be evaluated (see config.json: failed_keys)")
    if replicate_offset_drift and not per_key.empty:
        per_key = _merge_offset_drift(per_key, [r for r in drift_results if r.get("space") is not None])
        stats["offset_drift"] = drift_info
    summary = summarize_rotation_results(per_key, model_name=model_name, metric_version=metric_version)
    config = {
        "model_name": model_name,
        "metric_version": metric_version,
        "metric_spec": spec.to_dict(),
        "distance_spaces": metrics,
        "layout": source.layout,
        "offset_mode": index.attrs.get("offset_mode"),
        "selection": stats,
        "n_jobs": int(n_jobs),
        "runtime_seconds": round(time.time() - t0, 2),
    }
    try:
        from three_dbench import __version__

        config["three_dbench_version"] = __version__
    except Exception:  # pragma: no cover
        pass
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        per_key.to_parquet(output_dir / f"{model_name}_per_key.parquet", index=False)
        summary.to_csv(output_dir / "summary.csv", index=False)
        (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return {"per_key": per_key, "summary": summary, "config": config}


__all__ = [
    "LAYOUTS",
    "OFFSET_MODES",
    "TABLE1_METRICS",
    "ShardedEmbeddings",
    "build_rotation_index",
    "evaluate_rotation_embeddings",
    "find_shard_files",
    "read_molecule_list",
    "select_rows",
    "summarize_rotation_results",
    "validate_per_shard_offsets",
    "write_flat_embeddings",
]
