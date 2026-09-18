"""Trajectory (energy) benchmark evaluation for user-provided embeddings.

Defaults reproduce the protocol of the published energy tables (paper Tables 3, 6 and 7):
float64 energies, the ``legacy`` window scheme (per-molecule ``default_rng(2025)``, 100 windows of
2,000 frames drawn from ``[0, 98,000)``), ``paper`` metric definitions, cosine distance for vectors and
Tanimoto distance for fingerprints, and mean +/- 1.96 sd/sqrt(n) pooled over all molecule-windows.
See ``docs/metrics/energy.md``.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from three_dbench.datasets.traj import check_energy_precision, detect_quantized_energies, load_traj_energy_dataset
from three_dbench.traj.io import embedding_length, resolve_embedding
from three_dbench.traj.protocol import (
    LEGACY_TRAJ_LEN,
    METRIC_VERSIONS,
    WINDOW_SCHEMES,
    compute_window_metrics,
    draw_window_starts,
    order_molecules,
    resolve_embed_metric,
    resolve_molecules,
    single_threaded_blas,
    window_distances,
)


def _mean_ci(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size < 2:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    ci95 = 1.96 * std / np.sqrt(arr.size)
    return mean, float(ci95)


def _window_rows(task: dict[str, Any]) -> list[dict[str, Any]]:
    D = window_distances(
        task["X"],
        metric=task["metric"],
        metric_version=task["metric_version"],
        block_size=task["block_size"],
    )
    metrics = compute_window_metrics(
        D,
        task["E"],
        metric_version=task["metric_version"],
        time_ordered=task["time_ordered"],
    )
    return [
        {
            "mol_type": task["mol_type"],
            "sample_start": task["start"],
            "sample_end": task["end"],
            "model": task["model"],
            "metric": metric,
            "value": float(value) if np.isscalar(value) else np.nan,
        }
        for metric, value in metrics.items()
    ]


_WORKER_LIMITS: Any = None


def _worker_init() -> None:
    global _WORKER_LIMITS
    try:
        from threadpoolctl import threadpool_limits

        _WORKER_LIMITS = threadpool_limits(limits=1)
    except ImportError:  # pragma: no cover
        _WORKER_LIMITS = None


def _summarize(details_df: pd.DataFrame) -> pd.DataFrame:
    if details_df.empty:
        return pd.DataFrame(columns=["model", "metric", "mean", "ci95", "n"])
    grouped = details_df.groupby(["model", "metric"])["value"].apply(list).reset_index()
    summary_rows = []
    for _, row in grouped.iterrows():
        mean_val, ci_val = _mean_ci(row["value"])
        summary_rows.append(
            {
                "model": row["model"],
                "metric": row["metric"],
                "mean": mean_val,
                "ci95": ci_val,
                "n": int(np.isfinite(np.asarray(row["value"], dtype=float)).sum()),
            }
        )
    return pd.DataFrame(summary_rows)


def evaluate_trajectory_embeddings(
    *,
    dataset_dir: Path,
    embeddings_by_mol: Mapping[str, Any],
    output_dir: Path | None = None,
    model_name: str = "custom",
    n_samples: int = 100,
    window: int = 2000,
    metric_embed: str | None = None,
    block_size: int = 4096,
    random_seed: int = 2025,
    window_scheme: str = "legacy",
    metric_version: str = "paper",
    n_jobs: int = 1,
    molecules: Sequence[str] | None = None,
    energy_precision_check: str = "error",
    time_ordered: bool = False,
    legacy_traj_len: int = LEGACY_TRAJ_LEN,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate trajectory embeddings against the energy dataset saved at ``dataset_dir``.

    Args:
        dataset_dir: ``save_to_disk`` copy of the ``traj_energies`` config (columns ``mol_type``,
            ``n_frames``, ``energies``).
        embeddings_by_mol: ``{mol_type: array | list of RDKit bit vectors | zero-arg loader}``.
        metric_embed: ``cosine`` / ``euclidean`` / ``tanimoto``; ``None`` picks Tanimoto for
            fingerprint lists and cosine for vectors (the published setting).
        window_scheme: ``legacy`` (published) or ``shared`` (0.1.0 CLI behaviour).
        metric_version: ``paper`` (published definitions) or ``v2`` (appendix definitions).
        n_jobs: worker processes (windows are evaluated in parallel; results are order-preserving).
        molecules: optional subset (``aspirin`` or ``rmd17_aspirin``).
        energy_precision_check: ``error`` / ``warn`` / ``ignore`` for quantized energies.
        time_ordered: declare frames to be in simulation-time order (v2 ``TS``/``Smoothness`` are NaN
            otherwise). rMD17 frames are not time-ordered.
        legacy_traj_len: trajectory length assumed by the legacy scheme (100,000 for rMD17).

    Returns:
        ``(details_df, summary_df)``; also written to ``output_dir`` as ``details.csv``,
        ``summary.csv`` and ``config.json`` when given.
    """
    if window_scheme not in WINDOW_SCHEMES:
        raise ValueError(f"window_scheme must be one of {WINDOW_SCHEMES}, got {window_scheme!r}")
    if metric_version not in METRIC_VERSIONS:
        raise ValueError(f"metric_version must be one of {METRIC_VERSIONS}, got {metric_version!r}")
    t0 = time.time()
    dataset = load_traj_energy_dataset(dataset_dir)
    dataset_mols = [str(m) for m in dataset["mol_type"]]
    n_frames_all = {m: int(n) for m, n in zip(dataset_mols, dataset["n_frames"])}

    selected = resolve_molecules(molecules, dataset_mols)
    missing = [m for m in selected if m not in embeddings_by_mol]
    if missing:
        raise KeyError(f"No embeddings for {missing}; provide them or restrict with molecules=/--molecules")

    starts_all = draw_window_starts(
        n_frames_all if window_scheme == "shared" else {m: n_frames_all[m] for m in selected},
        window_scheme=window_scheme,
        n_samples=n_samples,
        window=window,
        random_seed=random_seed,
        legacy_traj_len=legacy_traj_len,
    )
    ordered = order_molecules(selected, window_scheme)

    energies: dict[str, np.ndarray] = {}
    reports = []
    energy_sha256: dict[str, str] = {}
    row_index = {m: i for i, m in enumerate(dataset_mols)}
    for mol_type in ordered:
        row = dataset[row_index[mol_type]]
        E = np.asarray(row["energies"], dtype=np.float64)
        if E.shape[0] != n_frames_all[mol_type]:
            raise ValueError(f"n_frames {n_frames_all[mol_type]} != len(energies) {E.shape[0]} for {mol_type}")
        energies[mol_type] = E
        energy_sha256[mol_type] = hashlib.sha256(np.ascontiguousarray(E).tobytes()).hexdigest()
        reports.append(detect_quantized_energies(E, mol_type=mol_type))
    flagged = check_energy_precision(reports, policy=energy_precision_check)

    resolved_metric: dict[str, str] = {}

    def _tasks() -> Iterator[dict[str, Any]]:
        for mol_type in ordered:
            X_all = resolve_embedding(embeddings_by_mol[mol_type])
            E = energies[mol_type]
            if embedding_length(X_all) != E.shape[0]:
                raise ValueError(f"Length mismatch for {mol_type}: {embedding_length(X_all)} != {E.shape[0]}")
            metric = resolve_embed_metric(metric_embed, X_all)
            resolved_metric[mol_type] = metric
            if verbose:
                print(
                    f"[{model_name}] {mol_type}: {len(starts_all[mol_type])} windows x {window} frames "
                    f"({metric}, {metric_version}, {window_scheme})",
                    flush=True,
                )
            for start in starts_all[mol_type]:
                s = int(start)
                e = s + int(window)
                yield {
                    "mol_type": mol_type,
                    "start": s,
                    "end": e,
                    "model": model_name,
                    "E": E[s:e],
                    "X": X_all[s:e],
                    "metric": metric,
                    "metric_version": metric_version,
                    "block_size": block_size,
                    "time_ordered": time_ordered,
                }

    details_rows: list[dict[str, Any]] = []
    if n_jobs is None or n_jobs <= 1:
        with single_threaded_blas():
            for task in _tasks():
                details_rows.extend(_window_rows(task))
    else:
        with mp.get_context().Pool(processes=int(n_jobs), initializer=_worker_init) as pool:
            for rows in pool.imap(_window_rows, _tasks(), chunksize=1):
                details_rows.extend(rows)

    details_df = pd.DataFrame(
        details_rows, columns=["mol_type", "sample_start", "sample_end", "model", "metric", "value"]
    )
    summary_df = _summarize(details_df)
    elapsed = time.time() - t0

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        details_df.to_csv(output_dir / "details.csv", index=False)
        summary_df.to_csv(output_dir / "summary.csv", index=False)
        config = {
            "model": model_name,
            "n_samples": n_samples,
            "window": window,
            "metric_embed": metric_embed,
            "metric_embed_resolved": resolved_metric,
            "block_size": block_size,
            "random_seed": random_seed,
            "window_scheme": window_scheme,
            "legacy_traj_len": legacy_traj_len if window_scheme == "legacy" else None,
            "metric_version": metric_version,
            "time_ordered": time_ordered,
            "molecules": ordered,
            "n_jobs": n_jobs,
            "energy_precision_check": energy_precision_check,
            "energy_precision_flagged": [r.as_dict() for r in flagged],
            "energy_sha256": energy_sha256,
            "window_starts": {m: [int(s) for s in starts_all[m]] for m in ordered},
            "dataset_dir": str(dataset_dir),
            "elapsed_seconds": round(elapsed, 3),
            "versions": _versions(),
        }
        (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    return details_df, summary_df


def _versions() -> dict[str, str]:
    import platform

    import scipy
    import sklearn

    out = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}
    out["scikit-learn"] = sklearn.__version__
    out["pandas"] = pd.__version__
    try:
        from importlib.metadata import version

        out["3dcs"] = version("3dcs")
    except Exception:
        pass
    return out
