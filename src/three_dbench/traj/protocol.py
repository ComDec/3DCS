"""Sampling protocol and per-window computation for the trajectory (energy) benchmark.

Window schemes
--------------
``legacy`` (default; reproduces the published Tables 3, 6 and 7)
    For every molecule a fresh ``numpy.random.default_rng(seed)`` draws ``n_samples`` starts with
    ``integers(0, legacy_traj_len - window)``; ``legacy_traj_len`` is 100,000 (the nominal rMD17
    length, also used for azobenzene with 99,988 frames). Every molecule therefore gets the same
    starts, and a molecule's windows do not depend on which other molecules are evaluated. This is
    exactly :func:`three_dbench.traj.evaluation.run_once_for_molecule`, the runner behind the paper.
    Molecules are processed in the legacy order (``DEFAULT_MOL_TYPES``), which fixes the row order of
    the details and the floating-point summation order of the summary.
``shared``
    The scheme of the 0.1.0 CLI: one generator shared across molecules in dataset row order, with
    ``integers(0, n_frames - window, size=min(n_samples, n_frames - window))``.

Metric versions
---------------
``paper`` (default): the implementation used for the published numbers (float16 distances, robust
EJS sigma, shared CKA bandwidth, ...). ``v2``: definitions from the paper appendix, see
:mod:`three_dbench.traj.metrics_v2` and ``docs/metrics/energy.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np

from .evaluation import (
    DEFAULT_MOL_TYPES,
    compute_energy_metrics_from_condensed,
    ks_wasserstein_against_energy_diff,
    pairwise_distances_from_embeddings_large,
    pairwise_tanimoto_fps_large,
    thresholded_smoothness,
)
from .metrics_v2 import compute_energy_metrics_v2

WINDOW_SCHEMES = ("legacy", "shared")
METRIC_VERSIONS = ("paper", "v2")
EMBED_METRICS = ("cosine", "euclidean", "tanimoto")
LEGACY_TRAJ_LEN = 100_000
LEGACY_MOL_ORDER = tuple(DEFAULT_MOL_TYPES)

DISTANCE_DTYPE = {"paper": np.float16, "v2": np.float32}


def resolve_molecules(requested: Sequence[str] | None, available: Sequence[str]) -> list[str]:
    """Match requested molecule names (``aspirin`` or ``rmd17_aspirin``) against dataset names."""
    available = list(available)
    if not requested:
        return available
    out = []
    for name in requested:
        name = str(name).strip()
        if name in available:
            match = name
        elif f"rmd17_{name}" in available:
            match = f"rmd17_{name}"
        else:
            raise KeyError(f"Molecule {name!r} not in dataset (available: {available})")
        if match not in out:
            out.append(match)
    return out


def order_molecules(molecules: Sequence[str], window_scheme: str) -> list[str]:
    """Evaluation order: legacy order for the legacy scheme, otherwise the given (dataset) order."""
    molecules = list(molecules)
    if window_scheme != "legacy":
        return molecules
    rank = {m: i for i, m in enumerate(LEGACY_MOL_ORDER)}
    return sorted(molecules, key=lambda m: (rank.get(m, len(rank)), molecules.index(m)))


def draw_window_starts(
    n_frames_by_mol: Mapping[str, int],
    *,
    window_scheme: str = "legacy",
    n_samples: int = 100,
    window: int = 2000,
    random_seed: int = 2025,
    legacy_traj_len: int = LEGACY_TRAJ_LEN,
) -> dict[str, np.ndarray]:
    """Draw window start indices for every molecule of ``n_frames_by_mol`` (in dataset order).

    For the ``shared`` scheme the mapping must contain *all* dataset molecules in dataset order,
    because each molecule's draw depends on the draws before it.
    """
    if window_scheme not in WINDOW_SCHEMES:
        raise ValueError(f"window_scheme must be one of {WINDOW_SCHEMES}, got {window_scheme!r}")
    if window < 2:
        raise ValueError("window must be >= 2")
    starts: dict[str, np.ndarray] = {}
    if window_scheme == "legacy":
        max_start = int(legacy_traj_len) - int(window)
        if max_start <= 0:
            raise ValueError("legacy_traj_len must be greater than window to draw samples.")
        for mol_type, n_frames in n_frames_by_mol.items():
            rng = np.random.default_rng(random_seed)
            s = rng.integers(0, max_start, size=n_samples, endpoint=False)
            if s.size and int(s.max()) + window > int(n_frames):
                raise ValueError(
                    f"Legacy window [{int(s.max())}, {int(s.max()) + window}) exceeds the {n_frames} frames of "
                    f"{mol_type}; use --window-scheme shared or a smaller --legacy-traj-len."
                )
            starts[mol_type] = s
        return starts

    rng = np.random.default_rng(random_seed)
    for mol_type, n_frames in n_frames_by_mol.items():
        max_start = int(n_frames) - int(window)
        if max_start <= 0:
            raise ValueError(f"Window size {window} exceeds trajectory length for {mol_type}.")
        starts[mol_type] = rng.integers(0, max_start, size=min(n_samples, max_start), endpoint=False)
    return starts


def _is_fingerprints(X: Any) -> bool:
    return isinstance(X, (list, tuple))


def resolve_embed_metric(metric_embed: str | None, X: Any) -> str:
    """``None`` selects Tanimoto for fingerprint lists and cosine for vectors."""
    if metric_embed is None:
        return "tanimoto" if _is_fingerprints(X) else "cosine"
    if metric_embed not in EMBED_METRICS:
        raise ValueError(f"metric_embed must be one of {EMBED_METRICS}, got {metric_embed!r}")
    if _is_fingerprints(X) and metric_embed != "tanimoto":
        raise ValueError("Fingerprint (bit-vector) embeddings support only the tanimoto distance.")
    return metric_embed


def tanimoto_distances_dense(X: np.ndarray, *, dtype_out=np.float16) -> np.ndarray:
    """Condensed ``1 - Tanimoto`` distances for a dense binary matrix (non-zero entries are on-bits).

    Matches RDKit ``BulkTanimotoSimilarity`` on the equivalent bit vectors (similarity 0 when both
    vectors are empty), including the float32 intermediate used by the fingerprint path.
    """
    B = (np.asarray(X) != 0).astype(np.float64)
    n = B.shape[0]
    counts = B.sum(axis=1)
    inter = B @ B.T
    union = counts[:, None] + counts[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        sim = np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)
    iu = np.triu_indices(n, 1)
    return (1.0 - sim.astype(np.float32)[iu]).astype(dtype_out, copy=False)


def window_distances(
    X: Any,
    *,
    metric: str,
    metric_version: str = "paper",
    block_size: int = 4096,
) -> np.ndarray:
    """Condensed representation distances for one window (``X`` already sliced)."""
    dtype_out = DISTANCE_DTYPE[metric_version]
    if metric == "tanimoto":
        if _is_fingerprints(X):
            return pairwise_tanimoto_fps_large(
                list(X), block_size=block_size, out_mode="condensed", dtype_out=dtype_out, progress=False
            )
        return tanimoto_distances_dense(X, dtype_out=dtype_out)
    return pairwise_distances_from_embeddings_large(
        X, metric=metric, block_size=block_size, out_mode="condensed", dtype_out=dtype_out, progress=False
    )


def compute_window_metrics(
    D: np.ndarray,
    E: np.ndarray,
    *,
    metric_version: str = "paper",
    time_ordered: bool = False,
) -> dict[str, float]:
    """All energy metrics for one window."""
    if metric_version == "paper":
        m1 = compute_energy_metrics_from_condensed(D, E)
        m2 = thresholded_smoothness(D, E, eps=1e-12)
        m3 = ks_wasserstein_against_energy_diff(D, E)
        return {**m1, **m2, **m3}
    if metric_version == "v2":
        return compute_energy_metrics_v2(D, E, time_ordered=time_ordered)
    raise ValueError(f"metric_version must be one of {METRIC_VERSIONS}, got {metric_version!r}")


@contextmanager
def single_threaded_blas():
    """Limit BLAS/OpenMP pools to one thread (reproducible float32 BLAS results, no oversubscription)."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - threadpoolctl ships with scikit-learn
        yield
        return
    with threadpool_limits(limits=1):
        yield
