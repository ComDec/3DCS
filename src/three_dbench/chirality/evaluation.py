"""Chirality (stereoisomer separation) metrics for 3DCS.

The per-molecule protocol and every metric definition are documented in
``docs/metrics/chirality.md``. Two switches control the definitions:

``distance``
    Representation distance used for continuous embeddings: ``"euclidean"`` (default; this is
    what produced the published Table 2) or ``"cosine"``. RDKit fingerprints always use Tanimoto
    distance, whatever ``distance`` is.

``metric_version``
    ``"paper"`` (default) reproduces the code path that produced the published numbers.
    ``"v2"`` applies the corrected definitions listed in ``docs/metrics/chirality.md``
    (tie-aware NN1 restricted to points with a same-class partner, centroid DBI, explicit Hopkins
    population, best-k silhouette on the selected distance).

Importing this module has no side effects (no directories are created, no files are read).
"""

import functools
import itertools
import json
import os
import pickle
import re
import warnings
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

# ===== RDKit fingerprint support =====
try:
    from rdkit import DataStructs as DS
    from rdkit import rdBase

    _HAS_RDKIT = True
except Exception:
    DS = None
    rdBase = None
    _HAS_RDKIT = False

# ===== Optional sklearn support (preferred when available) =====
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances, roc_auc_score, silhouette_score

    _HAS_SK = True
except Exception:
    roc_auc_score = None
    silhouette_score = None
    KMeans = None
    pairwise_distances = None
    _HAS_SK = False

_silhouette_score = silhouette_score

DISTANCES = ("euclidean", "cosine")
METRIC_VERSIONS = ("paper", "v2")
#: Hopkins statistic is only computed for molecules with at least this many conformers.
HOPKINS_MIN_N = 10
#: Small constant added to between-cluster distances in the v2 Davies-Bouldin index (App. C.5).
DBI_EPS = 1e-12


# ===================== 1) Parse key: mol_id + en_id =====================
_KEY_RE = re.compile(r"^(?P<mol>[^:]+)::en(?P<en>\d+)(?:_.*)?$")


def parse_key_en(key: str) -> tuple[str, str]:
    """Split a composite key into ``(mol_id, en_id)``.

    Example: ``'CHEMBL100259::en3_A5:S;A7:S;A10:R;A12:S'`` -> ``('CHEMBL100259', '3')``.
    """
    m = _KEY_RE.match(key)
    if m is None:
        return key, "0"
    return m.group("mol"), m.group("en")


# ================== 2) Flatten indices (aligned with embedding order) ==================
@dataclass
class FlatIndex:
    keys: list[str]  # Keys in flattened order
    counts: list[int]  # Sample count per key
    offsets: list[int]  # Global offset per key
    mol_to_indices: dict[str, np.ndarray]  # mol_id -> global indices
    en_labels: np.ndarray  # Global array of en labels
    mol_labels: np.ndarray  # Global array of mol_ids


def build_flat_index_from_counts(key_to_counts: dict[str, int]) -> FlatIndex:
    """Build flat indices from a mapping of key to conformer counts (dataset row order)."""
    keys, counts, offsets = [], [], []
    mol_idx_map: dict[str, list[int]] = {}
    en_labels = []
    mol_labels = []

    cursor = 0
    for k, n in key_to_counts.items():
        n = int(n)
        if n < 0:
            raise ValueError(f"Negative conformer count {n} for key {k!r}.")
        keys.append(k)
        counts.append(n)
        offsets.append(cursor)

        mol_id, en_id = parse_key_en(k)
        if n > 0:
            idxs = list(range(cursor, cursor + n))
            mol_idx_map.setdefault(mol_id, []).extend(idxs)
            en_labels.extend([en_id] * n)
            mol_labels.extend([mol_id] * n)

        cursor += n

    en_labels = np.asarray(en_labels, dtype=object)
    mol_labels = np.asarray(mol_labels, dtype=object)
    mol_to_indices = {mol: np.asarray(ix, dtype=int) for mol, ix in mol_idx_map.items()}

    return FlatIndex(keys, counts, offsets, mol_to_indices, en_labels, mol_labels)


def build_flat_index(key_to_mols: dict[str, list[Any]]) -> FlatIndex:
    """Build flat indices from a mapping of key to conformer lists."""
    return build_flat_index_from_counts(_counts_from_mols(key_to_mols))


def _counts_from_mols(key_to_mols: dict[str, list[Any]]) -> dict[str, int]:
    return {k: (len(v) if v is not None else 0) for k, v in key_to_mols.items()}


# ===================== 3) Distance computation (continuous / fingerprint) =====================
def is_fingerprint_list(embeddings: Any) -> bool:
    """Heuristically detect whether a sequence contains RDKit fingerprints."""
    if isinstance(embeddings, (list, tuple)) and len(embeddings) > 0:
        e0 = embeddings[0]
        # Explicit RDKit bit vectors expose these methods
        if _HAS_RDKIT and (hasattr(e0, "GetNumBits") or hasattr(e0, "GetNonzeroElements")):
            return True
    return False


def euclidean_distances(X: np.ndarray) -> np.ndarray:
    if _HAS_SK and pairwise_distances is not None:
        return pairwise_distances(X, metric="euclidean")
    # Manual numpy fallback
    G = X @ X.T
    s = np.sum(X**2, axis=1, keepdims=True)
    D2 = np.maximum(s + s.T - 2 * G, 0.0)
    return np.sqrt(D2, dtype=float)


def tanimoto_distance_matrix(fps: Sequence[Any]) -> np.ndarray:
    """Compute a Tanimoto distance matrix with RDKit (``D_ij = 1 - sim_ij``)."""
    if not _HAS_RDKIT:
        raise RuntimeError("RDKit is required to compute Tanimoto distance.")
    n = len(fps)
    D = np.zeros((n, n), dtype=float)
    # For each row, compare against the remaining vectors in bulk
    for i in range(n):
        sims = DS.BulkTanimotoSimilarity(fps[i], fps)
        # Convert similarity to distance
        for j in range(i + 1, n):
            dij = 1.0 - float(sims[j])
            D[i, j] = dij
            D[j, i] = dij
    return D


def pairwise_distances_from_embeddings(
    Z: np.ndarray,
    metric: str = "euclidean",
    normalize_cosine: bool = True,
) -> np.ndarray:
    """Compute pairwise distances (Delta) from embeddings.

    - metric='euclidean' -> L2 distances
    - metric='cosine'    -> 1 - cosine similarity (optionally unit-normalised first)
    - otherwise          -> fallback to sklearn pairwise distance
    """
    Z = np.asarray(Z, dtype=np.float64)
    if metric == "euclidean":
        sq = np.sum(Z**2, axis=1, keepdims=True)
        D2 = sq + sq.T - 2.0 * (Z @ Z.T)
        np.maximum(D2, 0.0, out=D2)
        return np.sqrt(D2, dtype=np.float64)
    elif metric == "cosine":
        X = Z.copy()
        if normalize_cosine:
            nrm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
            X = X / nrm
        S = X @ X.T
        np.clip(S, -1.0, 1.0, out=S)
        return 1.0 - S
    else:
        if pairwise_distances is None:
            raise RuntimeError(f"scikit-learn is required for metric='{metric}'")
        return pairwise_distances(Z, metric=metric)


def cosine_distances(X: np.ndarray) -> np.ndarray:
    """Cosine distance ``1 - cos(z_i, z_j)`` in float64.

    Rows are L2-normalised with a 1e-12 guard (``pairwise_distances_from_embeddings``) and the
    diagonal is set to exactly 0, so that precomputed-distance routines such as
    ``sklearn.metrics.silhouette_score`` accept the matrix.
    """
    D = pairwise_distances_from_embeddings(X, metric="cosine", normalize_cosine=True)
    np.fill_diagonal(D, 0.0)
    return D


def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    """Return ``X`` with unit-norm rows (1e-12 guard), keeping the input dtype."""
    X = np.asarray(X)
    nrm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return (X / nrm).astype(X.dtype, copy=False)


def _check_distance(distance: str) -> None:
    if distance not in DISTANCES:
        raise ValueError(f"Unknown distance {distance!r}; expected one of {DISTANCES}.")


def _check_metric_version(metric_version: str) -> None:
    if metric_version not in METRIC_VERSIONS:
        raise ValueError(f"Unknown metric_version {metric_version!r}; expected one of {METRIC_VERSIONS}.")


def distance_matrix_for_subset(
    embeddings: Union[np.ndarray, Sequence[Any]],
    idxs: np.ndarray,
    mode: str,
    distance: str = "euclidean",
) -> np.ndarray:
    """Return the representation distance matrix Delta for the selected rows.

    Continuous embeddings use ``distance`` (``"euclidean"`` or ``"cosine"``); fingerprints always
    use Tanimoto distance.
    """
    if mode == "continuous":
        _check_distance(distance)
        X = embeddings[idxs]  # (n,d)
        if distance == "cosine":
            return cosine_distances(X)
        return euclidean_distances(X)
    elif mode == "fingerprint":
        fps = [embeddings[i] for i in idxs]
        return tanimoto_distance_matrix(fps)
    else:
        raise ValueError(f"Unknown mode={mode}")


# ===================== 4) Metric utilities (distance-matrix based) =====================
def auc_diff_pairs_large_when_different(D: np.ndarray, y: np.ndarray) -> float:
    """Treat differing en labels as positives; score is the distance; return ROC-AUC."""
    n = len(y)
    if n < 2:
        return np.nan
    iu = np.triu_indices(n, 1)
    d = D[iu]
    pos = (y[iu[0]] != y[iu[1]]).astype(int)
    if pos.sum() == 0 or pos.sum() == pos.size:
        return np.nan
    if roc_auc_score is not None:
        try:
            return float(roc_auc_score(pos, d))
        except Exception:
            pass
    # Fallback: Mann-Whitney rank approximation
    order = np.argsort(d)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(order.size)
    U = ranks[pos == 1].sum() - pos.sum() * (pos.sum() - 1) / 2.0
    auc = U / (pos.sum() * (pos.size - pos.sum()) + 1e-12)
    return float(auc)


def silhouette_with_labels_from_D(D: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2 or D.shape[0] < 3:
        return np.nan
    if silhouette_score is not None:
        try:
            return float(silhouette_score(D, y, metric="precomputed"))
        except Exception:
            pass
    # Direct definition-based implementation
    n = D.shape[0]
    labels = np.unique(y)
    s_vals = []
    for i in range(n):
        same = y == y[i]
        if same.sum() <= 1:
            continue
        a = D[i, same].sum() / (same.sum() - 1)
        b = np.inf
        for c in labels:
            if c == y[i]:
                continue
            mask = y == c
            if mask.sum() == 0:
                continue
            b = min(b, D[i, mask].mean())
        if not np.isfinite(b):
            continue
        s = (b - a) / max(a, b, 1e-12)
        s_vals.append(s)
    return float(np.mean(s_vals)) if s_vals else np.nan


def _cluster_medoids_from_D(D: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return medoid indices per class and the associated intra-class scatter."""
    labels = np.unique(y)
    medoids = []
    S = []
    for c in labels:
        idx = np.where(y == c)[0]
        if idx.size == 0:
            medoids.append(-1)
            S.append(0.0)
            continue
        # Choose the point with the smallest within-class distance sum
        sub = D[np.ix_(idx, idx)]
        row_sum = sub.sum(axis=1)
        k = int(idx[np.argmin(row_sum)])
        medoids.append(k)
        S.append(float(np.mean(D[k, idx])))
    return np.asarray(medoids, dtype=int), np.asarray(S, dtype=float)


def davies_bouldin_from_D(D: np.ndarray, y: np.ndarray, mode: str, eps: float = 0.0) -> float:
    """Medoid approximation of the Davies-Bouldin index on a distance matrix.

    This is the ``paper`` definition (the published code path; ``eps=0``). Cluster "centres" are
    medoids of ``D`` and the scatter is the mean distance to the medoid.
    """
    y = np.asarray(y)
    if len(np.unique(y)) < 2 or D.shape[0] < 3:
        return np.nan
    medoids, S = _cluster_medoids_from_D(D, y)
    # Distances between medoids
    M = D[np.ix_(medoids, medoids)].copy()
    np.fill_diagonal(M, np.inf)
    R = (S[:, None] + S[None, :]) / (M + eps)
    np.fill_diagonal(R, -np.inf)
    DBI = np.mean(np.max(R, axis=1))
    return float(DBI)


def davies_bouldin_centroid(X: np.ndarray, y: np.ndarray, eps: float = DBI_EPS) -> float:
    """Davies-Bouldin index with Euclidean centroids, as written in App. C.5 (``v2``).

    ``S_i`` is the mean Euclidean distance of the members of class ``i`` to its centroid ``mu_i``,
    ``M_ij = ||mu_i - mu_j||`` and ``R_ij = (S_i + S_j) / (M_ij + eps)``.
    """
    y = np.asarray(y)
    labels = np.unique(y)
    if labels.size < 2 or X.shape[0] < 3:
        return np.nan
    X = np.asarray(X, dtype=np.float64)
    cents = np.stack([X[y == c].mean(axis=0) for c in labels])
    S = np.array([np.linalg.norm(X[y == c] - cents[i], axis=1).mean() for i, c in enumerate(labels)])
    M = np.linalg.norm(cents[:, None, :] - cents[None, :, :], axis=2)
    R = (S[:, None] + S[None, :]) / (M + eps)
    np.fill_diagonal(R, -np.inf)
    return float(np.mean(np.max(R, axis=1)))


def nn1_leave_one_out_from_D(D: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out 1-NN accuracy (``paper``): ties go to the lowest row index (``np.argmin``)."""
    n = D.shape[0]
    if n < 2 or len(np.unique(y)) < 2:
        return np.nan
    D2 = D.copy()
    np.fill_diagonal(D2, np.inf)
    nn = np.argmin(D2, axis=1)
    pred = np.asarray(y)[nn]
    return float(np.mean(pred == y))


def nn1_leave_one_out_v2(D: np.ndarray, y: np.ndarray) -> float:
    """Tie-aware leave-one-out 1-NN accuracy over points that have a same-class partner (``v2``).

    For each point ``i`` whose class has at least one other member, the score is the fraction of
    its exactly-tied nearest neighbours (``j != i``) that share its label, i.e. the expected
    accuracy under uniformly random tie-breaking. Points whose class has no other member cannot be
    classified correctly by construction and are excluded. Returns NaN when no point is eligible.
    """
    y = np.asarray(y)
    n = D.shape[0]
    if n < 2 or np.unique(y).size < 2:
        return np.nan
    same = y[:, None] == y[None, :]
    np.fill_diagonal(same, False)
    eligible = same.any(axis=1)
    if not eligible.any():
        return np.nan
    D2 = np.array(D, dtype=np.float64, copy=True)
    np.fill_diagonal(D2, np.inf)
    tied = D2 == D2.min(axis=1, keepdims=True)
    hits = (tied & same).sum(axis=1) / tied.sum(axis=1)
    return float(np.mean(hits[eligible]))


def boundary_clarity_from_D(D: np.ndarray, y: np.ndarray, q_intra: float = 0.90, q_inter: float = 0.10) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    classes = np.unique(y)
    intra_q = {}
    for c in classes:
        idx = np.where(y == c)[0]
        if idx.size < 2:
            intra_q[c] = 0.0
            continue
        Dij = D[np.ix_(idx, idx)]
        iu = np.triu_indices(idx.size, 1)
        v = Dij[iu]
        intra_q[c] = float(np.quantile(v, q_intra)) if v.size else 0.0

    vals = []
    for ca, cb in itertools.combinations(classes, 2):
        ia = np.where(y == ca)[0]
        ib = np.where(y == cb)[0]
        M = D[np.ix_(ia, ib)].reshape(-1)
        if M.size == 0:
            continue
        q_ab = float(np.quantile(M, q_inter))
        denom = max(q_ab, 1e-12)
        val = (q_ab - intra_q[ca] - intra_q[cb]) / denom
        vals.append(val)
    return float(min(vals)) if vals else np.nan


# ===================== 5) Unsupervised clustering (KMeans / K-medoids) =====================
def hopkins_statistic(X: np.ndarray, m: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> float:
    """Hopkins statistic on raw vectors (Euclidean); NaN when ``n < HOPKINS_MIN_N``.

    ``m = max(10, int(0.1 n))`` uniform points are drawn in the bounding box of ``X`` and ``m`` data
    points are sampled without replacement (``default_rng(0)`` unless ``rng`` is given).
    """
    rng = rng or np.random.default_rng(0)
    n, d = X.shape
    if n < HOPKINS_MIN_N:
        return np.nan
    if m is None:
        m = max(10, int(0.1 * n))
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    U = rng.uniform(mins, maxs, size=(m, d))

    def nn_dist(A, B):
        D2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
        return np.sqrt(D2.min(axis=1))

    W = nn_dist(U, X).sum()
    idx = rng.choice(n, size=m, replace=False)
    Xs = X[idx]
    D2 = ((Xs[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    for i in range(m):
        D2[i, idx[i]] = np.inf
    Y = np.sqrt(D2.min(axis=1)).sum()
    return float(W / (W + Y + 1e-12))


def pam_kmedoids(
    D: np.ndarray, k: int, max_iter: int = 50, rng: Optional[np.random.Generator] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Simple PAM k-medoids over a precomputed distance matrix; returns labels and medoids."""
    rng = rng or np.random.default_rng(0)
    n = D.shape[0]
    # Initialise with random, non-duplicated medoids
    medoids = rng.choice(n, size=k, replace=False)
    labels = np.argmin(D[:, medoids], axis=1)

    def total_cost(lab, meds):
        return float(np.sum(D[np.arange(n), meds[lab]]))

    best_cost = total_cost(labels, medoids)

    for _ in range(max_iter):
        improved = False
        for m_idx in range(k):
            for cand in range(n):
                if cand in medoids:
                    continue
                new_meds = medoids.copy()
                new_meds[m_idx] = cand
                new_labels = np.argmin(D[:, new_meds], axis=1)
                c = total_cost(new_labels, new_meds)
                if c + 1e-9 < best_cost:
                    medoids = new_meds
                    labels = new_labels
                    best_cost = c
                    improved = True
        if not improved:
            break
    return labels, medoids


@functools.cache
def _kmedoids_class():
    """Return ``sklearn_extra.cluster.KMedoids`` if importable (lazy; optional dependency)."""
    try:
        from sklearn_extra.cluster import KMedoids

        return KMedoids
    except Exception:
        return None


@functools.cache
def _agglomerative_class():
    try:
        from sklearn.cluster import AgglomerativeClustering

        return AgglomerativeClustering
    except Exception:
        return None


def resolve_unsup_kmax(n: int, unsup_kmax: Optional[int]) -> int:
    """Largest k scanned by the best-k silhouette for a molecule with ``n`` conformers.

    ``None`` means ``n - 1`` (unbounded; the setting of the published run). An integer caps the scan
    at ``min(unsup_kmax, n - 1)`` (the first public release hard-coded ``min(10, n - 1)``).
    """
    if unsup_kmax is None:
        return n - 1
    unsup_kmax = int(unsup_kmax)
    if unsup_kmax < 2:
        raise ValueError(f"unsup_kmax must be >= 2 or None, got {unsup_kmax}.")
    return min(unsup_kmax, n - 1)


def best_unsup_silhouette_from_D_or_X(
    mode: str,
    D: Optional[np.ndarray],
    X: Optional[np.ndarray],
    kmin: int = 2,
    kmax: int = 6,
    n_init: int = 10,
    random_state: int = 0,
    fp_backend: str = "kmedoids",  # {"kmedoids","agglomerative","custom"}
    silhouette_on: str = "X",  # {"X","D"}; continuous mode only
) -> tuple[float, Optional[int], Optional[np.ndarray]]:
    """Unified unsupervised interface across continuous and fingerprint modes.

    - ``mode == "continuous"``: run sklearn KMeans on ``X``. The silhouette is computed on ``X`` with
      Euclidean distance (``silhouette_on="X"``, ``paper``) or on the precomputed ``D``
      (``silhouette_on="D"``, ``v2``).
    - Otherwise, operate on the distance matrix ``D`` using the selected backend
      (``"kmedoids"``, ``"agglomerative"``, or ``"custom"``).

    Returns ``(best_silhouette, best_k, labels)``.
    """
    # ---------- Continuous embeddings: sklearn KMeans ----------
    if mode == "continuous":
        if X is None or not _HAS_SK or KMeans is None or _silhouette_score is None:
            return np.nan, None, None
        if silhouette_on not in ("X", "D"):
            raise ValueError(f"silhouette_on must be 'X' or 'D', got {silhouette_on!r}")
        if silhouette_on == "D" and D is None:
            raise ValueError("silhouette_on='D' requires D")
        n = X.shape[0]
        if n <= kmin:
            return np.nan, None, None
        kmax_eff = min(kmax, n - 1)
        best_s, best_k, best_lab = -1.0, None, None
        for k in range(kmin, max(kmin, kmax_eff) + 1):
            try:
                with warnings.catch_warnings():
                    # k close to n with duplicate rows: sklearn warns "Number of distinct clusters ... smaller
                    # than n_clusters"; the fit result is unchanged, only the log noise is suppressed.
                    warnings.filterwarnings("ignore", message="Number of distinct clusters")
                    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
                    lab = km.fit_predict(X)
                if silhouette_on == "D":
                    s = _silhouette_score(D, lab, metric="precomputed")
                else:
                    s = _silhouette_score(X, lab, metric="euclidean")
                if s > best_s:
                    best_s, best_k, best_lab = s, k, lab
            except Exception:
                continue
        if best_k is None:
            return np.nan, None, None
        return float(best_s), int(best_k), best_lab

    # ---------- Fingerprint/distance mode: clustering on D ----------
    if D is None:
        return np.nan, None, None
    n = D.shape[0]
    if n <= kmin:
        return np.nan, None, None
    kmax_eff = min(kmax, n - 1)

    # Silhouette helper for precomputed distances
    def _sil_from_D(dist: np.ndarray, labels: np.ndarray) -> float:
        if not _HAS_SK or _silhouette_score is None:
            return np.nan
        try:
            return float(_silhouette_score(dist, labels, metric="precomputed"))
        except Exception:
            return np.nan

    best_s, best_k, best_lab = -1.0, None, None

    # sklearn-extra KMedoids backend
    def _try_kmedoids(dist: np.ndarray, k: int) -> Optional[np.ndarray]:
        kmedoids_cls = _kmedoids_class()
        if kmedoids_cls is None:
            return None
        try:
            model = kmedoids_cls(
                n_clusters=k,
                metric="precomputed",
                method="pam",
                init="k-medoids++",
                random_state=random_state,
            )
            return model.fit_predict(dist)
        except Exception:
            return None

    # sklearn Agglomerative backend (handles metric vs affinity kwarg changes)
    def _try_agglomerative(dist: np.ndarray, k: int) -> Optional[np.ndarray]:
        agglo_cls = _agglomerative_class()
        if agglo_cls is None:
            return None
        try:
            # Handle sklearn API differences: metric="precomputed" vs affinity="precomputed"
            try:
                model = agglo_cls(n_clusters=k, metric="precomputed", linkage="average")
            except TypeError:
                model = agglo_cls(n_clusters=k, affinity="precomputed", linkage="average")
            lab = model.fit(dist).labels_
            return lab
        except Exception:
            return None

    # Custom PAM implementation
    def _try_custom_pam(dist: np.ndarray, k: int) -> Optional[np.ndarray]:
        try:
            lab, _ = pam_kmedoids(dist, k, max_iter=50, rng=np.random.default_rng(random_state))
            return lab
        except Exception:
            return None

    # Try backends in priority order
    for k in range(kmin, max(kmin, kmax_eff) + 1):
        labels = None
        if fp_backend == "kmedoids":
            labels = _try_kmedoids(D, k)
            if labels is None:
                # Fallback to the custom PAM implementation
                labels = _try_custom_pam(D, k)
        elif fp_backend == "agglomerative":
            labels = _try_agglomerative(D, k)
        else:  # "custom"
            labels = _try_custom_pam(D, k)

        if labels is None:
            continue

        s = _sil_from_D(D, labels)
        if np.isfinite(s) and s > best_s:
            best_s, best_k, best_lab = s, k, labels

    if best_k is None:
        return np.nan, None, None
    return float(best_s), int(best_k), best_lab


# ===================== 6) Evaluation interface (one row per molecule) =====================
@dataclass(frozen=True)
class ChiralitySettings:
    """Per-molecule evaluation settings (see ``docs/metrics/chirality.md``)."""

    per_mol_min_n: int = 2
    do_unsup_when_single_en: bool = False
    unsup_kmax: Optional[int] = None
    distance: str = "euclidean"
    metric_version: str = "paper"

    def validate(self) -> None:
        _check_distance(self.distance)
        _check_metric_version(self.metric_version)
        if self.unsup_kmax is not None:
            resolve_unsup_kmax(3, self.unsup_kmax)


def _empty_row(mol_id: str, n: int, mode_tag: str, embedding_mode: str, n_en: int, hop: float = np.nan) -> dict:
    return {
        "mol_id": mol_id,
        "n": int(n),
        "mode": mode_tag,
        "ESA_AUC": np.nan,
        "NN1_acc": np.nan,
        "sil_sup": np.nan,
        "DBI": np.nan,
        "clarity": np.nan,
        "hopkins": hop,
        "sil_unsup": np.nan,
        "k_unsup": np.nan,
        "clarity_unsup": np.nan,
        "n_en_classes": n_en,
        "embedding_mode": embedding_mode,
    }


def evaluate_molecule(
    mol_id: str,
    sub_embeddings: Union[np.ndarray, Sequence[Any]],
    y_en: np.ndarray,
    mode: str,
    settings: ChiralitySettings,
) -> dict:
    """Compute all chirality metrics for one molecule.

    ``sub_embeddings`` holds the rows of this molecule (array ``(n, d)`` or list of fingerprints)
    and ``y_en`` their stereoisomer labels, in dataset order.
    """
    y_en = np.asarray(y_en, dtype=object)
    n = int(len(y_en))
    v2 = settings.metric_version == "v2"
    if n < settings.per_mol_min_n:
        return _empty_row(mol_id, n, "skip_small", mode, n_en=0)

    all_idx = np.arange(n)
    D = distance_matrix_for_subset(sub_embeddings, all_idx, mode, distance=settings.distance)
    n_en = int(np.unique(y_en).size)

    if mode == "continuous":
        X_raw = sub_embeddings
        # v2: clustering / Hopkins geometry follows the selected distance (unit sphere for cosine)
        X_sub = l2_normalize_rows(X_raw) if (v2 and settings.distance == "cosine") else X_raw
        hop = hopkins_statistic(X_sub)
    else:
        X_raw = None
        X_sub = None
        hop = np.nan

    fp_backend_single = "custom" if v2 else "kmedoids"
    silhouette_on = "D" if v2 else "X"

    if n_en >= 2:
        auc = auc_diff_pairs_large_when_different(D, y_en)
        sils = silhouette_with_labels_from_D(D, y_en)
        clar = boundary_clarity_from_D(D, y_en)
        if v2:
            nn1 = nn1_leave_one_out_v2(D, y_en)
            has_pair = bool(np.any(np.unique(y_en, return_counts=True)[1] >= 2))
            if not has_pair:
                dbi = np.nan
            elif mode == "continuous":
                dbi = davies_bouldin_centroid(X_sub, y_en)
            else:
                dbi = davies_bouldin_from_D(D, y_en, mode=mode, eps=DBI_EPS)
        else:
            nn1 = nn1_leave_one_out_from_D(D, y_en)
            dbi = davies_bouldin_from_D(D, y_en, mode=mode)

        kmax_unsup = resolve_unsup_kmax(n, settings.unsup_kmax)
        silu, k_star, lab_star = best_unsup_silhouette_from_D_or_X(
            mode=mode, D=D, X=X_sub, kmin=2, kmax=kmax_unsup, fp_backend="custom", silhouette_on=silhouette_on
        )
        clar_unsup = boundary_clarity_from_D(D, lab_star) if lab_star is not None else np.nan

        return {
            "mol_id": mol_id,
            "n": int(n),
            "mode": "supervised+unsup",
            "ESA_AUC": auc,
            "NN1_acc": nn1,
            "sil_sup": sils,
            "DBI": dbi,
            "clarity": clar,
            "hopkins": hop,
            "sil_unsup": silu,
            "k_unsup": (np.nan if k_star is None else int(k_star)),
            "clarity_unsup": clar_unsup,
            "n_en_classes": n_en,
            "embedding_mode": mode,
        }

    # Only one stereoisomer class present.
    if not settings.do_unsup_when_single_en:
        # v2: Hopkins is reported on the same molecule population as the supervised metrics.
        return _empty_row(mol_id, n, "skip_single_en", mode, n_en=n_en, hop=(np.nan if v2 else hop))

    kmax_unsup = resolve_unsup_kmax(n, settings.unsup_kmax)
    silu, k_star, lab_star = best_unsup_silhouette_from_D_or_X(
        mode=mode, D=D, X=X_sub, kmin=2, kmax=kmax_unsup, fp_backend=fp_backend_single, silhouette_on=silhouette_on
    )
    clar_unsup = boundary_clarity_from_D(D, lab_star) if lab_star is not None else np.nan
    row = _empty_row(mol_id, n, "unsupervised_only", mode, n_en=n_en, hop=(np.nan if v2 else hop))
    row.update(
        {
            "sil_unsup": silu,
            "k_unsup": (np.nan if k_star is None else int(k_star)),
            "clarity_unsup": clar_unsup,
        }
    )
    return row


def summarize_rows(rows: list[dict]) -> dict:
    """Macro-average per-molecule rows (NaNs are skipped) into the summary dict."""

    def _agg_mean(xs):
        a = np.asarray(xs, float)
        a = a[np.isfinite(a)]
        return float(a.mean()) if a.size else np.nan

    def _agg_med(xs):
        a = np.asarray(xs, float)
        a = a[np.isfinite(a)]
        return float(np.median(a)) if a.size else np.nan

    def col(name):
        return [r[name] for r in rows]

    return {
        "ESA_AUC_mean": _agg_mean(col("ESA_AUC")),
        "ESA_AUC_median": _agg_med(col("ESA_AUC")),
        "NN1_acc_mean": _agg_mean(col("NN1_acc")),
        "NN1_acc_median": _agg_med(col("NN1_acc")),
        "sil_sup_mean": _agg_mean(col("sil_sup")),
        "sil_sup_median": _agg_med(col("sil_sup")),
        "DBI_mean": _agg_mean(col("DBI")),
        "DBI_median": _agg_med(col("DBI")),
        "clarity_mean": _agg_mean(col("clarity")),
        "clarity_median": _agg_med(col("clarity")),
        "hopkins_mean": _agg_mean(col("hopkins")),
        "hopkins_median": _agg_med(col("hopkins")),
        "sil_unsup_mean": _agg_mean(col("sil_unsup")),
        "sil_unsup_median": _agg_med(col("sil_unsup")),
        "k_unsup_median": _agg_med(col("k_unsup")),
        "clarity_unsup_mean": _agg_mean(col("clarity_unsup")),
        "clarity_unsup_median": _agg_med(col("clarity_unsup")),
        "n_molecules": len(rows),
    }


def coverage_from_rows(rows: list[dict]) -> dict:
    """Number of molecules that contribute a finite value to each summary metric."""

    def n_finite(name):
        return int(np.isfinite(np.asarray([r[name] for r in rows], dtype=float)).sum())

    modes = [r["mode"] for r in rows]
    return {
        "n_molecules": len(rows),
        "n_supervised": modes.count("supervised+unsup"),
        "n_skip_small": modes.count("skip_small"),
        "n_skip_single_en": modes.count("skip_single_en"),
        "n_unsupervised_only": modes.count("unsupervised_only"),
        **{f"n_finite_{k}": n_finite(k) for k in ("ESA_AUC", "NN1_acc", "sil_sup", "DBI", "hopkins", "sil_unsup")},
    }


def _detect_mode(embeddings: Union[np.ndarray, Sequence[Any]], n_total: int, n_keys: int) -> str:
    """Validate the embedding container against the dataset and return ``continuous``/``fingerprint``."""
    if isinstance(embeddings, np.ndarray):
        if embeddings.ndim != 2:
            raise ValueError(
                f"Continuous embeddings must be a 2-D array (n_conformers, dim); got shape {embeddings.shape}."
            )
        if embeddings.shape[0] != n_total:
            raise ValueError(
                f"Embedding rows ({embeddings.shape[0]}) != total conformers in the dataset ({n_total} = sum of "
                f"n_conformers over {n_keys} keys). Embeddings must have exactly one row per conformer, in dataset "
                "row order (see the dataset 'offset' column)."
            )
        if not np.issubdtype(embeddings.dtype, np.number):
            raise ValueError(f"Embeddings must be numeric; got dtype {embeddings.dtype}.")
        n_bad = int((~np.isfinite(embeddings)).any(axis=1).sum())
        if n_bad:
            raise ValueError(
                f"Embeddings contain NaN/Inf in {n_bad} rows; the chirality metrics require finite values."
            )
        return "continuous"
    if is_fingerprint_list(embeddings):
        if len(embeddings) != n_total:
            raise ValueError(
                f"Fingerprint list length ({len(embeddings)}) != total conformers in the dataset ({n_total} = sum "
                f"of n_conformers over {n_keys} keys). Fingerprints must be one per conformer, in dataset row order."
            )
        if not _HAS_RDKIT:
            raise RuntimeError("RDKit is required to compute Tanimoto distance.")
        return "fingerprint"
    raise ValueError(
        "Unrecognised embeddings: provide a 2-D numpy array (one row per conformer) or a list of RDKit "
        f"fingerprints (e.g. ExplicitBitVect); got {type(embeddings).__name__}."
    )


def _take_rows(embeddings: Union[np.ndarray, Sequence[Any]], idxs: np.ndarray, mode: str):
    if mode == "continuous":
        return embeddings[idxs]
    return [embeddings[i] for i in idxs]


# ---- parallel helpers (module-level so that they can be pickled) ----
def _evaluate_batch(batch, mode: str, settings: ChiralitySettings) -> list[tuple[int, dict]]:
    return [(pos, evaluate_molecule(mol_id, sub, labels, mode, settings)) for pos, mol_id, sub, labels in batch]


def _run_molecules(
    tasks: list[tuple[str, np.ndarray]],
    embeddings,
    en_labels: np.ndarray,
    mode: str,
    settings: ChiralitySettings,
    n_jobs: int,
    progress: bool,
) -> list[dict]:
    if n_jobs is None or n_jobs == 0:
        n_jobs = 1
    if n_jobs < 0:
        n_jobs = os.cpu_count() or 1
    n_jobs = int(min(n_jobs, max(1, len(tasks))))

    if n_jobs == 1:
        it = tqdm(tasks, disable=not progress, desc="molecules")
        return [
            evaluate_molecule(mol_id, _take_rows(embeddings, idxs, mode), en_labels[idxs], mode, settings)
            for mol_id, idxs in it
        ]

    # Every molecule is evaluated independently with its own fixed seeds, so the split into batches
    # and the number of workers do not change any per-molecule value. Each batch carries only the rows
    # of its molecules; large molecules are spread round-robin over the batches.
    from joblib import Parallel, delayed

    order = sorted(range(len(tasks)), key=lambda i: -int(tasks[i][1].size))
    n_batches = min(len(tasks), n_jobs * 8)
    batches: list[list] = [[] for _ in range(n_batches)]
    for j, i in enumerate(order):
        mol_id, idxs = tasks[i]
        batches[j % n_batches].append((i, mol_id, _take_rows(embeddings, idxs, mode), en_labels[idxs]))

    try:  # joblib >= 1.3
        from joblib import parallel_config as _backend_config
    except ImportError:  # pragma: no cover
        from joblib import parallel_backend as _backend_config

    jobs = (delayed(_evaluate_batch)(b, mode, settings) for b in batches)
    # One BLAS/OpenMP thread per worker: avoids oversubscription; KMeans results do not depend on it.
    with _backend_config("loky", inner_max_num_threads=1):
        parts = Parallel(n_jobs=n_jobs)(jobs)
    rows: list[Optional[dict]] = [None] * len(tasks)
    for part in tqdm(parts, total=n_batches, disable=not progress, desc="batches"):
        for pos, row in part:
            rows[pos] = row
    return rows  # type: ignore[return-value]


def evaluate_en_separation_from_counts(
    key_to_counts: dict[str, int],
    embeddings: Union[np.ndarray, Sequence[Any]],
    *,
    per_mol_min_n: int = 2,
    do_unsup_when_single_en: bool = False,
    unsup_kmax: Optional[int] = None,
    max_molecules: Optional[int] = None,
    distance: str = "euclidean",
    metric_version: str = "paper",
    n_jobs: int = 1,
    progress: bool = False,
):
    """Evaluate embeddings per molecule using only conformer counts per key.

    Args:
        key_to_counts: ordered mapping ``key -> n_conformers`` in dataset row order.
        embeddings: ``(sum(n_conformers), dim)`` array or list of RDKit fingerprints, dataset order.
        per_mol_min_n: molecules with fewer conformers are skipped.
        do_unsup_when_single_en: also run the best-k silhouette for single-stereoisomer molecules.
        unsup_kmax: largest k for the best-k silhouette; ``None`` = ``n - 1`` (published setting).
        max_molecules: evaluate only the first N molecules (quick tests).
        distance: ``"euclidean"`` (published Table 2) or ``"cosine"``; ignored for fingerprints.
        metric_version: ``"paper"`` (published definitions) or ``"v2"`` (corrected definitions).
        n_jobs: worker processes (``-1`` = all CPUs). Results do not depend on ``n_jobs``.
        progress: show a tqdm progress bar.

    Returns:
        ``(rows, summary)``: one dict per molecule and the macro-averaged summary.
    """
    settings = ChiralitySettings(
        per_mol_min_n=per_mol_min_n,
        do_unsup_when_single_en=do_unsup_when_single_en,
        unsup_kmax=unsup_kmax,
        distance=distance,
        metric_version=metric_version,
    )
    settings.validate()

    flat = build_flat_index_from_counts(key_to_counts)
    n_total = sum(flat.counts)
    mode = _detect_mode(embeddings, n_total, len(flat.keys))

    mol_items = list(flat.mol_to_indices.items())
    if max_molecules is not None:
        mol_items = mol_items[:max_molecules]
        print(f"Quick test mode: processing only {len(mol_items)} molecules (out of {len(flat.mol_to_indices)})")

    rows = _run_molecules(mol_items, embeddings, flat.en_labels, mode, settings, n_jobs, progress)
    return rows, summarize_rows(rows)


def evaluate_en_separation(
    key_to_mols: dict[str, list[Any]],
    embeddings: Union[np.ndarray, Sequence[Any]],
    *,
    per_mol_min_n: int = 2,
    do_unsup_when_single_en: bool = False,  # Run unsupervised metrics even with a single en-class
    unsup_kmax: Optional[int] = None,
    max_molecules: Optional[int] = None,  # Limit number of molecules for quick testing
    distance: str = "euclidean",
    metric_version: str = "paper",
    n_jobs: int = 1,
    progress: bool = True,
):
    """Evaluate embeddings per molecule from a ``key -> [Mol]`` mapping (see the ``_from_counts`` variant)."""
    return evaluate_en_separation_from_counts(
        _counts_from_mols(key_to_mols),
        embeddings,
        per_mol_min_n=per_mol_min_n,
        do_unsup_when_single_en=do_unsup_when_single_en,
        unsup_kmax=unsup_kmax,
        max_molecules=max_molecules,
        distance=distance,
        metric_version=metric_version,
        n_jobs=n_jobs,
        progress=progress,
    )


# ========= Legacy driver (authors' original data/ layout; not used by the CLI) =========
N_WORKERS = min(6, os.cpu_count() or 2)


def _legacy_paths() -> tuple[Path, Path]:
    from three_dbench.utils.paths import DATA_ROOT, RESULTS_ROOT

    return DATA_ROOT / "chirality", RESULTS_ROOT / "chirality" / "en_sep_results"


def legacy_model_specs(data_root: Optional[Path] = None) -> list[tuple[str, str, str, str]]:
    """(name, path, loader, key) of the published Table 2 inputs in the original ``data/chirality`` layout."""
    root = Path(data_root) if data_root is not None else _legacy_paths()[0]
    return [
        ("molspectra", str(root / "molspectra" / "sampled_mol_feature.npz"), "npz", "arr_0"),
        ("unimol", str(root / "unimol" / "1.npz"), "npz", "arr_0"),
        ("gemnet", str(root / "gemnet" / "sampled_feature.npz"), "npz", "gemnet"),
        ("molae", str(root / "molae" / "1.npz"), "npz", "arr_0"),
        ("e3fp", str(root / "fingerprint" / "sampled_chi.pkl"), "pkl_dict", "e3fp"),
        ("fmg", str(root / "fmg" / "chirality_bench_conformers_noised_only_aslist_embed.npz"), "npz", "embeddings"),
        ("mace", str(root / "mace" / "chirality.npz"), "npz", "arr_0"),
    ]


BASE_DICT = None  # Read-only cache shared by worker processes


def _worker_init(base_dict_pkl: str):
    """Load the base dictionary once per worker to avoid repeated large IPC transfers."""
    if rdBase is None:
        raise RuntimeError("RDKit is required to load chirality conformer pickles.")
    if rdBase.rdkitVersion < "2023.09":
        raise RuntimeError(
            "RDKit >= 2023.09 is required to load chirality conformer pickles. "
            f"Detected {rdBase.rdkitVersion}. Please upgrade your environment."
        )
    global BASE_DICT
    with open(base_dict_pkl, "rb") as f:
        BASE_DICT = pickle.load(f)


def _load_array(path: str, loader: str, key: str):
    if loader == "npz":
        with np.load(path) as data:
            return data[key]
    elif loader == "pkl_dict":
        with open(path, "rb") as f:
            d = pickle.load(f)
        return d[key]
    else:
        raise ValueError(f"Unknown loader: {loader}")


def _run_one(
    model_name: str,
    path: str,
    loader: str,
    key: str,
    max_molecules: Optional[int] = None,
    eval_kwargs: Optional[dict] = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Evaluate a single model and return ``(model_name, result_dict, summary_dict)``."""
    arr = _load_array(path, loader, key)
    result, summary = evaluate_en_separation(
        BASE_DICT, arr, max_molecules=max_molecules, progress=False, **(eval_kwargs or {})
    )
    return model_name, result, summary


def run_chirality_benchmark(
    *,
    max_workers: Optional[int] = None,
    base_dict_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    model_specs: Optional[list[tuple[str, str, str, str]]] = None,
    max_molecules: Optional[int] = None,  # Quick test: limit number of molecules
    **eval_kwargs,
) -> None:
    """Execute the chirality benchmark over several models from the original pickle layout."""
    data_root, default_out = _legacy_paths()
    workers = max_workers or N_WORKERS
    base_path = str(base_dict_path or (data_root / "chirality_bench_conformers_noised_only.pkl"))
    out_dir = output_dir or default_out
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = model_specs or legacy_model_specs(data_root)

    summaries = {}  # model_name -> summary dict (shared keys)
    json_paths = []  # Paths of written result JSON files

    # Submit jobs in parallel
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(base_path,)) as ex:
        futs = {
            ex.submit(_run_one, name, path, loader, key, max_molecules, eval_kwargs): (name, path)
            for (name, path, loader, key) in specs
        }

        for fut in tqdm(as_completed(futs), total=len(futs), desc="Evaluating models", ncols=90):
            name, _ = futs[fut]
            try:
                model_name, result, summary = fut.result()
                # Write per-model result JSON
                out_json = out_dir / f"{model_name}.json"
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False)
                json_paths.append(out_json)
                summaries[model_name] = summary
            except Exception as e:
                # Log failure details briefly
                err_log = out_dir / "errors.log"
                with open(err_log, "a", encoding="utf-8") as f:
                    f.write(f"[{name}] {type(e).__name__}: {e}\n")
                # Continue processing other tasks
                continue

    # Aggregate summary -> CSV
    if summaries:
        any_summary = next(iter(summaries.values()))
        cols = list(any_summary.keys())

        df = pd.DataFrame.from_dict(summaries, orient="index")[cols]
        csv_path = out_dir / "summary.csv"
        df.to_csv(csv_path, index_label="model")

        print(f"\nDone. JSON files: {len(json_paths)} written to {out_dir}")
        print(f"Summary CSV: {csv_path}")
    else:
        print("\nNo summaries were produced. Check errors.log.")


def main():
    run_chirality_benchmark()


if __name__ == "__main__":
    main()
