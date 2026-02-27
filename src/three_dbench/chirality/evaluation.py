import re
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Sequence, Union

import numpy as np

# ===== RDKit fingerprint support =====
try:
    from rdkit import DataStructs as DS
    _HAS_RDKIT = True
except Exception:
    DS = None
    _HAS_RDKIT = False

# ===== Optional sklearn support (preferred when available) =====
try:
    from sklearn.metrics import roc_auc_score, silhouette_score, pairwise_distances
    from sklearn.cluster import KMeans
    from sklearn.metrics import davies_bouldin_score
    _HAS_SK = True
except Exception:
    roc_auc_score = None
    silhouette_score = None
    KMeans = None
    davies_bouldin_score = None
    pairwise_distances = None
    _HAS_SK = False


# ===================== 1) Parse key: mol_id + en_id =====================
_KEY_RE = re.compile(r"^(?P<mol>[^:]+)::en(?P<en>\d+)(?:_.*)?$")

def parse_key_en(key: str) -> Tuple[str, str]:
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
    keys: List[str]  # Keys in flattened order
    counts: List[int]  # Sample count per key
    offsets: List[int]  # Global offset per key
    mol_to_indices: Dict[str, np.ndarray]  # mol_id -> global indices
    en_labels: np.ndarray  # Global array of en labels
    mol_labels: np.ndarray  # Global array of mol_ids

def build_flat_index(key_to_mols: Dict[str, List[Any]]) -> FlatIndex:
    keys, counts, offsets = [], [], []
    mol_idx_map: Dict[str, List[int]] = {}
    en_labels = []
    mol_labels = []

    cursor = 0
    for k, mol_list in key_to_mols.items():
        n = len(mol_list) if mol_list is not None else 0
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


def build_flat_index_from_counts(key_to_counts: Dict[str, int]) -> FlatIndex:
    """Build flat indices from a mapping of key to conformer counts."""
    keys, counts, offsets = [], [], []
    mol_idx_map: Dict[str, List[int]] = {}
    en_labels = []
    mol_labels = []

    cursor = 0
    for k, n in key_to_counts.items():
        keys.append(k)
        counts.append(int(n))
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
    D2 = np.maximum(s + s.T - 2*G, 0.0)
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
        for j in range(i+1, n):
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
            raise RuntimeError("scikit-learn is required for metric='%s'" % metric)
        return pairwise_distances(Z, metric=metric)

def distance_matrix_for_subset(
    embeddings: Union[np.ndarray, Sequence[Any]],
    idxs: np.ndarray,
    mode: str
) -> np.ndarray:
    """Return the distance matrix for the selected subset."""
    if mode == "continuous":
        X = embeddings[idxs]  # (n,d)
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
        same = (y == y[i])
        if same.sum() <= 1:
            continue
        a = D[i, same].sum() / (same.sum() - 1)
        b = np.inf
        for c in labels:
            if c == y[i]:
                continue
            mask = (y == c)
            if mask.sum() == 0:
                continue
            b = min(b, D[i, mask].mean())
        if not np.isfinite(b):
            continue
        s = (b - a) / max(a, b, 1e-12)
        s_vals.append(s)
    return float(np.mean(s_vals)) if s_vals else np.nan

def _cluster_medoids_from_D(D: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

def davies_bouldin_from_D(D: np.ndarray, y: np.ndarray, mode: str) -> float:
    """Compute the Davies-Bouldin index, falling back to a medoid approximation."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2 or D.shape[0] < 3:
        return np.nan
    if mode == "continuous" and _HAS_SK and davies_bouldin_score is not None:
        # Requires features rather than distances; fall back to the medoid version to avoid extra deps
        pass
    # medoid-DBI
    labels = np.unique(y)
    medoids, S = _cluster_medoids_from_D(D, y)
    # Distances between medoids
    M = D[np.ix_(medoids, medoids)].copy()
    np.fill_diagonal(M, np.inf)
    R = (S[:, None] + S[None, :]) / M
    np.fill_diagonal(R, -np.inf)
    DBI = np.mean(np.max(R, axis=1))
    return float(DBI)

def nn1_leave_one_out_from_D(D: np.ndarray, y: np.ndarray) -> float:
    n = D.shape[0]
    if n < 2 or len(np.unique(y)) < 2:
        return np.nan
    D2 = D.copy()
    np.fill_diagonal(D2, np.inf)
    nn = np.argmin(D2, axis=1)
    pred = np.asarray(y)[nn]
    return float(np.mean(pred == y))

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
    rng = rng or np.random.default_rng(0)
    n, d = X.shape
    if n < 10:
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

def pam_kmedoids(D: np.ndarray, k: int, max_iter: int = 50, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
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
            m_cur = medoids[m_idx]
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

from typing import Optional, Tuple
import numpy as np

# Optional third-party clustering libraries
try:
    from sklearn_extra.cluster import KMedoids as _KMedoids
except Exception:
    _KMedoids = None

try:
    from sklearn.cluster import AgglomerativeClustering as _AgglomerativeClustering
except Exception:
    _AgglomerativeClustering = None

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score as _silhouette_score
    _HAS_SK = True
except Exception:
    KMeans = None
    _silhouette_score = None
    _HAS_SK = False


def best_unsup_silhouette_from_D_or_X(
    mode: str,
    D: Optional[np.ndarray],
    X: Optional[np.ndarray],
    kmin: int = 2,
    kmax: int = 6,
    n_init: int = 10,
    random_state: int = 0,
    fp_backend: str = "kmedoids",   # {"kmedoids","agglomerative","custom"}
) -> Tuple[float, Optional[int], Optional[np.ndarray]]:
    """Unified unsupervised interface across continuous and fingerprint modes.

    - ``mode == "continuous"``: run sklearn KMeans on ``X`` with Euclidean silhouette.
    - Otherwise, operate on the distance matrix ``D`` using the selected backend
      (``"kmedoids"``, ``"agglomerative"``, or ``"custom"``).

    Returns ``(best_silhouette, best_k, labels)``.
    """
    # ---------- Continuous embeddings: sklearn KMeans ----------
    if mode == "continuous":
        if X is None or not _HAS_SK or KMeans is None or _silhouette_score is None:
            return np.nan, None, None
        n = X.shape[0]
        if n <= kmin:
            return np.nan, None, None
        kmax_eff = min(kmax, n - 1)
        best_s, best_k, best_lab = -1.0, None, None
        for k in range(kmin, max(kmin, kmax_eff) + 1):
            try:
                km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
                lab = km.fit_predict(X)
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
        if _KMedoids is None:
            return None
        try:
            model = _KMedoids(
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
        if _AgglomerativeClustering is None:
            return None
        try:
            # Handle sklearn API differences: metric="precomputed" vs affinity="precomputed"
            try:
                model = _AgglomerativeClustering(
                    n_clusters=k, metric="precomputed", linkage="average"
                )
            except TypeError:
                model = _AgglomerativeClustering(
                    n_clusters=k, affinity="precomputed", linkage="average"
                )
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
def evaluate_en_separation(
    key_to_mols: Dict[str, List[Any]],
    embeddings: Union[np.ndarray, Sequence[Any]],
    *,
    per_mol_min_n: int = 2,
    do_unsup_when_single_en: bool = False,  # Run unsupervised metrics even with a single en-class
    unsup_kmax: int = 50,
    max_molecules: Optional[int] = None,  # Limit number of molecules for quick testing
):
    """Evaluate embeddings per molecule and return detailed and summary metrics."""
    flat = build_flat_index(key_to_mols)
    N = sum(flat.counts)

    # Auto-detect embedding mode
    if isinstance(embeddings, np.ndarray):
        mode = "continuous"
        if embeddings.shape[0] != N:
            raise ValueError(f"Embeddings count {embeddings.shape[0]} != total molecules {N}.")
    elif is_fingerprint_list(embeddings):
        mode = "fingerprint"
        if len(embeddings) != N:
            raise ValueError(f"Fingerprint list length {len(embeddings)} != total molecules {N}.")
        if not _HAS_RDKIT:
            raise RuntimeError("RDKit is required to compute Tanimoto distance.")
    else:
        raise ValueError("Unrecognised embeddings: provide np.ndarray vectors or RDKit fingerprints.")

    rows = []

    cur_idx = 0
    from tqdm import tqdm

    mol_items = list(flat.mol_to_indices.items())
    if max_molecules is not None:
        mol_items = mol_items[:max_molecules]
        print(f"Quick test mode: processing only {len(mol_items)} molecules (out of {len(flat.mol_to_indices)})")

    for mol_id, idxs in tqdm(mol_items):
        n = idxs.size
        if n < per_mol_min_n:
            rows.append({
                "mol_id": mol_id, "n": int(n),
                "mode": "skip_small",
                "ESA_AUC": np.nan, "NN1_acc": np.nan, "sil_sup": np.nan,
                "DBI": np.nan, "clarity": np.nan,
                "hopkins": np.nan, "sil_unsup": np.nan,
                "k_unsup": np.nan, "clarity_unsup": np.nan,
                "n_en_classes": 0,
                "embedding_mode": mode
            })
            continue

        # Subset distance matrix
        D = distance_matrix_for_subset(embeddings, idxs, mode)
        y_en = flat.en_labels[idxs]
        n_en = int(np.unique(y_en).size)

        # Hopkins statistic only applies in continuous mode; fingerprints return NaN
        if mode == "continuous":
            X_sub = embeddings[idxs]  # for KMeans/Hopkins
            hop = hopkins_statistic(X_sub)
        else:
            X_sub = None
            hop = np.nan

        # Supervised metrics (requires n_en >= 2)
        if n_en >= 2:
            auc = auc_diff_pairs_large_when_different(D, y_en)
            nn1 = nn1_leave_one_out_from_D(D, y_en)
            sils = silhouette_with_labels_from_D(D, y_en)
            dbi = davies_bouldin_from_D(D, y_en, mode=mode)
            clar = boundary_clarity_from_D(D, y_en)

            # Optional unsupervised metrics even when supervised metrics are available
            # Limit kmax to 10 for speed (most molecules have < 10 enantiomers anyway)
            kmax_unsup = min(10, D.shape[0] - 1)
            silu, k_star, lab_star = best_unsup_silhouette_from_D_or_X(
                mode=mode, D=D, X=X_sub, kmin=2, kmax=kmax_unsup, fp_backend="custom"
            )
            clar_unsup = boundary_clarity_from_D(D, lab_star) if lab_star is not None else np.nan

            rows.append({
                "mol_id": mol_id, "n": int(n),
                "mode": "supervised+unsup",
                "ESA_AUC": auc, "NN1_acc": nn1, "sil_sup": sils,
                "DBI": dbi, "clarity": clar,
                "hopkins": hop, "sil_unsup": silu,
                "k_unsup": (np.nan if k_star is None else int(k_star)),
                "clarity_unsup": clar_unsup,
                "n_en_classes": n_en,
                "embedding_mode": mode
            })

        else:
            # Default: skip entirely when there is only one en-class
            if not do_unsup_when_single_en:
                rows.append({
                    "mol_id": mol_id, "n": int(n),
                    "mode": "skip_single_en",
                    "ESA_AUC": np.nan, "NN1_acc": np.nan, "sil_sup": np.nan,
                    "DBI": np.nan, "clarity": np.nan,
                    "hopkins": hop, "sil_unsup": np.nan,
                    "k_unsup": np.nan, "clarity_unsup": np.nan,
                    "n_en_classes": n_en,
                    "embedding_mode": mode
                })
            else:
                silu, k_star, lab_star = best_unsup_silhouette_from_D_or_X(
                    mode=mode, D=D, X=X_sub, kmin=2, kmax=unsup_kmax
                )
                clar_unsup = boundary_clarity_from_D(D, lab_star) if lab_star is not None else np.nan
                rows.append({
                    "mol_id": mol_id, "n": int(n),
                    "mode": "unsupervised_only",
                    "ESA_AUC": np.nan, "NN1_acc": np.nan, "sil_sup": np.nan,
                    "DBI": np.nan, "clarity": np.nan,
                    "hopkins": hop, "sil_unsup": silu,
                    "k_unsup": (np.nan if k_star is None else int(k_star)),
                    "clarity_unsup": clar_unsup,
                    "n_en_classes": n_en,
                    "embedding_mode": mode
                })
        cur_idx += 1

    # Aggregate macro-average across molecules
    def _agg_mean(xs): 
        a = np.asarray(xs, float); a = a[np.isfinite(a)]
        return float(a.mean()) if a.size else np.nan
    def _agg_med(xs):
        a = np.asarray(xs, float); a = a[np.isfinite(a)]
        return float(np.median(a)) if a.size else np.nan

    ESA = [r["ESA_AUC"] for r in rows]
    NN1 = [r["NN1_acc"] for r in rows]
    SIL = [r["sil_sup"] for r in rows]
    DBI = [r["DBI"] for r in rows]
    CLR = [r["clarity"] for r in rows]
    HOP = [r["hopkins"] for r in rows]
    SUS = [r["sil_unsup"] for r in rows]
    KUS = [r["k_unsup"] for r in rows]
    CUS = [r["clarity_unsup"] for r in rows]

    summary = {
        "ESA_AUC_mean": _agg_mean(ESA), "ESA_AUC_median": _agg_med(ESA),
        "NN1_acc_mean": _agg_mean(NN1), "NN1_acc_median": _agg_med(NN1),
        "sil_sup_mean": _agg_mean(SIL), "sil_sup_median": _agg_med(SIL),
        "DBI_mean": _agg_mean(DBI), "DBI_median": _agg_med(DBI),
        "clarity_mean": _agg_mean(CLR), "clarity_median": _agg_med(CLR),
        "hopkins_mean": _agg_mean(HOP), "hopkins_median": _agg_med(HOP),
        "sil_unsup_mean": _agg_mean(SUS), "sil_unsup_median": _agg_med(SUS),
        "k_unsup_median": _agg_med(KUS),
        "clarity_unsup_mean": _agg_mean(CUS), "clarity_unsup_median": _agg_med(CUS),
        "n_molecules": len(rows)
    }

    return rows, summary


def evaluate_en_separation_from_counts(
    key_to_counts: Dict[str, int],
    embeddings: Union[np.ndarray, Sequence[Any]],
    *,
    per_mol_min_n: int = 2,
    do_unsup_when_single_en: bool = False,
    unsup_kmax: int = 50,
    max_molecules: Optional[int] = None,
):
    """Evaluate embeddings using only conformer counts per key."""
    flat = build_flat_index_from_counts(key_to_counts)
    N = sum(flat.counts)

    if isinstance(embeddings, np.ndarray):
        mode = "continuous"
        if embeddings.shape[0] != N:
            raise ValueError(f"Embeddings count {embeddings.shape[0]} != total molecules {N}.")
    elif is_fingerprint_list(embeddings):
        mode = "fingerprint"
        if len(embeddings) != N:
            raise ValueError(f"Fingerprint list length {len(embeddings)} != total molecules {N}.")
        if not _HAS_RDKIT:
            raise RuntimeError("RDKit is required to compute Tanimoto distance.")
    else:
        raise ValueError("Unrecognised embeddings: provide np.ndarray vectors or RDKit fingerprints.")

    rows = []
    mol_items = list(flat.mol_to_indices.items())
    if max_molecules is not None:
        mol_items = mol_items[:max_molecules]
        print(f"Quick test mode: processing only {len(mol_items)} molecules (out of {len(flat.mol_to_indices)})")

    for mol_id, idxs in mol_items:
        n = idxs.size
        if n < per_mol_min_n:
            rows.append({
                "mol_id": mol_id, "n": int(n),
                "mode": "skip_small",
                "ESA_AUC": np.nan, "NN1_acc": np.nan, "sil_sup": np.nan,
                "DBI": np.nan, "clarity": np.nan,
                "hopkins": np.nan, "sil_unsup": np.nan,
                "k_unsup": np.nan, "clarity_unsup": np.nan,
                "n_en_classes": 0,
                "embedding_mode": mode,
            })
            continue

        D = distance_matrix_for_subset(embeddings, idxs, mode)
        y_en = flat.en_labels[idxs]
        n_en = int(np.unique(y_en).size)

        if mode == "continuous":
            X_sub = embeddings[idxs]
            hop = hopkins_statistic(X_sub)
        else:
            X_sub = None
            hop = np.nan

        if n_en >= 2:
            auc = auc_diff_pairs_large_when_different(D, y_en)
            nn1 = nn1_leave_one_out_from_D(D, y_en)
            sils = silhouette_with_labels_from_D(D, y_en)
            dbi = davies_bouldin_from_D(D, y_en, mode=mode)
            clar = boundary_clarity_from_D(D, y_en)

            kmax_unsup = min(10, D.shape[0] - 1)
            silu, k_star, lab_star = best_unsup_silhouette_from_D_or_X(
                mode=mode, D=D, X=X_sub, kmin=2, kmax=kmax_unsup, fp_backend="custom"
            )
            clar_unsup = boundary_clarity_from_D(D, lab_star) if lab_star is not None else np.nan

            rows.append({
                "mol_id": mol_id, "n": int(n),
                "mode": "supervised+unsup",
                "ESA_AUC": auc, "NN1_acc": nn1, "sil_sup": sils,
                "DBI": dbi, "clarity": clar,
                "hopkins": hop, "sil_unsup": silu,
                "k_unsup": (np.nan if k_star is None else int(k_star)),
                "clarity_unsup": clar_unsup,
                "n_en_classes": n_en,
                "embedding_mode": mode,
            })
        else:
            if not do_unsup_when_single_en:
                rows.append({
                    "mol_id": mol_id, "n": int(n),
                    "mode": "skip_single_en",
                    "ESA_AUC": np.nan, "NN1_acc": np.nan, "sil_sup": np.nan,
                    "DBI": np.nan, "clarity": np.nan,
                    "hopkins": hop, "sil_unsup": np.nan,
                    "k_unsup": np.nan, "clarity_unsup": np.nan,
                    "n_en_classes": n_en,
                    "embedding_mode": mode,
                })
            else:
                silu, k_star, lab_star = best_unsup_silhouette_from_D_or_X(
                    mode=mode, D=D, X=X_sub, kmin=2, kmax=unsup_kmax
                )
                clar_unsup = boundary_clarity_from_D(D, lab_star) if lab_star is not None else np.nan
                rows.append({
                    "mol_id": mol_id, "n": int(n),
                    "mode": "unsupervised_only",
                    "ESA_AUC": np.nan, "NN1_acc": np.nan, "sil_sup": np.nan,
                    "DBI": np.nan, "clarity": np.nan,
                    "hopkins": hop, "sil_unsup": silu,
                    "k_unsup": (np.nan if k_star is None else int(k_star)),
                    "clarity_unsup": clar_unsup,
                    "n_en_classes": n_en,
                    "embedding_mode": mode,
                })

    def _agg_mean(xs):
        a = np.asarray(xs, float)
        a = a[np.isfinite(a)]
        return float(a.mean()) if a.size else np.nan

    def _agg_med(xs):
        a = np.asarray(xs, float)
        a = a[np.isfinite(a)]
        return float(np.median(a)) if a.size else np.nan

    ESA = [r["ESA_AUC"] for r in rows]
    NN1 = [r["NN1_acc"] for r in rows]
    SIL = [r["sil_sup"] for r in rows]
    DBI = [r["DBI"] for r in rows]
    CLR = [r["clarity"] for r in rows]
    HOP = [r["hopkins"] for r in rows]
    SUS = [r["sil_unsup"] for r in rows]
    KUS = [r["k_unsup"] for r in rows]
    CUS = [r["clarity_unsup"] for r in rows]

    summary = {
        "ESA_AUC_mean": _agg_mean(ESA), "ESA_AUC_median": _agg_med(ESA),
        "NN1_acc_mean": _agg_mean(NN1), "NN1_acc_median": _agg_med(NN1),
        "sil_sup_mean": _agg_mean(SIL), "sil_sup_median": _agg_med(SIL),
        "DBI_mean": _agg_mean(DBI), "DBI_median": _agg_med(DBI),
        "clarity_mean": _agg_mean(CLR), "clarity_median": _agg_med(CLR),
        "hopkins_mean": _agg_mean(HOP), "hopkins_median": _agg_med(HOP),
        "sil_unsup_mean": _agg_mean(SUS), "sil_unsup_median": _agg_med(SUS),
        "k_unsup_median": _agg_med(KUS),
        "clarity_unsup_mean": _agg_mean(CUS), "clarity_unsup_median": _agg_med(CUS),
        "n_molecules": len(rows),
    }

    return rows, summary


import os
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from rdkit import rdBase
from three_dbench.utils.paths import DATA_ROOT as GLOBAL_DATA_ROOT, RESULTS_ROOT as GLOBAL_RESULTS_ROOT

# ========= Configuration =========
DATA_ROOT = GLOBAL_DATA_ROOT / "chirality"
RESULT_ROOT = GLOBAL_RESULTS_ROOT / "chirality"
BASE_DICT_PKL = DATA_ROOT / "chirality_bench_conformers_noised_only.pkl"
OUT_DIR = RESULT_ROOT / "en_sep_results"  # Output directory (per-model JSON + summary CSV)
N_WORKERS = min(6, os.cpu_count() or 2)

# Model definitions and loader settings (name, path, loader_type, key)
MODEL_SPECS = [
    ("molspectra", str(DATA_ROOT / "molspectra" / "sampled_mol_feature.npz"), "npz", "arr_0"),
    ("unimol",     str(DATA_ROOT / "unimol" / "1.npz"),              "npz", "arr_0"),
    ("gemnet",     str(DATA_ROOT / "gemnet" / "sampled_feature.npz"), "npz", "gemnet"),
    ("molae",      str(DATA_ROOT / "molae" / "1.npz"),               "npz", "arr_0"),
    ("e3fp",       str(DATA_ROOT / "fingerprint" / "sampled_chi.pkl"),       "pkl_dict", "e3fp"),
]

# ========= Existing evaluation function =========
# Assumes ``evaluate_en_separation`` is imported and available.

BASE_DICT = None  # Read-only cache shared by worker processes

def _worker_init(base_dict_pkl: str):
    """Load the base dictionary once per worker to avoid repeated large IPC transfers."""
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
        d = pickle.load(open(path, "rb"))
        return d[key]
    else:
        raise ValueError(f"Unknown loader: {loader}")

def _run_one(model_name: str, path: str, loader: str, key: str, max_molecules: Optional[int] = None) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Evaluate a single model and return ``(model_name, result_dict, summary_dict)``."""
    arr = _load_array(path, loader, key)
    # Delegate to the user-provided evaluation function
    result, summary = evaluate_en_separation(BASE_DICT, arr, max_molecules=max_molecules)
    return model_name, result, summary

def run_chirality_benchmark(
    *,
    max_workers: Optional[int] = None,
    base_dict_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    model_specs: Optional[List[Tuple[str, str, str, str]]] = None,
    max_molecules: Optional[int] = None,  # Quick test: limit number of molecules
) -> None:
    """Execute the chirality benchmark with optional overrides."""
    workers = max_workers or N_WORKERS
    base_path = str(base_dict_path or BASE_DICT_PKL)
    out_dir = output_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = model_specs or MODEL_SPECS

    summaries = {}  # model_name -> summary dict (shared keys)
    json_paths = []  # Paths of written result JSON files

    # Submit jobs in parallel
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(base_path,)) as ex:
        futs = {
            ex.submit(_run_one, name, path, loader, key, max_molecules): (name, path)
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
        # Use any summary's key order for columns (keys are shared)
        any_summary = next(iter(summaries.values()))
        cols = list(any_summary.keys())

        df = pd.DataFrame.from_dict(summaries, orient="index")[cols]
        csv_path = out_dir / "summary.csv"
        df.to_csv(csv_path, index_label="model")

        # Short report to stdout
        print(f"\nDone. JSON files: {len(json_paths)} written to {out_dir}")
        print(f"Summary CSV: {csv_path}")
    else:
        print("\nNo summaries were produced. Check errors.log.")

def main():
    run_chirality_benchmark()

if __name__ == "__main__":
    main()
