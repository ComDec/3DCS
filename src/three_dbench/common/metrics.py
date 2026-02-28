"""Shared metric utilities for 3DBench evaluations."""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import kendalltau, spearmanr
from sklearn.isotonic import IsotonicRegression

Number = Union[int, float]
RandomState = Optional[Union[int, np.random.Generator]]


def condensed_length(n: int) -> int:
    """Return the length of the condensed upper-triangular vector for an ``n`` x ``n`` matrix."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return n * (n - 1) // 2


def n_from_condensed_length(m: int) -> int:
    """Infer matrix dimension from condensed vector length.

    Raises:
        ValueError: if ``m`` is not a valid condensed length.
    """
    if m < 0:
        raise ValueError("Condensed length must be non-negative")
    n = int((1 + math.isqrt(1 + 8 * m)) // 2)
    if n * (n - 1) // 2 != m:
        raise ValueError(f"{m} is not a valid condensed length")
    return n


def to_square(distance: ArrayLike, *, dtype: np.dtype | None = np.float64) -> np.ndarray:
    """Convert a distance representation to a full square matrix.

    Accepts either a square matrix or a condensed upper-triangular vector.
    """
    arr = np.asarray(distance)
    if arr.ndim == 2:
        if arr.shape[0] != arr.shape[1]:
            raise ValueError(f"Square matrix expected, got shape {arr.shape}")
        return arr.astype(dtype, copy=False) if dtype is not None else arr
    if arr.ndim == 1:
        n = n_from_condensed_length(arr.size)
        sq = np.zeros((n, n), dtype=dtype or arr.dtype)
        idx = 0
        for i in range(n - 1):
            length = n - i - 1
            block = arr[idx : idx + length].astype(dtype or arr.dtype, copy=False)
            sq[i, i + 1 :] = block
            sq[i + 1 :, i] = block
            idx += length
        return sq
    raise ValueError("Distance must be a square matrix or condensed vector")


def to_condensed(distance: ArrayLike) -> np.ndarray:
    """Return the condensed upper-triangular vector for a distance matrix."""
    arr = np.asarray(distance)
    if arr.ndim == 1:
        return arr.astype(np.float64, copy=False)
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        iu = np.triu_indices(arr.shape[0], 1)
        return arr[iu].astype(np.float64, copy=False)
    raise ValueError("Distance must be a square matrix or condensed vector")


def _rng(random_state: RandomState) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _filter_finite(v1: np.ndarray, v2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(v1) & np.isfinite(v2)
    return v1[mask], v2[mask]


def spearman_correlation(
    distance_a: ArrayLike,
    distance_b: ArrayLike,
    *,
    max_pairs: int | None = None,
    random_state: RandomState = None,
) -> float:
    """Spearman correlation between two distance representations."""
    v1 = to_condensed(distance_a)
    v2 = to_condensed(distance_b)
    v1, v2 = _filter_finite(v1, v2)
    if v1.size < 2:
        return float("nan")
    if max_pairs is not None and v1.size > max_pairs:
        rng = _rng(random_state)
        idx = rng.choice(v1.size, size=max_pairs, replace=False)
        v1 = v1[idx]
        v2 = v2[idx]
    rho, _ = spearmanr(v1, v2)
    return float(rho)


def kendall_correlation(
    distance_a: ArrayLike,
    distance_b: ArrayLike,
    *,
    max_pairs: int | None = None,
    random_state: RandomState = None,
) -> float:
    """Kendall correlation between two distance representations."""
    v1 = to_condensed(distance_a)
    v2 = to_condensed(distance_b)
    v1, v2 = _filter_finite(v1, v2)
    if v1.size < 2:
        return float("nan")
    if max_pairs is not None and v1.size > max_pairs:
        rng = _rng(random_state)
        idx = rng.choice(v1.size, size=max_pairs, replace=False)
        v1 = v1[idx]
        v2 = v2[idx]
    tau, _ = kendalltau(v1, v2, nan_policy="omit")
    return float(tau)


def isotonic_r2(distance_reference: ArrayLike, distance_model: ArrayLike) -> float:
    """Coefficient of determination for isotonic regression fit between distance sets."""
    x = to_condensed(distance_reference)
    y = to_condensed(distance_model)
    x, y = _filter_finite(x, y)
    if x.size < 2 or np.allclose(x.min(), x.max()):
        return float("nan")
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    yhat = model.fit_transform(xs, ys)
    ss_res = float(np.sum((ys - yhat) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2)) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def distance_correlation(distance_a: ArrayLike, distance_b: ArrayLike) -> float:
    """Distance correlation between two distance matrices."""
    A = to_square(distance_a, dtype=np.float64)
    B = to_square(distance_b, dtype=np.float64)
    if A.shape != B.shape:
        raise ValueError("Distance matrices must share the same shape")
    n = A.shape[0]

    def _double_center(X: np.ndarray) -> np.ndarray:
        row_mean = X.mean(axis=1, keepdims=True)
        col_mean = X.mean(axis=0, keepdims=True)
        grand_mean = X.mean()
        return X - row_mean - col_mean + grand_mean

    Ac = _double_center(A)
    Bc = _double_center(B)
    n2 = float(n * n)
    cov = (Ac * Bc).sum() / n2
    var_a = (Ac * Ac).sum() / n2
    var_b = (Bc * Bc).sum() / n2
    if var_a <= 1e-20 or var_b <= 1e-20:
        return 0.0
    value = cov / math.sqrt(var_a * var_b)
    return float(np.clip(value, 0.0, 1.0))


def mantel_test(
    distance_a: ArrayLike,
    distance_b: ArrayLike,
    *,
    n_permutations: int = 999,
    method: str = "spearman",
    random_state: RandomState = None,
) -> tuple[float, float]:
    """Perform a Mantel test between two distance matrices."""
    D = to_square(distance_a, dtype=np.float64)
    Delta = to_square(distance_b, dtype=np.float64)
    if D.shape != Delta.shape:
        raise ValueError("Distance matrices must have identical shapes")
    n = D.shape[0]
    iu = np.triu_indices(n, 1)
    d = D[iu]
    x = Delta[iu]
    if method == "pearson":

        def stat(a, b):
            return np.corrcoef(a, b)[0, 1]
    else:

        def stat(a, b):
            return spearmanr(a, b).correlation

    r0 = float(stat(d, x))
    rng = _rng(random_state)
    ge = 0
    for _ in range(max(1, n_permutations)):
        perm = rng.permutation(n)
        x_perm = Delta[perm][:, perm][iu]
        r = stat(d, x_perm)
        if abs(r) >= abs(r0):
            ge += 1
    p_value = (ge + 1) / (n_permutations + 1)
    return r0, float(p_value)


def kruskal_stress(
    distance_reference: ArrayLike,
    distance_model: ArrayLike,
    *,
    weights: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Return Kruskal stress-1 along with optimal linear fit parameters."""
    D = to_square(distance_reference, dtype=np.float64)
    Delta = to_square(distance_model, dtype=np.float64)
    if D.shape != Delta.shape:
        raise ValueError("Distance matrices must have identical shapes")
    n = D.shape[0]
    iu = np.triu_indices(n, 1)
    d = D[iu]
    x = Delta[iu]
    if weights is None:
        w = np.ones_like(d, dtype=np.float64)
    else:
        W = to_square(weights, dtype=np.float64)
        w = W[iu]
    wx = w * x
    sxx = float(np.dot(wx, x))
    sx1 = float(wx.sum())
    s11 = float(w.sum())
    wd = w * d
    sxd = float(np.dot(wd, x))
    s1d = float(wd.sum())
    det = sxx * s11 - sx1 * sx1 + 1e-24
    a = (s11 * sxd - sx1 * s1d) / det
    b = (sxx * s1d - sx1 * sxd) / det
    d_hat = a * x + b
    num = float(np.sum(w * (d - d_hat) ** 2))
    den = float(np.sum(w * d * d)) + 1e-20
    stress1 = math.sqrt(num / den)
    return float(stress1), float(a), float(b)


def triplet_order_preservation(
    distance_reference: ArrayLike,
    distance_model: ArrayLike,
    *,
    n_triplets: int = 100_000,
    random_state: RandomState = None,
) -> float:
    """Fraction of triplets with preserved ordering."""
    D = to_square(distance_reference, dtype=np.float64)
    Delta = to_square(distance_model, dtype=np.float64)
    if D.shape != Delta.shape:
        raise ValueError("Distance matrices must have identical shapes")
    n = D.shape[0]
    if n < 3:
        return float("nan")
    rng = _rng(random_state)
    i = rng.integers(0, n, size=n_triplets)
    j = rng.integers(0, n, size=n_triplets)
    k = rng.integers(0, n, size=n_triplets)
    bad = (i == j) | (i == k) | (j == k)
    while bad.any():
        count = int(bad.sum())
        j[bad] = rng.integers(0, n, size=count)
        k[bad] = rng.integers(0, n, size=count)
        bad = (i == j) | (i == k) | (j == k)
    s1 = np.sign(D[i, j] - D[i, k])
    s2 = np.sign(Delta[i, j] - Delta[i, k])
    mask = s1 != 0
    if not mask.any():
        return float("nan")
    return float((s1[mask] == s2[mask]).mean())


def _median_bandwidth(distances: np.ndarray, eps: float = 1e-12) -> float:
    positive = distances[np.isfinite(distances) & (distances > 0.0)]
    if positive.size == 0:
        return 1.0
    med_sq = np.median(positive * positive)
    return float(math.sqrt(med_sq + eps))


def _rbf_kernel(distance: np.ndarray, sigma: float | None) -> np.ndarray:
    if sigma is None:
        iu = np.triu_indices(distance.shape[0], 1)
        sigma = _median_bandwidth(distance[iu])
    K = np.exp(-(distance**2) / (2.0 * sigma**2 + 1e-12))
    np.fill_diagonal(K, 1.0)
    return K


def _center_kernel(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n, dtype=K.dtype) - np.ones((n, n), dtype=K.dtype) / n
    return H @ K @ H


def cka_rbf(
    distance_a: ArrayLike,
    distance_b: ArrayLike,
    *,
    sigma_a: float | None = None,
    sigma_b: float | None = None,
    share_sigma: bool = False,
) -> float:
    """CKA similarity using RBF kernels derived from distance matrices."""
    D = to_square(distance_a, dtype=np.float64)
    Delta = to_square(distance_b, dtype=np.float64)
    if D.shape != Delta.shape:
        raise ValueError("Distance matrices must have identical shapes")
    if share_sigma:
        i_upper = np.triu_indices(D.shape[0], 1)
        sigma = _median_bandwidth(np.concatenate([D[i_upper], Delta[i_upper]]))
        sigma_a = sigma_b = sigma
    KA = _center_kernel(_rbf_kernel(D, sigma_a))
    KB = _center_kernel(_rbf_kernel(Delta, sigma_b))
    num = float(np.sum(KA * KB))
    den = float(np.linalg.norm(KA, "fro") * np.linalg.norm(KB, "fro") + 1e-20)
    if den == 0.0:
        return float("nan")
    return num / den


def local_isometry_error(
    distance_reference: ArrayLike,
    distance_model: ArrayLike,
    *,
    k: int = 10,
    aggregation: str = "mean",
) -> float:
    """Compute local isometry error (LIE@k) between distance matrices."""
    D = to_square(distance_reference, dtype=np.float64)
    Delta = to_square(distance_model, dtype=np.float64)
    if D.shape != Delta.shape:
        raise ValueError("Distance matrices must have identical shapes")
    n = D.shape[0]
    if n <= 1:
        return float("nan")
    k = int(min(max(1, k), n - 1))
    neighbor_idx = np.argpartition(D, kth=k, axis=1)[:, :k]
    lies = np.empty(n, dtype=np.float64)
    for i in range(n):
        nbrs = neighbor_idx[i]
        Di = D[i, nbrs]
        Delta_i = Delta[i, nbrs]
        Di_bar = Di.mean() + 1e-12
        Delta_bar = Delta_i.mean() + 1e-12
        diff = (Di / Di_bar) - (Delta_i / Delta_bar)
        lies[i] = math.sqrt(float(np.mean(diff * diff)))
    if aggregation == "median":
        return float(np.median(lies))
    return float(np.mean(lies))


def torsion_embedding_spearman(
    distance_model: ArrayLike,
    torsion_deg: ArrayLike,
    *,
    max_pairs: int | None = None,
    random_state: RandomState = None,
) -> float:
    """Spearman correlation between embedding distances and torsion differences."""
    Delta = to_square(distance_model, dtype=np.float64)
    torsion = np.asarray(torsion_deg, dtype=np.float64)
    if torsion.ndim == 1:
        A = _angular_distance_matrix(torsion)
    elif torsion.ndim == 2:
        A = _torus_distance_matrix(torsion)
    else:
        raise ValueError("torsion_deg must have shape (n,) or (n, k)")
    return spearman_correlation(Delta, A, max_pairs=max_pairs, random_state=random_state)


def _angular_distance_matrix(phi_deg: np.ndarray) -> np.ndarray:
    phi = np.deg2rad(phi_deg)
    diff = np.abs(phi[:, None] - phi[None, :])
    two_pi = 2.0 * np.pi
    return np.minimum(diff, two_pi - diff, out=diff)


def _torus_distance_matrix(Phi_deg: np.ndarray) -> np.ndarray:
    Phi = np.deg2rad(Phi_deg)
    if Phi.ndim != 2:
        raise ValueError("Phi_deg must be two-dimensional")
    n, k = Phi.shape
    acc = np.zeros((n, n), dtype=np.float64)
    two_pi = 2.0 * np.pi
    for t in range(k):
        diff = np.abs(Phi[:, [t]] - Phi[:, [t]].T)
        diff = np.minimum(diff, two_pi - diff, out=diff)
        acc += diff * diff
    return np.sqrt(acc, out=acc)


def angular_smoothness(
    distance_model: ArrayLike,
    torsion_deg: ArrayLike,
    *,
    circular: bool = True,
) -> dict[str, float]:
    """Assess angular smoothness along sorted torsion angles."""
    Delta = to_square(distance_model, dtype=np.float64)
    phi = np.deg2rad(np.asarray(torsion_deg, dtype=np.float64).reshape(-1))
    order = np.argsort(phi)
    n = order.size
    if n < 2:
        return {"AS": float("nan"), "path_length": 0.0, "mean_step": float("nan")}
    if circular:
        idx_pairs = [(order[i], order[(i + 1) % n]) for i in range(n)]
    else:
        idx_pairs = [(order[i], order[i + 1]) for i in range(n - 1)]
    step_angles = np.empty(len(idx_pairs), dtype=np.float64)
    step_dists = np.empty(len(idx_pairs), dtype=np.float64)
    two_pi = 2.0 * np.pi
    for t, (i, j) in enumerate(idx_pairs):
        dphi = abs(phi[j] - phi[i])
        step_angles[t] = min(dphi, two_pi - dphi)
        step_dists[t] = Delta[i, j]
    path_length = float(step_dists.sum())
    mask = step_angles > 1e-9
    slope = np.divide(step_dists[mask], step_angles[mask], out=np.zeros_like(step_angles[mask]), where=mask)
    return {
        "AS": float(np.mean(slope)) if slope.size else float("nan"),
        "path_length": path_length,
        "mean_step": float(np.mean(step_dists)) if step_dists.size else float("nan"),
    }
