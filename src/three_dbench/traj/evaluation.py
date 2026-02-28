"""Trajectory benchmark evaluation combining distance, energy, and sampling metrics.

The pipeline produces pairwise distances (vector and fingerprint), aligns them with energy
signals (Spearman, Kendall, isotonic R2, distance correlation, CKA), and reports energy-centric
metrics (EJS, smoothness, TS, KS, W1). The original parallel sampling workflow is preserved.

Dependencies: numpy, scipy, scikit-learn, pandas, rdkit.
"""

import json
import math
import os
import pickle
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy.special import erfinv

from three_dbench.common.metrics import (
    cka_rbf,
    distance_correlation,
    isotonic_r2,
    kendall_correlation,
    spearman_correlation,
)
from three_dbench.utils.paths import DATA_ROOT as GLOBAL_DATA_ROOT
from three_dbench.utils.paths import RESULTS_ROOT as GLOBAL_RESULTS_ROOT

# ============================================================
# Common type aliases
# ============================================================
ArrayLike = Union[np.ndarray, "np.memmap"]

# ============================================================
# ============ 1. Upper-triangular helpers and containers =====
# ============================================================


def _condensed_len(n: int) -> int:
    return n * (n - 1) // 2


def n_from_condensed_len(m: int) -> int:
    n = int((1 + math.isqrt(1 + 8 * m)) // 2)
    if n * (n - 1) // 2 != m:
        raise ValueError(f"Invalid condensed length {m}")
    return n


def condensed_index(i: int, j: int, n: int) -> int:
    if not (0 <= i < j < n):
        raise IndexError(f"Need 0<=i<j<n, got i={i}, j={j}, n={n}")
    return (n * i) - (i * (i + 1)) // 2 + (j - i - 1)


def _pair_indices_from_n(n: int) -> tuple[np.ndarray, np.ndarray]:
    ii, jj = [], []
    for i in range(n - 1):
        j0 = i + 1
        ii.extend([i] * (n - j0))
        jj.extend(range(j0, n))
    return np.asarray(ii, int), np.asarray(jj, int)


def expand_condensed_to_square(
    v: ArrayLike, out: Optional[Union[str, ArrayLike]] = None, dtype=np.float32
) -> ArrayLike:
    v = np.asarray(v)
    n = n_from_condensed_len(v.shape[0])
    if isinstance(out, str):
        if os.path.dirname(out):
            os.makedirs(os.path.dirname(out), exist_ok=True)
        M = np.memmap(out, dtype=dtype, mode="w+", shape=(n, n))
    elif isinstance(out, (np.ndarray, np.memmap)):
        M = out
        if M.shape != (n, n):
            raise ValueError(f"Provided square has shape {M.shape}, expected {(n, n)}")
    else:
        M = np.zeros((n, n), dtype=dtype)

    idx = 0
    for i in range(n - 1):
        cnt = n - i - 1
        block = v[idx : idx + cnt].astype(dtype, copy=False)
        M[i, i + 1 :] = block
        M[i + 1 :, i] = block
        idx += cnt
    np.fill_diagonal(M, 0.0)
    return M


def _condensed_row_start(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.int64)
    return (n * i - i * (i + 1) // 2).astype(np.int64)


def allocate_out_container(
    n: int,
    mode: Literal["condensed", "square"] = "condensed",
    dtype: np.dtype = np.float16,
    out: Optional[Union[str, ArrayLike]] = None,
) -> ArrayLike:
    if mode == "condensed":
        shape = (_condensed_len(n),)
    elif mode == "square":
        shape = (n, n)
    else:
        raise ValueError("mode must be 'condensed' or 'square'.")

    if isinstance(out, (np.ndarray, np.memmap)):
        if out.shape != shape or out.dtype != dtype:
            raise ValueError(f"Provided 'out' has shape {out.shape}, dtype {out.dtype}, expected {shape}, {dtype}")
        return out

    if isinstance(out, str):
        if os.path.dirname(out):
            os.makedirs(os.path.dirname(out), exist_ok=True)
        return np.memmap(out, dtype=dtype, mode="w+", shape=shape)

    return np.zeros(shape, dtype=dtype)


def _write_block_to_condensed(
    out_cond: ArrayLike,
    block: np.ndarray,  # (bi_len, bj_len)
    n: int,
    i0: int,
    i1: int,
    j0: int,
    j1: int,
    starts: np.ndarray,
    cast_dtype: np.dtype = np.float16,
):
    bi_len, _ = block.shape
    for ii in range(bi_len):
        gi = i0 + ii
        start_j = max(gi + 1, j0)
        if start_j >= j1:
            continue
        jj0 = start_j - j0
        length = j1 - start_j
        if length <= 0:
            continue
        base = int(starts[gi])
        idx0 = base + (start_j - gi - 1)
        out_cond[idx0 : idx0 + length] = block[ii, jj0 : jj0 + length].astype(cast_dtype, copy=False)


# ============================================================
# ============ 2. Continuous embedding distances (scaled) ======
# ============================================================


def pairwise_distances_from_embeddings_large(
    Z: np.ndarray,
    metric: Literal["euclidean", "cosine"] = "cosine",
    unit_normalize: bool = True,
    block_size: int = 4096,
    out_mode: Literal["condensed", "square"] = "condensed",
    out: Optional[Union[str, ArrayLike]] = None,
    dtype_out: np.dtype = np.float16,
    compute_dtype: np.dtype = np.float32,
    progress: bool = True,
    compress_to_npz: Optional[str] = None,
) -> ArrayLike:
    """Normalize pairwise distances to the [0, 1] range.

    - If ``unit_normalize`` is True:
        cosine:  ``d = (1 - cos) / 2``
        euclidean: ``d = ||x - y||^2 / 4`` (equivalent for unit vectors)
    - If ``unit_normalize`` is False and the metric is Euclidean, fall back to
      global min-max normalisation (kept only for backwards compatibility).
    """
    Z = np.asarray(Z, dtype=compute_dtype, order="C")
    n, d = Z.shape
    out_arr = allocate_out_container(n, mode=out_mode, dtype=dtype_out, out=out)

    if unit_normalize:
        norms = np.linalg.norm(Z, axis=1, keepdims=True).astype(compute_dtype) + 1e-12
        X = Z / norms
    else:
        X = Z

    n_blocks = math.ceil(n / block_size)
    total_blocks = n_blocks * (n_blocks + 1) // 2
    block_cnt = 0
    starts = _condensed_row_start(n) if out_mode == "condensed" else None

    need_minmax_scan = (metric == "euclidean") and (not unit_normalize) and (out_mode == "condensed")
    global_min, global_max = np.inf, -np.inf
    if need_minmax_scan:
        for bi in range(n_blocks):
            i0 = bi * block_size
            i1 = min(n, (bi + 1) * block_size)
            Xi = X[i0:i1]
            sq_i = (Xi * Xi).sum(axis=1, keepdims=True)
            for bj in range(bi, n_blocks):
                j0 = bj * block_size
                j1 = min(n, (bj + 1) * block_size)
                Xj = X[j0:j1]
                sq_j = (Xj * Xj).sum(axis=1, keepdims=True).T
                G = Xi @ Xj.T
                D2 = sq_i + sq_j - 2.0 * G
                np.maximum(D2, 0.0, out=D2)
                D = np.sqrt(D2, out=D2)
                if bi == bj:
                    mask = np.triu(np.ones_like(D, dtype=bool), 1)
                    blk = D[mask]
                else:
                    blk = D
                if blk.size:
                    global_min = min(global_min, float(np.nanmin(blk)))
                    global_max = max(global_max, float(np.nanmax(blk)))
        if not np.isfinite(global_min) or not np.isfinite(global_max) or global_max <= global_min + 1e-12:
            global_min, global_max = 0.0, 1.0

    for bi in range(n_blocks):
        i0 = bi * block_size
        i1 = min(n, (bi + 1) * block_size)
        Xi = X[i0:i1]

        if metric == "euclidean":
            sq_i = (Xi * Xi).sum(axis=1, keepdims=True)
        else:
            ni = np.linalg.norm(Xi, axis=1, keepdims=True) + 1e-12

        for bj in range(bi, n_blocks):
            j0 = bj * block_size
            j1 = min(n, (bj + 1) * block_size)
            Xj = X[j0:j1]

            if metric == "euclidean":
                sq_j = (Xj * Xj).sum(axis=1, keepdims=True).T
                G = Xi @ Xj.T
                D2 = sq_i + sq_j - 2.0 * G
                np.maximum(D2, 0.0, out=D2)
                if unit_normalize:
                    Dij01 = 0.25 * D2
                    np.clip(Dij01, 0.0, 1.0, out=Dij01)
                else:
                    D = np.sqrt(D2, out=D2)
                    if global_max > global_min + 1e-12:
                        Dij01 = (D - global_min) / (global_max - global_min)
                    else:
                        Dij01 = np.zeros_like(D)
            else:
                nj = np.linalg.norm(Xj, axis=1, keepdims=True) + 1e-12
                S = (Xi @ Xj.T) / (ni * nj.T)
                np.clip(S, -1.0, 1.0, out=S)
                Dij01 = 0.5 * (1.0 - S)

            if out_mode == "square":
                out_arr[i0:i1, j0:j1] = Dij01.astype(dtype_out, copy=False)
                if bj != bi:
                    out_arr[j0:j1, i0:i1] = Dij01.T.astype(dtype_out, copy=False)
                else:
                    diag_len = i1 - i0
                    if diag_len > 0:
                        out_arr[i0:i1, i0:i1].flat[:: diag_len + 1] = dtype_out.type(0.0)
            else:
                _write_block_to_condensed(out_arr, Dij01, n, i0, i1, j0, j1, starts, cast_dtype=dtype_out)

            block_cnt += 1
            if progress and (block_cnt % 10 == 0 or block_cnt == total_blocks):
                print(
                    f"[pairwise:{metric}->[0,1]] blocks {block_cnt}/{total_blocks} "
                    f"({bi + 1}/{n_blocks} x {bj + 1}/{n_blocks})"
                )

    if isinstance(compress_to_npz, str):
        arr = np.array(out_arr, copy=False) if isinstance(out_arr, np.memmap) else out_arr
        np.savez_compressed(compress_to_npz, data=arr)
        print(f"Compressed saved to {compress_to_npz}")
        return arr

    return out_arr


# ============================================================
# ============ 3. Fingerprint Tanimoto distance (blocked) =====
# ============================================================


def pairwise_tanimoto_fps_large(
    fps: list,
    block_size: int = 4096,
    out_mode: Literal["condensed", "square"] = "condensed",
    out: Optional[Union[str, ArrayLike]] = None,
    dtype_out: np.dtype = np.float16,
    progress: bool = True,
    compress_to_npz: Optional[str] = None,
) -> ArrayLike:
    from rdkit import DataStructs

    n = len(fps)
    out_arr = allocate_out_container(n, mode=out_mode, dtype=dtype_out, out=out)

    n_blocks = math.ceil(n / block_size)
    total_blocks = n_blocks * (n_blocks + 1) // 2
    block_cnt = 0
    starts = _condensed_row_start(n) if out_mode == "condensed" else None

    if out_mode == "square":
        if isinstance(out_arr, (np.memmap, np.ndarray)):
            for i in range(n):
                out_arr[i, i] = dtype_out.type(0.0)

    for bi in range(n_blocks):
        i0 = bi * block_size
        i1 = min(n, (bi + 1) * block_size)
        Fi = fps[i0:i1]

        for bj in range(bi, n_blocks):
            j0 = bj * block_size
            j1 = min(n, (bj + 1) * block_size)
            Fj = fps[j0:j1]

            sim_block = np.empty((i1 - i0, j1 - j0), dtype=np.float32)
            for ii, fpi in enumerate(Fi):
                sims = DataStructs.BulkTanimotoSimilarity(fpi, Fj)
                sim_block[ii, :] = np.asarray(sims, dtype=np.float32)

            Dij = 1.0 - sim_block

            if out_mode == "square":
                out_arr[i0:i1, j0:j1] = Dij.astype(dtype_out, copy=False)
                if bj != bi:
                    out_arr[j0:j1, i0:i1] = Dij.T.astype(dtype_out, copy=False)
            else:
                _write_block_to_condensed(out_arr, Dij, n, i0, i1, j0, j1, starts, cast_dtype=dtype_out)

            block_cnt += 1
            if progress and (block_cnt % 10 == 0 or block_cnt == total_blocks):
                print(
                    f"[pairwise:tanimoto] blocks {block_cnt}/{total_blocks} ({bi + 1}/{n_blocks} x {bj + 1}/{n_blocks})"
                )

    if isinstance(compress_to_npz, str):
        arr = np.array(out_arr, copy=False) if isinstance(out_arr, np.memmap) else out_arr
        np.savez_compressed(compress_to_npz, data=arr)
        print(f"Compressed saved to {compress_to_npz}")
        return arr

    return out_arr


# ============================================================
# ============ 4. Geometric alignment metrics =================
# ============================================================


def spearman_from_condensed(D1: ArrayLike, D2: ArrayLike) -> float:
    """Compatibility wrapper using the shared Spearman implementation."""
    return spearman_correlation(D1, D2)


def kendall_from_condensed(D1: ArrayLike, D2: ArrayLike) -> float:
    """Compatibility wrapper using the shared Kendall implementation."""
    return kendall_correlation(D1, D2)


def isotonic_r2_from_condensed(D_ref: ArrayLike, D_emb: ArrayLike) -> float:
    """Compatibility wrapper for the shared isotonic R^2 routine."""
    return isotonic_r2(D_ref, D_emb)


def cka_rbf_from_condensed(
    D1_cond: ArrayLike,
    D2_cond: ArrayLike,
    sigma1: Optional[float] = None,
    sigma2: Optional[float] = None,
    expand_memmap1: Optional[str] = None,
    expand_memmap2: Optional[str] = None,
    in_mem_threshold: int = 8000,
    shared_sigma: bool = True,
) -> float:
    """Compatibility wrapper that delegates to the shared CKA implementation.

    The additional arguments are accepted for backward compatibility but ignored
    because the shared implementation already handles bandwidth selection.
    """
    _ = (expand_memmap1, expand_memmap2, in_mem_threshold)
    return cka_rbf(D1_cond, D2_cond, sigma_a=sigma1, sigma_b=sigma2, share_sigma=shared_sigma)


def distance_correlation_from_condensed(x: ArrayLike, y: ArrayLike) -> float:
    """Compatibility wrapper for the shared distance correlation implementation."""
    return distance_correlation(x, y)


# ============================================================
# ============ 5. Energy-derived references and metrics =======
# ============================================================


def vector_to_absdiff_condensed(x: np.ndarray, dtype=np.float32) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = x.size
    if n < 2:
        return np.array([], dtype=dtype)
    i_idx = []
    j_idx = []
    for i in range(n - 1):
        j0 = i + 1
        cnt = n - j0
        i_idx.extend([i] * cnt)
        j_idx.extend(range(j0, n))
    i_idx = np.asarray(i_idx, dtype=np.int64)
    j_idx = np.asarray(j_idx, dtype=np.int64)
    v = np.abs(x[i_idx] - x[j_idx]).astype(dtype, copy=False)
    return v


def energy_diff_condensed(E: np.ndarray) -> np.ndarray:
    E = np.asarray(E, float).reshape(-1)
    n = E.size
    ii, jj = [], []
    for i in range(n - 1):
        j0 = i + 1
        ii.extend([i] * (n - j0))
        jj.extend(range(j0, n))
    ii = np.asarray(ii, int)
    jj = np.asarray(jj, int)
    return np.abs(E[ii] - E[jj])


# ---- EJS ----


def _lam_tag(lam: float) -> str:
    """Convert lambda to safe key tag: 0.5 -> '0p5', 2 -> '2'."""
    s = f"{lam:g}"
    return s.replace(".", "p").replace("-", "m")


def energy_jump_sensitivity_auto(
    E: np.ndarray,
    Delta_cond: ArrayLike,
    lam_grid: Optional[Iterable[float]] = (0.1, 0.5, 1.0, 2.0, 3.0),  # Multiple lambda values
    lam_rep: float = 2.0,  # Representative lambda for backward compatibility
    robust_sigma: bool = True,  # use a robust sigma estimate
    winsor: float = 0.0,  # optional winsorization of E
    dist_quantile: float = 0.75,  # quantile for distance threshold tau
    use_global_quantile: bool = True,  # use global delta quantile or jump subset
) -> dict[str, float]:
    E = np.asarray(E, float).reshape(-1)
    Delta_cond = np.asarray(Delta_cond, float).reshape(-1)

    # Infer n from condensed length
    try:
        n = n_from_condensed_len(Delta_cond.size)
    except Exception:
        return {"EJS": np.nan, "num_jumps": 0.0, "theta": np.nan, "tau": np.nan}

    if E.size != n or n < 2:
        return {"EJS": np.nan, "num_jumps": 0.0, "theta": np.nan, "tau": np.nan}

    # Optional winsorization
    if winsor and winsor > 0:
        finite_E = E[np.isfinite(E)]
        if finite_E.size > 0:
            lo, hi = np.quantile(finite_E, [winsor, 1.0 - winsor])
            E = np.clip(E, lo, hi)

    # Pairwise differences
    ii, jj = _pair_indices_from_n(n)
    dE = np.abs(E[ii] - E[jj])  # |delta E|
    dZ = Delta_cond  # delta

    valid = np.isfinite(dE) & np.isfinite(dZ)
    if valid.sum() == 0:
        return {"EJS": np.nan, "num_jumps": 0.0, "theta": np.nan, "tau": np.nan}

    dE = dE[valid]
    dZ = dZ[valid]

    # Half-normal scale for sigma
    sigma_hat = _sigma_halfnormal(dE, robust=robust_sigma)

    # Lambda grid processing
    lam_list = list(lam_grid) if lam_grid is not None else [lam_rep]

    # Global distance threshold
    tau_global = float(np.quantile(dZ, np.clip(dist_quantile, 0.0, 1.0))) if use_global_quantile else np.nan

    out: dict[str, float] = {}
    rep_best = {"EJS": np.nan, "num_jumps": 0, "theta": np.nan, "tau": np.nan}
    rep_idx = int(np.argmin(np.abs(np.asarray(lam_list, float) - float(lam_rep))))

    # Compute EJS for each lambda value
    for k, lam in enumerate(lam_list):
        # theta = lambda * sigma_hat (same for both modes)
        theta = float(lam * sigma_hat)

        jump_mask = dE > theta
        num_jumps = int(jump_mask.sum())

        if num_jumps == 0:
            tau = tau_global if use_global_quantile else np.nan
            ejs_val = np.nan
        else:
            if use_global_quantile:
                tau = tau_global
            else:
                base = dZ[jump_mask]
                tau = float(np.quantile(base, np.clip(dist_quantile, 0.0, 1.0))) if base.size > 0 else np.nan
            ejs_val = float((dZ[jump_mask] > tau).mean()) if np.isfinite(tau) else np.nan

        # Store with lambda tag
        tag = _lam_tag(lam)
        out[f"EJS_lam{tag}"] = ejs_val
        out[f"EJS_num_jumps_lam{tag}"] = float(num_jumps)
        out[f"EJS_tau_lam{tag}"] = float(tau) if np.isfinite(tau) else np.nan
        out[f"EJS_theta_lam{tag}"] = theta

        # Store representative lambda values
        if k == rep_idx:
            rep_best["EJS"] = ejs_val
            rep_best["num_jumps"] = float(num_jumps)
            rep_best["theta"] = theta
            rep_best["tau"] = float(tau) if np.isfinite(tau) else np.nan

    # Backward-compatible single-value keys
    out.update(rep_best)
    # Optionally report sigma_d for summaries
    out["EJS_sigma_hat"] = float(sigma_hat)
    out["EJS_dist_quantile"] = float(dist_quantile)
    out["EJS_use_global_tau"] = float(1.0 if use_global_quantile else 0.0)

    return out


def _sigma_halfnormal(dE: np.ndarray, robust: bool = False) -> float:
    dE = np.asarray(dE, float)
    if robust:
        med = np.nanmedian(dE)
        c = np.sqrt(2.0) * erfinv(0.5)  # ~ 0.67449
        return float(med / (c + 1e-12))
    else:
        return float(np.sqrt(np.nanmean(dE**2) + 1e-12))


def ejs_auc_halfnormal(
    Delta_cond: np.ndarray,
    E: np.ndarray,
    p_jump: float = 0.95,  # Optional half-normal quantile threshold
    lam: Optional[float] = None,  # If provided, use theta = lam * sigma_d (overrides p_jump)
    robust_sigma: bool = False,  # Whether to use a robust estimator for sigma_d
    max_pairs: Optional[int] = None,
    random_state: int = 0,
    return_pr_auc: bool = True,
) -> dict[str, float]:
    """Half-normal threshold + ROC-AUC check for distance separation of energy jumps.

    Parameters
    ----------
    Delta_cond: array-like
        Condensed upper-triangular distance vector aligned with the energy ordering.
    E: array-like
        Energy values (length ``n``).
    p_jump: float
        Half-normal quantile (used when ``lam`` is None).
    lam: float, optional
        Explicit ``theta = lam * sigma_d`` (overrides ``p_jump`` when provided).
    robust_sigma: bool
        Use a median-based robust estimate for ``sigma_d``.
    max_pairs: int, optional
        Optional subsampling limit when ``O(n^2)`` comparisons are too large.
    return_pr_auc: bool
        Also compute PR-AUC (average precision) if True.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    Delta_cond = np.asarray(Delta_cond, float).ravel()
    m = Delta_cond.size
    n = n_from_condensed_len(m)
    E = np.asarray(E, float).reshape(-1)
    if E.size != n:
        raise ValueError("len(E) mismatch with Delta_cond length.")

    dE = energy_diff_condensed(E)
    # Align the valid mask between |delta E| and delta
    mask = np.isfinite(dE) & np.isfinite(Delta_cond)
    dE = dE[mask]
    dZ = Delta_cond[mask]
    if dE.size < 10:
        return {
            "ROC_AUC": np.nan,
            "PR_AUC": np.nan,
            "pos_rate": np.nan,
            "theta": np.nan,
            "sigma_hat": np.nan,
            "num_pairs": int(dE.size),
        }

    # Optional subsampling of pairs
    if (max_pairs is not None) and (dE.size > max_pairs):
        rng = np.random.default_rng(random_state)
        idx = rng.integers(0, dE.size, size=max_pairs)
        dE = dE[idx]
        dZ = dZ[idx]

    # Estimate sigma_d and the threshold theta
    sigma_hat = _sigma_halfnormal(dE, robust=robust_sigma)
    if lam is not None:
        theta = float(lam * sigma_hat)
        pj = np.nan
    else:
        p = float(np.clip(p_jump, 1e-9, 1 - 1e-9))
        theta = float(sigma_hat * np.sqrt(2.0) * erfinv(p))
        pj = p

    y = (dE > theta).astype(int)  # Positive class: energy jump pair
    s = dZ  # Score: embedding distance (larger means more separated)

    # Handle degenerate cases
    n_pos = int(y.sum())
    n_all = int(y.size)
    if n_pos == 0 or n_pos == n_all:
        return {
            "ROC_AUC": np.nan,
            "PR_AUC": np.nan,
            "pos_rate": n_pos / max(n_all, 1),
            "theta": theta,
            "sigma_hat": sigma_hat,
            "p_jump": pj,
            "num_pairs": n_all,
        }

    # Compute ROC/PR AUC via sklearn
    try:
        auc = float(roc_auc_score(y, s))
    except Exception:
        # Fallback: Mann-Whitney U rank statistic approximation
        order = np.argsort(s)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(order.size)
        U = ranks[y == 1].sum() - n_pos * (n_pos - 1) / 2.0
        auc = float(U / (n_pos * (n_all - n_pos) + 1e-12))

    pr = np.nan
    if return_pr_auc:
        try:
            pr = float(average_precision_score(y, s))
        except Exception:
            pr = np.nan

    return {
        "ROC_AUC": auc,
        "PR_AUC": pr,
        "pos_rate": n_pos / n_all,
        "theta": theta,
        "sigma_hat": sigma_hat,
        "p_jump": pj,
        "num_pairs": n_all,
    }


def _condensed_fetch_pairs(Delta_cond: ArrayLike, pairs: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    Delta_cond = np.asarray(Delta_cond, float)
    n = n_from_condensed_len(Delta_cond.shape[0])
    ii, jj = pairs
    vals = np.empty(ii.shape[0], dtype=np.float64)
    for k, (i, j) in enumerate(zip(ii, jj)):
        if i == j:
            vals[k] = 0.0
        else:
            if i > j:
                i, j = j, i
            idx = condensed_index(i, j, n)
            vals[k] = Delta_cond[idx]
    return vals


def energy_embedding_smoothness(
    E: np.ndarray, Delta_cond: ArrayLike, order: Optional[np.ndarray] = None, eps_energy: float = 1e-8
) -> dict[str, float]:
    E = np.asarray(E, float).reshape(-1)
    Delta_cond = np.asarray(Delta_cond, float)
    n = n_from_condensed_len(Delta_cond.shape[0])
    if E.size != n or n < 2:
        return {"Smoothness": np.nan, "segments": 0}

    if order is None:
        order = np.arange(n, dtype=int)
    else:
        order = np.asarray(order, int).reshape(-1)
        if order.size != n:
            raise ValueError("order length must equal n")

    ii = order[:-1]
    jj = order[1:]
    dE = np.abs(E[jj] - E[ii])
    dZ = _condensed_fetch_pairs(Delta_cond, (ii, jj))

    m = np.isfinite(dE) & np.isfinite(dZ)
    if m.sum() == 0:
        return {"Smoothness": np.nan, "segments": 0}
    seg_vals = np.exp(-dZ[m] / (dE[m] + eps_energy))
    return {"Smoothness": float(np.mean(seg_vals)), "segments": int(m.sum())}


def _q90_condensed(Delta_cond: np.ndarray, eps: float = 1e-12) -> float:
    v = np.asarray(Delta_cond, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 1.0
    return float(np.quantile(v, 0.90) + eps)


def _q90_absdiff_energy(E: np.ndarray, max_pairs: int = 200_000, eps: float = 1e-12) -> float:
    E = np.asarray(E, float).reshape(-1)
    n = E.size
    if n < 2:
        return 1.0
    rng = np.random.default_rng(0)
    total = n * (n - 1) // 2
    m = min(max_pairs, total)
    ii = rng.integers(0, n - 1, size=m)
    jj = rng.integers(1, n, size=m)
    mask = ii < jj
    di = np.abs(E[ii[mask]] - E[jj[mask]])
    if di.size == 0:
        return 1.0
    return float(np.quantile(di, 0.90) + eps)


def _resolve_energy_threshold(
    E: np.ndarray,
    theta_abs: Optional[float] = None,
    gamma_rel: Optional[float] = None,
    qE90_cache: Optional[float] = None,
) -> tuple[float, float, str]:
    if theta_abs is not None:
        return float(theta_abs), (qE90_cache if qE90_cache is not None else _q90_absdiff_energy(E)), "abs"
    if gamma_rel is None:
        gamma_rel = 0.20
    q = qE90_cache if qE90_cache is not None else _q90_absdiff_energy(E)
    return float(gamma_rel * q), q, "rel"


def thresholded_smoothness(
    Delta_cond: np.ndarray,
    E: np.ndarray,
    theta_abs: Optional[float] = None,
    gamma_rel: Optional[float] = 0.20,
    eps: float = 1e-12,
) -> dict[str, float]:
    Delta_cond = np.asarray(Delta_cond, float)
    E = np.asarray(E, float).reshape(-1)
    n = n_from_condensed_len(Delta_cond.size)
    if n != E.size:
        raise ValueError("len(E) mismatch with Delta_cond length.")

    qZ = _q90_condensed(Delta_cond, eps=eps)
    qE = _q90_absdiff_energy(E, eps=eps)
    ii, jj = _pair_indices_from_n(n)
    dE = np.abs(E[ii] - E[jj])
    dZ = _condensed_fetch_pairs(Delta_cond, (ii, jj))

    TE, qE90, mode = _resolve_energy_threshold(E, theta_abs, gamma_rel, qE90_cache=qE)
    m = np.isfinite(dE) & np.isfinite(dZ) & (dE >= TE)
    if m.sum() == 0:
        return {"TS": np.nan, "used": 0, "T_E": TE, "qE90": qE90, "qZ90": qZ}

    num = dZ[m] / qZ
    den = dE[m] / qE + eps
    val = np.exp(-num / den)
    return {"TS": float(np.mean(val)), "used": int(m.sum()), "T_E": TE, "qE90": qE90, "qZ90": qZ}


# ---- KS/W1: two-sample comparison against |delta E| ----
def ks_wasserstein_against_energy_diff(
    Delta_cond: np.ndarray, E: np.ndarray, q_grid: Iterable[float] = tuple(np.linspace(0.0, 1.0, 1001)[1:-1])
) -> dict[str, float]:
    """Compare delta distances and |delta E| via KS and Wasserstein-1 statistics.

    Returns the Kolmogorov-Smirnov statistic, the Wasserstein-1 distance, and
    ``sigma_hat`` from a half-normal fit to ``|delta E|`` (reported for reference).
    """
    Delta = np.asarray(Delta_cond, float).ravel()
    dE = energy_diff_condensed(E)

    m = np.isfinite(Delta) & np.isfinite(dE)
    if m.sum() < 10:
        return {"KS": np.nan, "W1": np.nan, "sigma_hat": np.nan}

    x = Delta[m].astype(float, copy=False)
    y = dE[m].astype(float, copy=False)

    sigma_hat = float(np.sqrt(np.mean(y**2) + 1e-12))

    try:
        from scipy.stats import ks_2samp, wasserstein_distance

        KS = float(ks_2samp(x, y, alternative="two-sided", mode="auto").statistic)
        W1 = float(wasserstein_distance(x, y))
    except Exception:
        xs, ys = np.sort(x), np.sort(y)
        grid = np.unique(np.concatenate([xs, ys], axis=0))
        Fx = np.searchsorted(xs, grid, side="right") / xs.size
        Fy = np.searchsorted(ys, grid, side="right") / ys.size
        KS = float(np.max(np.abs(Fx - Fy)))
        qs = np.asarray(list(q_grid), float)
        W1 = float(np.mean(np.abs(np.quantile(xs, qs) - np.quantile(ys, qs)))) if qs.size else np.nan

    return {"KS": KS, "W1": W1, "sigma_hat": sigma_hat}


def compute_energy_metrics_from_condensed(
    Delta_cond: ArrayLike,
    E: np.ndarray,
    order: Optional[np.ndarray] = None,
    # CKA memory control flags
    cka_memmap_ref: Optional[str] = None,
    cka_memmap_emb: Optional[str] = None,
    cka_in_mem_threshold: int = 8000,
    # Optional EJS-AUC computation
    compute_ejs_auc: bool = True,
) -> dict[str, float]:
    """Compute energy-aligned metrics from a condensed distance vector.

    Returns rank/statistics metrics (Spearman, Kendall, isotonic R2, dCor,
    CKA), point estimates for EJS and smoothness, and optionally EJS-AUC.
    """
    Delta_cond = np.asarray(Delta_cond, dtype=float).reshape(-1)
    m = Delta_cond.shape[0]
    n = n_from_condensed_len(m)

    E = np.asarray(E, dtype=float).reshape(-1)
    if E.size != n:
        raise ValueError(f"Energy length {E.size} != n inferred from Delta_cond ({n}).")

    Dref_cond = vector_to_absdiff_condensed(E, dtype=np.float32)

    out: dict[str, float] = {}
    # Correlation metrics
    out["spearman"] = spearman_from_condensed(Dref_cond, Delta_cond)
    out["kendall"] = kendall_from_condensed(Dref_cond, Delta_cond)
    out["iso_R2"] = isotonic_r2_from_condensed(Dref_cond, Delta_cond)
    # Distance correlation
    try:
        out["dCor"] = distance_correlation_from_condensed(Dref_cond, Delta_cond)
    except Exception:
        out["dCor"] = np.nan
    # CKA (RBF)
    try:
        out["cka_rbf"] = cka_rbf_from_condensed(
            Dref_cond,
            Delta_cond,
            expand_memmap1=cka_memmap_ref,
            expand_memmap2=cka_memmap_emb,
            in_mem_threshold=cka_in_mem_threshold,
        )
    except Exception:
        out["cka_rbf"] = np.nan

    # EJS with multiple lambda values
    ejs_dict = energy_jump_sensitivity_auto(
        E,
        Delta_cond,
        lam_grid=(0.1, 0.5, 1.0, 2.0, 3.0),  # Compute for all lambda values
        lam_rep=2.0,
        robust_sigma=True,
        dist_quantile=0.75,
        use_global_quantile=True,
    )
    # Copy all EJS-related keys
    for k, v in ejs_dict.items():
        if k.startswith("EJS") and isinstance(v, (int, float, np.floating, np.integer)):
            out[k] = float(v)
    # Backward compatibility
    out["EJS"] = float(ejs_dict.get("EJS", np.nan))
    out["EJS_num_jumps"] = float(ejs_dict.get("num_jumps", np.nan))
    out["EJS_tau"] = float(ejs_dict.get("tau", np.nan))

    # Optional EJS-AUC curve and scores
    if compute_ejs_auc:
        try:
            auc_res = ejs_auc_halfnormal(
                Delta_cond,
                E,  # Ensure delta precedes E
                lam=2.0,  # theta = 2 * sigma_d (tune if needed)
            )
            out["ROC_AUC"] = float(auc_res.get("ROC_AUC", np.nan))
            out["PR_AUC"] = float(auc_res.get("PR_AUC", np.nan))
        except Exception:
            out["ROC_AUC"] = np.nan
            out["PR_AUC"] = np.nan

    # Smoothness summary
    sm = energy_embedding_smoothness(E, Delta_cond, order=order)
    out["Smoothness"] = float(sm.get("Smoothness", np.nan))
    out["Smoothness_segments"] = float(sm.get("segments", np.nan))

    return out


# ============================================================
# ============ 6. Sampling, evaluation, and aggregation =======
# ============================================================


@dataclass
class Config:
    mol_types: list[str] = None
    root_dir: str = str(GLOBAL_DATA_ROOT / "traj" / "results")
    energy_dir: str = str(GLOBAL_DATA_ROOT / "traj" / "npz_data")
    # Sampling configuration
    n_samples: int = 100
    window: int = 2000
    traj_len: int = 100_000
    random_seed: int = 2025
    # Parallelism
    n_jobs: int = max(1, cpu_count() - 1)
    # Embedding distance settings
    block_size: int = 4096
    metric_embed: str = "cosine"
    # Output paths
    out_dir: str = str(GLOBAL_RESULTS_ROOT / "traj" / "energy_metrics_out")
    summary_csv: str = "summary_overall.csv"
    details_csv: str = "details_long.csv"
    save_intermediate: bool = True


DEFAULT_MOL_TYPES = [
    "rmd17_aspirin",
    "rmd17_azobenzene",
    "rmd17_benzene",
    "rmd17_ethanol",
    "rmd17_malonaldehyde",
    "rmd17_toluene",
    "rmd17_naphthalene",
    "rmd17_paracetamol",
    "rmd17_salicylic",
    "rmd17_uracil",
]


def mean_ci(values: list[float], alpha: float = 0.05) -> tuple[float, float]:
    arr = np.asarray(values, float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return (float(np.nan), float(np.nan))
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=1))
    half = 1.96 * s / math.sqrt(arr.size)
    return m, half


def _one_sample_worker(args: dict[str, Any]) -> dict[str, Any]:
    mol_type: str = args["mol_type"]
    idx_start: int = args["idx_start"]
    window: int = args["window"]
    cfg: Config = args["cfg"]
    idx_end = idx_start + window

    # 1) Energy fragment
    energy_path = os.path.join(cfg.energy_dir, mol_type + ".npz")
    energy = np.load(energy_path, mmap_mode="r")["energies"][idx_start:idx_end].astype(np.float64, copy=False)

    results_rows = []

    # ========== E3FP fingerprints ==========
    try:
        with open(os.path.join(cfg.root_dir, "e3fp", mol_type + ".pkl"), "rb") as f:
            fps_all = pickle.load(f)
        fps_sub = fps_all[idx_start:idx_end]
        D_tani = pairwise_tanimoto_fps_large(fps_sub, block_size=cfg.block_size, out_mode="condensed", progress=False)
        m1 = compute_energy_metrics_from_condensed(D_tani, energy)
        m2 = thresholded_smoothness(D_tani, energy, eps=1e-12)
        m3 = ks_wasserstein_against_energy_diff(D_tani, energy)
        metrics_e3fp = {**m1, **m2, **m3}
        for k, v in metrics_e3fp.items():
            results_rows.append(
                {
                    "mol_type": mol_type,
                    "sample_start": idx_start,
                    "sample_end": idx_end,
                    "model": "E3FP",
                    "metric": k,
                    "value": float(v) if np.isscalar(v) else np.nan,
                }
            )
    except Exception as e:
        results_rows.append(
            {
                "mol_type": mol_type,
                "sample_start": idx_start,
                "sample_end": idx_end,
                "model": "E3FP",
                "metric": "ERROR",
                "value": np.nan,
                "error": str(e),
            }
        )

    # ========== Continuous embedding models ==========
    embed_specs = [
        ("UniMol", "unimol", "arr_0"),
        ("MolSpectra", "molspec", "arr_0"),
        ("GemNet", "gemnet", "gemnet"),
        ("MolAE", "molae", "arr_0"),
        ("MACE", "mace", "arr_0"),
        ("FMG", "FMG", "embeddings"),  # FMG uses 'embeddings' key
    ]
    for model_name, subdir, key in embed_specs:
        try:
            path = os.path.join(cfg.root_dir, subdir, mol_type + ".npz")
            arr = np.load(path, mmap_mode="r")[key]
            Z = arr[idx_start:idx_end]
            D = pairwise_distances_from_embeddings_large(
                Z, metric=cfg.metric_embed, block_size=cfg.block_size, out_mode="condensed", progress=False
            )
            m1 = compute_energy_metrics_from_condensed(D, energy)
            m2 = thresholded_smoothness(D, energy, eps=1e-12)
            m3 = ks_wasserstein_against_energy_diff(D, energy)
            metrics = {**m1, **m2, **m3}
            for k, v in metrics.items():
                results_rows.append(
                    {
                        "mol_type": mol_type,
                        "sample_start": idx_start,
                        "sample_end": idx_end,
                        "model": model_name,
                        "metric": k,
                        "value": float(v) if np.isscalar(v) else np.nan,
                    }
                )
        except Exception as e:
            results_rows.append(
                {
                    "mol_type": mol_type,
                    "sample_start": idx_start,
                    "sample_end": idx_end,
                    "model": model_name,
                    "metric": "ERROR",
                    "value": np.nan,
                    "error": str(e),
                }
            )

    return {"rows": results_rows}


def run_once_for_molecule(cfg: Config, mol_type: str) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_seed)
    max_start = cfg.traj_len - cfg.window
    if max_start <= 0:
        raise ValueError("traj_len must be greater than window to draw samples.")
    starts = rng.integers(0, max_start, size=cfg.n_samples, endpoint=False)

    tasks = [{"mol_type": mol_type, "idx_start": int(s), "window": cfg.window, "cfg": cfg} for s in starts]

    print(f"[{mol_type}] Sampling {cfg.n_samples} windows of size {cfg.window}...")

    if cfg.n_jobs > 1:
        with Pool(processes=cfg.n_jobs) as pool:
            out_list = pool.map(_one_sample_worker, tasks)
    else:
        out_list = [_one_sample_worker(t) for t in tasks]

    rows_all = []
    for obj in out_list:
        rows_all.extend(obj["rows"])
    df = pd.DataFrame(rows_all)

    if cfg.save_intermediate:
        os.makedirs(cfg.out_dir, exist_ok=True)
        tmp_csv = os.path.join(cfg.out_dir, f"details_{mol_type}.csv")
        df.to_csv(tmp_csv, index=False)
        print(f"Saved details for {mol_type}: {tmp_csv}")

    return df


def aggregate_overall(details_df: pd.DataFrame) -> pd.DataFrame:
    df = details_df[details_df["metric"] != "ERROR"].copy()
    grouped = df.groupby(["model", "metric"])["value"].apply(list).reset_index()

    summary_rows = []
    for _, row in grouped.iterrows():
        model = row["model"]
        metric = row["metric"]
        vals = [float(v) for v in row["value"] if np.isfinite(v)]
        m, ci = mean_ci(vals, alpha=0.05)
        summary_rows.append({"model": model, "metric": metric, "mean": m, "ci95": ci, "n": len(vals)})
    summary = pd.DataFrame(summary_rows)

    mean_table = summary.pivot(index="model", columns="metric", values="mean")
    ci_table = summary.pivot(index="model", columns="metric", values="ci95")

    cols = []
    data = []
    for metric in mean_table.columns:
        cols.append(f"{metric}_mean")
        cols.append(f"{metric}_ci95")
    for model in mean_table.index:
        row_vals = []
        for metric in mean_table.columns:
            row_vals.append(mean_table.loc[model, metric])
            row_vals.append(ci_table.loc[model, metric])
        data.append(row_vals)

    summary_wide = pd.DataFrame(data, index=mean_table.index, columns=cols)
    summary_wide = summary_wide.sort_index()
    return summary_wide


def run_trajectory_benchmark(cfg: Optional[Config] = None) -> None:
    """Run the trajectory benchmark with an optional :class:`Config` override."""
    cfg = cfg or Config(mol_types=DEFAULT_MOL_TYPES)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    details_all = [run_once_for_molecule(cfg, mol) for mol in cfg.mol_types]
    details_df = pd.concat(details_all, axis=0, ignore_index=True)

    details_csv = out_dir / cfg.details_csv
    details_df.to_csv(details_csv, index=False)
    print(f"Details saved: {details_csv}  (rows={len(details_df)})")

    summary_wide = aggregate_overall(details_df)
    summary_csv = out_dir / cfg.summary_csv
    summary_wide.to_csv(summary_csv)
    print(f"Summary saved: {summary_csv}")
    print(summary_wide)


def main():
    run_trajectory_benchmark()


if __name__ == "__main__":
    main()
