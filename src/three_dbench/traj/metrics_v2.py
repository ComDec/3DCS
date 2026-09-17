"""Corrected ("v2") energy metrics for the trajectory benchmark.

The published tables (paper Tables 3, 6, 7) were computed with the definitions in
:func:`three_dbench.traj.evaluation.compute_energy_metrics_from_condensed`,
:func:`~three_dbench.traj.evaluation.thresholded_smoothness` and
:func:`~three_dbench.traj.evaluation.ks_wasserstein_against_energy_diff` (``--metric-version paper``).
Those implementations differ from the definitions written in the paper appendix (C.3, C.6) in
several places. ``--metric-version v2`` implements the appendix definitions, with the choices the
appendix leaves open made explicit. See ``docs/metrics/energy.md`` for formulas and rationale.

Differences from ``paper`` (per window of ``n`` frames, ``dE_ij = |E_i - E_j|``, ``dZ_ij = Delta_ij``):

* Distances are kept as float32 (``paper`` stores them as float16) and ``dE`` as float64 (``paper``
  casts ``dE`` to float32 for the rank/CKA/isotonic metrics).
* ``cka_rbf``: separate median-heuristic RBF bandwidths for ``dE`` and ``dZ`` (appendix C.3). The
  ``paper`` code uses one bandwidth from the concatenation of both, which is dominated by the
  representation distances and makes the energy kernel nearly an indicator of tied energies.
* ``iso_R2``: isotonic fit ``dE ~ f(dZ)`` and R^2 of ``dE`` (appendix C.3). ``paper`` fits the
  reverse direction (a constant representation then scores 1.0); v2 returns NaN for constant ``dZ``.
* ``EJS_*`` and ``EJS_num_jumps_*``: ``sigma_d = sqrt(mean(dE^2))`` (appendix C.6). ``paper`` uses the
  robust ``median(dE) / 0.6745``. ``ROC_AUC``/``PR_AUC`` already used ``sqrt(mean(dE^2))`` in ``paper``.
* ``KS``/``W1``: computed between commensurate, dimensionless quantities ``dZ / Q90(dZ)`` and
  ``dE / Q90(dE)``. ``paper`` compares ``dZ`` in [0, 1] with ``dE`` in kcal/mol, so KS is ~1 for every
  continuous representation. Smaller is better.
* ``TS`` and ``Smoothness`` are defined on consecutive frames and are only meaningful when the frames
  are in simulation-time order. They are NaN unless ``time_ordered=True``. rMD17 frames are **not**
  time-ordered (the ``old_indices`` are shuffled; consecutive-frame |dE| equals random-pair |dE|), so
  these two metrics are not reported for rMD17 in v2.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from three_dbench.common.metrics import (
    cka_rbf,
    distance_correlation,
    isotonic_r2,
    kendall_correlation,
    spearman_correlation,
)

from .evaluation import (
    _pair_indices_from_n,
    ejs_auc_halfnormal,
    energy_jump_sensitivity_auto,
    n_from_condensed_len,
)

LAMBDA_GRID = (0.1, 0.5, 1.0, 2.0, 3.0)
TS_GAMMA = 0.2


def _q90(values: np.ndarray, eps: float) -> float:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return float("nan")
    return float(np.quantile(v, 0.90)) + eps


def consecutive_segments(Delta_cond: np.ndarray, E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(dZ_k, dE_k)`` for consecutive frames ``k -> k+1`` (``k = 0..n-2``)."""
    n = E.size
    k = np.arange(n - 1, dtype=np.int64)
    idx = n * k - (k * (k + 1)) // 2  # condensed index of (k, k+1)
    return np.asarray(Delta_cond, dtype=np.float64)[idx], np.abs(np.diff(E))


def isotonic_r2_reference_from_representation(dE: np.ndarray, dZ: np.ndarray) -> float:
    """Appendix C.3 isoR^2: fit ``dE_hat = f_iso(dZ)`` (non-decreasing) and return R^2 of ``dE``."""
    return isotonic_r2(dZ, dE)


def compute_energy_metrics_v2(
    Delta_cond: np.ndarray,
    E: np.ndarray,
    *,
    time_ordered: bool = False,
    ts_gamma: float = TS_GAMMA,
    eps: float = 1e-12,
    compute_ejs_auc: bool = True,
) -> dict[str, float]:
    """Compute the v2 energy metrics for one window (see module docstring)."""
    D = np.asarray(Delta_cond, dtype=np.float64).reshape(-1)
    n = n_from_condensed_len(D.size)
    E = np.asarray(E, dtype=np.float64).reshape(-1)
    if E.size != n:
        raise ValueError(f"Energy length {E.size} != n inferred from Delta_cond ({n}).")

    ii, jj = _pair_indices_from_n(n)
    dE = np.abs(E[ii] - E[jj])

    out: dict[str, float] = {}
    out["spearman"] = spearman_correlation(dE, D)
    out["kendall"] = kendall_correlation(dE, D)
    out["iso_R2"] = isotonic_r2_reference_from_representation(dE, D)
    try:
        out["dCor"] = distance_correlation(dE, D)
    except Exception:
        out["dCor"] = np.nan
    try:
        out["cka_rbf"] = cka_rbf(dE, D, share_sigma=False)
    except Exception:
        out["cka_rbf"] = np.nan

    ejs = energy_jump_sensitivity_auto(
        E,
        D,
        lam_grid=LAMBDA_GRID,
        lam_rep=2.0,
        robust_sigma=False,
        dist_quantile=0.75,
        use_global_quantile=True,
    )
    for key, value in ejs.items():
        if key.startswith("EJS") and isinstance(value, (int, float, np.floating, np.integer)):
            out[key] = float(value)
    out["EJS"] = float(ejs.get("EJS", np.nan))
    out["EJS_num_jumps"] = float(ejs.get("num_jumps", np.nan))
    out["EJS_tau"] = float(ejs.get("tau", np.nan))

    if compute_ejs_auc:
        try:
            auc = ejs_auc_halfnormal(D, E, lam=2.0, robust_sigma=False)
            out["ROC_AUC"] = float(auc.get("ROC_AUC", np.nan))
            out["PR_AUC"] = float(auc.get("PR_AUC", np.nan))
        except Exception:
            out["ROC_AUC"] = np.nan
            out["PR_AUC"] = np.nan

    qZ = _q90(D, eps)
    qE = _q90(dE, eps)
    out["qZ90"] = qZ
    out["qE90"] = qE
    mask = np.isfinite(D) & np.isfinite(dE)
    if mask.sum() >= 10 and np.isfinite(qZ) and np.isfinite(qE):
        x = D[mask] / qZ
        y = dE[mask] / qE
        out["KS"] = float(ks_2samp(x, y, alternative="two-sided", mode="auto").statistic)
        out["W1"] = float(wasserstein_distance(x, y))
    else:
        out["KS"] = np.nan
        out["W1"] = np.nan

    out["time_ordered"] = 1.0 if time_ordered else 0.0
    out["T_E"] = float(ts_gamma * qE) if np.isfinite(qE) else np.nan
    if time_ordered and n >= 2:
        dZk, dEk = consecutive_segments(D, E)
        ok = np.isfinite(dZk) & np.isfinite(dEk)
        ratio = np.exp(-(dZk / qZ) / (dEk / qE + eps))
        out["Smoothness"] = float(np.mean(ratio[ok])) if ok.any() else np.nan
        out["Smoothness_segments"] = float(ok.sum())
        jump = ok & (dEk > out["T_E"])
        out["TS"] = float(np.mean(ratio[jump])) if jump.any() else np.nan
        out["used"] = float(jump.sum())
    else:
        out["Smoothness"] = np.nan
        out["Smoothness_segments"] = 0.0
        out["TS"] = np.nan
        out["used"] = 0.0
    return out
