"""Geometry (rotation) metric definitions and versioned presets.

Three presets are provided (see ``docs/metrics/geometry.md`` for the full rationale):

``paper``
    Reproduces the published Table 1 numbers. The definitions were recovered by matching
    per-molecule values in the backup outputs of the original runs, because the original
    evaluation scripts (``eval_geo_single.py`` / ``eval_geo_single_sup.py``) are not available.
``v2``
    Definitions as written in the paper's Table 1 caption and Appendix C.3/C.4
    (LIE with k=3 nearest neighbours excluding the conformer itself, AS as the median
    ``||z_{i+1} - z_i|| / |phi_{i+1} - phi_i|`` in radians, isotonic fit of D on Delta,
    RBF-CKA with median-heuristic bandwidths).
``legacy``
    The behaviour of ``compute_all_geometry_metrics`` in the initial public release (06dee1c).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np
from sklearn.isotonic import IsotonicRegression

from three_dbench.common.metrics import (
    cka_rbf,
    distance_correlation,
    isotonic_r2,
    kendall_correlation,
    kruskal_stress,
    mantel_test,
    spearman_correlation,
    to_condensed,
    torsion_embedding_spearman,
    triplet_order_preservation,
)

METRIC_VERSIONS = ("paper", "v2", "legacy")

LIE_VARIANTS = ("released", "exclude_self")
AS_VARIANTS = (
    "mean_delta_circular",  # legacy release: mean of Delta per radian, sorted torsions incl. wrap-around step
    "median_delta_circular",  # published 10 % run (not used in Table 1)
    "median_halfdelta_circular",  # published full "sup" run (Table 1 AS row)
    "median_dz_linear",  # v2 / Appendix C.4: median ||z_{i+1} - z_i|| / |dphi|, no wrap-around step
)
CKA_VARIANTS = ("median_2sigma2", "paper_run")
ISOR2_VARIANTS = ("model_on_reference", "reference_on_model", "reference_on_model_noguard")
LIE_DELTA_SCALES = ("raw", "unit_range")


@dataclass(frozen=True)
class GeometryMetricSpec:
    """Configuration of the per-molecule geometry metrics."""

    name: str = "custom"
    lie_k: int = 10
    lie_include_self: bool = True
    lie_delta_scale: str = "raw"
    lie_min_conformers: int = 2
    as_variant: str = "mean_delta_circular"
    cka_variant: str = "median_2sigma2"
    isor2_variant: str = "model_on_reference"
    kendall_min_conformers: int = 0
    kendall_max_pairs: int | None = 100_000
    extra_metrics: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS: dict[str, GeometryMetricSpec] = {
    "legacy": GeometryMetricSpec(
        name="legacy",
        lie_k=10,
        lie_include_self=True,
        lie_delta_scale="raw",
        lie_min_conformers=2,
        as_variant="mean_delta_circular",
        cka_variant="median_2sigma2",
        isor2_variant="model_on_reference",
        kendall_min_conformers=0,
        extra_metrics=True,
    ),
    "paper": GeometryMetricSpec(
        name="paper",
        lie_k=10,
        lie_include_self=True,
        lie_delta_scale="unit_range",
        lie_min_conformers=2,
        as_variant="median_halfdelta_circular",
        cka_variant="paper_run",
        isor2_variant="reference_on_model_noguard",
        kendall_min_conformers=11,
        extra_metrics=False,
    ),
    "v2": GeometryMetricSpec(
        name="v2",
        lie_k=3,
        lie_include_self=False,
        lie_delta_scale="raw",
        lie_min_conformers=3,
        as_variant="median_dz_linear",
        cka_variant="median_2sigma2",
        isor2_variant="reference_on_model",
        kendall_min_conformers=0,
        extra_metrics=False,
    ),
}


def resolve_metric_spec(
    metric_version: str = "paper",
    *,
    lie_k: int | None = None,
    lie_include_self: bool | None = None,
    as_variant: str | None = None,
    extra_metrics: bool | None = None,
) -> GeometryMetricSpec:
    """Return the preset for ``metric_version`` with optional explicit overrides."""
    if metric_version not in PRESETS:
        raise ValueError(f"metric_version must be one of {sorted(PRESETS)}, got {metric_version!r}")
    spec = PRESETS[metric_version]
    overrides = {}
    if lie_k is not None:
        overrides["lie_k"] = int(lie_k)
    if lie_include_self is not None:
        overrides["lie_include_self"] = bool(lie_include_self)
    if as_variant is not None:
        if as_variant not in AS_VARIANTS:
            raise ValueError(f"as_variant must be one of {AS_VARIANTS}, got {as_variant!r}")
        overrides["as_variant"] = as_variant
    if extra_metrics is not None:
        overrides["extra_metrics"] = bool(extra_metrics)
    if overrides:
        spec = replace(spec, name=f"{spec.name}+custom", **overrides)
    return spec


# ---------------------------------------------------------------------------
# Individual metric implementations
# ---------------------------------------------------------------------------


def local_isometry_error_knn(
    D: np.ndarray,
    Delta: np.ndarray,
    *,
    k: int,
    include_self: bool,
    eps: float = 1e-12,
) -> float:
    """LIE@k: mean over conformers of the RMS difference of locally normalised distances.

    ``include_self=True`` reproduces the released implementation, whose neighbour set
    (``argpartition(D)[:, :k]``) contains the conformer itself (distance 0).
    ``include_self=False`` uses the k nearest *other* conformers, as in Appendix C.4.
    ``k`` is capped at ``n - 1``.
    """
    D = np.asarray(D, dtype=np.float64)
    Delta = np.asarray(Delta, dtype=np.float64)
    n = D.shape[0]
    if n <= 1:
        return float("nan")
    k = int(min(max(1, k), n - 1))
    if include_self:
        nbrs = np.argpartition(D, kth=k, axis=1)[:, :k]
    else:
        Dm = D.copy()
        np.fill_diagonal(Dm, np.inf)
        if k < n - 1:
            nbrs = np.argpartition(Dm, kth=k - 1, axis=1)[:, :k]
        else:
            nbrs = np.argsort(Dm, axis=1, kind="stable")[:, :k]
    lies = np.empty(n, dtype=np.float64)
    for i in range(n):
        Di = D[i, nbrs[i]]
        Xi = Delta[i, nbrs[i]]
        diff = Di / (Di.mean() + eps) - Xi / (Xi.mean() + eps)
        lies[i] = math.sqrt(float(np.mean(diff * diff)))
    return float(np.mean(lies))


def _sorted_steps(torsion_deg: np.ndarray, circular: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Consecutive conformer pairs along the torsion scan and their angular increments (radians)."""
    phi = np.deg2rad(np.asarray(torsion_deg, dtype=np.float64).reshape(-1))
    order = np.argsort(phi, kind="stable")
    n = order.size
    if circular:
        i_idx = order
        j_idx = np.roll(order, -1)
        dphi = np.abs(phi[j_idx] - phi[i_idx])
        dphi = np.minimum(dphi, 2.0 * np.pi - dphi)
    else:
        i_idx = order[: n - 1]
        j_idx = order[1:]
        dphi = np.abs(phi[j_idx] - phi[i_idx])
    return i_idx, j_idx, dphi


def angular_smoothness_variant(
    torsion_deg: np.ndarray,
    *,
    variant: str,
    Delta: np.ndarray | None = None,
    Z: np.ndarray | None = None,
) -> float:
    """Angular smoothness (AS) under one of :data:`AS_VARIANTS`.

    Steps with an angular increment below 1e-9 rad are ignored.
    For ``median_dz_linear`` the displacement is ``||z_{i+1} - z_i||`` (Euclidean norm of the raw
    embedding difference); when ``Z`` is not a numeric array (e.g. fingerprints) the representation
    distance ``Delta`` is used instead.
    """
    torsion_deg = np.asarray(torsion_deg, dtype=np.float64).reshape(-1)
    n = torsion_deg.size
    if n < 2:
        return float("nan")
    if variant == "mean_delta_circular":
        i_idx, j_idx, dphi = _sorted_steps(torsion_deg, circular=True)
        steps = np.asarray(Delta, dtype=np.float64)[i_idx, j_idx]
        agg = np.mean
    elif variant == "median_delta_circular":
        i_idx, j_idx, dphi = _sorted_steps(torsion_deg, circular=True)
        steps = np.asarray(Delta, dtype=np.float64)[i_idx, j_idx]
        agg = np.median
    elif variant == "median_halfdelta_circular":
        i_idx, j_idx, dphi = _sorted_steps(torsion_deg, circular=True)
        steps = 0.5 * np.asarray(Delta, dtype=np.float64)[i_idx, j_idx]
        agg = np.median
    elif variant == "median_dz_linear":
        i_idx, j_idx, dphi = _sorted_steps(torsion_deg, circular=False)
        if Z is not None and isinstance(Z, np.ndarray) and Z.ndim == 2 and np.issubdtype(Z.dtype, np.number):
            Zd = np.asarray(Z, dtype=np.float64)
            steps = np.linalg.norm(Zd[j_idx] - Zd[i_idx], axis=1)
        else:
            steps = np.asarray(Delta, dtype=np.float64)[i_idx, j_idx]
        agg = np.median
    else:
        raise ValueError(f"Unknown AS variant {variant!r}; expected one of {AS_VARIANTS}")
    mask = dphi > 1e-9
    if not mask.any():
        return float("nan")
    return float(agg(steps[mask] / dphi[mask]))


def cka_rbf_paper_run(D: np.ndarray, Delta: np.ndarray) -> float:
    """RBF-CKA as computed in the published geometry runs.

    ``K = exp(-d^2 / (median(d) + 1e-12)^2)`` with the median over the upper triangle,
    computed separately for D and Delta, followed by HKH centring. (The appendix and
    :func:`three_dbench.common.metrics.cka_rbf` use ``exp(-d^2 / (2 sigma^2))``.)
    """
    D = np.asarray(D, dtype=np.float64)
    Delta = np.asarray(Delta, dtype=np.float64)
    n = D.shape[0]
    if n < 2:
        return float("nan")
    iu = np.triu_indices(n, 1)

    def _kernel(M: np.ndarray) -> np.ndarray:
        sigma = float(np.median(M[iu])) + 1e-12
        return np.exp(-(M**2) / sigma**2)

    H = np.eye(n) - 1.0 / n
    A = H @ _kernel(D) @ H
    B = H @ _kernel(Delta) @ H
    den = float(np.linalg.norm(A) * np.linalg.norm(B))
    if den == 0.0:
        return float("nan")
    return float(np.sum(A * B) / (den + 1e-20))


def isotonic_r2_reference_on_model(D: np.ndarray, Delta: np.ndarray, *, guard: bool = True) -> float:
    """Isotonic R^2 of the fit ``D_hat = f_iso(Delta)`` (Appendix C.3 direction).

    With ``guard=True`` the value is NaN for fewer than two pairs or a constant Delta
    (same guard as :func:`three_dbench.common.metrics.isotonic_r2`). With ``guard=False`` no
    guard is applied, which reproduces the published runs (a single pair gives 1.0).
    """
    if guard:
        return isotonic_r2(Delta, D)
    x = to_condensed(Delta)
    y = to_condensed(D)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size == 0:
        return float("nan")
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    yhat = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(xs, ys)
    ss_res = float(np.sum((ys - yhat) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2)) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def _unit_range(Delta: np.ndarray, delta_metric: str) -> np.ndarray:
    """Map a distance matrix to [0, 1] as described in Appendix C.2."""
    if delta_metric == "cosine":
        return 0.5 * Delta
    return Delta


def compute_geometry_metrics(
    D: np.ndarray,
    Delta: np.ndarray,
    *,
    torsion_deg: np.ndarray | None,
    spec: GeometryMetricSpec,
    Z: np.ndarray | None = None,
    delta_metric: str = "cosine",
    random_state: int | None = 0,
) -> dict[str, float]:
    """Compute per-molecule geometry metrics for one representation distance matrix.

    Keys follow the naming of the original outputs: ``A1_spearman``, ``A2_kendall``,
    ``G_cka_rbf``, ``H_LIE@k``, ``J_isotonic_R2``, ``torsion_sp``, ``AS`` (+ ``B_dcor``,
    ``C_mantel_r``, ``C_mantel_p``, ``D_stress1``, ``D_fit_a``, ``D_fit_b``, ``E_triplet_OP``
    when ``spec.extra_metrics``).
    """
    D = np.asarray(D, dtype=np.float64)
    Delta = np.asarray(Delta, dtype=np.float64)
    if D.ndim != 2 or D.shape != Delta.shape:
        raise ValueError(f"D and Delta must be square matrices of equal shape, got {D.shape} and {Delta.shape}")
    n = D.shape[0]
    out: dict[str, float] = {}
    out["A1_spearman"] = spearman_correlation(D, Delta)
    if n >= max(2, spec.kendall_min_conformers):
        out["A2_kendall"] = kendall_correlation(D, Delta, max_pairs=spec.kendall_max_pairs, random_state=random_state)
    else:
        out["A2_kendall"] = float("nan")
    if spec.cka_variant == "paper_run":
        out["G_cka_rbf"] = cka_rbf_paper_run(D, Delta)
    elif spec.cka_variant == "median_2sigma2":
        out["G_cka_rbf"] = cka_rbf(D, Delta) if n >= 2 else float("nan")
    else:
        raise ValueError(f"Unknown CKA variant {spec.cka_variant!r}")

    if n >= max(2, spec.lie_min_conformers):
        X_lie = _unit_range(Delta, delta_metric) if spec.lie_delta_scale == "unit_range" else Delta
        out["H_LIE@k"] = local_isometry_error_knn(D, X_lie, k=spec.lie_k, include_self=spec.lie_include_self)
    else:
        out["H_LIE@k"] = float("nan")

    if spec.isor2_variant == "model_on_reference":
        out["J_isotonic_R2"] = isotonic_r2(D, Delta)
    elif spec.isor2_variant == "reference_on_model":
        out["J_isotonic_R2"] = isotonic_r2_reference_on_model(D, Delta, guard=True)
    elif spec.isor2_variant == "reference_on_model_noguard":
        out["J_isotonic_R2"] = isotonic_r2_reference_on_model(D, Delta, guard=False)
    else:
        raise ValueError(f"Unknown isoR2 variant {spec.isor2_variant!r}")

    if torsion_deg is not None and len(torsion_deg) == n:
        deg = np.asarray(torsion_deg, dtype=np.float64)
        out["torsion_sp"] = torsion_embedding_spearman(Delta, deg)
        out["AS"] = angular_smoothness_variant(deg, variant=spec.as_variant, Delta=Delta, Z=Z)
    else:
        out["torsion_sp"] = float("nan")
        out["AS"] = float("nan")

    if spec.extra_metrics:
        out["B_dcor"] = distance_correlation(D, Delta)
        if n >= 3:
            rM, pM = mantel_test(D, Delta, n_permutations=199, method="spearman", random_state=random_state)
        else:
            rM, pM = float("nan"), float("nan")
        out["C_mantel_r"] = rM
        out["C_mantel_p"] = pM
        s1, a_fit, b_fit = kruskal_stress(D, Delta)
        out["D_stress1"] = s1
        out["D_fit_a"] = a_fit
        out["D_fit_b"] = b_fit
        out["E_triplet_OP"] = triplet_order_preservation(D, Delta, n_triplets=100_000, random_state=random_state)
    return out


__all__ = [
    "AS_VARIANTS",
    "METRIC_VERSIONS",
    "PRESETS",
    "GeometryMetricSpec",
    "angular_smoothness_variant",
    "cka_rbf_paper_run",
    "compute_geometry_metrics",
    "isotonic_r2_reference_on_model",
    "local_isometry_error_knn",
    "resolve_metric_spec",
]
