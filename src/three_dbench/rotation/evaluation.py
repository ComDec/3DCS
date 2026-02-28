"""Rotation geometry evaluation with optional shard sampling and batching.

Highlights:
- Sample-level control via SAMPLE_RATIO_SAMPLES / SAMPLE_MAX_SAMPLES / SAMPLE_SEED.
- Non-sampled entries only advance offsets without cutting embeddings.
- Ensures cosine distances are handled vector-wise and fixes the MOLAE directory path.
- Aggregates shard outputs into gzipped JSON summaries.
"""

import gzip
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import lmdb
import numpy as np
from numpy.linalg import norm
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolAlign
from rdkit.Chem.rdchem import Conformer
from sklearn.metrics import pairwise_distances

from three_dbench.common.metrics import (
    angular_smoothness,
    cka_rbf,
    distance_correlation,
    isotonic_r2,
    kendall_correlation,
    kruskal_stress,
    local_isometry_error,
    mantel_test,
    spearman_correlation,
    torsion_embedding_spearman,
    triplet_order_preservation,
)
from three_dbench.utils.paths import DATA_ROOT as GLOBAL_DATA_ROOT
from three_dbench.utils.paths import RESULTS_ROOT as GLOBAL_RESULTS_ROOT

# =========================
# Global evaluation configuration
# =========================
FAST = True  # True = favour speed; False = prefer exhaustive statistics
CONF = {
    "max_pairs_kendall": 100_000 if FAST else None,
    "mantel_permutations": 199 if FAST else 999,
    "triplets": 100_000 if FAST else 200_000,
    "lie_k": 10,
}

# =========================
# Sampling configuration per shard
# =========================
SAMPLE_RATIO_SAMPLES = 0.1  # Fraction of keys to evaluate; 1.0 = evaluate everything
SAMPLE_MAX_SAMPLES = None  # Optional hard cap per shard (e.g. 500)
SAMPLE_SEED = 12345  # Reproducible sampling seed


# =========================
# Utility helpers
# =========================
def _set_rng(seed: Optional[int] = None):
    return np.random.default_rng(seed)


# =========================
# Angle and RMSD utilities
# =========================
_TWOPI = 2.0 * np.pi


def _deg_to_rad_wrap(phi_deg: np.ndarray) -> np.ndarray:
    phi_rad = np.deg2rad(phi_deg.astype(np.float64, copy=False))
    return np.mod(phi_rad, _TWOPI)


def _angular_diff_matrix_from_degrees(phi_deg: np.ndarray) -> np.ndarray:
    phi = _deg_to_rad_wrap(np.asarray(phi_deg).reshape(-1))
    d = np.abs(phi[:, None] - phi[None, :])
    return np.minimum(d, _TWOPI - d, out=d)


def _torus_diff_matrix_from_degrees(Phi_deg: np.ndarray) -> np.ndarray:
    Phi = np.asarray(Phi_deg, dtype=np.float64)
    if Phi.ndim != 2:
        raise ValueError("Phi_deg must have shape (n, k)")
    n, k = Phi.shape
    Phi_rad = _deg_to_rad_wrap(Phi)
    acc = np.zeros((n, n), dtype=np.float64)
    for t in range(k):
        d = np.abs(Phi_rad[:, [t]] - Phi_rad[:, [t]].T)
        d = np.minimum(d, _TWOPI - d, out=d)
        acc += d * d
    return np.sqrt(acc, out=acc)


def merge_mols_as_conformers(mols: list[Chem.Mol], remove_hs: bool = True) -> Chem.Mol:
    if not mols:
        raise ValueError("Received an empty molecule list")
    proc = [Chem.RemoveHs(m) if remove_hs else Chem.AddHs(m, addCoords=True) for m in mols]
    base = Chem.Mol(proc[0])
    for cid in [c.GetId() for c in base.GetConformers()]:
        base.RemoveConformer(cid)
    natoms = base.GetNumAtoms()
    for m in proc:
        if m.GetNumAtoms() != natoms:
            raise ValueError("Failed to merge conformers: atom counts do not match")
        conf_src = m.GetConformer()
        if not conf_src.Is3D():
            raise ValueError("Encountered a non-3D conformer")
        conf_new = Conformer(natoms)
        conf_new.Set3D(True)
        for idx in range(natoms):
            conf_new.SetAtomPosition(idx, conf_src.GetAtomPosition(idx))
        base.AddConformer(conf_new, assignId=True)
    return base


def rmsd_matrix_from_one_mol(mol_list: list) -> np.ndarray:
    mol = merge_mols_as_conformers(mol_list, remove_hs=True)
    cids = [conf.GetId() for conf in mol.GetConformers()]
    n = len(cids)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        ci = cids[i]
        for j in range(i + 1, n):
            rms = rdMolAlign.GetBestRMS(mol, mol, prbId=ci, refId=cids[j])
            D[i, j] = D[j, i] = float(rms)
    return D


# =========================
# Distance matrices from embeddings/fingerprints
# =========================
def pairwise_distances_from_embeddings(
    Z: np.ndarray,
    metric: str = "euclidean",
    normalize_cosine: bool = True,
) -> np.ndarray:
    Z = np.asarray(Z, dtype=np.float64)
    if metric == "euclidean":
        sq = (Z * Z).sum(axis=1, keepdims=True)
        D2 = sq + sq.T - 2.0 * (Z @ Z.T)
        np.maximum(D2, 0.0, out=D2)
        return np.sqrt(D2, out=D2)
    elif metric == "cosine":
        X = Z.copy()
        if normalize_cosine:
            nrm = norm(X, axis=1, keepdims=True) + 1e-12
            X /= nrm
        S = X @ X.T
        np.clip(S, -1.0, 1.0, out=S)
        return 1.0 - S
    else:
        return pairwise_distances(Z, metric=metric)


def pairwise_distances_from_fingerprint(Z: list) -> np.ndarray:
    L = len(Z)
    sim = np.eye(L, dtype=np.float32)
    for i in range(L - 1):
        sims = DataStructs.BulkTanimotoSimilarity(Z[i], Z[i + 1 :])
        sim[i, i + 1 :] = sims
        sim[i + 1 :, i] = sims
    return (1.0 - sim).astype(np.float64, copy=False)


# =========================
# Metric computation (handled via shared module)
# =========================
def compute_all_geometry_metrics(
    D: np.ndarray,
    Z: Optional[np.ndarray] = None,
    Delta: Optional[np.ndarray] = None,
    deg: Optional[list[float]] = None,
    delta_metric: str = "cosine",
    embedding_type: str = "fp",
    max_pairs_spearman: Optional[int] = None,
    max_pairs_kendall: Optional[int] = CONF["max_pairs_kendall"],
    mantel_permutations: int = CONF["mantel_permutations"],
    triplets: int = CONF["triplets"],
    k_neighbors: int = CONF["lie_k"],
    lie_agg: str = "mean",
    cka_sigma_D: Optional[float] = None,
    cka_sigma_Delta: Optional[float] = None,
    random_state: Optional[int] = 0,
) -> dict:
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square distance matrix")
    if Delta is None:
        if Z is None:
            raise ValueError("Provide either Delta or Z")
        if embedding_type == "vector":
            Delta = pairwise_distances_from_embeddings(Z, metric=delta_metric)
        elif embedding_type == "fp":
            Delta = pairwise_distances_from_fingerprint(Z)  # type: ignore
        else:
            raise ValueError(f"embedding_type must be 'vector' or 'fp', got {embedding_type}")

    out = {}
    out["A1_spearman"] = spearman_correlation(D, Delta, max_pairs=max_pairs_spearman, random_state=random_state)
    out["A2_kendall"] = kendall_correlation(D, Delta, max_pairs=max_pairs_kendall, random_state=random_state)
    out["B_dcor"] = distance_correlation(D, Delta)
    rM, pM = mantel_test(D, Delta, n_permutations=mantel_permutations, method="spearman", random_state=random_state)
    out["C_mantel_r"] = rM
    out["C_mantel_p"] = pM
    stress1, a_fit, b_fit = kruskal_stress(D, Delta)
    out["D_stress1"] = stress1
    out["D_fit_a"] = a_fit
    out["D_fit_b"] = b_fit
    out["E_triplet_OP"] = triplet_order_preservation(D, Delta, n_triplets=triplets, random_state=random_state)
    out["G_cka_rbf"] = cka_rbf(D, Delta, sigma_a=cka_sigma_D, sigma_b=cka_sigma_Delta)
    k_neighbors = int(min(max(1, k_neighbors), D.shape[0] - 1))
    out["H_LIE@k"] = local_isometry_error(D, Delta, k=k_neighbors, aggregation=lie_agg)
    out["J_isotonic_R2"] = isotonic_r2(D, Delta)
    if deg is not None:
        deg_arr = np.asarray(deg, dtype=np.float64)
        out["torsion_sp"] = torsion_embedding_spearman(Delta, deg_arr)
        out["AS"] = angular_smoothness(Delta, deg_arr, circular=True)["AS"]
    else:
        out["torsion_sp"] = np.nan
        out["AS"] = np.nan
    return out


# =========================
# Parallel shard processing (sample-level)
# =========================
DATA_ROOT = GLOBAL_DATA_ROOT / "rotation"
RESULT_ROOT = GLOBAL_RESULTS_ROOT / "rotation"
SOURCE_ROOT = DATA_ROOT / "results"

LMDB_FP_WITH_SOURCE_DIR = SOURCE_ROOT / "fingerprint"
LMDB_DEG_DIR = SOURCE_ROOT / "sources"
UNIMOL_DIR = SOURCE_ROOT / "unimol"
MOLAE_DIR = SOURCE_ROOT / "molae"
MOLSPEC_DIR = SOURCE_ROOT / "molspectra"
OUT_DIR = RESULT_ROOT / "metrics_dict"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_SEC = 5.0
METRICS = ("cosine", "euclidean")  # Use ("cosine",) to reduce runtime if needed


def _pin_threads_single():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _fmt_eta(remain_items: int, rate: float) -> str:
    if rate <= 0 or remain_items < 0:
        return "ETA: --"
    secs = int(remain_items / rate)
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    return f"ETA: {h:02d}:{m:02d}:{s:02d}"


def _select_sample_indices(total: int, ratio: float, max_n: Optional[int], seed: int) -> np.ndarray:
    """Return sorted key indices selected for evaluation."""
    if not (0.0 < ratio <= 1.0):
        raise ValueError("SAMPLE_RATIO_SAMPLES must be in (0, 1].")
    n_take = int(np.ceil(total * ratio))
    if max_n is not None:
        n_take = min(n_take, int(max_n))
    n_take = max(1, min(n_take, total))
    rng = _set_rng(seed)
    idx = rng.choice(total, size=n_take, replace=False)
    return np.sort(idx)


def process_shard(
    i: int,
    sample_ratio: float = SAMPLE_RATIO_SAMPLES,
    sample_max: Optional[int] = SAMPLE_MAX_SAMPLES,
    sample_seed: int = SAMPLE_SEED,
):
    _pin_threads_single()

    lmdb_fp_with_source_path = LMDB_FP_WITH_SOURCE_DIR / f"rot_{i}.lmdb"
    lmdb_deg_path = LMDB_DEG_DIR / f"rot_conf_deg_{i}.lmdb"
    unimol_path = UNIMOL_DIR / f"{i}.npz"
    molae_path = MOLAE_DIR / f"{i}.npz"
    molspec_path = MOLSPEC_DIR / f"rot_{i}.npz"

    # Load embedding arrays
    arr_unimol = np.load(str(unimol_path))["arr_0"]
    arr_molae = np.load(str(molae_path))["arr_0"]
    arr_molspec = np.load(str(molspec_path))["arr_0"]

    # Open LMDB stores
    env_fp = lmdb.open(
        str(lmdb_fp_with_source_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
        max_readers=64,
    )
    env_deg = lmdb.open(
        str(lmdb_deg_path), subdir=False, readonly=True, lock=False, readahead=True, meminit=False, max_readers=64
    )

    # Enumerate all keys in the shard
    with env_fp.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))
    total = len(keys)
    sel_idx = _select_sample_indices(total, sample_ratio, sample_max, sample_seed + i)  # Diversify seed per shard
    sel_set = set(int(x) for x in sel_idx)
    print(f"[shard {i}] total={total}, take={len(sel_idx)} (ratio={sample_ratio}, max={sample_max})", flush=True)

    out_e3fp: dict[str, dict] = {}
    out_unimol: dict[str, dict] = {}
    out_molae: dict[str, dict] = {}
    out_molspec: dict[str, dict] = {}

    cur_idx = 0
    t0 = time.time()
    last_t = t0
    processed = 0
    seen = 0

    # Long-lived read transactions
    with env_fp.begin() as txn_fp, env_deg.begin() as txn_deg:
        for idx, k in enumerate(keys):
            seen += 1
            # Read fingerprint payload to determine conformer count (advance cursor)
            data_fp = pickle.loads(txn_fp.get(k))
            mol_list = [x[0] for x in data_fp["base"]]
            L = len(mol_list)

            if idx not in sel_set:
                # Non-sampled entry: advance cursor only, skip computations
                cur_idx += L
                continue

            # Sampled entry: perform full evaluation
            data_deg = pickle.loads(txn_deg.get(k))
            D = rmsd_matrix_from_one_mol(mol_list)
            deg_list = [x[2] for x in data_deg]  # torsion angles in degrees

            Z_e3fp = data_fp["e3fp"]["fps"]
            Delta_e3fp = pairwise_distances_from_fingerprint(Z_e3fp)

            Z_unimol = arr_unimol[cur_idx : cur_idx + L, :]
            Z_molae = arr_molae[cur_idx : cur_idx + L, :]
            Z_molspec = arr_molspec[cur_idx : cur_idx + L, :]
            cur_idx += L

            str_key = k.decode("utf-8")

            # e3fp
            out_e3fp[str_key] = compute_all_geometry_metrics(D, Z=None, Delta=Delta_e3fp, deg=deg_list)

            # Evaluate each embedding family
            out_unimol[str_key] = {}
            out_molae[str_key] = {}
            out_molspec[str_key] = {}

            for metric in METRICS:
                Delta_unimol = pairwise_distances_from_embeddings(Z_unimol, metric=metric)
                Delta_molae = pairwise_distances_from_embeddings(Z_molae, metric=metric)
                Delta_molspec = pairwise_distances_from_embeddings(Z_molspec, metric=metric)

                out_unimol[str_key][metric] = compute_all_geometry_metrics(D, Z=None, Delta=Delta_unimol, deg=deg_list)
                out_molae[str_key][metric] = compute_all_geometry_metrics(D, Z=None, Delta=Delta_molae, deg=deg_list)
                out_molspec[str_key][metric] = compute_all_geometry_metrics(
                    D, Z=None, Delta=Delta_molspec, deg=deg_list
                )

            processed += 1
            now = time.time()
            if now - last_t >= REPORT_SEC:
                rate = processed / (now - t0) if (now - t0) > 0 else 0.0
                pct = 100.0 * seen / total if total else 100.0
                print(
                    f"[shard {i}] seen {seen}/{total} ({pct:.1f}%), processed {processed}, "
                    f"{rate:.2f} it/s | {_fmt_eta(len(sel_idx) - processed, rate)}",
                    flush=True,
                )
                last_t = now

    dt = time.time() - t0
    print(f"[shard {i}] done in {dt:.1f}s | processed {processed}/{len(sel_idx)}", flush=True)

    return {"e3fp": out_e3fp, "unimol": out_unimol, "molae": out_molae, "molspec": out_molspec}


def merge_dicts(big, part):
    big["e3fp"].update(part["e3fp"])
    big["unimol"].update(part["unimol"])
    big["molae"].update(part["molae"])
    big["molspec"].update(part["molspec"])


def main(max_workers: Optional[int] = None):
    # Limit BLAS threads to prevent process x thread oversubscription
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    shards = list(range(16))
    combined = {"e3fp": {}, "unimol": {}, "molae": {}, "molspec": {}}

    if max_workers is None:
        cpu = os.cpu_count() or 8
        max_workers = min(8, cpu)

    t0 = time.time()
    done = 0
    print(
        f"[main] launch {len(shards)} shards with max_workers={max_workers}; "
        f"sample_ratio={SAMPLE_RATIO_SAMPLES}, sample_max={SAMPLE_MAX_SAMPLES}",
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_pin_threads_single) as ex:
        futs = [ex.submit(process_shard, i, SAMPLE_RATIO_SAMPLES, SAMPLE_MAX_SAMPLES, SAMPLE_SEED) for i in shards]
        for fut in as_completed(futs):
            part = fut.result()
            merge_dicts(combined, part)
            done += 1
            print(f"[main] shard {done}/{len(shards)} merged | elapsed {time.time() - t0:.1f}s", flush=True)

    out_path = OUT_DIR / "metrics_all_shards.json.gz"
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[Done] {out_path} | total {time.time() - t0:.1f}s", flush=True)


def run_rotation_benchmark(*, max_workers: Optional[int] = None) -> None:
    """Public entry point mirroring :func:`main`."""
    main(max_workers=max_workers)


if __name__ == "__main__":
    main()
