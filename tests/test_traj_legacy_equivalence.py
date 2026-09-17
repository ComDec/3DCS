"""The CLI protocol reproduces the legacy runner behind the published tables bit for bit."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from traj_helpers import random_fingerprints, save_energy_dataset, synthetic_embeddings, synthetic_energies

import three_dbench.traj.evaluation as ev
from three_dbench.benchmarks import evaluate_trajectory_embeddings

# ---- reference loop implementations of the index helpers (0.1.0 code) ----


def _pairs_loop(n):
    ii, jj = [], []
    for i in range(n - 1):
        ii.extend([i] * (n - i - 1))
        jj.extend(range(i + 1, n))
    return np.asarray(ii, int), np.asarray(jj, int)


def _fetch_loop(Delta_cond, pairs):
    Delta_cond = np.asarray(Delta_cond, float)
    n = ev.n_from_condensed_len(Delta_cond.shape[0])
    vals = np.empty(len(pairs[0]), dtype=np.float64)
    for k, (i, j) in enumerate(zip(*pairs)):
        if i == j:
            vals[k] = 0.0
        else:
            i, j = min(i, j), max(i, j)
            vals[k] = Delta_cond[ev.condensed_index(i, j, n)]
    return vals


@pytest.mark.parametrize("n", [0, 1, 2, 3, 57])
def test_vectorised_index_helpers_match_loops(n):
    ii, jj = ev._pair_indices_from_n(n)
    ri, rj = _pairs_loop(n)
    np.testing.assert_array_equal(ii, ri)
    np.testing.assert_array_equal(jj, rj)
    assert ii.dtype == ri.dtype
    x = synthetic_energies(n, seed=n) if n else np.zeros(0)
    ref = np.abs(x[ri] - x[rj])
    np.testing.assert_array_equal(ev.energy_diff_condensed(x), ref)
    np.testing.assert_array_equal(ev.vector_to_absdiff_condensed(x), ref.astype(np.float32))
    if n >= 2:
        D = np.random.default_rng(n).random(n * (n - 1) // 2).astype(np.float16)
        rng = np.random.default_rng(0)
        pairs = (rng.integers(0, n, 200), rng.integers(0, n, 200))
        np.testing.assert_array_equal(ev._condensed_fetch_pairs(D, pairs), _fetch_loop(D, pairs))
        consecutive = (np.arange(n - 1), np.arange(1, n))
        np.testing.assert_array_equal(ev._condensed_fetch_pairs(D, consecutive), _fetch_loop(D, consecutive))


@pytest.fixture
def legacy_layout(tmp_path):
    """Two molecules in the legacy on-disk layout plus the equivalent HF-style dataset."""
    n = 400
    mols = ["rmd17_toluene", "rmd17_naphthalene"]  # dataset order differs from the legacy order
    energies = {m: synthetic_energies(n, seed=i) for i, m in enumerate(mols)}
    root = tmp_path / "results"
    (root / "gemnet").mkdir(parents=True)
    (root / "e3fp").mkdir()
    energy_dir = tmp_path / "npz_data"
    energy_dir.mkdir()
    gem, fps = {}, {}
    for i, m in enumerate(mols):
        np.savez(energy_dir / f"{m}.npz", energies=energies[m])
        gem[m] = synthetic_embeddings(energies[m], 12, seed=100 + i)
        np.savez(root / "gemnet" / f"{m}.npz", gemnet=gem[m])
        fps[m] = random_fingerprints(n, 128, seed=200 + i, density=0.2)
        with (root / "e3fp" / f"{m}.pkl").open("wb") as f:
            pickle.dump(fps[m], f)
    dataset_dir = save_energy_dataset(tmp_path / "hf", energies)
    return {
        "n": n,
        "mols": mols,
        "root": root,
        "energy_dir": energy_dir,
        "dataset_dir": dataset_dir,
        "gem": gem,
        "fps": fps,
        "out": tmp_path,
    }


def _legacy_details(layout, n_samples, window):
    cfg = ev.Config(
        mol_types=list(ev.DEFAULT_MOL_TYPES),
        root_dir=str(layout["root"]),
        energy_dir=str(layout["energy_dir"]),
        n_samples=n_samples,
        window=window,
        traj_len=layout["n"],
        random_seed=2025,
        n_jobs=1,
        out_dir=str(layout["out"] / "legacy"),
        save_intermediate=False,
    )
    frames = [ev.run_once_for_molecule(cfg, m) for m in ev.DEFAULT_MOL_TYPES if m in layout["mols"]]
    df = pd.concat(frames, ignore_index=True)
    return df[df["metric"] != "ERROR"]


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_cli_protocol_equals_legacy_runner(legacy_layout, n_jobs):
    n_samples, window = 3, 60
    legacy = _legacy_details(legacy_layout, n_samples, window)
    legacy_summary = ev.aggregate_overall(legacy)
    for model, embeddings in (("GemNet", legacy_layout["gem"]), ("E3FP", legacy_layout["fps"])):
        details, summary = evaluate_trajectory_embeddings(
            dataset_dir=legacy_layout["dataset_dir"],
            embeddings_by_mol=embeddings,
            model_name=model,
            n_samples=n_samples,
            window=window,
            n_jobs=n_jobs,
            legacy_traj_len=legacy_layout["n"],
            verbose=False,
        )
        ref = legacy[legacy["model"] == model].reset_index(drop=True)
        new = details.reset_index(drop=True)
        cols = ["mol_type", "sample_start", "sample_end", "model", "metric"]
        pd.testing.assert_frame_equal(new[cols], ref[cols], check_dtype=False)
        np.testing.assert_array_equal(new["value"].to_numpy(float), ref["value"].to_numpy(float))
        for _, row in summary.iterrows():
            assert row["mean"] == legacy_summary.loc[model, f"{row['metric']}_mean"] or (
                np.isnan(row["mean"]) and np.isnan(legacy_summary.loc[model, f"{row['metric']}_mean"])
            )
            ci_ref = legacy_summary.loc[model, f"{row['metric']}_ci95"]
            assert row["ci95"] == ci_ref or (np.isnan(row["ci95"]) and np.isnan(ci_ref))
