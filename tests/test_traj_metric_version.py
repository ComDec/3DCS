"""paper vs v2 energy metric definitions."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
from scipy.special import erfinv
from traj_helpers import save_energy_dataset, synthetic_embeddings, synthetic_energies

from three_dbench.common.metrics import cka_rbf
from three_dbench.traj.evaluation import energy_diff_condensed
from three_dbench.traj.protocol import compute_window_metrics, window_distances


@pytest.fixture
def window():
    E = synthetic_energies(120, seed=0)
    Z = synthetic_embeddings(E, 16, seed=1)
    return E, Z


def test_ejs_sigma_paper_robust_v2_rms(window):
    E, Z = window
    dE = energy_diff_condensed(E)
    robust = np.median(dE) / (np.sqrt(2.0) * erfinv(0.5) + 1e-12)
    rms = np.sqrt(np.mean(dE**2) + 1e-12)
    paper = compute_window_metrics(window_distances(Z, metric="cosine", metric_version="paper"), E)
    v2 = compute_window_metrics(window_distances(Z, metric="cosine", metric_version="v2"), E, metric_version="v2")
    for lam, tag in [(0.5, "0p5"), (2.0, "2")]:
        assert paper[f"EJS_theta_lam{tag}"] == pytest.approx(lam * robust)
        assert paper[f"EJS_num_jumps_lam{tag}"] == float((dE > lam * robust).sum())
        assert v2[f"EJS_theta_lam{tag}"] == pytest.approx(lam * rms)
        assert v2[f"EJS_num_jumps_lam{tag}"] == float((dE > lam * rms).sum())


def test_distance_precision_per_version(window):
    _, Z = window
    assert window_distances(Z, metric="cosine", metric_version="paper").dtype == np.float16
    assert window_distances(Z, metric="cosine", metric_version="v2").dtype == np.float32


def test_cka_bandwidth(window):
    E, Z = window
    D = window_distances(Z, metric="cosine", metric_version="v2")
    v2 = compute_window_metrics(D, E, metric_version="v2")
    dE = energy_diff_condensed(E)
    assert v2["cka_rbf"] == pytest.approx(cka_rbf(dE, D.astype(np.float64), share_sigma=False))
    assert v2["cka_rbf"] != pytest.approx(cka_rbf(dE, D.astype(np.float64), share_sigma=True))


@pytest.mark.filterwarnings("ignore:An input array is constant")
def test_isotonic_direction_constant_representation():
    E = synthetic_energies(80, seed=2)
    D = np.zeros(80 * 79 // 2, dtype=np.float32)  # constant representation
    paper = compute_window_metrics(D.astype(np.float16), E)
    v2 = compute_window_metrics(D, E, metric_version="v2")
    assert paper["iso_R2"] == pytest.approx(1.0)  # published implementation fits Delta ~ f(dE)
    assert np.isnan(v2["iso_R2"])  # appendix: dE ~ f(Delta), undefined for constant Delta


def test_v2_ks_is_scale_free(window):
    E, Z = window
    D = window_distances(Z, metric="cosine", metric_version="v2")
    E_scaled = E.mean() + 50.0 * (E - E.mean())
    v2 = compute_window_metrics(D, E, metric_version="v2")
    v2_scaled = compute_window_metrics(D, E_scaled, metric_version="v2")
    assert v2["KS"] == pytest.approx(v2_scaled["KS"], abs=1e-12)
    D16 = D.astype(np.float16)
    paper = compute_window_metrics(D16, E)
    paper_scaled = compute_window_metrics(D16, E_scaled)
    assert paper_scaled["KS"] > paper["KS"] + 0.05  # Delta in [0, 1] vs |dE| in energy units


def test_v2_ts_requires_time_order(window):
    E, Z = window
    D = window_distances(Z, metric="cosine", metric_version="v2")
    unordered = compute_window_metrics(D, E, metric_version="v2")
    assert np.isnan(unordered["TS"]) and np.isnan(unordered["Smoothness"])
    ordered = compute_window_metrics(D, E, metric_version="v2", time_ordered=True)
    assert 0.0 <= ordered["TS"] <= 1.0
    assert 0.0 <= ordered["Smoothness"] <= 1.0
    assert ordered["Smoothness_segments"] == E.size - 1
    paper = compute_window_metrics(D.astype(np.float16), E)
    assert np.isfinite(paper["TS"]) and np.isfinite(paper["Smoothness"])


def test_unknown_metric_version(window):
    E, Z = window
    with pytest.raises(ValueError):
        compute_window_metrics(window_distances(Z, metric="cosine"), E, metric_version="v3")


def test_cli_metric_version_switch(tmp_path):
    n = 160
    E = synthetic_energies(n, seed=3)
    dataset_dir = save_energy_dataset(tmp_path / "ds", {"rmd17_aspirin": E, "rmd17_ethanol": E[::-1].copy()})
    emb_dir = tmp_path / "emb"
    emb_dir.mkdir()
    np.savez(emb_dir / "rmd17_aspirin.npz", arr_0=synthetic_embeddings(E, 8, seed=4))
    np.savez(emb_dir / "rmd17_ethanol.npz", arr_0=synthetic_embeddings(E[::-1], 8, seed=5))
    summaries = {}
    for version in ("paper", "v2"):
        out = tmp_path / version
        cmd = [
            sys.executable,
            "-m",
            "three_dbench",
            "evaluate",
            "traj",
            "--dataset-dir",
            str(dataset_dir),
            "--embeddings",
            str(emb_dir),
            "--output-dir",
            str(out),
            "--n-samples",
            "2",
            "--window",
            "60",
            "--legacy-traj-len",
            str(n),
            "--molecules",
            "ethanol",
            "--metric-version",
            version,
            "--n-jobs",
            "2",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        config = json.loads((out / "config.json").read_text())
        assert config["metric_version"] == version
        assert config["window_scheme"] == "legacy"
        assert config["molecules"] == ["rmd17_ethanol"]
        import pandas as pd

        summaries[version] = pd.read_csv(out / "summary.csv").set_index("metric")["mean"]
    assert np.isfinite(summaries["paper"]["TS"]) and np.isnan(summaries["v2"]["TS"])
    assert summaries["paper"]["EJS_num_jumps_lam2"] != summaries["v2"]["EJS_num_jumps_lam2"]


def test_cli_help_lists_trajectory_options():
    result = subprocess.run(
        [sys.executable, "-m", "three_dbench", "evaluate", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    for flag in ("--window-scheme", "--metric-version", "--n-jobs", "--molecules", "--energy-precision-check"):
        assert flag in result.stdout
