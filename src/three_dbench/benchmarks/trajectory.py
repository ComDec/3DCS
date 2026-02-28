"""Trajectory benchmark evaluation for user-provided embeddings."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from three_dbench.datasets.traj import load_traj_energy_dataset
from three_dbench.traj.evaluation import (
    compute_energy_metrics_from_condensed,
    ks_wasserstein_against_energy_diff,
    pairwise_distances_from_embeddings_large,
    thresholded_smoothness,
)


def _mean_ci(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size < 2:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    ci95 = 1.96 * std / np.sqrt(arr.size)
    return mean, float(ci95)


def evaluate_trajectory_embeddings(
    *,
    dataset_dir: Path,
    embeddings_by_mol: dict[str, np.ndarray],
    output_dir: Path | None = None,
    model_name: str = "custom",
    n_samples: int = 100,
    window: int = 2000,
    metric_embed: str = "cosine",
    block_size: int = 4096,
    random_seed: int = 2025,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate trajectory embeddings against the HF energy dataset."""
    dataset = load_traj_energy_dataset(dataset_dir)
    rng = np.random.default_rng(random_seed)
    details_rows = []

    for row in dataset:
        mol_type = row["mol_type"]
        energies = np.asarray(row["energies"], dtype=np.float64)
        embeddings = embeddings_by_mol[mol_type]
        if embeddings.shape[0] != energies.shape[0]:
            raise ValueError(f"Length mismatch for {mol_type}: {embeddings.shape[0]} != {energies.shape[0]}")

        max_start = energies.shape[0] - window
        if max_start <= 0:
            raise ValueError(f"Window size {window} exceeds trajectory length for {mol_type}.")
        starts = rng.integers(0, max_start, size=min(n_samples, max_start), endpoint=False)

        for start in starts:
            end = int(start + window)
            energy_window = energies[start:end]
            Z = embeddings[start:end]
            D = pairwise_distances_from_embeddings_large(
                Z,
                metric=metric_embed,
                block_size=block_size,
                out_mode="condensed",
                progress=False,
            )
            m1 = compute_energy_metrics_from_condensed(D, energy_window)
            m2 = thresholded_smoothness(D, energy_window, eps=1e-12)
            m3 = ks_wasserstein_against_energy_diff(D, energy_window)
            metrics = {**m1, **m2, **m3}

            for metric, value in metrics.items():
                details_rows.append(
                    {
                        "mol_type": mol_type,
                        "sample_start": int(start),
                        "sample_end": int(end),
                        "model": model_name,
                        "metric": metric,
                        "value": float(value) if np.isscalar(value) else np.nan,
                    }
                )

    details_df = pd.DataFrame(details_rows)
    grouped = details_df.groupby(["model", "metric"])["value"].apply(list).reset_index()
    summary_rows = []
    for _, row in grouped.iterrows():
        mean_val, ci_val = _mean_ci(row["value"])
        summary_rows.append(
            {
                "model": row["model"],
                "metric": row["metric"],
                "mean": mean_val,
                "ci95": ci_val,
                "n": len(row["value"]),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        details_path = output_dir / "details.csv"
        summary_path = output_dir / "summary.csv"
        details_df.to_csv(details_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        config_path = output_dir / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "model": model_name,
                    "n_samples": n_samples,
                    "window": window,
                    "metric_embed": metric_embed,
                    "block_size": block_size,
                    "random_seed": random_seed,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return details_df, summary_df
