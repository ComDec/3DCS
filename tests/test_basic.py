"""Basic unit tests for 3DBench."""

from __future__ import annotations

from three_dbench.chirality.evaluation import evaluate_en_separation_from_counts
from three_dbench.traj.evaluation import (
    compute_energy_metrics_from_condensed,
    pairwise_distances_from_embeddings_large,
)


def test_chirality_counts(chirality_key_counts, rng):
    total = sum(chirality_key_counts.values())
    embeddings = rng.standard_normal((total, 6))
    rows, summary = evaluate_en_separation_from_counts(chirality_key_counts, embeddings, per_mol_min_n=2)
    assert len(rows) == 2
    assert summary["n_molecules"] == 2
    assert "ESA_AUC_mean" in summary


def test_trajectory_metrics(rng):
    energies = rng.normal(size=10)
    embeddings = rng.normal(size=(10, 8))
    D = pairwise_distances_from_embeddings_large(
        embeddings,
        metric="cosine",
        block_size=8,
        out_mode="condensed",
        progress=False,
    )
    metrics = compute_energy_metrics_from_condensed(D, energies, compute_ejs_auc=False)
    assert "spearman" in metrics
    assert "kendall" in metrics
