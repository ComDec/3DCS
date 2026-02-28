"""Rotation benchmark evaluation for user-provided embeddings."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from three_dbench.datasets.rotation import load_rotation_dataset
from three_dbench.datasets.serialization import blocks_to_mols
from three_dbench.embeddings import EmbeddingArray
from three_dbench.rotation.evaluation import compute_all_geometry_metrics, rmsd_matrix_from_one_mol


def _slice_flat_embeddings(embeddings: EmbeddingArray, offset: int, count: int):
    if isinstance(embeddings.array, np.ndarray):
        return embeddings.array[offset : offset + count]
    return embeddings.array[offset : offset + count]


def _aggregate_metric(values: Iterable[float | None]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(np.median(arr))


def evaluate_rotation_embeddings(
    *,
    dataset_dir: Path,
    embeddings: EmbeddingArray | None = None,
    embeddings_by_key: dict[str, np.ndarray] | None = None,
    output_dir: Path | None = None,
    model_name: str = "custom",
    metrics: Iterable[str] = ("cosine", "euclidean"),
) -> dict[str, dict[str, dict[str, float]]]:
    """Evaluate rotation embeddings using the HF dataset."""
    if embeddings is None and embeddings_by_key is None:
        raise ValueError("Provide either embeddings or embeddings_by_key.")

    dataset = load_rotation_dataset(dataset_dir)
    results: dict[str, dict[str, dict[str, float]]] = {}
    metrics = list(metrics)

    for row in dataset:
        key = row["key"]
        if not row["mol_blocks"]:
            raise ValueError("Rotation dataset is missing MolBlocks. Rebuild without --no-mol-blocks.")
        mols = blocks_to_mols(row["mol_blocks"], sanitize=False)
        D = rmsd_matrix_from_one_mol(mols)
        deg = row.get("torsion_deg")

        if embeddings_by_key is not None:
            Z = embeddings_by_key[key]
            entry_metrics = {}
            for metric in metrics:
                entry_metrics[metric] = compute_all_geometry_metrics(
                    D,
                    Z=Z,
                    Delta=None,
                    deg=deg,
                    delta_metric=metric,
                    embedding_type="vector",
                )
        else:
            assert embeddings is not None
            emb = embeddings
            offset = int(row["offset"])
            count = int(row["n_conformers"])
            Z = _slice_flat_embeddings(emb, offset, count)
            entry_metrics = {}
            if emb.kind == "fingerprint":
                entry_metrics["fingerprint"] = compute_all_geometry_metrics(
                    D,
                    Z=Z,
                    Delta=None,
                    deg=deg,
                    embedding_type="fp",
                )
            else:
                for metric in metrics:
                    entry_metrics[metric] = compute_all_geometry_metrics(
                        D,
                        Z=Z,
                        Delta=None,
                        deg=deg,
                        delta_metric=metric,
                        embedding_type="vector",
                    )

        results[key] = entry_metrics

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_json = output_dir / f"{model_name}_per_key.json"
        out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

        summary_rows = []
        for metric_name in next(iter(results.values())).keys():
            metric_keys = next(iter(results.values()))[metric_name].keys()
            for metric_key in metric_keys:
                values = [results[k][metric_name].get(metric_key) for k in results]
                mean_val, med_val = _aggregate_metric(values)
                summary_rows.append(
                    {
                        "model": model_name,
                        "metric": f"{metric_name}:{metric_key}",
                        "mean": mean_val,
                        "median": med_val,
                        "n": len(values),
                    }
                )
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_dir / "summary.csv", index=False)

    return results
