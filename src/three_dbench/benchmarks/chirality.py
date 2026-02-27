"""Chirality benchmark evaluation for user-provided embeddings."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Tuple

import json
import pandas as pd

from three_dbench.chirality.evaluation import evaluate_en_separation_from_counts
from three_dbench.datasets.chirality import load_chirality_dataset
from three_dbench.embeddings import EmbeddingArray


def _key_counts_from_dataset(dataset) -> Dict[str, int]:
    counts = OrderedDict()
    for row in dataset:
        counts[row["key"]] = int(row["n_conformers"])
    return counts


def evaluate_chirality_embeddings(
    *,
    dataset_dir: Path,
    embeddings: EmbeddingArray,
    output_dir: Optional[Path] = None,
    model_name: str = "custom",
    per_mol_min_n: int = 2,
    do_unsup_when_single_en: bool = False,
    unsup_kmax: int = 50,
    max_molecules: Optional[int] = None,
) -> Tuple[Dict, Dict]:
    """Evaluate chirality embeddings against the HF dataset."""
    dataset = load_chirality_dataset(dataset_dir)
    key_to_counts = _key_counts_from_dataset(dataset)

    rows, summary = evaluate_en_separation_from_counts(
        key_to_counts,
        embeddings.array,
        per_mol_min_n=per_mol_min_n,
        do_unsup_when_single_en=do_unsup_when_single_en,
        unsup_kmax=unsup_kmax,
        max_molecules=max_molecules,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        per_mol_path = output_dir / f"{model_name}_per_molecule.json"
        with per_mol_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)

        summary_df = pd.DataFrame([summary])
        summary_df.insert(0, "model", model_name)
        summary_path = output_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False)

    return {"rows": rows}, summary
