#!/usr/bin/env python
"""Turn rotation per-key outputs into results.csv rows for Table 1.

Table 1 of the paper combines two original runs (see docs/metrics/geometry.md):
- Spearman, Kendall, CKA, isotonic R^2 and Torsion-SP are means over the 10 % molecule sample
  (``sampled_molecules_seed2027.txt``); that run used correctly aligned embeddings;
- LIE@k and AS are means over all evaluated molecules of the full run, in which the embeddings of
  shards 1 and 2 were misaligned after a molecule that failed (``<metric>__offset_drift`` columns,
  written by ``evaluate rotation --replicate-offset-drift``).
Variant ``paper`` reproduces the published values; variant ``paper_aligned`` gives LIE@k and AS with
correctly aligned embeddings; variant ``v2`` gives all rows (means over all evaluated molecules)
with the corrected definitions. Means exclude non-finite per-molecule values.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROWS = [
    ("spearman", "A1_spearman", "sample"),
    ("kendall", "A2_kendall", "sample"),
    ("cka", "G_cka_rbf", "sample"),
    ("isotonic_r2", "J_isotonic_R2", "sample"),
    ("lie_k", "H_LIE@k", "all_drift"),
    ("torsion_sp", "torsion_sp", "sample"),
    ("as", "AS", "all_drift"),
]


def _mean(values: pd.Series) -> float:
    v = values.to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def collect(per_key: Path, *, model: str, variant: str, table: str, space: str, sample_keys: set[str] | None):
    df = pd.read_parquet(per_key)
    df = df[df["space"] == space]
    rows = []

    def add(name: str, var: str, series: pd.Series) -> None:
        rows.append({"table": table, "model": model, "metric": name, "variant": var, "value": f"{_mean(series):.6f}"})

    for name, col, subset in ROWS:
        if variant != "paper":
            add(name, variant, df[col])
        elif subset == "sample":
            add(name, "paper", df.loc[df["key"].isin(sample_keys), col])
        else:
            drift_col = f"{col}__offset_drift"
            if drift_col not in df.columns:
                raise SystemExit(f"{per_key}: missing {drift_col}; run evaluate rotation with --replicate-offset-drift")
            add(name, "paper", df[drift_col])
            add(name, "paper_aligned", df[col])
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs=3, action="append", metavar=("VARIANT", "MODEL", "PER_KEY_PARQUET"), required=True)
    ap.add_argument("--table", default="1")
    ap.add_argument("--space", default="cosine")
    ap.add_argument("--molecule-list", type=Path, default=HERE / "sampled_molecules_seed2027.txt")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    sample_keys = None
    if any(v == "paper" for v, _, _ in args.run):
        with args.molecule_list.open() as fh:
            sample_keys = {line.strip() for line in fh if line.strip() and not line.startswith("#")}
    rows = []
    for variant, model, path in args.run:
        rows.extend(
            collect(
                Path(path), model=model, variant=variant, table=args.table, space=args.space, sample_keys=sample_keys
            )
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["table", "model", "metric", "variant", "value"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
