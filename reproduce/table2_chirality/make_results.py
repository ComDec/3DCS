"""Collect Table 2 summaries written by run.sh into results.csv (table,model,metric,variant,value)."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

METRICS = [  # (paper row label, summary.csv column)
    ("ES-AUC", "ESA_AUC_mean"),
    ("NN@1-Acc", "NN1_acc_mean"),
    ("Hopkins", "hopkins_mean"),
    ("SCI", "sil_sup_mean"),
    ("SCI_unsup", "sil_unsup_mean"),
]
MODELS = ["e3fp", "gemnet", "molae", "molspectra", "unimol", "fmg", "mace"]
VARIANTS = ["euclidean", "cosine", "v2_euclidean", "v2_cosine"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, required=True, help="OUT_DIR of run.sh (<variant>/<model>/summary.csv)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for variant in VARIANTS:
        for model in MODELS:
            path = args.runs / variant / model / "summary.csv"
            if not path.exists():
                continue
            with path.open() as f:
                summary = next(csv.DictReader(f))
            for label, col in METRICS:
                value = float(summary[col])
                if math.isnan(value):  # e.g. Hopkins for fingerprints ("-" in the paper)
                    continue
                rows.append({"table": "table2", "model": model, "metric": label, "variant": variant, "value": value})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["table", "model", "metric", "variant", "value"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "value": repr(r["value"])})
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
