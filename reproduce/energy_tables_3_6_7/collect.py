"""Collect Table 3/6/7 cells from per-model trajectory runs into results.csv.

Input: ``<runs>/<metric_version>/<model>/summary.csv`` written by ``python -m three_dbench evaluate traj``.
Output columns: ``table,model,metric,variant,value`` (variant = metric version: ``paper`` or ``v2``).

Cell naming
  Table 3: metric in {Spearman, Kendall, CKA, isoR2, EJS, EJS_ROCAUC, TS, KS} (means).
  Table 6: ``<row>:mean`` and ``<row>:ci95`` for rows Spearman, Kendall, CKA, isoR2, EJS(<l>sigma) for
           l in {0.1, 0.5, 1, 2, 3}, ROC_AUC, Smoothness, TS, KS.
  Table 7: model ``all``; ``jumps(<l>sigma):mean`` / ``:ci95``. The counts depend only on the energies and
           windows, so they are identical for every model; the script checks this.
Non-finite values (v2 TS/Smoothness on rMD17, which is not time-ordered) are omitted.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

DISPLAY = {
    "e3fp": "E3FP",
    "gemnet": "GemNet",
    "molae": "MolAE",
    "molspectra": "MolSpectra",
    "unimol": "UniMol",
    "fmg": "FMG",
    "mace": "MACE",
}
TABLE3_MODELS = ["E3FP", "GemNet", "MolAE", "MolSpectra", "UniMol", "FMG", "MACE"]
TABLE6_MODELS = ["E3FP", "GemNet", "MolAE", "MolSpectra", "MACE", "FMG"]
TABLE3_ROWS = [
    ("Spearman", "spearman"),
    ("Kendall", "kendall"),
    ("CKA", "cka_rbf"),
    ("isoR2", "iso_R2"),
    ("EJS", "EJS"),
    ("EJS_ROCAUC", "ROC_AUC"),
    ("TS", "TS"),
    ("KS", "KS"),
]
LAMBDAS = [("0.1", "0p1"), ("0.5", "0p5"), ("1", "1"), ("2", "2"), ("3", "3")]
TABLE6_ROWS = (
    [("Spearman", "spearman"), ("Kendall", "kendall"), ("CKA", "cka_rbf"), ("isoR2", "iso_R2")]
    + [(f"EJS({lam}sigma)", f"EJS_lam{tag}") for lam, tag in LAMBDAS]
    + [("ROC_AUC", "ROC_AUC"), ("Smoothness", "Smoothness"), ("TS", "TS"), ("KS", "KS")]
)
TABLE7_ROWS = [(f"jumps({lam}sigma)", f"EJS_num_jumps_lam{tag}") for lam, tag in LAMBDAS]


def _finite(x: float) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def collect(runs: Path) -> pd.DataFrame:
    rows = []
    for version_dir in sorted(p for p in runs.iterdir() if p.is_dir()):
        variant = version_dir.name
        summaries = {}
        for model_dir in sorted(p for p in version_dir.iterdir() if p.is_dir()):
            summary_path = model_dir / "summary.csv"
            if not summary_path.exists():
                continue
            df = pd.read_csv(summary_path)
            name = DISPLAY.get(model_dir.name, model_dir.name)
            summaries[name] = df.set_index("metric")
        for name, s in summaries.items():
            if name in TABLE3_MODELS:
                for cell, metric in TABLE3_ROWS:
                    if metric in s.index and _finite(s.loc[metric, "mean"]):
                        rows.append(("3", name, cell, variant, float(s.loc[metric, "mean"])))
            if name in TABLE6_MODELS:
                for cell, metric in TABLE6_ROWS:
                    if metric in s.index and _finite(s.loc[metric, "mean"]):
                        rows.append(("6", name, f"{cell}:mean", variant, float(s.loc[metric, "mean"])))
                        rows.append(("6", name, f"{cell}:ci95", variant, float(s.loc[metric, "ci95"])))
        for cell, metric in TABLE7_ROWS:
            values = {n: (float(s.loc[metric, "mean"]), float(s.loc[metric, "ci95"])) for n, s in summaries.items()}
            if not values:
                continue
            distinct = set(values.values())
            if len(distinct) != 1:
                raise SystemExit(f"Table 7 {cell} ({variant}) differs between models: {values}")
            mean, ci = distinct.pop()
            rows.append(("7", "all", f"{cell}:mean", variant, mean))
            rows.append(("7", "all", f"{cell}:ci95", variant, ci))
    return pd.DataFrame(rows, columns=["table", "model", "metric", "variant", "value"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected", type=Path, default=None, help="expected.csv to subset")
    parser.add_argument(
        "--expected-out", type=Path, default=None, help="write the expected rows for the computed cells"
    )
    args = parser.parse_args()
    results = collect(args.runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False, float_format="%.10g")
    print(f"wrote {len(results)} cells to {args.out}")
    if args.expected is not None and args.expected_out is not None and args.expected.exists():
        expected = pd.read_csv(args.expected, dtype={"table": str})
        keys = set(map(tuple, results[["table", "model", "metric", "variant"]].astype(str).values))
        mask = [tuple(map(str, r)) in keys for r in expected[["table", "model", "metric", "variant"]].values]
        expected[mask].to_csv(args.expected_out, index=False)
        missing = len(expected[expected["variant"].isin(results["variant"].unique())]) - int(sum(mask))
        print(f"wrote {int(sum(mask))} expected rows to {args.expected_out} ({missing} expected cells not computed)")


if __name__ == "__main__":
    main()
