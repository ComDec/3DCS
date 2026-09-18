"""Build expected.csv from a reference results.csv and the printed paper values (maintainer tool).

    python make_expected.py --results out/results.csv --paper paper_values.csv --out expected.csv

expected_value is the recomputed value rounded to 6 decimals. For variant ``paper`` each cell is
classified against the printed value (3 decimals): it rounds to the printed value, matches it only by
truncation, or differs from it. Known issues are added to ``notes``.
"""

from __future__ import annotations

import argparse
import math
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

TOLERANCE = 0.001

T3_VS_T6 = {  # Table 3 cells printed differently from the same Table 6 mean
    ("MACE", "Kendall"): ("0.159", "0.160"),
    ("MACE", "isoR2"): ("0.080", "0.083"),
    ("FMG", "Kendall"): ("0.009", "0.010"),
    ("FMG", "CKA"): ("0.011", "0.012"),
    ("FMG", "TS"): ("0.582", "0.583"),
    ("FMG", "KS"): ("0.999", "1.000"),
    ("MolSpectra", "EJS_ROCAUC"): ("0.526", "0.527"),
    ("MolSpectra", "KS"): ("0.977", "0.976"),
    ("E3FP", "KS"): ("0.916", "0.913"),
}


def _decimals(printed: str) -> int:
    return len(printed.split(".")[1]) if "." in printed else 0


def classify(value: float, printed: str) -> str:
    q = Decimal(1).scaleb(-_decimals(printed))
    v = Decimal(repr(value))
    if v.quantize(q, rounding=ROUND_HALF_UP) == Decimal(printed):
        return "round"
    if v.quantize(q, rounding=ROUND_DOWN) == Decimal(printed):
        return "trunc"
    return "mismatch"


def notes_for(table: str, model: str, metric: str, value: float, printed: str | None) -> str:
    if printed is None:
        return ""
    status = classify(value, printed)
    notes = []
    if status == "round":
        notes.append("recomputed value rounds to the printed value")
    elif status == "trunc":
        notes.append("printed value is the 3-dp truncation (not rounding) of the recomputed value")
    else:
        notes.append("recomputed value does not match the printed value")

    base = metric.split(":")[0]
    stat = metric.split(":")[1] if ":" in metric else "mean"
    if table == "3" and (model, base) in T3_VS_T6:
        t3, t6 = T3_VS_T6[(model, base)]
        notes.append(f"Table 3 prints {t3} while Table 6 prints {t6} for the same mean")
    if table == "3" and base == "KS" and model in ("E3FP", "MolSpectra"):
        notes.append(
            "the Table 3 value is carried over from the submission draft and is not produced by any backed-up run"
        )
    if model in ("MACE", "FMG") and status == "mismatch":
        if stat == "ci95":
            notes.append(
                "published MACE/FMG CIs are about 4-5x wider than this 1000-window run (consistent with roughly "
                "40-47 windows); no backed-up run reproduces the published MACE/FMG CIs"
            )
        else:
            notes.append(
                "MACE/FMG energy columns are only partially reproducible from the backed-up embeddings; no "
                "backed-up run reproduces this published value"
            )
    if model == "FMG" and base == "Smoothness":
        notes.append(
            "published FMG Smoothness 0.972 +/- 0.002 is not attainable with the backed-up FMG embeddings: "
            "every one of the 1000 per-window values is >= 0.999"
        )
    if table == "7":
        notes.append("model-independent; printed with thousands separators in the paper")
    return "; ".join(notes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    results = pd.read_csv(args.results, dtype={"table": str})
    paper = pd.read_csv(args.paper, dtype={"table": str, "paper_value": str})
    printed = {(r.table, r.model, r.metric): r.paper_value for r in paper.itertuples()}

    rows = []
    for r in results.itertuples():
        if not math.isfinite(r.value):
            continue
        pv = printed.get((r.table, r.model, r.metric)) if r.variant == "paper" else None
        note = notes_for(r.table, r.model, r.metric, float(r.value), pv)
        if r.variant != "paper":
            note = f"{r.variant} metric definitions (docs/metrics/energy.md); no published counterpart"
        rows.append(
            {
                "table": r.table,
                "model": r.model,
                "metric": r.metric,
                "variant": r.variant,
                "paper_value": pv if pv is not None else "",
                "expected_value": f"{float(r.value):.6f}",
                "tolerance": TOLERANCE,
                "notes": note,
                "_status": classify(float(r.value), pv) if pv is not None else "",
            }
        )
    missing = set(printed) - {(r["table"], r["model"], r["metric"]) for r in rows if r["variant"] == "paper"}
    if missing:
        raise SystemExit(f"printed cells without a recomputed value: {sorted(missing)[:10]}")
    out = pd.DataFrame(rows)
    order = {"paper": 0, "v2": 1}
    out["_v"] = out["variant"].map(order)
    out["_t"] = out["table"].astype(int)
    out = out.sort_values(["_v", "_t"], kind="stable").drop(columns=["_v", "_t"])
    counts = out.loc[out["_status"] != "", "_status"].value_counts()
    out = out.drop(columns=["_status"])
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} rows to {args.out}; paper cells: {counts.to_dict()}")


if __name__ == "__main__":
    main()
