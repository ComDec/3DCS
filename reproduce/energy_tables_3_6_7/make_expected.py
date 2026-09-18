"""Build expected.csv from a reference results.csv and the printed paper values (maintainer tool).

    python make_expected.py --results out/results.csv --paper paper_values.csv --out expected.csv

``expected_value`` is the reference value of this release, rounded to 6 decimals; ``paper_value`` is
the value as printed in the paper; ``notes`` describes the metric version and the row. The console
summary reports, for the ``paper`` rows, how each reference value relates to the printed precision.
"""

from __future__ import annotations

import argparse
import math
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

TOLERANCE = 0.001

PAPER_NOTE = (
    "paper metric definitions (docs/metrics/energy.md); reference value computed with this code from "
    "the published trajectory embeddings and the float64 rMD17 energies (legacy windows, "
    "100 x 2000 frames, seed 2025)"
)
V2_NOTE = "v2 metric definitions (docs/metrics/energy.md); not printed in the paper"
TABLE7_NOTE = (
    "jump counts depend only on the energies and the windows, not on the representation; the paper "
    "prints them with thousands separators"
)


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
    """Describe what the row computes (metric version, variant, provenance of the reference value)."""
    if printed is None:
        return ""
    notes = [PAPER_NOTE]
    if table == "7":
        notes.append(TABLE7_NOTE)
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
            note = V2_NOTE if r.variant == "v2" else f"{r.variant} metric definitions (docs/metrics/energy.md)"
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
