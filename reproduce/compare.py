#!/usr/bin/env python
"""Compare recomputed results with the expected values of a paper table.

usage: python reproduce/compare.py --expected <expected.csv> --results <results.csv> [--strict]

expected.csv columns: table,model,metric,variant,paper_value,expected_value,tolerance,notes
results.csv  columns: table,model,metric,variant,value

Rows are matched on (table, model, metric, variant). Status per expected row:
  PASS         |value - expected_value| <= tolerance (tolerance defaults to 0.001)
  FAIL         the difference exceeds the tolerance, or the value is not finite
  MISSING      no result row (e.g. a model whose embeddings are not available); a failure with --strict
  NO_EXPECTED  expected_value is empty (this file holds no reference value for the row); never a failure
Exit code: 0 if no FAIL, 1 if any FAIL, 2 if no row could be compared.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

KEY = ("table", "model", "metric", "variant")
DEFAULT_TOLERANCE = 0.001


def _float(text):
    if text is None:
        return None
    text = str(text).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _read(path: Path, required: tuple[str, ...]) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path}: missing columns {sorted(missing)}")
        return [{k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()} for row in reader]


def compare(expected_rows: list[dict], result_rows: list[dict], *, strict: bool = False) -> tuple[list[dict], int]:
    results: dict[tuple, dict] = {}
    for row in result_rows:
        key = tuple(row[k] for k in KEY)
        if key in results:
            raise SystemExit(f"duplicate result row for {key}")
        results[key] = row
    report = []
    n_fail = 0
    n_compared = 0
    for exp in expected_rows:
        key = tuple(exp[k] for k in KEY)
        expected = _float(exp.get("expected_value"))
        tol = _float(exp.get("tolerance"))
        tol = DEFAULT_TOLERANCE if tol is None else tol
        res = results.get(key)
        value = _float(res.get("value")) if res is not None else None
        diff = None
        if expected is None:
            status = "NO_EXPECTED"
        elif res is None:
            status = "MISSING"
            if strict:
                n_fail += 1
                status = "MISSING(FAIL)"
        else:
            n_compared += 1
            if value is None or not math.isfinite(value):
                status = "FAIL"
                n_fail += 1
            else:
                diff = value - expected
                if abs(diff) <= tol:
                    status = "PASS"
                else:
                    status = "FAIL"
                    n_fail += 1
        report.append(
            {
                **{k: exp[k] for k in KEY},
                "paper_value": exp.get("paper_value", ""),
                "expected_value": "" if expected is None else f"{expected:.6f}",
                "value": "" if value is None else f"{value:.6f}",
                "diff": "" if diff is None else f"{diff:+.2e}",
                "tolerance": f"{tol:g}",
                "status": status,
            }
        )
    unmatched = sorted(set(results) - {tuple(e[k] for k in KEY) for e in expected_rows})
    for key in unmatched:
        print(f"note: result row without expected value: {key}", file=sys.stderr)
    return report, (1 if n_fail else (2 if n_compared == 0 else 0))


def _print_table(report: list[dict]) -> None:
    cols = [
        "table",
        "model",
        "metric",
        "variant",
        "paper_value",
        "expected_value",
        "value",
        "diff",
        "tolerance",
        "status",
    ]
    widths = {c: max(len(c), *(len(str(r[c])) for r in report)) if report else len(c) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in report:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expected", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--strict", action="store_true", help="treat MISSING rows as failures")
    ap.add_argument("--tables", nargs="*", default=None, help="only compare these table ids")
    ap.add_argument("--variants", nargs="*", default=None, help="only compare these variants")
    args = ap.parse_args(argv)

    expected = _read(args.expected, KEY + ("expected_value",))
    results = _read(args.results, KEY + ("value",))
    if args.tables:
        expected = [r for r in expected if r["table"] in args.tables]
        results = [r for r in results if r["table"] in args.tables]
    if args.variants:
        expected = [r for r in expected if r["variant"] in args.variants]
        results = [r for r in results if r["variant"] in args.variants]
    report, code = compare(expected, results, strict=args.strict)
    _print_table(report)
    counts: dict[str, int] = {}
    for r in report:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("\n" + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    if code == 2:
        print("No expected row had a matching result; nothing was compared.", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
