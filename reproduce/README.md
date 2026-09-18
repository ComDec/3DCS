# Reproducing the paper tables

Each directory recomputes one or more tables of the ICLR 2026 paper from the released datasets
(`EscheWang/3dcs`), the published embeddings (`EscheWang/3dcs-embeddings`) and this code.

| Directory | Tables | Command | Notes |
|---|---|---|---|
| [`table1_geometry/`](table1_geometry/README.md) | 1 | `bash reproduce/table1_geometry/run.sh` | GemNet only (the other rotation embeddings are not available); `QUICK=1` runs shard 1 |
| [`table2_chirality/`](table2_chirality/README.md) | 2 | `bash reproduce/table2_chirality/run.sh` | 7 models; Euclidean (published) and cosine / v2 variants |
| [`energy_tables_3_6_7/`](energy_tables_3_6_7/README.md) | 3, 6, 7 | `bash reproduce/energy_tables_3_6_7/run.sh` | 7 models; float64 energies, legacy windows |

Every directory contains:

- `run.sh`: downloads what it needs, runs `python -m three_dbench evaluate …`, writes `results.csv`
  and calls `compare.py`. The header of each script lists its environment variables (output
  directory, number of workers, reuse of local copies).
- `expected.csv`: one row per table cell and variant.

## File formats

`expected.csv`

| Column | Meaning |
|---|---|
| `table` | table id (`1`, `2`, `3`, …; `1-shard1` for the quick Table 1 check) |
| `model` | lower-case model name (`e3fp`, `gemnet`, `molae`, `molspectra`, `unimol`, `fmg`, `mace`) |
| `metric` | metric id used by the table's `collect` script |
| `variant` | e.g. `paper`, `euclidean`, `cosine`, `v2` (see the table's README) |
| `paper_value` | value as printed in the paper (empty if the variant is not in the paper) |
| `expected_value` | value recomputed with this code or taken from the original outputs, 6 decimals; empty if not available yet |
| `tolerance` | allowed absolute difference (default 0.001) |
| `notes` | provenance and known differences from the printed value (rounding vs truncation, cells that cannot be recomputed, …) |

`results.csv`: `table,model,metric,variant,value`.

## Comparing

```bash
python reproduce/compare.py --expected reproduce/table1_geometry/expected.csv --results results/reproduce/table1_geometry/results.csv
```

Status per expected row:

- `PASS`: `|value − expected_value| ≤ tolerance`;
- `FAIL`: larger difference or a non-finite value;
- `MISSING`: no result row, e.g. a model whose embeddings are not published (a failure with `--strict`);
- `NO_EXPECTED`: the reference value is not available yet (never a failure).

The exit code is 1 if any row fails and 2 if no row could be compared. `--tables` and `--variants`
restrict the comparison.

The `expected_value` is compared, not the printed `paper_value`: printed values are rounded or, in
some cells, truncated, and a few printed cells differ from the recomputed values for reasons listed in
`notes`.
