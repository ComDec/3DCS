# Tables 3, 6 and 7: zero-shot energy benchmark (rMD17)

```bash
bash reproduce/energy_tables_3_6_7/run.sh                          # published definitions
N_JOBS=24 METRIC_VERSIONS="paper v2" bash reproduce/energy_tables_3_6_7/run.sh   # plus the v2 definitions
```

`run.sh` does four things:

1. Downloads the float64 rMD17 energies (`EscheWang/3dcs`, config `traj_energies`).
2. Downloads the backed-up trajectory embeddings of the 7 baselines (`EscheWang/3dcs-embeddings`,
   `traj/<model>/`, about 7 GB).
3. Runs `python -m three_dbench evaluate traj` once per model with the published protocol: legacy
   windows, 100 × 2,000 frames per molecule, seed 2025, cosine distance (Tanimoto for E3FP).
4. Writes `out/results.csv` (`table,model,metric,variant,value`) with `collect.py` and compares it
   with `expected.csv` using `reproduce/compare.py`.

Existing local copies can be reused with `ENERGY_DATASET_DIR=...` and `EMB_ROOT=...` (see the header
of `run.sh`). Runs whose `summary.csv` already exists are skipped unless `FORCE=1`.

**Runtime.** On a shared 128-thread box with `N_JOBS=24`, one metric version took about 50 minutes
for the 7 models (1,000 windows per model). See `timings.tsv` in the output folder.

**Files.**

| File | Content |
|---|---|
| `expected.csv` | One row per table cell and metric version: the value as printed in the paper, the reference value computed with this code (6 decimals), a tolerance of 0.001 and a note describing the row. Variant `v2` rows are not printed in the paper. |
| `paper_values.csv` | Values as printed in the final paper. Table 7 is written without thousands separators. |
| `make_expected.py` | Regenerates `expected.csv` from a reference `results.csv` and `paper_values.csv`. |
| `collect.py` | Maps `summary.csv` keys to table cells (see `docs/metrics/energy.md` §5). |

**Reference values.** `expected_value` is what this code computes from the published trajectory
embeddings and the float64 rMD17 energies with the published protocol (legacy windows, 100 × 2,000
frames, seed 2025); `compare.py` compares against it. `paper_value` carries the value as printed in
the table of that row.

`v2` does not report `TS` and `Smoothness` here: both are defined along consecutive frames and are
computed only with `--time-ordered` (`docs/metrics/energy.md` §1).
