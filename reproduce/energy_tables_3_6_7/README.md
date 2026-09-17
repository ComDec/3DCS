# Tables 3, 6 and 7: zero-shot energy benchmark (rMD17)

```bash
bash reproduce/energy_tables_3_6_7/run.sh                          # published definitions
N_JOBS=24 METRIC_VERSIONS="paper v2" bash reproduce/energy_tables_3_6_7/run.sh   # plus corrected definitions
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
| `expected.csv` | For every published cell (variant `paper`): the printed value, the value recomputed with this code (6 decimals), a tolerance of 0.001 and notes. Variant `v2` rows hold the corrected definitions, which have no printed counterpart. |
| `paper_values.csv` | Values as printed in the final paper. Table 7 is written without thousands separators. |
| `make_expected.py` | Regenerates `expected.csv` from a reference `results.csv` and `paper_values.csv`, classifying each cell as rounding to the printed value, matching only by truncation, or differing. |
| `collect.py` | Maps `summary.csv` keys to table cells (see `docs/metrics/energy.md` §5). |

**Where the recomputed values differ from the printed tables.** The `notes` column of
`expected.csv` records each case:

- Table 3 KS for E3FP (0.916) and MolSpectra (0.977) differ from the Table 6 values for the same
  cells (0.913, 0.976). The recomputation reproduces the Table 6 values.
- Some Table 3 cells are printed truncated rather than rounded, e.g. UniMol EJS 0.30155 → 0.301.
- MACE and FMG columns are only partially reproducible from the backed-up embeddings. Most of their
  published 95% CIs are 4–5 times wider than a 1,000-window run gives, and the published FMG
  Smoothness (0.972 ± 0.002) is not attainable with these embeddings.
- The v2 `TS` and `Smoothness` are not reported, because rMD17 frames are not time-ordered
  (`docs/metrics/energy.md` §1).
