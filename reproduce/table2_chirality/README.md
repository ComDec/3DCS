# Table 2: zero-shot chirality

This directory recomputes Table 2 of the paper from the published embeddings.

```bash
pip install -e .
bash reproduce/table2_chirality/run.sh                  # all 7 models x 4 variants
VARIANTS=euclidean N_JOBS=16 bash reproduce/table2_chirality/run.sh   # published protocol only
```

`run.sh` runs four steps:

1. It saves the `chirality` config of `EscheWang/3dcs` to `data/hf/chirality`.
2. It downloads the seven files listed in `models.csv` from `EscheWang/3dcs-embeddings` into
   `data/embeddings/` (about 382 MB) and checks their SHA-256.
3. It runs `python -m three_dbench evaluate chirality` for each model and variant.
4. It writes `results/reproduce/table2_chirality/results.csv` and compares it with `expected.csv`
   using `reproduce/compare.py`.

To use local copies of the files, set `DATASET_DIR`, `EMB_ROOT` (a directory laid out as
`chirality/<model>/<file>`) and `SKIP_DOWNLOAD=1`. The header of `run.sh` lists all variables.

| Variant | CLI options | In the paper? |
|---|---|---|
| `euclidean` | `--distance euclidean --metric-version paper` (the defaults) | yes, this is Table 2 |
| `cosine` | `--distance cosine --metric-version paper` | no |
| `v2_euclidean` | `--distance euclidean --metric-version v2` | no |
| `v2_cosine` | `--distance cosine --metric-version v2` | no |

All variants scan the best-k silhouette up to `k = n - 1` (`--unsup-kmax n-1`, the default).

## What to expect

- **`euclidean`** is the published protocol. `compare.py` reports PASS for all 34 cells of
  `expected.csv` when the run uses the published embeddings. E3FP Hopkins is not defined for
  fingerprints (NaN) and has no row.
- **`euclidean` against the original per-molecule outputs** (`en_sep_results`, in the embeddings repo
  under `results/chirality/`), which cover E3FP, GemNet, MolAE, MolSpectra and UniMol:
  - ES-AUC, NN@1-Acc, SCI, Hopkins and DBI are bit-identical for every molecule.
  - SCI_unsup is bit-identical for E3FP, MolAE and MolSpectra; GemNet and UniMol differ by about
    2e-6, because KMeans local optima depend on the BLAS kernel.
  - For FMG and MACE the release has no per-molecule output; their reference values are computed from
    the published embeddings.
- **Other variants.** The `cosine` and `v2_*` rows are not printed in the paper. They are given so
  that the alternative definitions can be checked. See `docs/metrics/chirality.md` for what each one
  computes.

## Run time

On 22 processes of an AMD EPYC 7513, each model and variant takes about 40–65 s for E3FP and 1–3 min
for the other models, so all 28 runs take roughly 40–60 min. With `N_JOBS=1`, expect roughly 10–20
times longer.
