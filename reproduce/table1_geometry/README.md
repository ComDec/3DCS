# Table 1: geometry (rotation dataset)

```bash
bash reproduce/table1_geometry/run.sh             # all 16 shards
QUICK=1 bash reproduce/table1_geometry/run.sh     # shard 1 only
```

`run.sh`:

1. downloads the `rotation` config of `EscheWang/3dcs` to `data/hf/rotation` and the GemNet rotation
   embeddings (`rotation/gemnet/rotation_conformers_{0..15}.npz`) to `data/embeddings`
   (skip with `SKIP_DOWNLOAD=1` when they are already there);
2. evaluates GemNet with `--metric-version paper --replicate-offset-drift` and with
   `--metric-version v2` (cosine space, `--layout by-shard`);
3. writes `results.csv` with `collect.py`;
4. compares it with `expected.csv` (or `expected_quick.csv` for `QUICK=1`).

## Variants

| Variant | Rows | Definition |
|---|---|---|
| `paper` | all 7 | Spearman, Kendall, CKA, isotonic R², Torsion-SP: mean over the 146,389 molecules in `sampled_molecules_seed2027.txt`; LIE@k and AS: mean over all molecules, with the embedding-offset shift of the original full run (`<metric>__offset_drift` columns). This is Table 1. |
| `paper_aligned` | LIE@k, AS | same definitions, correctly aligned embeddings |
| `v2` | all 7 | `--metric-version v2`, mean over all evaluated molecules |

Definitions, provenance and the offset shift are described in
[docs/metrics/geometry.md](../../docs/metrics/geometry.md).

## Files

| File | Content |
|---|---|
| `expected.csv` | Table 1 cells for the five models. `expected_value` for `paper` rows = mean of the original per-molecule outputs (`metrics_all_0.1_1.json.gz` for the sampled rows, `metrics_sup_100.json.gz` for LIE@k and AS). Only GemNet can be recomputed; rows of the other models report `MISSING`. The GemNet `paper_aligned` and `v2` reference values need a full 16-shard run and are left empty (`NO_EXPECTED`). |
| `expected_quick.csv` | shard 1 only: `paper` rows from the original outputs restricted to shard 1; `paper_aligned` and `v2` rows are regression values computed with this code. |
| `sampled_molecules_seed2027.txt` | the 146,389 molecule keys of the original 10 % sample run, in dataset order (read from the backup output; the sampling code is not available). SHA-256 `35da6a9ea84ee1112bd99c43028269cba9035ffa913f0eaafbfba7fab83163ac`. |
| `collect.py` | per-key parquet → `results.csv` |

## Run time

Measured on the A100 box with 24 worker processes (shared with another 24-process job), cosine space
only: shard 1 took 129 s with `--metric-version paper --replicate-offset-drift` (including the second
pass over the 80,901 shifted molecules) and 75 s with `--metric-version v2`. A full run is expected to
take about 20–25 min per metric version, plus downloads (dataset ~7.5 GB, embeddings ~5.2 GB).
