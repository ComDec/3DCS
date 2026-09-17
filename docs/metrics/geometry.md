# Geometry metrics (rotation dataset, Table 1)

This page documents how `python -m three_dbench evaluate rotation` computes the geometry
metrics, which definitions were used for the published Table 1, and what `--metric-version v2`
changes. Chirality and energy metrics are described in [chirality.md](chirality.md) and
[energy.md](energy.md); [../METRICS.md](../METRICS.md) is the index.

## Data and alignment

- Dataset: `EscheWang/3dcs`, config `rotation`: 1,559,779 molecules (rows), 10,097,643
  conformers, 16 shards. Each row has `key`, `shard`, `n_conformers`, `offset`, `mol_blocks`
  and `torsion_deg`.
- **`offset` is per shard.** It restarts at 0 in every shard, and the rows are ordered by the
  shard id as a string (`0, 1, 10, 11, …, 15, 2, …, 9`). The initial release (0.1.0) sliced a single
  flat array with this offset, which picks the wrong conformers for every shard other than 0.
  The evaluator now supports three layouts:

  | `--layout` | Embeddings | Slicing |
  |---|---|---|
  | `by-shard` (default for a directory) | one file per shard, e.g. `rotation_conformers_{shard}.npz` (the layout of `EscheWang/3dcs-embeddings`) | file of `row.shard`, rows `offset : offset + n_conformers` |
  | `flat` (default for a file) | one array in **dataset row order** | cumulative sum of `n_conformers` over the dataset rows |
  | `by-key` | dict `{key: (n_conformers, dim)}` | by key |

  Datasets written by the original converter (`convert rotation`) store global offsets in numeric
  shard order; `--offset-mode auto` detects this for complete datasets and converts them to per-shard
  offsets. For a filtered subset of such a dataset, pass `--offset-mode global`.
- Reference distance `D`: RMSD after optimal alignment, `rdMolAlign.GetBestRMS` on heavy atoms
  (`Chem.RemoveHs`), symmetry-aware, for all conformer pairs of a molecule.
- Representation distance `Δ`: cosine distance `1 − cos` (not rescaled), Euclidean distance, or
  Tanimoto distance for RDKit bit vectors. `--metrics` selects the spaces; Table 1 uses cosine
  (Tanimoto for E3FP).
- Molecules with fewer than 2 conformers are skipped (`--min-conformers 2`), as in the published
  runs (1,464,495 of 1,559,779 molecules qualify).
- Molecules whose conformers cannot be merged for RMSD (different heavy-atom counts after
  `RemoveHs`) are skipped and listed in `config.json` (`failed_keys`). On the published data this
  happens for `1-R5B2G8_6-R7B1G14_0_22` (shard 1) and, according to the backup outputs,
  `6-R8B2G15_35-R2B1G6_0_36` (shard 2); the published full run also has no result for these two.
- Aggregation: one value per molecule; `summary.csv` reports the mean and median over molecules
  with a finite value.

Outputs: `<model>_per_key.parquet` (one row per molecule and distance space), `summary.csv` and
`config.json` (definitions, selection counts, failures, runtime).

## Definitions by metric version

`--metric-version paper` is the default. The Python API also exposes `legacy` (the behaviour of
`compute_all_geometry_metrics` in release 0.1.0) for comparison.

| Metric (output key) | `paper` (published runs) | `v2` (paper text) | `legacy` (release 0.1.0) |
|---|---|---|---|
| Spearman (`A1_spearman`) | Spearman of upper-triangular `D` vs `Δ` | same | same |
| Kendall (`A2_kendall`) | as Spearman, **only for molecules with ≥ 11 conformers** (NaN otherwise) | all molecules with ≥ 2 pairs | all molecules |
| CKA (`G_cka_rbf`) | `K = exp(−d² / (median(d) + 1e-12)²)` per matrix (median over the upper triangle), HKH centring | `K = exp(−d² / 2σ²)`, `σ² = median` of the positive squared distances, per matrix (median heuristic, Appendix C.3) | as v2 |
| Isotonic R² (`J_isotonic_R2`) | fit `D̂ = f_iso(Δ)`, R² on `D`; no guard (a single pair gives 1.0) | fit `D̂ = f_iso(Δ)`, R² on `D`; NaN for < 2 pairs or constant `Δ` | fit `Δ̂ = f_iso(D)`, R² on `Δ` (a constant representation scores 1.0) |
| LIE@k (`H_LIE@k`) | k = 10, neighbour set `argpartition(D)[:, :k]` (**includes the conformer itself**), `Δ` rescaled to [0, 1] (`(1 − cos)/2` for cosine) | k = 3 nearest **other** conformers (Table 1 caption, Appendix C.4); molecules with ≥ 3 conformers | k = 10 including self, `Δ` not rescaled |
| Torsion-SP (`torsion_sp`) | Spearman of `Δ` vs circular torsion distance `min(|φi − φj|, 2π − |φi − φj|)` | same | same |
| AS (`AS`) | **median** over consecutive conformers sorted by torsion, **including the wrap-around step**, of `((1 − cos)/2) / Δφ` (radians) | **median** of `‖z_{i+1} − z_i‖ / |φ_{i+1} − φ_i|` (radians) over consecutive conformers sorted by torsion, no wrap-around step (Appendix C.4); Tanimoto distance for bit vectors | **mean** of `Δ / Δφ` including the wrap-around step |

Individual choices can be overridden: `--lie-k`, `--lie-self {include,exclude}`,
`--as-variant {mean_delta_circular,median_delta_circular,median_halfdelta_circular,median_dz_linear}`.
`--extra-metrics` adds distance correlation, Mantel r/p, Kruskal stress and triplet order
preservation (not in Table 1; they dominate the runtime).

Notes on `v2`:

- **LIE**: the neighbour set of the paper definition does not contain the conformer itself.
  With the conformer included, one of the k "neighbours" has distance 0 in both spaces, which
  dilutes the score, and k = 10 exceeds the conformer count of most molecules (median 6).
- **AS**: the appendix defines AS as the median representation change per angular increment along
  the scan. `‖z_{i+1} − z_i‖` is in the units of the embedding, so AS values are comparable
  between conformers of one model but not across models with different embedding scales.
  The wrap-around step (largest torsion back to the smallest) is excluded because conformers do not
  necessarily cover the full circle after redundancy removal.
- **Isotonic R²**: the appendix fits `D` as a monotone function of `Δ`. The reverse fit rewards a
  constant representation with R² = 1.
- **Kendall**: no conformer-count threshold.
- The `paper` column is kept as the default so that the published numbers can be regenerated.

## Provenance of Table 1

The scripts that produced Table 1 (`eval_geo_single.py`, `eval_geo_single_sup.py`) are not
available. The `paper` definitions above were recovered by recomputing GemNet metrics from the
published GemNet rotation embeddings and matching them, molecule by molecule, against the backed-up
per-molecule outputs of the two original runs (the per-molecule JSON files are published in
`EscheWang/3dcs-embeddings` under `results/rotation/`):

| Table 1 rows | Original run | Molecules | Backup file |
|---|---|---|---|
| Spearman, Kendall, CKA, Isotonic R², Torsion-SP | `eval_geo_single.py --sample-ratio 0.1 --sample-seed 2027 --pairs-cap 100` | 146,389 (10 % sample) | `metrics_all_0.1_1.json.gz` (= `all_metric.csv`) |
| LIE@k, AS | `eval_geo_single_sup.py --sample-ratio 1 --sample-seed 2027 --pairs-cap 100` | 1,464,493 | `metrics_sup_100.json.gz` |

The 10 % sample cannot be regenerated from the seed because the sampling code is not available.
Its molecule keys, read from the backup file, are distributed as
[`reproduce/table1_geometry/sampled_molecules_seed2027.txt`](../../reproduce/table1_geometry/sampled_molecules_seed2027.txt)
(146,389 keys in dataset order; use it with `--molecule-list`). `--sample-ratio/--sample-seed`
implement a documented per-shard sampler (`numpy.random.default_rng(seed + shard)`), which does not
reproduce that list.

### Per-molecule agreement (GemNet, cosine)

Recomputed with `--metric-version paper` from the HF MolBlocks and the published embeddings.

| Metric | Molecules compared | Max. abs. difference | Difference of the mean |
|---|---|---|---|
| Spearman | 16,717 (10 % sample, shards 0–1) | 2.1e-2 (54 molecules > 1e-6) | 4.7e-7 |
| Kendall | 3,003 | 1.4e-3 | −3.9e-7 |
| CKA | 18,262 | 4.9e-5 | −2.3e-8 |
| Isotonic R² | 18,262 | 2.6e-3 | −2.9e-7 |
| Torsion-SP | 16,717 | 1.0e-2 (1 molecule > 1e-6) | 6.2e-7 |
| LIE@k | 91,499 (all of shard 0) | 9.9e-2 (99th percentile 1.9e-5) | 3.6e-7 |
| AS | 91,499 | 2.0e-3 (31 molecules > 1e-6) | −1.2e-8 |

The remaining per-molecule differences come from the reference distances: the HF MolBlocks store
coordinates with 4 decimals, so RMSD values differ from the original coordinates by about 1e-5 Å,
which can reorder near-tied pairs. LIE is sensitive for GemNet because some GemNet cosine distances
are of the order of the 1e-12 stabiliser.

Two further properties of the published runs were found this way:

- **Kendall** is finite for every sampled molecule with ≥ 11 conformers and NaN for every molecule
  with ≤ 10 conformers (24,198 of 146,389 molecules have a value). The exact rule of the original
  script is not recoverable; any pair-count threshold between 46 and 55 selects the same molecules.
- **Embedding offsets in the full run.** In `metrics_sup_100.json.gz`, the molecules of shard 1 after
  `1-R5B2G8_6-R7B1G14_0_22` (local offset 69,515, 3 conformers) and of shard 2 after
  `6-R8B2G15_35-R2B1G6_0_36` (local offset 560,074, 7 conformers) were evaluated with embeddings
  shifted back by 3 and 7 rows: the failing molecule was skipped without advancing the embedding
  cursor. For GemNet this reproduces 530/530 and 67/67 sampled molecules after the failure in
  shards 1 and 2 (LIE and AS), while the correctly aligned embeddings do not. 91,093 of the
  1,464,493 molecules (6.2 %) are affected. The 10 % run is not affected (neither molecule is in the
  sample), and E3FP is not affected (its fingerprints are stored per molecule).
  `--replicate-offset-drift` (by-shard layout) recomputes all metrics with this shift and stores
  them as `<metric>__offset_drift` columns; the regular columns always use aligned embeddings.

For the four learned models, the effect on the published LIE@k and AS means can be estimated from
the backup outputs by averaging over the unaffected molecules only (an estimate; exact aligned
values require the embeddings, which are available for GemNet only):

| Model | LIE@k published | LIE@k, unaffected molecules | AS published | AS, unaffected molecules |
|---|---|---|---|---|
| E3FP | 0.3238 | 0.3239 | 2.7576 | 2.7850 |
| GemNet | 0.3901 | 0.3662 | 0.001825 | 0.001166 |
| MolAE | 0.3489 | 0.3321 | 0.004755 | 0.004022 |
| MolSpectra | 0.2379 | 0.2254 | 0.021443 | 0.020741 |
| UniMol | 0.3059 | 0.2863 | 0.005987 | 0.005291 |

(For E3FP the small differences reflect the subset only.)

### Not reproduced

- The Euclidean-space LIE@k values of the full run (not used in Table 1). In that run AS is the same
  in both spaces (computed from the cosine distance).
- Rotation embeddings for E3FP, UniMol, MolAE and MolSpectra are not available, so their Table 1
  values can only be compared with the backed-up per-molecule outputs.

## Runtime

GemNet, 24 worker processes on the A100 box, cosine and Euclidean spaces: shard 0 (91,499
molecules) took 94 s with `--metric-version paper`; shards 0–1 (182,981 molecules) took 188 s
(`paper`) and 191 s (`v2`) while another 24-process job shared the machine. RMSD takes about 3 ms per
molecule. A full run over the 16 shards (1,464,495 molecules) is therefore expected to take
roughly 25 minutes per metric version with 24 workers; `--replicate-offset-drift` adds a second pass
over the affected molecules. `--extra-metrics` is about 20× slower (Mantel permutations).
