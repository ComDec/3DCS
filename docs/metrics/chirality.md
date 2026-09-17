# Chirality metrics (Table 2)

This page documents how `python -m three_dbench evaluate chirality` computes the zero-shot chirality
metrics, which settings produced the published Table 2, and what the `v2` definitions change.

- Code: `src/three_dbench/chirality/evaluation.py` (metrics, per-molecule loop) and
  `src/three_dbench/benchmarks/chirality.py` (dataset loading, validation, output files).
- Reproduction script: `reproduce/table2_chirality/run.sh`, expected values in
  `reproduce/table2_chirality/expected.csv`.

## 1. Options at a glance

| Option | Values | Default | Default reproduces the published Table 2? |
|---|---|---|---|
| `--distance` | `euclidean`, `cosine` | `euclidean` | yes |
| `--metric-version` | `paper`, `v2` | `paper` | yes |
| `--unsup-kmax` | `n-1` or an integer >= 2 | `n-1` | yes |
| `--embedding-key` | NPZ array name or pickle dict entry | none | use the key listed in the table in section 6 |
| `--n-jobs` | integer (`-1` = all CPUs) | `1` | results do not depend on it |

The Python API takes the same options: `evaluate_chirality_embeddings(..., distance=, metric_version=, unsup_kmax=, n_jobs=)`
and `evaluate_en_separation_from_counts(...)`. `unsup_kmax=None` means `n - 1`.

Each run writes `<model>_per_molecule.json`, `summary.csv` (the same 18 columns as the original
Sept-2025 output) and `config.json`. `config.json` records the options, the library versions and the
number of molecules behind each mean (`coverage`).

## 2. Data and evaluation unit

- Dataset: `load_dataset("EscheWang/3dcs", name="chirality", split="train")`. It has 14,903 rows,
  one per stereoisomer (`key = <mol_id>::en<k>_<CIP labels>`), with 52,391 conformers and 3,903 molecules.
- Embeddings: one row per conformer, in dataset row order. Row `i` of the dataset owns embedding rows
  `offset[i] : offset[i] + n_conformers[i]`. The evaluator checks three things before any metric is
  computed, and raises an error naming the problem if one fails:
  - the number of embedding rows equals `sum(n_conformers)`;
  - `offset` is the running sum of `n_conformers`;
  - keys are unique and embeddings are finite.
- Unit of evaluation: a molecule (`mol_id`), which pools the conformers of all its stereoisomer rows.
  The label is `en_id`, the index of the stereoisomer within the molecule. There are up to 5 per
  molecule, and they include diastereomers, not only enantiomer pairs.
- Aggregation: every summary value is the unweighted mean over molecules of the finite per-molecule
  values. NaN values are skipped.

Molecule populations in the published protocol (`paper`, identical for all 7 models):

| Population | Molecules |
|---|---|
| all molecules (`n_molecules` in `summary.csv`) | 3,903 |
| skipped, fewer than 2 conformers (`skip_small`) | 19 |
| skipped, one stereoisomer only (`skip_single_en`) | 42 |
| supervised metrics computed (`supervised+unsup`) | 3,842 |
| of which every stereoisomer has exactly one conformer | 1,019 |
| ES-AUC and SCI finite (at least one same-class pair) | 2,823 |
| NN@1-Acc finite | 3,842 (the 1,019 molecules above score 0 by construction) |
| Hopkins finite (continuous embeddings, n >= 10) | 1,764 (including 11 single-stereoisomer molecules) |
| SCI_unsup finite (n >= 3) | 3,576 |

## 3. Representation distance (`--distance`)

| | Definition | Code |
|---|---|---|
| `euclidean` (default) | `Delta_ij = ||z_i - z_j||_2` via `sklearn.metrics.pairwise_distances` (float32 input stays float32) | `euclidean_distances` |
| `cosine` | `Delta_ij = 1 - <z_i, z_j> / ((||z_i|| + 1e-12)(||z_j|| + 1e-12))` in float64, with the diagonal set to exactly 0 | `cosine_distances` |
| RDKit fingerprints | Tanimoto distance `1 - |F_i & F_j| / |F_i | F_j|` (RDKit `BulkTanimotoSimilarity`), whatever `--distance` is | `tanimoto_distance_matrix` |

- **The published Table 2 was computed with Euclidean distance.** The paper text (§4.1 and App. C.2)
  says cosine distance is the default for learned representations. The released evaluator
  (`chirality/evaluation.py`, `distance_matrix_for_subset`) used Euclidean distance. Recomputing
  Table 2 with Euclidean distance matches all published cells to within 0.001, except for the sign of
  GemNet SCI (section 7). Cosine distance does not match: for example, MolAE SCI is 0.189 against the
  printed 0.115, and GemNet ES-AUC is 0.602 against 0.577.
- App. C.2 also says distance matrices are normalised to [0, 1]. Normalisation is not applied, and it
  would not change any chirality metric. Each metric is computed within one molecule, and each is
  invariant to rescaling that molecule's `Delta` by a positive constant: ES-AUC is rank based, NN@1
  uses an argmin, and silhouette, DBI and clarity are ratios.
- The diagonal of the cosine matrix is set to 0. Without that, the 1e-12 guard leaves diagonal values
  around 1e-12, and `sklearn.metrics.silhouette_score(metric="precomputed")` rejects the matrix. The
  code then silently falls back to a hand-written silhouette that skips points in singleton classes
  instead of scoring them 0. Without the fix, cosine SCI changes for 6 (MolSpectra, large norms) to 959
  (FMG) molecules per model; for example, MolAE gives 0.1938 instead of 0.1895. Cosine SCI is not
  published.
- In `paper` mode, `--distance` changes ES-AUC, NN@1-Acc, SCI, DBI and clarity. Hopkins and SCI_unsup
  are still computed on the raw vectors with Euclidean distance, as in the published code. In `v2`
  they follow the selected distance (section 4).

## 4. Metric definitions: `paper` and `v2`

`paper` is the code path that produced the published numbers. `v2` changes a definition only where
the published code departs from App. C.5 of the paper, or where a per-molecule value is fixed by
construction rather than by the representation. Every such case is listed below with the measured
evidence. The remaining metrics are identical in both versions.

Notation: a molecule has `n` conformers with labels `y`, representation distances `Delta`, and
vectors `X` (rows L2-normalised when `v2` is combined with `--distance cosine`).

### ES-AUC (identical in both versions)

`ESA-AUC = ROC-AUC({(Delta_ij, 1[y_i != y_j])}_{i<j})` (`sklearn.metrics.roc_auc_score`, ties count 1/2).
It is NaN if all pairs share a label or all pairs differ. Code: `auc_diff_pairs_large_when_different`.

### NN@1-Acc

- **paper** (`nn1_leave_one_out_from_D`): `mean_i 1[y_{argmin_{j != i} Delta_ij} = y_i]` over all `n` points.
  Ties are broken by `np.argmin`, which picks the lowest row index.
- **v2** (`nn1_leave_one_out_v2`): the mean is taken only over points whose class has at least one other
  member. Each point scores the fraction of its exactly tied nearest neighbours that share its label,
  which is the expected accuracy under uniformly random tie-breaking. A molecule with no eligible point
  is NaN.
- **Rationale:**
  1. *Ties.* Tanimoto distances on 1024-bit E3FP fingerprints are often exactly tied. On the published
     E3FP file, 1,272 conformers in 923 of the 3,842 molecules have tied nearest neighbours. The
     `paper` value therefore depends on the dataset row order: 339 molecules change under tie-aware
     scoring. No ties were found for the six continuous embeddings.
  2. *Structural zeros.* A conformer whose stereoisomer has no other conformer in the molecule can
     never be classified correctly by leave-one-out 1-NN. In 1,019 molecules every stereoisomer has one
     conformer, so `paper` NN@1-Acc is 0 for them regardless of the representation. The same molecules
     are already NaN for ES-AUC and SCI. For example, MolAE `paper` NN@1-Acc is 0.4975 over 3,842
     molecules, and 0.677 over the other 2,823.

### SCI (supervised silhouette; identical in both versions)

`s_i = (b_i - a_i) / max(a_i, b_i)`, with `a_i` the mean distance to the other members of the point's
class and `b_i` the smallest mean distance to another class. SCI is the mean of `s_i`, computed with
`sklearn.metrics.silhouette_score(Delta, y, metric="precomputed")`.

- Points in singleton classes score 0 (the sklearn convention).
- The value is NaN when `n < 3` or when every class is a singleton, because sklearn requires
  `2 <= n_labels <= n - 1`.

Code: `silhouette_with_labels_from_D`.

### DBI (Davies-Bouldin index; not in Table 2)

- **paper** (`davies_bouldin_from_D`): a medoid approximation on `Delta`.
  - The medoid `m_c` is the class member with the smallest sum of distances to its class.
  - `S_c` is the mean distance from the class members to `m_c`, and `M_cd = Delta(m_c, m_d)`.
  - `DBI = mean_c max_{d != c} (S_c + S_d) / M_cd`, with no epsilon.
- **v2**: the centroid definition of App. C.5 (`davies_bouldin_centroid`).
  - `mu_c` is the class mean of `X`, `S_c = mean ||x - mu_c||_2`, `M_cd = ||mu_c - mu_d||_2`, and
    `R_cd = (S_c + S_d) / (M_cd + 1e-12)`.
  - This matches `sklearn.metrics.davies_bouldin_score` on non-degenerate input.
  - Fingerprints have no centroid under Tanimoto, so they keep the medoid version, with the 1e-12 term.
  - A molecule in which no class has two or more members is NaN.
- **Rationale:** App. C.5 defines DBI with Euclidean centroids. In `paper` mode, the 753 molecules
  with `n >= 3` and only singleton classes get `S = 0` and therefore `DBI = 0`, the best possible
  score, whatever the representation.

### Hopkins statistic

- **paper** (`hopkins_statistic`): computed on the raw vectors with Euclidean distance.
  - `m = max(10, int(0.1 n))` points are drawn uniformly in the bounding box of `X`, and `m` data
    points are sampled without replacement. Both use `numpy.random.default_rng(0)` for every molecule.
  - `H = sum u_i / (sum u_i + sum w_i)`, where `u` are nearest-data distances of the uniform points and
    `w` are nearest-other-point distances of the sampled points.
  - The value is NaN for `n < 10` and for fingerprints.
  - It is also computed for molecules with a single stereoisomer (11 molecules with `n >= 10`).
- **v2**: the same estimator and the same `n >= 10` threshold, with two changes.
  1. It is computed on the L2-normalised vectors when `--distance cosine` is used.
  2. It is reported only for the molecules that enter the supervised metrics, so all Table 2 columns
     describe the same molecule set. The single-stereoisomer molecules become NaN.
- **Why `n < 10` stays NaN in v2.** On the published embeddings this estimator depends strongly on
  `n`. With the sample size capped at `n` (`m = min(n, max(10, int(0.1 n)))`), the mean Hopkins value
  for molecules with 2 conformers is 0.35 for every model. It rises steadily with `n`:

  | Conformers per molecule | 2 | 3 | 5 | 7 | 9 | 10 | >= 12 |
  |---|---|---|---|---|---|---|---|
  | Molecules | 276 | 110 | 719 | 165 | 151 | 138 | 1,512 |
  | GemNet | 0.349 | 0.430 | 0.496 | 0.520 | 0.545 | 0.553 | 0.599 |
  | MolAE | 0.359 | 0.429 | 0.485 | 0.520 | 0.539 | 0.552 | 0.611 |
  | FMG | 0.349 | 0.460 | 0.552 | 0.613 | 0.652 | 0.674 | 0.765 |

  Adding the 2,139 molecules with `n < 10` would mostly add this size effect to the mean. `v2`
  therefore keeps the threshold, and `config.json` reports how many molecules contribute
  (`coverage.n_finite_hopkins`). The Hopkins mean is best read as a relative score between
  representations on the same molecule set. It is not an absolute test for clusters.

### SCI_unsup (best-k silhouette)

- **paper**: for `k = 2 .. kmax`, labels come from `sklearn.cluster.KMeans(n_clusters=k, n_init=10,
  random_state=0)` on the raw `X`. The silhouette is computed on `X` with Euclidean distance, and the
  largest silhouette over `k` is reported.
  - Fingerprints use a PAM k-medoids on the Tanimoto matrix (`pam_kmedoids`, `default_rng(0)`), with
    the silhouette on that matrix.
  - `kmax = n - 1` (`--unsup-kmax n-1`, the default).
- **v2**: the silhouette is computed on the selected `Delta`, as App. C.5 writes
  `silhouette(Delta, k-means_k)`.
  - With `--distance cosine`, k-means runs on L2-normalised vectors.
  - With `--distance euclidean` the value equals `paper` up to floating-point rounding.
  - With `--do-unsup-when-single-en`, the single-stereoisomer branch uses the same deterministic PAM.
    In `paper` mode that branch uses `sklearn_extra.KMedoids` when it is importable, and falls back to
    PAM otherwise.
- **`kmax` and the published run.** The first public release (0.1.0) hard-coded
  `kmax = min(10, n - 1)`, with the comment "Limit kmax to 10 for speed". The Sept-2025 run that
  produced Table 2 did not have this cap:
  - its per-molecule output contains `k_unsup` values up to 82 for E3FP and 16 for MolAE;
  - with `kmax = n - 1` the current code reproduces those per-molecule values, except for 3 molecules
    whose KMeans optimum depends on the BLAS kernel (section 7);
  - with the cap, SCI_unsup differs by up to 3.9e-4 (E3FP 0.033437 against 0.033825).

  Both settings are within 0.001 of every printed SCI_unsup value. `--unsup-kmax 10` restores the
  0.1.0 behaviour. (In 0.1.0 the opt-in single-stereoisomer branch used `k <= min(50, n - 1)`; it
  now follows `--unsup-kmax` as well.)

### Clarity (not in the paper)

`boundary_clarity_from_D` and `clarity_unsup` are unchanged. They are written to `summary.csv` for
continuity with the original output files.

## 5. Expected values

All values below come from the published embeddings (section 6) with the current code, on
numpy 2.4.6, scikit-learn 1.9.1, rdkit 2026.03.6 and an AMD EPYC 7513. `expected.csv` holds 6-decimal
values for all 7 models × 4 variants, with a tolerance of 0.001.

The `euclidean` column is the published protocol. Every cell is within 0.001 of the printed value,
except for GemNet SCI (sign). E3FP is a fingerprint, so its cosine rows equal its Euclidean rows and its
Hopkins value is NaN. In `v2`, NN@1-Acc is averaged over 2,823 molecules instead of 3,842 and Hopkins
over 1,753 instead of 1,764. The other columns use the same molecules in both versions.

| Metric | Model | Paper | `euclidean` (published protocol) | `cosine` | `v2_euclidean` | `v2_cosine` |
|---|---|---|---|---|---|---|
| ES-AUC | E3FP | 0.486 | 0.485935 | 0.485935 | 0.485935 | 0.485935 |
| ES-AUC | GemNet | 0.577 | 0.577410 | 0.601837 | 0.577410 | 0.601837 |
| ES-AUC | MolAE | 0.782 | 0.782544 | 0.781942 | 0.782544 | 0.781942 |
| ES-AUC | MolSpectra | 0.545 | 0.544744 | 0.543114 | 0.544744 | 0.543114 |
| ES-AUC | UniMol | 0.622 | 0.622998 | 0.622620 | 0.622998 | 0.622620 |
| ES-AUC | FMG | 0.706 | 0.705652 | 0.712555 | 0.705652 | 0.712555 |
| ES-AUC | MACE | 0.485 | 0.485611 | 0.484971 | 0.485611 | 0.484971 |
| NN@1-Acc | E3FP | 0.178 | 0.177939 | 0.177939 | 0.256400 | 0.256400 |
| NN@1-Acc | GemNet | 0.292 | 0.291641 | 0.312251 | 0.415357 | 0.444804 |
| NN@1-Acc | MolAE | 0.497 | 0.497546 | 0.497357 | 0.709640 | 0.709405 |
| NN@1-Acc | MolSpectra | 0.235 | 0.235250 | 0.235108 | 0.337786 | 0.337668 |
| NN@1-Acc | UniMol | 0.339 | 0.339582 | 0.338899 | 0.481977 | 0.480923 |
| NN@1-Acc | FMG | 0.412 | 0.411765 | 0.422004 | 0.587976 | 0.601655 |
| NN@1-Acc | MACE | 0.199 | 0.199113 | 0.200708 | 0.283468 | 0.285157 |
| Hopkins | E3FP | – | NaN | NaN | NaN | NaN |
| Hopkins | GemNet | 0.593 | 0.592812 | 0.592812 | 0.592823 | 0.577979 |
| Hopkins | MolAE | 0.602 | 0.602659 | 0.602659 | 0.602748 | 0.602690 |
| Hopkins | MolSpectra | 0.533 | 0.532642 | 0.532642 | 0.532736 | 0.533677 |
| Hopkins | UniMol | 0.559 | 0.559572 | 0.559572 | 0.559674 | 0.559664 |
| Hopkins | FMG | 0.752 | 0.751741 | 0.751741 | 0.752106 | 0.751049 |
| Hopkins | MACE | 0.637 | 0.637379 | 0.637379 | 0.637289 | 0.637111 |
| SCI | E3FP | -0.012 | -0.012543 | -0.012543 | -0.012543 | -0.012543 |
| SCI | GemNet | 0.015 (sign typo) | -0.015151 | 0.004663 | -0.015151 | 0.004663 |
| SCI | MolAE | 0.115 | 0.115274 | 0.189480 | 0.115274 | 0.189480 |
| SCI | MolSpectra | -0.020 | -0.020326 | -0.037315 | -0.020326 | -0.037315 |
| SCI | UniMol | 0.012 | 0.011623 | 0.021010 | 0.011623 | 0.021010 |
| SCI | FMG | 0.117 | 0.117174 | 0.158030 | 0.117174 | 0.158030 |
| SCI | MACE | -0.094 | -0.093877 | -0.151823 | -0.093877 | -0.151823 |
| SCI_unsup | E3FP | 0.033 | 0.033825 | 0.033825 | 0.033825 | 0.033825 |
| SCI_unsup | GemNet | 0.272 | 0.271961 | 0.271961 | 0.271961 | 0.380451 |
| SCI_unsup | MolAE | 0.247 | 0.247099 | 0.247099 | 0.247099 | 0.408498 |
| SCI_unsup | MolSpectra | 0.127 | 0.127115 | 0.127115 | 0.127115 | 0.232817 |
| SCI_unsup | UniMol | 0.152 | 0.152108 | 0.152108 | 0.152108 | 0.270063 |
| SCI_unsup | FMG | 0.509 | 0.509364 | 0.509364 | 0.509364 | 0.682689 |
| SCI_unsup | MACE | 0.369 | 0.369336 | 0.369336 | 0.369336 | 0.566272 |

## 6. Published embeddings and keys

The files are the original bytes from the authors' backup, published at `EscheWang/3dcs-embeddings`
(see `docs/EMBEDDINGS.md`). SHA-256 values are listed in `reproduce/table2_chirality/models.csv`.

| Model | File (repo path) | `--embedding-key` | Content |
|---|---|---|---|
| E3FP | `chirality/e3fp/sampled_chi.pkl` | `e3fp` | pickled dict, `e3fp`: list of 52,391 RDKit `ExplicitBitVect` (1024 bits); `morgan`: 2048-bit Morgan (not used) |
| GemNet | `chirality/gemnet/sampled_feature.npz` | `gemnet` | float32 (52391, 128) |
| MolAE | `chirality/molae/1.npz` | `arr_0` | float32 (52391, 512) |
| MolSpectra | `chirality/molspectra/sampled_mol_feature.npz` | `arr_0` | float32 (52391, 256) |
| UniMol | `chirality/unimol/1.npz` | `arr_0` | float32 (52391, 512) |
| FMG | `chirality/fmg/chirality_bench_conformers_noised_only_aslist_embed.npz` | `embeddings` | float32 (52391, 128); the same file also stores `smiles` |
| MACE | `chirality/mace/chirality.npz` | `arr_0` | float32 (52391, 256) |

- **E3FP.** Reading the pickle requires RDKit. The fingerprints were regenerated bit-exactly (3,000 of
  3,000 random conformers) from the original RDKit molecules with e3fp 1.2.7:

  ```python
  fprints_from_mol(mol, fprint_params=dict(bits=1024, level=5, radius_multiplier=1.5, stereo=True,
                   include_disconnected=True, rdkit_invariants=True, first=1, counts=False))
  ```

  Starting from the HF MolBlocks instead gives the same bits for 2,846 of 3,000 conformers.
- **Other models.** The extraction settings (checkpoint, layer, pooling) of the other published
  embeddings have not been recovered. The files are provided so that the published numbers can be
  recomputed. They do not document how the embeddings were produced.

## 7. Known differences between the paper and the recomputed values

- **GemNet SCI.** The paper prints `0.015`. The recomputed value, and the Sept-2025 output, is
  `-0.015151`, so the minus sign is missing in the table.
- **Rounding.** The printed values mix rounding and truncation to three decimals. The largest gap is
  UniMol ES-AUC: 0.622998 is printed as 0.622. Every other cell is within 0.001 of the recomputed value.
- **Hopkins for E3FP** is printed as "–". Hopkins is not defined for fingerprints, and the evaluator
  returns NaN.
- **Sept-2025 outputs.** For E3FP, GemNet, MolAE, MolSpectra and UniMol, the backup holds the original
  per-molecule outputs (`en_sep_results/*.json`). With the defaults, ES-AUC, NN@1-Acc, SCI, DBI and
  Hopkins are bit-identical for every molecule of all five models.
  - SCI_unsup is bit-identical for E3FP, MolAE and MolSpectra.
  - It differs for 1 GemNet molecule and 2 UniMol molecules, which changes the means by +2.4e-6 and
    +2.0e-6. On these near-tied silhouette landscapes the KMeans local optimum depends on the OpenBLAS
    kernel.
  - The unpublished clarity columns differ by less than 2e-5, because of a numpy version difference
    (they match under numpy 2.2 and 2.3).
- **FMG and MACE** have no original metric output in the backup. Their expected values are recomputed
  from the published embeddings.

## 8. Performance

The run time is dominated by the best-k KMeans scan up to `k = n - 1` (at most 99). On 22 processes
(`--n-jobs 22`, AMD EPYC 7513, shared machine), one model and variant takes about 40–65 s for E3FP and
1–3 min for the continuous embeddings.

Molecules are evaluated independently with fixed seeds, in batches run by joblib (`loky`) with one
BLAS/OpenMP thread per worker. The per-molecule outputs for `--n-jobs 8` were bit-identical to
`--n-jobs 1` on the first 300 molecules for UniMol and E3FP, and the full 22-process runs are
bit-identical to the phase-1 sharded runs of the released evaluator.
