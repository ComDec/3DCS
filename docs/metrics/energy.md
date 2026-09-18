# Energy (trajectory) benchmark metrics

This page defines the metrics behind the zero-shot energy tables of the paper (Table 3 in the main
text, Tables 6 and 7 in Appendix B.2). It covers the data, the sampling protocol, and each metric in
its two implementations:

- `--metric-version paper` (the default) is the implementation that produced the published numbers.
- `--metric-version v2` follows the definitions written in Appendix C.3/C.6. Where the appendix
  leaves a choice open, v2 makes that choice explicit.

Both versions share the same windows and the same aggregation. `reproduce/energy_tables_3_6_7/`
contains the end-to-end script and the expected values for both.

Code map:

| Component | Location |
|---|---|
| CLI | `python -m three_dbench evaluate traj` (`src/three_dbench/__main__.py`) |
| Benchmark driver | `src/three_dbench/benchmarks/trajectory.py::evaluate_trajectory_embeddings` |
| Windows, distances, version dispatch | `src/three_dbench/traj/protocol.py` |
| `paper` metrics | `src/three_dbench/traj/evaluation.py` (`compute_energy_metrics_from_condensed`, `thresholded_smoothness`, `ks_wasserstein_against_energy_diff`) |
| `v2` metrics | `src/three_dbench/traj/metrics_v2.py::compute_energy_metrics_v2` |
| Shared estimators | `src/three_dbench/common/metrics.py` (`spearman_correlation`, `kendall_correlation`, `isotonic_r2`, `cka_rbf`) |
| Embedding loading | `src/three_dbench/traj/io.py::load_traj_embeddings` |
| Energy precision check | `src/three_dbench/datasets/traj.py::detect_quantized_energies` |
| Legacy runner used for the paper | `src/three_dbench/traj/evaluation.py::run_trajectory_benchmark` (numerically unchanged; the CLI reproduces it bit for bit, see `tests/test_traj_legacy_equivalence.py`) |

## 1. Data

**Energies.** The benchmark uses the 10 molecules of rMD17 with 100,000 frames each; azobenzene
has 99,988. The values are absolute total energies in kcal/mol, about −4×10⁵, stored as **float64**.
Download them with `load_dataset("EscheWang/3dcs", name="traj_energies", split="train")` or from the
standalone repo `EscheWang/3dcs-traj-energies`. Either source can be converted from the original
rMD17 npz files with `python -m three_dbench convert traj`.

**Earlier revision.** An earlier revision of `traj_energies` stored a float32 cast of these energies.
The schema was declared float64, but each molecule held only 1,096–3,334 distinct values, with a
resolution of 1/64–1/256 kcal/mol. Do not use that revision (see the dataset card). With those values:
- Pooled CKA means drop by up to about 40%, e.g. GemNet 0.0163 → 0.0113.
- Smoothness and KS shift by about 0.001.
- Table 7 no longer reproduces.

The evaluator therefore checks the precision of the energies of every evaluated molecule (option
`--energy-precision-check`, default `error`). A molecule is flagged as quantized when either:
- fewer than 50% of its frames have distinct energies, or
- every value is float32-representable **and** the float32 step at the largest magnitude is ≥ 10⁻⁴
  of the inter-quartile range.

Relative energies stored in float32 are not flagged.

**Frame order.** rMD17 frames are **not in simulation-time order**, based on checks on the rMD17 npz
files:
- `old_indices` is not monotonic.
- The lag-1 correlation of the energies is 0.002–0.004 (aspirin, ethanol, uracil).
- The median |ΔE| between consecutive frames equals the median over random pairs, e.g. aspirin
  5.847 vs 5.834 kcal/mol.

Metrics defined along consecutive frames (TS and Smoothness) are therefore computed on effectively
random pairs of frames. See §4.

**Embeddings.** `EscheWang/3dcs-embeddings` holds the backed-up files as
`traj/<model>/rmd17_<mol>.{npz,pkl}`, one file per molecule, with rows in frame order:

| Model | Format | Key / size |
|---|---|---|
| E3FP | pickle | list of RDKit `ExplicitBitVect` (1024 bits) |
| GemNet | npz | `gemnet` (128) |
| MolAE | npz | `arr_0` (512) |
| MolSpectra | npz | `arr_0` (256) |
| UniMol | npz | `arr_0` (512) |
| FMG | npz | `embeddings` (128), plus `smiles` |
| MACE | npz | `arr_0` (256) |

`load_traj_embeddings` accepts:
- a directory with one `rmd17_<mol>.{npz,npy,pkl}` per molecule, or
- a single `.npz`/`.pkl` keyed by molecule.

Each molecule is loaded only when it is evaluated.

## 2. Protocol (shared by both metric versions)

**Windows** (`--window-scheme`, `traj/protocol.py::draw_window_starts`):

- `legacy` (default, published): every molecule draws its starts from a fresh generator, so all
  molecules share the same 100 starts:

  ```python
  starts = numpy.random.default_rng(2025).integers(0, 100_000 - 2000, size=100)
  ```

  - The first three starts are 43852, 97456 and 97273, and the largest is 97636, so no window
    runs past the 99,988 azobenzene frames.
  - A molecule's windows do not depend on which other molecules are evaluated (`--molecules`).
  - Molecules are processed in the legacy order `aspirin, azobenzene, benzene, ethanol, malonaldehyde,
    toluene, naphthalene, paracetamol, salicylic, uracil`. This order fixes the floating-point
    summation order of the summary.
  - This is exactly `traj/evaluation.py::run_once_for_molecule`.
- `shared` (the 0.1.0 CLI): one generator is shared across molecules in dataset row order, and each
  molecule draws `integers(0, n_frames − 2000, size=100)`. Only aspirin gets the published windows.

Defaults are `--n-samples 100 --window 2000 --random-seed 2025`, i.e. 1,000 windows and
n = 2,000 frames per window. Every metric uses all n(n−1)/2 = 1,999,000 frame pairs of a window.

**Representation distance Δ** (`traj/protocol.py::window_distances`):
- *Vectors* (cosine, default): Δᵢⱼ = (1 − cos(zᵢ, zⱼ))/2 ∈ [0, 1]. It is computed in float32 from
  unit-normalised vectors and stored as **float16** in `paper` and **float32** in `v2`.
  `--metric-embed euclidean` gives ‖ẑᵢ − ẑⱼ‖²/4 on unit vectors.
- *Fingerprints* (Tanimoto, automatic for bit-vector pickles): Δᵢⱼ = 1 − |Fᵢ∩Fⱼ| / |Fᵢ∪Fⱼ|, with
  RDKit `BulkTanimotoSimilarity`. It uses the same storage dtype rule. Dense 0/1 arrays with
  `--metric-embed tanimoto` give identical values, with one exception: for two all-zero fingerprints
  the dense path always returns similarity 0, while RDKit returns 1.0 up to version 2025.09 and 0.0
  from 2026.03 on. The published E3FP fingerprints contain no all-zero vector (checked for all
  999,988 trajectory frames and all 52,391 chirality conformers), so the tables do not depend on
  this.

**Aggregation.** For every metric, the per-window values of all molecules are pooled, non-finite
values are dropped, and the result is reported as mean ± 1.96·sd(ddof = 1)/√n, with n = 1,000.

**Determinism.** BLAS pools are limited to one thread per worker, and `--n-jobs` workers evaluate
windows in parallel with order-preserving results. On one machine, repeated runs are bitwise
identical, and so are runs with different `--n-jobs`. Across machines, float32 BLAS kernels can
change per-window values slightly: up to ~4×10⁻⁵, and up to ~3×10⁻⁴ for TS, were observed between
the original run and a rerun on different hardware.

## 3. Notation

For one window: Eᵢ is the energy of frame i (float64), dEᵢⱼ = |Eᵢ − Eⱼ|, Δᵢⱼ is the representation
distance, and all sums and quantiles run over the pairs i < j unless stated otherwise. Q_p(x)
denotes the p-quantile. ε = 10⁻¹².

## 4. Metric definitions

### Spearman ρ and Kendall τ (keys `spearman`, `kendall`)

| | Definition |
|---|---|
| paper | `scipy.stats.spearmanr` / `kendalltau` (τ-b) between vec(dE) and vec(Δ). dE is cast to float32 first (`vector_to_absdiff_condensed`), and Δ is float16. |
| v2 | Same estimators, with dE in float64 and Δ in float32. |

### CKA with RBF kernels (key `cka_rbf`)

K⁽ᴰ⁾ᵢⱼ = exp(−dE²ᵢⱼ / (2σ_D² + ε)) and K⁽Δ⁾ᵢⱼ = exp(−Δ²ᵢⱼ / (2σ_Δ² + ε)), with unit diagonals.
With H = I − 11ᵀ/n:

CKA = ⟨HK⁽ᴰ⁾H, HK⁽Δ⁾H⟩_F / (‖HK⁽ᴰ⁾H‖_F ‖HK⁽Δ⁾H‖_F).

| | Bandwidth |
|---|---|
| paper | **One shared** σ = √median{d² : d ∈ vec(dE) ∪ vec(Δ), d > 0} over the concatenation of both distance sets (`cka_rbf(..., share_sigma=True)`). |
| v2 | **Separate** median heuristics σ_D = √median{dE² > 0} and σ_Δ = √median{Δ² > 0} (Appendix C.3). |

*Rationale for v2:* dE is in kcal/mol (median ≈ 6) and Δ ∈ [0, 1] (median ≈ 10⁻⁴–10⁻³ for the
continuous models). The shared median is therefore ≈ 0.003, far below almost every energy difference, and the
energy kernel degenerates to an indicator of near-tied energies. This is also why the published CKA
values depend strongly on the precision of the energies.

### Isotonic R² (key `iso_R2`)

| | Definition |
|---|---|
| paper | Fits a non-decreasing Δ̂ = f(dE) and returns 1 − Σ(Δ − Δ̂)² / (Σ(Δ − Δ̄)² + ε). A constant representation scores 1.0. |
| v2 | Fits a non-decreasing dÊ = f(Δ) and returns 1 − Σ(dE − dÊ)² / (Σ(dE − dĒ)² + ε), as in Appendix C.3 ("mapping representation distances onto the reference values"). The result is NaN when Δ is constant. |

### Energy-jump sensitivity EJS(λ) (keys `EJS_lam{0p1,0p5,1,2,3}`, `EJS` = λ = 2; counts `EJS_num_jumps_lam*`)

θ(λ) = λ·σ̂_d, J_λ = {(i, j) : dEᵢⱼ > θ(λ)}, τ = Q_0.75(vec Δ) over all pairs, and

EJS(λ) = |{(i, j) ∈ J_λ : Δᵢⱼ > τ}| / |J_λ|, for λ ∈ {0.1, 0.5, 1, 2, 3}.

| | σ̂_d |
|---|---|
| paper | Robust half-normal scale σ̂_d = median(dE) / (√2·erf⁻¹(0.5) + ε) ≈ median(dE)/0.6745. |
| v2 | σ̂_d = √(mean(dE²) + ε) (Appendix C.6). |

Table 7 reports the mean ± CI of |J_λ| for each λ. It depends only on the energies and the windows,
not on the representation. The published counts, e.g. 102,542.184 ± 492.257 at λ = 2, are
reproduced exactly with the `paper` σ̂_d, float64 energies and legacy windows. The appendix σ̂_d
gives 94,275.340 ± 163.305 at λ = 2.

### EJS–ROC (keys `ROC_AUC`, `PR_AUC`)

The labels are yᵢⱼ = 1{dEᵢⱼ > 2σ̂_rms} with σ̂_rms = √(mean(dE²) + ε), and the scores are sᵢⱼ = Δᵢⱼ.
The AUC is computed with `sklearn.metrics.roc_auc_score` (PR-AUC with `average_precision_score`).

Both versions use this definition. It already matched the appendix in `paper`.

### Thresholded smoothness TS (key `TS`)

| | Definition |
|---|---|
| paper | Uses **all pairs** i < j with dEᵢⱼ ≥ T_E, where T_E = 0.2·Q̃_E. Q̃_E is the 0.9-quantile of \|Eᵢ − Eⱼ\| over ≤ 200,000 random index pairs (`default_rng(0)`, i ∈ [0, n−1), j ∈ [1, n), pairs with i < j kept) plus ε, and Q_Z = Q_0.9(vec Δ) + ε. TS = mean exp(−(Δᵢⱼ/Q_Z) / (dEᵢⱼ/Q̃_E + ε)). |
| v2 | Uses **consecutive frames** k → k+1 along the declared time order (Appendix C.6). Q_Z = Q_0.9(vec Δ) + ε and Q_E = Q_0.9(vec dE) + ε over all pairs (exact). T_E = 0.2·Q_E; the appendix leaves T_E unspecified, and 0.2 is the relative threshold of the released code. K = {k : \|E_{k+1} − E_k\| > T_E}. TS = mean over k ∈ K of exp(−(Δ_{k,k+1}/Q_Z) / (\|E_{k+1} − E_k\|/Q_E + ε)). **It is computed only with `--time-ordered`; otherwise it is NaN.** |

### Smoothness (key `Smoothness`; Table 6)

The final paper does not define this metric. Its two definitions here are:

| | Definition |
|---|---|
| paper | Consecutive frames in file order, in raw units: Smoothness = mean_k exp(−Δ_{k,k+1} / (\|E_{k+1} − E_k\| + 10⁻⁸)). Δ is dimensionless and \|ΔE\| is in kcal/mol. |
| v2 | The scale-normalised TS expression over **all** consecutive segments (T_E = 0): mean_k exp(−(Δ_{k,k+1}/Q_Z) / (\|E_{k+1} − E_k\|/Q_E + ε)). **It is computed only with `--time-ordered`; otherwise it is NaN.** |

*Rationale for v2:* both quantities describe how the representation changes along a trajectory, which
requires time-ordered frames. rMD17 frames are not time-ordered (§1). The `paper` values for rMD17
therefore average over pairs of frames that are effectively random, and v2 does not report them.

### Distributional divergence KS / W1 (keys `KS`, `W1`)

| | Definition |
|---|---|
| paper | Two-sample KS statistic `scipy.stats.ks_2samp(vec Δ, vec dE)` and `wasserstein_distance(vec Δ, vec dE)`. Δ ∈ [0, 1] is compared directly with dE in kcal/mol, so KS is close to 1 for every continuous representation (≥ 0.976 in Table 6), and the value mostly reflects the unit mismatch. |
| v2 | The same statistics between the dimensionless quantities vec(Δ)/Q_Z and vec(dE)/Q_E, with Q_Z and Q_E the 0.9-quantiles above. **Smaller is better** (more similar distribution shapes). |

### Distance correlation (key `dCor`, not reported in the paper)

Both versions return the double-centred distance correlation of the two n × n matrices, clipped to [0, 1].

## 5. Table mapping

| Table cell | Summary key |
|---|---|
| Table 3: Spearman, Kendall, CKA, iso R², EJS, EJS-ROCAUC, TS, KS | `spearman`, `kendall`, `cka_rbf`, `iso_R2`, `EJS`, `ROC_AUC`, `TS`, `KS` (means) |
| Table 6: the 13 rows, mean ± CI | the Table 3 keys plus `EJS_lam0p1` … `EJS_lam3` and `Smoothness` |
| Table 7: jumps at 0.1σ … 3σ | `EJS_num_jumps_lam0p1` … `EJS_num_jumps_lam3` |

`reproduce/energy_tables_3_6_7/collect.py` performs this mapping. `expected.csv` in the same folder
lists, for every cell, the printed value, the value recomputed with this code, and notes on cells
where the two differ.

## 6. Usage

```bash
# Published protocol (defaults: legacy windows, paper metrics, 100 x 2000-frame windows, seed 2025)
python -m three_dbench evaluate traj \
  --dataset-dir data/hf/traj/energies \
  --embeddings embeddings/traj/gemnet --embedding-key gemnet \
  --model-name GemNet --output-dir results/traj/gemnet --n-jobs 16

# E3FP fingerprints (pickled RDKit bit vectors): Tanimoto distance is selected automatically
python -m three_dbench evaluate traj --dataset-dir data/hf/traj/energies \
  --embeddings embeddings/traj/e3fp --model-name E3FP --n-jobs 16

# Corrected definitions, subset of molecules
python -m three_dbench evaluate traj --dataset-dir data/hf/traj/energies \
  --embeddings embeddings/traj/unimol --metric-version v2 --molecules aspirin ethanol --n-jobs 8
```

Each run writes:
- `details.csv`: one row per molecule × window × metric.
- `summary.csv`: model, metric, mean, ci95, n.
- `config.json`: all options, window starts, the sha256 of each molecule's float64 energies, and
  package versions.
