# Usage guide

This guide covers downloading data, the `evaluate` options of each benchmark, and the Python API.
Embedding formats are described in [EMBEDDINGS.md](EMBEDDINGS.md), metric definitions in
[METRICS.md](METRICS.md), and the reproduction of the paper tables in
[../reproduce/README.md](../reproduce/README.md).

Default data and output directories are resolved relative to `$THREE_DBENCH_HOME` if it is set, and
relative to the current working directory otherwise. Importing the package does not create files.

## 1. Download

```bash
python -m three_dbench download dataset --task {chirality,traj,rotation,all} [--out data/hf] [--revision REV] [--overwrite]
python -m three_dbench download embeddings --task {chirality,traj,rotation,chirality_legacy_15218,all} \
    [--models gemnet unimol ...] [--results] [--out data/embeddings] [--revision REV] [--no-verify] [--dry-run]
```

`download dataset` loads the configs of [`EscheWang/3dcs`](https://huggingface.co/datasets/EscheWang/3dcs)
and writes them with `save_to_disk`:

| Task | Config | Written to |
|---|---|---|
| `chirality` | `chirality` | `data/hf/chirality` |
| `traj` | `traj_energies`, `traj_frames` | `data/hf/traj/energies`, `data/hf/traj/frames` |
| `rotation` | `rotation` | `data/hf/rotation` (~7.5 GB download; ~25 GB on disk) |

Existing directories are kept unless `--overwrite` is given.

`download embeddings` reads `manifest.csv` of
[`EscheWang/3dcs-embeddings`](https://huggingface.co/datasets/EscheWang/3dcs-embeddings), downloads
the files under `<task>/<model>/` (plus `results/<task>/` with `--results`) to `<out>/<path>`, and
verifies each file's SHA-256 (files that are already present with the right checksum are not
downloaded again). `--dry-run` lists the selected files and their sizes.

Equivalent Python:

```python
from three_dbench.download import download_dataset, download_embeddings

download_dataset("chirality", "data/hf")
download_embeddings("rotation", "data/embeddings", models=["gemnet"])
```

## 2. Chirality

```bash
python -m three_dbench evaluate chirality \
  --dataset-dir data/hf/chirality \
  --embeddings data/embeddings/chirality/unimol/1.npz --embedding-key arr_0 \
  --model-name unimol --output-dir results/chirality/unimol
```

Options (see [metrics/chirality.md](metrics/chirality.md)):

| Option | Default | Meaning |
|---|---|---|
| `--distance {euclidean,cosine}` | `euclidean` | distance for continuous embeddings; `euclidean` reproduces Table 2. Fingerprints always use Tanimoto. |
| `--metric-version {paper,v2}` | `paper` | metric definitions |
| `--unsup-kmax` | `n-1` | largest k for the best-k silhouette |
| `--per-mol-min-n` | 2 | minimum conformers per molecule |
| `--max-molecules` | all | evaluate the first N molecules (testing) |
| `--do-unsup-when-single-en` | off | also compute the unsupervised metrics for parents with a single stereoisomer |
| `--n-jobs` | 1 | worker processes (`-1` = all CPUs) |

Outputs: `<model>_per_molecule.json`, `summary.csv`.

## 3. Trajectory (energy)

```bash
python -m three_dbench evaluate traj \
  --dataset-dir data/hf/traj/energies \
  --embeddings data/embeddings/traj/unimol --embedding-key arr_0 \
  --model-name unimol --n-jobs 16 --output-dir results/traj/unimol
```

`--embeddings` is a directory with one `rmd17_<molecule>.npz` (or `.pkl` fingerprint list) per
molecule, or a single `.npz`/`.pkl` dict keyed by molecule.

Options (see [metrics/energy.md](metrics/energy.md)):

| Option | Default | Meaning |
|---|---|---|
| `--window-scheme {legacy,shared}` | `legacy` | window sampling; `legacy` reproduces the published runs |
| `--n-samples`, `--window`, `--random-seed` | 100, 2000, 2025 | windows per molecule, frames per window, seed |
| `--metric-embed {cosine,euclidean,tanimoto}` | cosine (Tanimoto for fingerprints) | representation distance |
| `--metric-version {paper,v2}` | `paper` | metric definitions |
| `--molecules` | all | subset of rMD17 molecules |
| `--energy-precision-check {error,warn,ignore}` | `error` | action when the energies look float32-quantized |
| `--time-ordered` | off | treat the frames as a time series (enables the `v2` TS / smoothness metrics; rMD17 frames are not ordered) |
| `--legacy-traj-len` | 100000 | trajectory length assumed by the `legacy` window scheme |
| `--block-size` | 4096 | block size of the pairwise distance computation |
| `--n-jobs` | 1 | worker processes (`-1` = all CPUs) |

Outputs: `details.csv`, `summary.csv`, `config.json`.

## 4. Rotation (geometry)

```bash
python -m three_dbench evaluate rotation \
  --dataset-dir data/hf/rotation \
  --embeddings data/embeddings/rotation/gemnet --layout by-shard --embedding-key gemnet \
  --model-name gemnet --metrics cosine --n-jobs 24 --output-dir results/rotation/gemnet
```

Embedding layouts (details in [metrics/geometry.md](metrics/geometry.md#data-and-alignment)):

| `--layout` | `--embeddings` | Alignment |
|---|---|---|
| `by-shard` (default for a directory) | directory with one file per shard (`rotation_conformers_{shard}.npz`, `rot{shard}.npz`, … or `--shard-file-pattern`) | per-shard `offset` of each row |
| `flat` (default for a file) | one `.npz`/`.npy`/`.pkl` array in dataset row order | cumulative `n_conformers` over the dataset rows |
| `by-key` | `.npz`/`.pkl` dict keyed by `key` | by key |

The published dataset stores per-shard offsets (they restart at 0 in every shard). Datasets written
by `convert rotation` store global offsets; `--offset-mode auto` (default) detects both.

Options:

| Option | Default | Meaning |
|---|---|---|
| `--metrics` | `cosine euclidean` | distance spaces for continuous embeddings (Table 1 uses cosine) |
| `--metric-version {paper,v2}` | `paper` | metric definitions |
| `--lie-k`, `--lie-self {include,exclude}` | from the metric version | override LIE@k |
| `--as-variant` | from the metric version | override angular smoothness (`mean_delta_circular`, `median_delta_circular`, `median_halfdelta_circular`, `median_dz_linear`) |
| `--extra-metrics` | off | also distance correlation, Mantel, stress, triplet order (slow) |
| `--shards` | all shards with embeddings | shard ids to evaluate |
| `--molecule-list FILE` | – | evaluate only these keys (one per line, `.gz` allowed), e.g. `reproduce/table1_geometry/sampled_molecules_seed2027.txt` |
| `--sample-ratio R --sample-seed S` | – , 2027 | per-shard random sample (`default_rng(S + shard)`); does not regenerate the published 10 % sample |
| `--min-conformers` | 2 | skip molecules with fewer conformers |
| `--max-keys` | – | evaluate at most N molecules (testing) |
| `--n-jobs` | 1 | worker processes (Linux, fork) |
| `--replicate-offset-drift` | off | by-shard only: also compute `<metric>__offset_drift` columns with the embedding shift of the published full run (Table 1 LIE@k / AS) |

Outputs: `<model>_per_key.parquet` (one row per molecule and distance space), `summary.csv`
(mean, median, number of finite values per metric) and `config.json` (definitions, selection counts,
molecules that could not be evaluated, runtime).

A full GemNet run over the 16 shards (1,464,495 molecules) with 24 workers is expected to take about
20–25 minutes per metric version for the cosine space.

`examples/run_rotation_from_hf.py` wraps the same evaluation and can also build a flat `.npy` cache
of the selected shards (`--flat-cache DIR`).

## 5. Python API

```python
from pathlib import Path

from three_dbench.benchmarks import evaluate_chirality_embeddings, evaluate_rotation_embeddings
from three_dbench.benchmarks.rotation import ShardedEmbeddings
from three_dbench.embeddings import load_embeddings, published_embedding

spec = published_embedding("rotation", "gemnet")  # path, key, layout of the published files
result = evaluate_rotation_embeddings(
    dataset_dir=Path("data/hf/rotation"),
    embeddings_by_shard=ShardedEmbeddings.from_directory(spec.local_path("data/embeddings"), key=spec.key),
    metrics=["cosine"],
    metric_version="v2",
    shards=[0],
    n_jobs=8,
)
print(result["summary"])
```

Per-molecule geometry metrics can be computed directly with
`three_dbench.rotation.metrics.compute_geometry_metrics(D, Delta, torsion_deg=..., spec=PRESETS["v2"], Z=...)`.

## 6. Generating your own embeddings

Load a dataset with `datasets.load_from_disk`, rebuild molecules with
`three_dbench.datasets.serialization.mol_from_block`, and write one vector per conformer in the
order described in [EMBEDDINGS.md](EMBEDDINGS.md). For rotation, writing one file per shard in
per-shard offset order lets you use `--layout by-shard`.

Common pitfalls:

- A flat array must have exactly one row per conformer, in dataset row order.
- For rotation, do not use the stored `offset` as a global index: it restarts in every shard.
- Rotation evaluation needs the MolBlocks (RMSD); do not convert with `--no-mol-blocks`.
- Energies must be float64 (`EscheWang/3dcs` config `traj_energies`); the trajectory evaluator checks
  for float32 quantization.

## 7. Converting raw data

The converters used to build the Hugging Face datasets are kept for completeness; the raw inputs are
not distributed.

```bash
python -m three_dbench convert chirality --input-pkl data/chirality/chirality_bench_conformers_noised_only.pkl --output-dir data/hf/chirality
python -m three_dbench convert traj --mol-pkl-dir data/traj/mol_pkl --energy-dir data/traj/npz_data --output-dir data/hf/traj
python -m three_dbench convert rotation --lmdb-root data/rotation/results --output-dir data/hf/rotation
```

`convert rotation` writes global offsets in numeric shard order, which differs from the published
dataset (per-shard offsets); the evaluator accepts both.
