# 3DCS

[![CI](https://github.com/ComDec/3DCS/actions/workflows/ci.yml/badge.svg)](https://github.com/ComDec/3DCS/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ComDec/3DCS/graph/badge.svg)](https://codecov.io/gh/ComDec/3DCS)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Evaluation toolkit for **3DCS: Datasets and Benchmark for Evaluating Conformational Sensitivity in
Molecular Representations** (ICLR 2026, [OpenReview](https://openreview.net/forum?id=JAb0y8lkqL)).

3DCS tests whether the representations of different conformers of the same molecule preserve
geometric variation (rotation / relaxed-scan dataset), separate stereoisomers (chirality dataset)
and reflect the energy landscape (rMD17 trajectories). The package name is `3dcs`; the Python module
is `three_dbench`.

| Resource | Location |
|---|---|
| Datasets | [`EscheWang/3dcs`](https://huggingface.co/datasets/EscheWang/3dcs), configs `chirality`, `rotation`, `traj_frames`, `traj_energies` (license CC BY-SA 4.0) |
| Baseline embeddings and original metric outputs | [`EscheWang/3dcs-embeddings`](https://huggingface.co/datasets/EscheWang/3dcs-embeddings) (see [docs/EMBEDDINGS.md](docs/EMBEDDINGS.md)) |
| Metric definitions (`paper` and `v2`) | [docs/METRICS.md](docs/METRICS.md) |
| Per-table reproduction scripts | [reproduce/](reproduce/README.md) |
| rMD17 splits | [splits/rmd17/](splits/rmd17/README.md) |

## Quick start

### Install

```bash
git clone https://github.com/ComDec/3DCS.git
cd 3DCS
pip install -e .            # RDKit is installed from PyPI as the `rdkit` package
pip install -e ".[dev]"     # optional: pytest, ruff, pre-commit
```

Python 3.9–3.12 are supported. Do not install the old `rdkit-pypi` package; it has no wheels for
Python 3.12 and fails to import with NumPy 2.

### Download data

```bash
# Datasets (save_to_disk copies under data/hf/)
python -m three_dbench download dataset --task chirality     # data/hf/chirality        (~0.1 GB)
python -m three_dbench download dataset --task traj          # data/hf/traj/{energies,frames}
python -m three_dbench download dataset --task rotation      # data/hf/rotation         (~7.5 GB download)

# Published baseline embeddings (verified against the SHA-256 in manifest.csv)
python -m three_dbench download embeddings --task chirality --models gemnet unimol --out data/embeddings
python -m three_dbench download embeddings --task rotation --models gemnet --out data/embeddings
```

The datasets can also be loaded directly with Hugging Face `datasets`:

```python
from datasets import load_dataset

ds = load_dataset("EscheWang/3dcs", name="chirality", split="train")
ds.save_to_disk("data/hf/chirality")
```

### Evaluate embeddings

```bash
# Chirality: one embedding per conformer, in dataset row order
python -m three_dbench evaluate chirality \
  --dataset-dir data/hf/chirality \
  --embeddings data/embeddings/chirality/gemnet/sampled_feature.npz --embedding-key gemnet \
  --model-name gemnet

# Trajectory (rMD17): a directory with one rmd17_<molecule>.npz per molecule
python -m three_dbench evaluate traj \
  --dataset-dir data/hf/traj/energies \
  --embeddings data/embeddings/traj/gemnet --embedding-key gemnet \
  --model-name gemnet --n-jobs 16

# Rotation: one file per shard (rows aligned with the per-shard `offset`)
python -m three_dbench evaluate rotation \
  --dataset-dir data/hf/rotation \
  --embeddings data/embeddings/rotation/gemnet --layout by-shard \
  --model-name gemnet --metrics cosine --shards 0 --n-jobs 16
```

Every evaluation writes a summary (`summary.csv`), per-molecule results and the configuration to
`--output-dir` (default `results/<task>/<model>`, relative to the working directory or
`$THREE_DBENCH_HOME`). `--metric-version paper` (default) uses the definitions behind the published
numbers; `--metric-version v2` uses the corrected definitions described in
[docs/METRICS.md](docs/METRICS.md). See [docs/USAGE.md](docs/USAGE.md) for all options, embedding
layouts and the Python API.

A small demo with bundled GemNet fixtures:

```bash
python examples/demo.py all
```

## Reproducing the paper

Each table has a directory under [`reproduce/`](reproduce/README.md) with a `run.sh` (download,
evaluation, `results.csv`) and an `expected.csv` (printed value, recomputed value with 6 decimals,
tolerance and notes). `reproduce/compare.py` prints PASS/FAIL per cell.

| Table | Script | What it recomputes | Runtime (24 workers) |
|---|---|---|---|
| Table 1 (geometry) | `bash reproduce/table1_geometry/run.sh` | GemNet from the published rotation embeddings | ~40–50 min for two metric versions (estimated from single-shard runs); `QUICK=1` (one shard): ~4 min |
| Table 2 (chirality, zero-shot) | `bash reproduce/table2_chirality/run.sh` | all 7 models from the published embeddings | ~40–60 min for all variants |
| Tables 3, 6, 7 (energy, zero-shot) | `bash reproduce/energy_tables_3_6_7/run.sh` | all 7 models from the published embeddings, float64 energies | ~50 min per metric version |

Download sizes: rotation dataset ~7.5 GB, rotation GemNet embeddings ~5.2 GB, trajectory embeddings
~7 GB, chirality embeddings ~0.4 GB.

### Status

"Reproducible" means that the published value is recomputed from released data, embeddings and code
within the tolerance in `expected.csv`; the notes in each `expected.csv` list the known exceptions.

| Table | E3FP | GemNet | MolAE | MolSpectra | UniMol | FMG | MACE |
|---|---|---|---|---|---|---|---|
| 1 Geometry | pending (rotation embeddings) | full 16-shard run pending | pending (rotation embeddings) | pending (rotation embeddings) | pending (rotation embeddings) | not in table | not in table |
| 2 Chirality (zero-shot) | reproducible | reproducible | reproducible | reproducible | reproducible | reproducible | reproducible |
| 3 / 6 / 7 Energy (zero-shot) | reproducible | reproducible | reproducible | reproducible | reproducible | partially | partially |
| 4 Chirality correlation | pending (evaluation code) | pending (evaluation code) | pending (evaluation code) | pending (evaluation code) | pending (evaluation code) | not in table | not in table |
| 5 Chirality fine-tuning | not in table | pending | pending | pending | pending | pending | pending |
| 8 / 9 rMD17 fine-tuning | pending | pending | pending | pending | pending | pending | pending |

Further notes:

- **Embedding extraction.** The published embeddings are the files used for the paper. Extraction
  scripts are not part of this repository; the per-model status (known, partially known, unknown) is
  listed in [docs/EMBEDDINGS.md](docs/EMBEDDINGS.md).
- **Table 1.** The published rows come from two runs: Spearman, Kendall, CKA, isotonic R² and
  Torsion-SP from a 10 % molecule sample (its key list is in `reproduce/table1_geometry/`), LIE@k and
  AS from all molecules. The per-molecule outputs of both runs for all five models are published in
  the embeddings repository (`results/rotation/`). GemNet is the only model whose rotation embeddings
  were kept, and a full 16-shard recomputation has not been run yet; `reproduce/table1_geometry/`
  currently verifies one shard against the backed-up per-molecule outputs. In the full published run,
  two molecules could not be processed and the embedding cursor was not advanced, so the molecules
  after them in shards 1 and 2 (6.2 % of the total) were scored with embeddings shifted by 3 and 7
  rows; this affects the LIE@k and AS columns. `--replicate-offset-drift` reproduces that behaviour.
  See [docs/metrics/geometry.md](docs/metrics/geometry.md).
- **Table 4.** The backed-up summary behind the Spearman, Kendall and CKA rows is published
  (`results/chirality/chirality_metrics_summary.csv` in the embeddings repository); outputs for the
  OPD rows were not found in our backups. The code for this table is not in the release.
- **Fine-tuning (Tables 5, 8, 9).** Fine-tuning code and checkpoints are not part of this release.
  Checkpoints for the chirality fine-tuning (Table 5) are being prepared and will be added later; no
  checkpoints are planned for the rMD17 tables (8, 9). For the rMD17 tables, the train/test indices we
  found correspond to the official split 01 ([splits/rmd17/](splits/rmd17/README.md)).

## Dataset structure

| Config | Rows | Fields |
|--------|------|--------|
| `chirality` | 14,903 (52,391 conformers) | key, mol_id, en_id, n_conformers, offset, mol_blocks |
| `rotation` | 1,559,779 (10,097,643 conformers) | key, shard, n_conformers, offset (per shard), mol_blocks, torsion_deg |
| `traj_frames` | 999,988 | mol_type, frame_idx, mol_block |
| `traj_energies` | 10 | mol_type, n_frames, energies (kcal/mol) |

`mol_blocks` stores MolBlock strings with 3D coordinates; rebuild RDKit molecules with
`three_dbench.datasets.serialization.mol_from_block`. Converters for the original raw files are
available as `python -m three_dbench convert {chirality,traj,rotation}` (the raw files are not
distributed).

## Documentation

- [docs/USAGE.md](docs/USAGE.md): CLI and Python API
- [docs/EMBEDDINGS.md](docs/EMBEDDINGS.md): embedding formats and the published baseline embeddings
- [docs/METRICS.md](docs/METRICS.md): metric definitions (`paper` and `v2`)
- [reproduce/README.md](reproduce/README.md): reproducing the paper tables
- [CONTRIBUTING.md](CONTRIBUTING.md): development setup

## Citation

```bibtex
@inproceedings{wang2026threedcs,
  title     = {3{DCS}: Datasets and Benchmark for Evaluating Conformational Sensitivity in Molecular Representations},
  author    = {Wang, Xi and Zhang, Yang and Zhang, Yingjia and Cai, Yejia and Wang, Shengjie},
  booktitle = {The Fourteenth International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=JAb0y8lkqL}
}
```

See also [CITATION.cff](CITATION.cff). When you use the trajectory data or the rMD17 splits, please
also cite rMD17 (Christensen & von Lilienfeld, 2020).

## License

The code is released under the MIT License (see [LICENSE](LICENSE)). The datasets on Hugging Face
are released under CC BY-SA 4.0; see the dataset cards for upstream terms (ChEMBL, rMD17).
