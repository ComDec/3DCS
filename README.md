# 3DCS

[![CI](https://github.com/EscheWang/3DBench/actions/workflows/ci.yml/badge.svg)](https://github.com/EscheWang/3DBench/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

3DCS is an evaluation toolkit for 3D molecular embeddings. It provides benchmarks for
chirality separation, trajectory alignment, and rotation invariance. The repository uses
Hugging Face datasets as the standard input format for all benchmarks, making it easy to
upload and reuse datasets.

## Highlights

- Hugging Face dataset integration for reproducible evaluation
- CLI and Python API for converting datasets and evaluating embeddings
- Three benchmarks: chirality, trajectory, and rotation
- Bundled example embeddings for instant demo
- Consistent outputs (per-sample JSON/CSV and aggregated summaries)

## Quick Start

Package name: **3DCS** (`pip install` name). Python module: `three_dbench`.

### Install

```bash
# Basic install
pip install -e .

# With RDKit (required for molecule parsing)
pip install rdkit-pypi

# Development install (includes pytest, ruff, pre-commit)
pip install -e ".[dev]"
```

### Try the demo

The repository includes small GemNet embedding fixtures so you can run an evaluation
immediately without generating your own embeddings:

```bash
# Run chirality demo (downloads HF dataset on first run)
python examples/demo.py chirality

# Run trajectory demo
python examples/demo.py trajectory

# Run both
python examples/demo.py all
```

### Download datasets from Hugging Face

The datasets are hosted at [`EscheWang/3dcs`](https://huggingface.co/datasets/EscheWang/3dcs)
with configs: `chirality`, `traj_energies`, `traj_frames`, `rotation`.

```python
from datasets import load_dataset

ds = load_dataset("EscheWang/3dcs", name="chirality", split="train")
ds.save_to_disk("data/hf/chirality")
```

### Evaluate your embeddings

```bash
# Chirality
python -m three_dbench evaluate chirality \
  --dataset-dir data/hf/chirality \
  --embeddings your_embeddings.npz \
  --embedding-key arr_0 \
  --model-name your_model

# Trajectory
python -m three_dbench evaluate traj \
  --dataset-dir data/hf/traj/energies \
  --embeddings your_traj_embeddings_dir/ \
  --embedding-key arr_0 \
  --model-name your_model

# Rotation
python -m three_dbench evaluate rotation \
  --dataset-dir data/hf/rotation \
  --embeddings your_rotation_embeddings.npz \
  --embedding-key arr_0 \
  --model-name your_model
```

### Convert raw data to Hugging Face format

If you have the original raw data (pickle, LMDB), you can convert it:

```bash
python -m three_dbench convert chirality \
  --input-pkl data/chirality/chirality_bench_conformers_noised_only.pkl \
  --output-dir data/hf/chirality

python -m three_dbench convert traj \
  --mol-pkl-dir data/traj/mol_pkl \
  --energy-dir data/traj/npz_data \
  --output-dir data/hf/traj

python -m three_dbench convert rotation \
  --lmdb-root data/rotation/results \
  --output-dir data/hf/rotation
```

## Dataset Structure

| Config | Fields |
|--------|--------|
| `chirality` | key, mol_id, en_id, n_conformers, offset, mol_blocks |
| `rotation` | key, shard, n_conformers, offset, mol_blocks, torsion_deg |
| `traj_frames` | mol_type, frame_idx, mol_block |
| `traj_energies` | mol_type, n_frames, energies |

The `mol_blocks` field stores MolBlock strings with 3D coordinates. Rebuild RDKit
objects with `three_dbench.datasets.serialization.mol_from_block`.

## Documentation

- [docs/USAGE.md](docs/USAGE.md) — end-to-end usage guide
- [docs/EMBEDDINGS.md](docs/EMBEDDINGS.md) — embedding format specifications
- [CONTRIBUTING.md](CONTRIBUTING.md) — development setup and contribution guidelines

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to set up a development environment,
run tests, and submit changes.

```bash
make dev      # Install with dev dependencies + pre-commit
make test     # Run test suite
make lint     # Check code style
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
