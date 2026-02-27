# Usage Guide

This guide describes dataset conversion, embedding formats, and evaluation workflows.
For embedding generation details, see `docs/EMBEDDINGS.md`.

## Quick demo

The fastest way to see 3DCS in action is to run the bundled demo with pre-computed
GemNet embeddings:

```bash
pip install -e .
python examples/demo.py all
```

This downloads the HF dataset and evaluates the included fixture embeddings for both
chirality and trajectory benchmarks. See `examples/demo.py` for details.

## 1. Dataset conversion

Convert the raw RDKit molecule data into Hugging Face datasets:

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

Use `--no-mol-blocks` to skip MolBlock storage when disk space is limited. Rotation
evaluation requires MolBlocks to compute RMSD.

## 2. Embedding formats

### Chirality

- **Flat array**: one embedding per conformer, aligned with the dataset order
- Use the `offset` field in the dataset rows to verify alignment if needed
- Supported files: `.npz`, `.npy`, `.pkl`
- Use `--embedding-key` for NPZ or pickled dicts

### Rotation

- **Flat array** (default): embeddings aligned with `offset` and `n_conformers`
- **By key**: a dict mapping `key` to a `(n_conformers, dim)` array
- Supported files: `.npz` or `.pkl` dicts for by-key layout

### Trajectory

- Directory of `rmd17_*.npz` files (default)
- Or a dict mapping `mol_type` to `(n_frames, dim)` arrays

## 3. Run evaluation

### Chirality

```bash
python -m three_dbench evaluate chirality \
  --dataset-dir data/hf/chirality \
  --embeddings data/chirality/unimol/1.npz \
  --embedding-key arr_0 \
  --model-name unimol
```

### Rotation

```bash
python -m three_dbench evaluate rotation \
  --dataset-dir data/hf/rotation \
  --embeddings /path/to/rotation_embeddings.npz \
  --embedding-key arr_0 \
  --model-name my_model
```

To use a dict keyed by rotation entry IDs:

```bash
python -m three_dbench evaluate rotation \
  --dataset-dir data/hf/rotation \
  --embeddings /path/to/rotation_by_key.pkl \
  --layout by-key \
  --model-name my_model
```

### Trajectory

```bash
python -m three_dbench evaluate traj \
  --dataset-dir data/hf/traj/energies \
  --embeddings data/traj/results/unimol \
  --embedding-key arr_0 \
  --model-name unimol
```

## 4. End-to-end examples (download from HF and evaluate)

The scripts under `examples/` will download the dataset from HF if it is not cached
locally, then run evaluation using existing embeddings.

### Chirality

```bash
python examples/run_chirality_from_hf.py \
  --repo-id EscheWang/3dcs \
  --config chirality \
  --embeddings data/chirality/unimol/1.npz \
  --embedding-key arr_0 \
  --model-name unimol \
  --output-dir results/chirality/unimol
```

### Trajectory

```bash
python examples/run_traj_from_hf.py \
  --repo-id EscheWang/3dcs \
  --config traj_energies \
  --embeddings-dir data/traj/results/unimol \
  --embedding-key arr_0 \
  --model-name unimol \
  --output-dir results/traj/unimol
```

### Rotation

```bash
python examples/run_rotation_from_hf.py \
  --repo-id EscheWang/3dcs \
  --config rotation \
  --embeddings-dir data/rotation/results/gemnet \
  --embedding-key gemnet \
  --model-name gemnet \
  --output-dir results/rotation/gemnet
```

#### Rotation quick test

```bash
python examples/run_rotation_from_hf.py \
  --repo-id EscheWang/3dcs \
  --config rotation \
  --embeddings-dir data/rotation/results/gemnet \
  --embedding-key gemnet \
  --model-name gemnet \
  --output-dir results/rotation/gemnet_quick \
  --shards 0 \
  --max-keys 200
```

## 5. End-to-end with your own embeddings

This section shows the full pipeline when you generate embeddings with your own model.
It includes download, embedding generation, format validation, and evaluation.

### Step 1: Download datasets from HF

```bash
# Chirality dataset
python - <<'PY'
from datasets import load_dataset

ds = load_dataset("EscheWang/3dcs", name="chirality", split="train")
ds.save_to_disk("data/hf/chirality")
PY

# Trajectory energies
python - <<'PY'
from datasets import load_dataset

ds = load_dataset("EscheWang/3dcs", name="traj_energies", split="train")
ds.save_to_disk("data/hf/traj/energies")
PY

# Rotation dataset
python - <<'PY'
from datasets import load_dataset

ds = load_dataset("EscheWang/3dcs", name="rotation", split="train")
ds.save_to_disk("data/hf/rotation")
PY
```

### Step 2: Generate embeddings

Use MolBlocks from the HF datasets to feed your model and generate fixed-length vectors.
Detailed formats and examples are in `docs/EMBEDDINGS.md`.

Minimal skeleton for chirality:

```python
from datasets import load_from_disk
from three_dbench.datasets.serialization import mol_from_block
import numpy as np

ds = load_from_disk("data/hf/chirality")
vectors = []

for row in ds:
    mols = [mol_from_block(b, sanitize=False) for b in row["mol_blocks"]]
    for mol in mols:
        vec = your_model(mol)  # shape (dim,)
        vectors.append(vec)

embeddings = np.stack(vectors, axis=0)
np.savez("embeddings/my_model_chirality.npz", arr_0=embeddings)
```

### Step 3: Validate embedding alignment

Embedding arrays must align with the dataset order. For flat arrays, the total number
of vectors must equal the sum of `n_conformers` in the dataset.

```python
from datasets import load_from_disk
import numpy as np

ds = load_from_disk("data/hf/chirality")
expected = int(sum(ds["n_conformers"]))
arr = np.load("embeddings/my_model_chirality.npz")["arr_0"]
assert arr.shape[0] == expected
```

### Step 4: Run evaluation

```bash
python -m three_dbench evaluate chirality \
  --dataset-dir data/hf/chirality \
  --embeddings embeddings/my_model_chirality.npz \
  --embedding-key arr_0 \
  --model-name my_model \
  --output-dir results/chirality/my_model
```

### Trajectory with custom embeddings

```bash
python -m three_dbench evaluate traj \
  --dataset-dir data/hf/traj/energies \
  --embeddings embeddings/traj_my_model \
  --embedding-key arr_0 \
  --model-name my_model \
  --output-dir results/traj/my_model
```

`embeddings/traj_my_model` should contain one `rmd17_*.npz` per molecule.

### Rotation with custom embeddings

For rotation, embeddings must be aligned with `offset` and `n_conformers` in the
rotation dataset. If you generate per-key embeddings, build a dict mapping `key` to
`(n_conformers, dim)` and save as a pickle.

```bash
python -m three_dbench evaluate rotation \
  --dataset-dir data/hf/rotation \
  --embeddings embeddings/rotation_by_key.pkl \
  --layout by-key \
  --model-name my_model \
  --output-dir results/rotation/my_model
```

### Common pitfalls

- If the evaluation errors with a length mismatch, check the total number of conformers.
- Rotation evaluation requires MolBlocks; do not use `--no-mol-blocks` during conversion.
- Trajectory evaluation uses energy windows; make sure embeddings cover all frames.

## 6. Outputs

Each evaluation writes:

- `summary.csv`: aggregated statistics
- `details.csv` or `*_per_molecule.json`: per-sample metrics
- `config.json` (trajectory)

Output directory defaults to `results/{task}/{model_name}` unless overridden.
