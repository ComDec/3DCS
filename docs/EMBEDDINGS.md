# Embedding Generation Guide

This guide explains what inputs 3DCS provides and what embedding formats are accepted.

## Overview

3DCS evaluates embeddings that represent 3D molecular conformations. For each task, the
dataset contains RDKit MolBlocks with 3D coordinates. Your workflow is:

1. Load the HF dataset for a task
2. Reconstruct RDKit molecules from MolBlocks
3. Call your model to produce a fixed-length vector per conformer
4. Save the embeddings in one of the supported formats

The embedding dimension must be fixed for a given model.

## Inputs provided by 3DCS

- Chirality dataset rows contain: `key`, `mol_id`, `en_id`, `n_conformers`, `offset`, `mol_blocks`
- Rotation dataset rows contain: `key`, `shard`, `n_conformers`, `offset`, `mol_blocks`, `torsion_deg`
- Trajectory frames dataset rows contain: `mol_type`, `frame_idx`, `mol_block`
- Trajectory energies dataset rows contain: `mol_type`, `n_frames`, `energies`

MolBlocks include 3D coordinates. Use `three_dbench.datasets.serialization.mol_from_block`
to rebuild RDKit Mol objects.

## Supported embedding formats

### Common formats

- NPZ: `np.savez(path, arr_0=embeddings)` or `np.savez(path, embeddings=embeddings)`
- NPY: `np.save(path, embeddings)`
- PKL: `pickle.dump(embeddings, f)` or a dict `{key: embeddings}`

### Chirality

- **Flat array** (default): shape `(total_conformers, dim)` aligned with dataset order
- **Fingerprint list**: a list of RDKit fingerprints aligned with dataset order

Alignment is determined by `offset` and `n_conformers` in the HF dataset rows.

### Rotation

- **Flat array** (default): shape `(total_conformers, dim)` aligned with dataset order
- **By key dict**: `{key: np.ndarray of shape (n_conformers, dim)}`
- **Fingerprint list**: supported for flat layout

### Trajectory

- **Directory of NPZ files**: one file per molecule (e.g. `rmd17_aspirin.npz`)
- **Dict**: `{mol_type: np.ndarray of shape (n_frames, dim)}`

## Example: chirality embedding generation

```python
from pathlib import Path
import numpy as np
from datasets import load_from_disk
from three_dbench.datasets.serialization import mol_from_block

ds = load_from_disk("data/hf/chirality")

vectors = []
for row in ds:
    mols = [mol_from_block(b, sanitize=False) for b in row["mol_blocks"]]
    # Replace this with your model call
    for mol in mols:
        vec = np.random.randn(256).astype(np.float32)
        vectors.append(vec)

embeddings = np.stack(vectors, axis=0)
np.savez("my_model_chirality.npz", arr_0=embeddings)
```

## Example: rotation embedding generation (by key)

```python
from pathlib import Path
import numpy as np
from datasets import load_from_disk
from three_dbench.datasets.serialization import mol_from_block

ds = load_from_disk("data/hf/rotation")
out = {}
for row in ds:
    mols = [mol_from_block(b, sanitize=False) for b in row["mol_blocks"]]
    vecs = [np.random.randn(256).astype(np.float32) for _ in mols]
    out[row["key"]] = np.stack(vecs, axis=0)

import pickle
with open("my_rotation_by_key.pkl", "wb") as f:
    pickle.dump(out, f)
```

## Example: trajectory embedding generation (per molecule)

```python
import numpy as np
from datasets import load_from_disk
from three_dbench.datasets.serialization import mol_from_block

frames = load_from_disk("data/hf/traj/frames")
mol_types = sorted(set(frames["mol_type"]))

for mol_type in mol_types:
    rows = frames.filter(lambda x: x["mol_type"] == mol_type)
    vecs = []
    for row in rows:
        mol = mol_from_block(row["mol_block"], sanitize=False)
        vecs.append(np.random.randn(256).astype(np.float32))
    embeddings = np.stack(vecs, axis=0)
    np.savez(f"embeddings/{mol_type}.npz", arr_0=embeddings)
```

## Tips

- Use float32 for most models. Float16 is acceptable for very large datasets.
- Keep the embedding order identical to the dataset order.
- For rotation, MolBlocks are required to compute RMSD; do not skip them during conversion.
