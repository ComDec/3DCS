# Embeddings

This page describes (1) the embedding formats the evaluators accept and (2) the published baseline
embeddings in [`EscheWang/3dcs-embeddings`](https://huggingface.co/datasets/EscheWang/3dcs-embeddings).

## 1. Inputs provided by 3DCS

Each dataset row stores RDKit MolBlocks with 3D coordinates (`three_dbench.datasets.serialization.mol_from_block`).

| Config | Row | Conformer order |
|---|---|---|
| `chirality` | one stereoisomer of a parent molecule: `key`, `mol_id`, `en_id`, `n_conformers`, `offset`, `mol_blocks` | `offset` is a global index (rows in dataset order) |
| `rotation` | one molecule: `key`, `shard`, `n_conformers`, `offset`, `mol_blocks`, `torsion_deg` | `offset` restarts at 0 in every shard; shards appear in string order `0, 1, 10, …, 15, 2, …, 9` |
| `traj_frames` | one rMD17 frame: `mol_type`, `frame_idx`, `mol_block` | `frame_idx` = row of the rMD17 `.npz` |
| `traj_energies` | one molecule: `mol_type`, `n_frames`, `energies` (kcal/mol) | aligned with `frame_idx` |

## 2. Accepted formats

- **NPZ**: `np.savez(path, arr_0=embeddings)` or any key; pass `--embedding-key`. Without a key the
  loader tries `arr_0`, `embeddings`, `gemnet`, `mol_feature`, then the first numeric array (string
  side-arrays such as FMG's `smiles` are skipped).
- **NPY**: `np.save(path, embeddings)` (memory-mapped for the rotation by-shard layout).
- **PKL**: a list of RDKit `ExplicitBitVect` (fingerprints, Tanimoto distance) or a dict; without a key
  the entries `e3fp`, `embeddings`, `arr_0` are tried.
- A directory holding exactly one embedding file can be passed instead of the file.

| Task | Layout | Shape |
|---|---|---|
| Chirality | flat, dataset row order | `(52391, dim)` or a list of 52,391 fingerprints |
| Trajectory | directory with `rmd17_<molecule>.npz` (or `.pkl` fingerprints), or a dict `{mol_type: array}` | `(n_frames, dim)` per molecule |
| Rotation | `--layout by-shard`: directory with one file per shard (`rotation_conformers_{shard}.npz`, or `--shard-file-pattern`), rows in per-shard offset order | `(conformers in shard, dim)` |
| Rotation | `--layout flat`: one array in dataset row order | `(10097643, dim)` |
| Rotation | `--layout by-key`: dict `{key: array}` in `.npz`/`.pkl` | `(n_conformers, dim)` per key |

Generating embeddings for the rotation dataset shard by shard keeps the per-shard layout:

```python
import numpy as np
from datasets import load_from_disk
from three_dbench.datasets.serialization import mol_from_block

ds = load_from_disk("data/hf/rotation")
for shard in range(16):
    part = ds.filter(lambda s: s == shard, input_columns=["shard"]).sort("offset")
    vectors = []
    for row in part:
        for block in row["mol_blocks"]:
            vectors.append(encode(mol_from_block(block)))  # your model, shape (dim,)
    np.savez(f"embeddings/my_model/rotation_conformers_{shard}.npz", arr_0=np.stack(vectors))
```

[`baselines/`](../baselines/README.md) holds a worked version of this for each of the seven
baseline models on the chirality set: input handling, the model call and the write, with the
environment each one needs. All seven read the dataset through the same `--dataset` syntax
(`hf:<repo>[:<config>]`, `hfdisk:<dir>` or a `save_to_disk` directory, a bare Hub id, a pickle of
RDKit molecules, or `lmdb:<file>`).

Check alignment before evaluating: for flat arrays the number of rows must equal the sum of
`n_conformers`; for by-shard files, each file must have `max(offset + n_conformers)` rows of its shard.

## 3. Published baseline embeddings

`python -m three_dbench download embeddings --task {chirality,traj,rotation,chirality_legacy_15218} [--models ...] [--results]`
downloads files listed in the repository's `manifest.csv` to `data/embeddings/<path>` and checks
their SHA-256. The files keep their original bytes and array keys; only directory names were
normalised. `three_dbench.embeddings.published_embedding(task, model)` returns the path, key and
layout of each entry.

| Path | Key | Shape | Used for |
|---|---|---|---|
| `chirality/e3fp/sampled_chi.pkl` | dict entry `e3fp` (also `morgan`) | 52,391 RDKit bit vectors (1024 bits) | Table 2 |
| `chirality/gemnet/sampled_feature.npz` | `gemnet` | (52391, 128) float32 | Table 2 |
| `chirality/molae/1.npz` | `arr_0` | (52391, 512) | Table 2 |
| `chirality/molspectra/sampled_mol_feature.npz` | `arr_0` | (52391, 256) | Table 2 |
| `chirality/unimol/1.npz` | `arr_0` | (52391, 512) | Table 2 |
| `chirality/fmg/chirality_bench_conformers_noised_only_aslist_embed.npz` | `embeddings` (+ `smiles`) | (52391, 128) | Table 2 |
| `chirality/mace/chirality.npz` | `arr_0` | (52391, 256) | Table 2 |
| `chirality_legacy_15218/…` | see manifest | 15,218 conformers | earlier chirality set (Table 4) |
| `traj/<model>/rmd17_<molecule>.npz` (`.pkl` for E3FP) | `gemnet` (GemNet), `embeddings` (FMG), `arr_0` (others); E3FP: pickled list | (100000, dim); azobenzene 99,988 | Tables 3, 6, 7 |
| `rotation/gemnet/rotation_conformers_{0..15}.npz` | `gemnet` | (conformers in shard, 128) | Table 1 |
| `rotation/fmg/rot_mol_list_0_embed.npz`, `rotation/mace/rot0.npz` | `embeddings` / `arr_0` | shard 0 only | not used in the paper |
| `results/chirality/`, `results/traj/`, `results/rotation/` | – | original metric outputs | reference values |

Rotation embeddings for E3FP, UniMol, MolAE and MolSpectra are not available. For those models the
per-molecule metric outputs of the original runs (`results/rotation/metrics_all_0.1_1.json.gz`,
`results/rotation/metrics_sup_100.json.gz`) are published instead.

### What each published file contains

| Model | Dimension | Notes |
|---|---|---|
| E3FP | 1024 bits | `e3fp` 1.2.7, `fprints_from_mol(mol, fprint_params=dict(bits=1024, level=5, radius_multiplier=1.5, stereo=True, include_disconnected=True, rdkit_invariants=True, first=1, counts=False))`, hydrogens kept. Recomputing with these parameters from the original RDKit molecules reproduces all 52,391 chirality fingerprints bit for bit; starting from the HF MolBlocks, about 5 % of fingerprints differ. |
| GemNet (GemNet-Q) | 128 | one 128-d vector per conformer, key `gemnet`. The GemNet-Q weights are not part of this release. |
| UniMol | 512 | one 512-d vector per conformer, key `arr_0`. |
| MolAE | 512 | one 512-d vector per conformer, key `arr_0`. |
| MolSpectra | 256 | one 256-d vector per conformer, key `arr_0`. |
| MACE | 256 | one 256-d vector per conformer, key `arr_0`. |
| FMG | 128 | one 128-d vector per conformer, key `embeddings` (third-party model: Dumitrescu et al., ICLR 2025). |

The reference values in [`reproduce/`](../reproduce/README.md) are computed from these files.
Embeddings produced with a different extractor are evaluated the same way, but their values are not
expected to match these reference values.

### Extracting these embeddings

[`baselines/<model>/extract_chirality.py`](../baselines/README.md) computes the chirality embedding
of each model from the dataset, in the same row order and under the same array key as the file
above. Each script drives an upstream checkout that you install yourself — no third-party code or
weights are redistributed — and `baselines/<model>/ENVIRONMENT.md` gives the upstream repository and
commit, the weight file with its SHA-256 and where to download it, and the exact install commands.
`--verify` compares a freshly extracted file with the published one and prints the checksums, the
elementwise differences and the per-row cosine similarity — over the rows the run covers, so a run
with `--limit` on a slice is compared with the matching rows of the published file;
[`baselines/README.md`](../baselines/README.md) tabulates those numbers for a full run of every
script.
