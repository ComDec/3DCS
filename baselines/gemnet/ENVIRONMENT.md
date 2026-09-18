# GemNet-Q — environment

`extract_chirality.py` drives an unmodified `gemnet_pytorch` checkout: the graph is built by
upstream's own `DataContainer`, and the per-atom representation `h` of the last interaction
block is read through a `register_forward_hook`, so upstream's `gemnet.py` runs exactly as
published. No third-party code and no weights are redistributed here.

## Upstream

```bash
git clone https://github.com/TUM-DAML/gemnet_pytorch.git
git -C gemnet_pytorch checkout a0164f74217155232d39c35f0bb2c016bd3f44da
```

Upstream `gemnet_pytorch` is licensed under the Hippocratic License 2.0. Check that it
permits your use before running it.

## Weights

They are part of that checkout — the released GemNet-Q pretraining, no fine-tuned weights.

| file in the checkout | sha256 |
|---|---|
| `pretrained/GemNet-Q/model.pth` | `d51e00af18cf9dd0097f0bc251386fdd539cfe82fdeb2790d0a0f9a241f169cd` |
| `pretrained/GemNet-Q/model_kwargs.json` | `a1283907379e62a9af447d7dc74410cdc9e0969265f87d1e842c5f72238825d0` |
| `pretrained/scaling_factors.json` | `f9c855d929b8774b003c20246733faa571befe9d999bda2025fe7618cc100635` |

`model_kwargs.json` selects GemNet-Q: `num_blocks 4`, `emb_size_atom 128`, `cutoff 5.0`,
`int_cutoff 10.0`, `triplets_only false`, `direct_forces false`, `extensive true`,
`activation swish`, `scale_file scaling_factors.json`.

## Install

```bash
conda create -y -p ./env python=3.10
./env/bin/pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
./env/bin/pip install torch_scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
./env/bin/pip install -r requirements.txt
```

Resolved versions: python 3.10.21, torch 2.1.0+cu121, torch_scatter 2.1.2+pt21cu121,
numpy 1.24.4, scipy 1.10.1, sympy 1.12, numba 0.58.1, rdkit 2025.03.5. The script prints all
of them at startup.

Three compatibility points, all handled by the script:

- **`torch_scatter` is optional.** If the compiled package is not importable,
  `compat/torch_scatter.py` is used instead — a pure-PyTorch implementation of the
  `add` / `mean` reductions GemNet needs. On this dataset the two agree to
  `max|diff| = 7.4e-06` (`np.allclose(atol=1e-5, rtol=1e-4)` passes).
- **numpy aliases.** Upstream targets numpy < 1.24 and uses names later releases removed:
  `np.bool` (`gemnet/training/data_container.py`) and `np.math.factorial`
  (`gemnet/model/layers/basis_utils.py`). The script restores those aliases at import time
  instead of editing the checkout.
- Upstream ships `gemnet/` without `__init__.py` files; the script creates the empty ones if
  they are missing.

## Run

```bash
python extract_chirality.py \
    --gemnet-repo ./gemnet_pytorch \
    --dataset hf:EscheWang/3dcs:chirality \
    --out gemnet_chirality.npz \
    --batch-size 8 --device cuda --checkpoint-every 8000 --verify
```

`--dataset` takes the input specification shared by every script in `baselines/` (see the
table in [`../README.md`](../README.md#running-one)): `hf:<repo>[:<config>]`, `hfdisk:<dir>` or a plain
`save_to_disk` directory, a bare Hub dataset id, a pickle of RDKit molecules, or `lmdb:<file>`.
`--limit`/`--start` run a slice of the conformers and `--verify-rows` says which rows of the
reference that slice covers.

Row order follows the dataset: for the Hugging Face forms, rows are ordered by `offset` and
then by position inside each row's `mol_blocks`. The script refuses to skip a conformer it
cannot featurise rather than shifting the rows.

Options: `--pooling {mean,add}`, `--hydrogens {all,remove,keep}`, `--cutoff`, `--int-cutoff`,
`--triplets-only`, `--round-coords`, `--fp16`, `--tf32`, `--start/--limit`, `--batch-size`,
`--device`, `--seed`, `--num-threads`, `--oom-retries/--oom-wait` (retry a batch in smaller
pieces after a CUDA OOM) and `--checkpoint-every` (write the partial `.npz` as it goes).
## Settings

`Chem.RemoveAllHs` (heavy atoms only, `--hydrogens all`), edges within 5 A, quadruplet
interactions within 10 A, float32, mean pooling of `h` over the atoms of the molecule, no
normalisation afterwards.

## Determinism

The script seeds python / numpy / torch, disables cuDNN autotuning and leaves TF32 off, so
repeated runs on the same machine agree bit for bit, and `--batch-size` does not change the
result (a run at `--batch-size 1` and one at `--batch-size 64` agreed to the last float32 bit
on the first 256 conformers). Results are not bit-identical across different GPUs and
CUDA / cuBLAS versions: the same code with TF32 on moves the output by `max|diff| = 2.0e-03`.

## Input precision

The published embedding file and the agreement numbers in [`../README.md`](../README.md) were
computed from the source RDKit molecules, whose coordinates carry full float precision. The public
`EscheWang/3dcs` dataset stores those geometries as V2000 MOL blocks, which hold four decimals.
Over the first 2,000 conformers, a run from `hf:EscheWang/3dcs:chirality` and a run from the
full-precision molecules, on the same machine in the same environment, differ by
`max|diff| = 5.1e-04` and `mean|diff| = 1.9e-05`, per-row cosine at least 0.999999996, no row below
1 - 1e-6. On the first 256 conformers the same comparison gives `max|diff| = 2.4e-04`, and
`--round-coords 4` on the full-precision molecules reproduces it, so the rounding of the MOL blocks
is the whole of the difference. GemNet-Q is, with MACE, among the least sensitive of the seven.
## Rotation shards

The same settings cover the rotation track. Pass `--dataset lmdb:<shard>.lmdb`: each key of a
shard holds a list of `(Mol, energy, torsion_deg)`, and the row order is the LMDB cursor
order of the keys followed by the position inside each list. For shard 0 that gives 630,021
conformers over 97,487 keys, which is the shape of the published
`rotation/gemnet/rotation_conformers_0.npz`. On the first 2,048 rows of that shard the script
agrees with the published file to a per-row cosine of at least 0.9999997
(`max|diff| = 3.4e-03`). Requires `lmdb`. Write the output with `--key gemnet` and the file
name `rotation_conformers_<shard>.npz` to match the `--layout by-shard` reader.

## Cost

52,391 conformers on one A100-80GB: 44 min at `--batch-size 8` (19.8 conformers/s) on a busy
machine. Peak GPU memory is a few GB at `--batch-size 8`; a handful of large molecules spike
much higher, so on a shared GPU use a small `--batch-size` and rely on `--oom-retries`.
`--hydrogens keep` makes the quadruplet index sets explode and can exhaust an 80 GB GPU.
