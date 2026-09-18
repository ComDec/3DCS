# MolSpectra — environment

`extract_chirality.py` needs (a) a clone of the upstream MolSpectra repository, (b) an
equivariant-Transformer checkpoint with `embedding_dimension = 256`, and (c) the environment
below. No third-party code or weights are redistributed here.

The MolSpectra authors publish their QM9S dataset but no pre-trained checkpoint, so the
weights have to be supplied by the caller. The output depends entirely on which checkpoint is
used: with a checkpoint of your own this script produces a MolSpectra-architecture embedding,
not a copy of the published file. The published
`chirality/molspectra/sampled_mol_feature.npz` in
[`EscheWang/3dcs-embeddings`](https://huggingface.co/datasets/EscheWang/3dcs-embeddings) is
the file the paper's MolSpectra values were computed from.

## Upstream

| item | source | pin |
|---|---|---|
| MolSpectra reference implementation | <https://github.com/AzureLeon1/MolSpectra> | commit `8846530e573a1dc2834eab738d20278b0c90e601` (2025-04-19, `main`) |
| architecture | TorchMD-NET equivariant Transformer, as forked by MolSpectra | — |
| checkpoint used for the runs reported in [`../README.md`](../README.md) | `checkpoints/denoised-pcqm4mv2.ckpt` of <https://github.com/shehzaidi/pre-training-via-denoising> | sha256 `f6b387ba3632e03d273939257969161b2615a7fbf41e9f514fd25e1b5d345e66`, 86,684,181 B, epoch 25 / step 400000 |

That checkpoint's `hyper_parameters` are the architecture MolSpectra pre-trains with:
`model=equivariant-transformer`, `embedding_dimension=256`, `num_layers=8`, `num_rbf=64`,
`rbf_type=expnorm`, `trainable_rbf=False`, `cutoff_lower=0.0`, `cutoff_upper=5.0`,
`max_z=100`, `max_num_neighbors=32`, `num_heads=8`, `distance_influence=both`,
`activation=silu`, `attn_activation=silu`, `neighbor_embedding=True`,
`layernorm_on_vec=whitened`, `reduce_op=add`.

## Install

```bash
uv venv --python 3.10 ./env
source ./env/bin/activate
uv pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install torch_scatter==2.1.2 torch_cluster==1.6.3 \
    -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
uv pip install -r requirements.txt
git clone https://github.com/AzureLeon1/MolSpectra.git upstream
git -C upstream checkout 8846530e573a1dc2834eab738d20278b0c90e601
```

Verified versions on an A100 80GB (driver 595.71.05): python 3.10.21, torch 2.3.1+cu121,
torch_scatter 2.1.2+pt23cu121, torch_cluster 1.6.3+pt23cu121, torch_geometric 2.6.1,
numpy 1.26.3, rdkit 2024.03.5.

Differences from the upstream `requirements.txt`, and why:

* `pytorch_lightning==2.3.3` instead of `1.3.8`. PL 1.3.8 pins `pyyaml<=5.4.1`, which no
  longer builds against Cython 3. PL is only needed to unpickle a Lightning checkpoint; the
  extraction instantiates `torchmdnet.models.torchmd_et.TorchMD_ET` directly and never
  touches the Lightning `LightningModule`.
* `setuptools<81` is required by `lightning_fabric`, which still calls `pkg_resources`.
* `rdkit`, `scipy` and `scikit-learn` are not in the upstream `requirements.txt`; they are
  needed to read the conformers and to run the 3DCS evaluator.

## Run

```bash
python extract_chirality.py \
  --dataset  chirality_bench_conformers_noised_only_aslist.pkl \
  --repo     ./upstream \
  --checkpoint ./denoised-pcqm4mv2.ckpt \
  --out      molspectra_chirality.npz \
  --batch-size 128 --device cuda \
  --arch torchmdnet --hydrogens remove --pool add
```

`--dataset` also accepts a `save_to_disk` directory of `EscheWang/3dcs` config `chirality`.

## What the flags mean

* `--arch molspectra` builds MolSpectra's `TorchMD_ET`, which inserts a per-layer
  `x_norms` / `vec_norms` pair that upstream TorchMD-NET does not have. A checkpoint that
  predates MolSpectra has no weights for those 24 tensors and they stay at LayerNorm init,
  which changes the function. `--arch torchmdnet` disables them (`use_dataset_md17=True`), so
  an upstream checkpoint loads with 0 missing tensors. Use `molspectra` with a MolSpectra
  checkpoint and `torchmdnet` with an upstream one; the script prints the missing-key count,
  so a mismatch is visible.
* `--hydrogens remove` (default) drops every atom with `Z == 1` before the forward pass;
  `keep` feeds the conformer as stored. The chirality conformers carry explicit hydrogens
  (75.0 atoms per conformer, 38.6 of them heavy).
* `--pool add` (default) is MolSpectra's `reduce_op: add`.

## Determinism

The configuration is fixed by the CLI flags, but the GPU forward pass is not bit-reproducible:
`scatter(..., reduce="add")` and the attention reductions use atomics, so two runs of the
identical command differ in the last float32 bits. Measured over two runs of the same
command: largest absolute element difference 2.7e-3 on values up to 377 (~7e-6 relative to
the row norm), per-row cosine at least 0.999999999. Use `--device cpu` for a bit-reproducible
output.

## Input precision

The `mol_blocks` of the Hugging Face config store coordinates with four decimals. For a
model with a 5 A cutoff that rounding can add or drop an edge: 1.19 % of rows have a per-row
cosine below 0.9999 between a run from the MOL blocks and a run from the full-precision
pickle of the same conformers. This applies to every cutoff-graph model here.

## Cost

119 s for 52,391 conformers on an otherwise idle A100 80GB (~440 conformers/s), about 290 s
when the GPU is shared. Under 3 GB of GPU memory at `--batch-size 128`.
