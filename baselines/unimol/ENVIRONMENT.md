# Uni-Mol — environment

`extract_chirality.py` drives an unmodified Uni-Mol checkout: it writes the LMDB that
`unimol.tasks.UniMolTask` reads, parses the same command line that `unimol/infer.py` parses,
and runs the encoder forward pass, keeping the `[CLS]` row. No third-party code and no model
weights are redistributed here.

## Upstream

| item | source | pin |
|---|---|---|
| Uni-Core | <https://github.com/dptech-corp/Uni-Core> | commit `ace6fae1c8479a9751f2bb1e1d6e4047427bc134` |
| Uni-Mol | <https://github.com/deepmodeling/Uni-Mol> | commit `90f52c41299a1a582da0f9765e9f87aa21faa16a`, the `unimol/` project |

## Weights and dictionary

| file | source | sha256 |
|---|---|---|
| `mol_pre_no_h_220816.pt` (190,540,187 bytes) | <https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/mol_pre_no_h_220816.pt> | `da27196af09a8c6d089e10b7764b6a716bcc33da227fc118f5b45b0e484585e9` |
| `dict.txt` | `Uni-Mol/unimol/example_data/molecule/dict.txt` of the checkout | `94135cb9a9198f988de684cb61e2c372882a3bd59b8320effbae704c38057127` |

## Install

Tested on Linux x86-64, one NVIDIA A100 80GB, CUDA 12.x driver.

```bash
conda create -y -p ./env python=3.10
./env/bin/python -m pip install --upgrade pip setuptools wheel
./env/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
./env/bin/python -m pip install \
    "numpy==1.26.4" "pandas==2.2.3" "scipy==1.14.1" "rdkit==2024.03.6" "lmdb==1.5.1" \
    tqdm ml_collections tensorboardX tokenizers iopath scikit-learn
./env/bin/python -m pip install datasets        # only for --dataset hf:... / a save_to_disk dir

git clone https://github.com/dptech-corp/Uni-Core.git
git -C Uni-Core checkout ace6fae1c8479a9751f2bb1e1d6e4047427bc134
./env/bin/python -m pip install --no-build-isolation --no-deps ./Uni-Core

git clone https://github.com/deepmodeling/Uni-Mol.git
git -C Uni-Mol checkout 90f52c41299a1a582da0f9765e9f87aa21faa16a
./env/bin/python -m pip install --no-build-isolation --no-deps -e ./Uni-Mol/unimol

curl -L -o mol_pre_no_h_220816.pt \
  https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/mol_pre_no_h_220816.pt
```

`Uni-Core`'s `setup.py` builds its fused CUDA kernels only with `--enable-cuda-ext`. The
install above uses the PyTorch reference implementations of layer norm and softmax-dropout,
which is what the numbers in [`../README.md`](../README.md) were produced with. The full
resolved environment is in `requirements.txt`.

## Run

```bash
python extract_chirality.py \
    --dataset hf:EscheWang/3dcs:chirality \
    --unimol-repo /path/to/Uni-Mol/unimol \
    --weights /path/to/mol_pre_no_h_220816.pt \
    --out chirality_unimol.npz \
    --batch-size 256 --device cuda:0 --verify
```

`--dataset` takes the input specification shared by every script in `baselines/` (see the
table in [`../README.md`](../README.md#running-one)): `hf:<repo>[:<config>]`, `hfdisk:<dir>` or a plain
`save_to_disk` directory, a bare Hub dataset id, a pickle of RDKit molecules, or `lmdb:<file>`.
`--limit`/`--start` run a slice of the conformers and `--verify-rows` says which rows of the
reference that slice covers.

Other options: `--dict`, `--num-workers`, `--seed`, `--work-dir`, `--keep-work`.

The script prints the versions of Python, PyTorch, Uni-Core, NumPy, RDKit and LMDB, the GPU
name, the sha256 of the weights and of `dict.txt`, the full Uni-Mol command line it parses,
and the shape and sha256 of the output.
## Settings

The command line the script parses is
`--only-polar 0 --conf-size 1 --random-token-prob 0 --leave-unmasked-prob 1.0 --mode infer`
on `unimol_base` with `mol_pre_no_h_220816.pt`, fp32, no `--fp16`, and the embedding is
`encoder_rep[:, 0, :]` (the `[CLS]` token of the last encoder layer) with no normalisation.
With `--random-token-prob 0 --leave-unmasked-prob 1.0`, Uni-Mol's `MaskPointsDataset`
computes `num_mask == 0`, so no token is replaced and no coordinate noise is added, and the
pipeline draws no effective randomness: `--seed` does not change the output. `LMDBDataset`
looks records up by `str(idx)` and the iterator runs with `shuffle=False`, so the row order
of the input is preserved.

## Input precision

The published embedding file and the agreement numbers in [`../README.md`](../README.md) were
computed from the source RDKit molecules, whose coordinates carry full float precision. The public
`EscheWang/3dcs` dataset stores those geometries as V2000 MOL blocks, which hold four decimals.
Running the script on the full-precision molecules instead of the MOL blocks gives a mean per-row
cosine of 0.9999994 between the two runs, with 62 conformers of 52,391 below 0.9999 and a worst
case of 0.9945 (a 65-atom molecule); each chirality metric moves by less than 0.0002.

## Cost

52,391 conformers on one A100 80GB at `--batch-size 256`: 64 s for the forward pass from a
pickle input, 159 s when the input is the Hugging Face config (the MOL blocks are parsed by
RDKit first), plus about a minute to write the LMDB.
