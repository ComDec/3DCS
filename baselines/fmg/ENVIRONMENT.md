# FMG — environment

`extract_chirality.py` is a wrapper. It imports `utils.align`,
`utils.create_gaussian_batch_pdf_values` and
`denoising_diffusion_pytorch.classifier_free_guidance.Unet3D` from a clone of the upstream
FMG repository and loads a checkpoint obtained from the FMG authors. Neither the FMG source
nor the FMG weights are redistributed here.

> Dumitrescu, Korpela, Heinonen, Verma, Iadarola, Marttinen, Garg.
> *E(3)-equivariant models cannot learn chirality: Field-based molecular generation.*
> ICLR 2025.

## Upstream

| item | source | pin |
|---|---|---|
| FMG | <https://github.com/Dumitrescu-Alexandru/FMG> | commit `13a0a7cc331d136a4028eda80c8b55b0adfd58c3` (2025-04-22) |

## Weights

| field | value |
|---|---|
| file | `model-120qm9_3rd_run.pt` (FMG QM9 3D U-Net) |
| size | 1,246,683,351 bytes |
| sha256 | `f55ec38f2b6c20ad3a2e4e6287efb77af3901d46548449543bdbab33357d2afa` |
| source | the FMG authors (the upstream repository does not host it) |

The script reads the 226 tensors of the `ema` state dict whose names start with
`online_model.model.`.

## Install

```bash
git clone https://github.com/Dumitrescu-Alexandru/FMG.git
git -C FMG checkout 13a0a7cc331d136a4028eda80c8b55b0adfd58c3

conda create -y -p ./env python=3.10
./env/bin/pip install "pip==24.0" "setuptools==67.8.0" wheel
./env/bin/pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
./env/bin/pip install -r requirements.txt
```

Resolved versions used for the numbers in [`../README.md`](../README.md): python 3.10.21,
torch 2.0.0+cu118 (cuDNN 8700), numpy 1.26.4, rdkit 2024.9.6, einops 0.7.0, datasets 4.0.0.

RDKit must be 2023 or newer to read the released conformer pickles (they carry RDKit pickle
version 16); the `rdkit-pypi==2022.9.5` of the upstream `requirements.txt` raises
`Bad pickle format: ENDMOL tag not found`. The remaining pins in `requirements.txt` are
transitive dependencies of `import utils` in the FMG checkout.

## Run

```bash
python extract_chirality.py \
    --fmg-repo ./FMG \
    --checkpoint ./model-120qm9_3rd_run.pt \
    --dataset EscheWang/3dcs \
    --out chirality_fmg.npz \
    --batch-size 32 --device cuda:0 --verify
```

`--dataset` takes the `chirality` config of `EscheWang/3dcs` (a Hub id or a `save_to_disk`
directory), or a pickle holding a list of RDKit molecules with one conformer each. Both
forms give rows in the order of the published embedding file; each row's `offset` is
checked against the running conformer count.

The output `.npz` holds `embeddings` (float32, N x 128) and `smiles`. The `smiles` array is
metadata written by the RDKit build in use: 108 of the 52,391 entries are written
differently by rdkit 2024.9.6 than in the published file (all of them `[O][Na]` against
`O[Na]`), and all 108 re-canonicalise to the same molecule.

## Determinism

The FMG forward pass runs in float32 with TF32 convolutions enabled by default, so two runs
of the same code on the same machine differ by up to ~6e-4 per element. `--deterministic`
forces deterministic cuDNN kernels and disables TF32; it is slower and moves the output
further from the published file, which was computed with the library defaults.

## Input precision

The `mol_blocks` of the Hugging Face config store coordinates with four decimals. FMG aligns
each conformer onto its PCA axes, and for 216 of the 52,391 conformers (0.41 %) that
rounding is enough to swap near-degenerate principal axes, which rotates the field (lowest
per-row cosine 0.9931 against the run from the full-precision molecules). The remaining
99.6 % agree to the 1e-6 level, and the chirality metrics move by at most 5.3e-4. Start from
a pickle of the molecules for the closest agreement.

## Cost

52,391 conformers on one A100-80GB at `--batch-size 32`: 89 s to read the molecules and
510 s for the forward pass (102.8 conformers/s). Peak GPU memory 1.79 GiB allocated, 9.93
GiB reserved by the caching allocator.
