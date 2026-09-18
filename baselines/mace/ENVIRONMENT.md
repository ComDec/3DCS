# MACE — environment

`extract_chirality.py` calls `mace-torch` directly. No third-party code or weights are
redistributed here: `mace_mp()` fetches the foundation model from the upstream GitHub
release and caches it under `~/.cache/mace/`, and the script reports what it used.

## Upstream

| item | source | pin |
|---|---|---|
| MACE | <https://github.com/ACEsuit/mace> (MIT) | `mace-torch==0.3.15` on PyPI |

## Weights

`mace_mp()` downloads these; the sha256 below were computed from the downloaded files.

| `--model` | file | URL | bytes | sha256 |
|---|---|---|---|---|
| `medium` (default) | `2023-12-03-mace-128-L1_epoch-199.model` | <https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model> | 44,422,970 | `01bfe22100139f424713cf921144e5509cbe353d67aa9fa1be9c6e1e0ed35845` |
| `small` | `2023-12-10-mace-128-L0_energy_epoch-249.model` | <https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model> | 32,581,838 | `2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736` |
| `large` | `MACE_MPtrj_2022.9.model` | <https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/MACE_MPtrj_2022.9.model> | 133,803,220 | `f80e992b65ab8f88fdf26964511357c022e92704e4d9bcd086652635a8495b32` |
| `medium-mpa-0` | `mace-mpa-0-medium.model` | <https://github.com/ACEsuit/mace-foundations/releases/download/mace_mpa_0/mace-mpa-0-medium.model> | 79,462,305 | `75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638` |

`mace-torch` strips punctuation from the cached file name, e.g.
`~/.cache/mace/20231203mace128L1_epoch199model`. Descriptor width per model (invariants of
all interaction layers, concatenated): `small` 256, `medium` 256, `large` 512,
`medium-mpa-0` 256. The MACE-MP-0 weights are released by the MACE authors under the terms
stated in that repository; check them before redistributing.

## Install

```bash
conda create -y -p ./env python=3.11
./env/bin/pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu124
./env/bin/pip install -r requirements.txt
./env/bin/pip install datasets          # only for --dataset hf:... / hfdisk:...
```

Resolved versions used for the numbers in [`../README.md`](../README.md): python 3.11,
torch 2.5.1+cu124, mace-torch 0.3.15, e3nn 0.4.4, ase 3.29.0, numpy 1.26.4, rdkit 2024.09.6,
scipy 1.17.1, opt-einsum 3.4.0, torchmetrics 1.9.0. Box driver 595.71.05.

## Run

```bash
python extract_chirality.py \
    --dataset hf:EscheWang/3dcs:chirality \
    --out chirality.npz \
    --device cuda --batch-size 1 --compress --verify
```

`--dataset` also accepts a local pickle of RDKit molecules or `hfdisk:<save_to_disk dir>`.
Other switches: `--model`, `--aggregation mean|sum`, `--num-layers`, `--full-features`,
`--conf-id`, `--dtype`, `--limit`. The script prints the versions of every numerically
relevant package and the sha256 of the file it wrote.

## Settings

`mace_mp(model="medium")` (MACE-MP-0 medium, 128 channels, two interaction layers) ->
`get_descriptors(invariants_only=True, num_layers=-1)`, which concatenates the l = 0 part of
the node features of both layers (256 values per atom) -> mean over **all** atoms, hydrogens
included, conformer 0, float32 -> `np.savez_compressed` under `arr_0`, input row order kept.
Being a Materials-Project model it covers every element in the benchmark, including the
Se / Si / B / As / Na / K / Ca / Mg / Zn atoms that appear in 459 of the 52,391 conformers.

`--batch-size 1` runs one conformer per forward pass and is the default. Larger batches are
faster but reorder the scatter reductions, so results can differ in the last float32 digits
(measured: at most 1.4e-7 on a 128-conformer sample, the same order as the difference
between CPU and GPU). float32 and float64 runs agree to ~1e-7, the float32 spacing at these
magnitudes.

## Input precision

The `mol_blocks` of the Hugging Face config store coordinates with four decimals. Atom order
and symbols are identical to the source pickle and coordinates differ by at most 5e-05 A;
embeddings extracted through the Hugging Face path differ from the published file by at most
3.78e-05, per-row cosine at least 0.9999998.

## Cost

52,391 conformers at `--batch-size 1`: 5,378 s on an A100 80GB shared with three other jobs
(11.8 GB of GPU memory), or about 24 minutes split over 24 CPU processes at
`--batch-size 16`.
