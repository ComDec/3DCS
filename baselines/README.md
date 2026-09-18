# Baseline embedding extraction

One script per baseline model that turns the 3DCS chirality conformers into the embedding
matrix the evaluator reads. Each script is standalone, runs in its own model environment and
writes a file in the same layout as the corresponding published embedding in
[`EscheWang/3dcs-embeddings`](https://huggingface.co/datasets/EscheWang/3dcs-embeddings).

```
baselines/
  common.py             input loading, npz writing, and the --verify comparison
  <model>/
    extract_chirality.py
    ENVIRONMENT.md      upstream repository + commit, weight file + sha256 + download
                        location, exact install commands, run command, cost
    requirements.txt    pinned Python environment
```

No third-party source code and no model weights are redistributed here. Every script drives
an upstream checkout or package that you install yourself, and `ENVIRONMENT.md` says which
commit and which weight file, with its SHA-256 and where it comes from.

## What each script produces

| Model | Dim | Output key | Hydrogens | Pooling | Upstream dependency |
|---|---|---|---|---|---|
| [E3FP](e3fp/) | 1024 bits | `e3fp` (pickle) | kept | — (fingerprint) | `e3fp==1.2.7` |
| [GemNet-Q](gemnet/) | 128 | `gemnet` | `RemoveAllHs`, heavy atoms only | mean over atoms | `TUM-DAML/gemnet_pytorch` @ `a0164f7` + its `pretrained/GemNet-Q` |
| [Uni-Mol](unimol/) | 512 | `arr_0` | removed (no-H checkpoint) | `[CLS]` token | `deepmodeling/Uni-Mol` @ `90f52c4`, `dptech-corp/Uni-Core` @ `ace6fae`, `mol_pre_no_h_220816.pt` |
| [Mol-AE](molae/) | 512 | `arr_0` | removed (no-H checkpoint) | `[CLS]` token | same Uni-Mol / Uni-Core, Mol-AE `checkpoint_7_1000000.pt` |
| [MolSpectra](molspectra/) | 256 | `arr_0` | removed (`Z != 1`) | sum over atoms | `AzureLeon1/MolSpectra` @ `8846530` + a checkpoint you supply |
| [MACE](mace/) | 256 | `arr_0` | kept | mean over atoms | `mace-torch==0.3.15`, MACE-MP-0 `medium` |
| [FMG](fmg/) | 128 | `embeddings` | dropped (C, O, N, F channels only) | mean over the 24³ grid | `Dumitrescu-Alexandru/FMG` @ `13a0a7c` + `model-120qm9_3rd_run.pt` |

Every file holds one row per conformer, in the row order of the `chirality` config of
[`EscheWang/3dcs`](https://huggingface.co/datasets/EscheWang/3dcs) (52,391 rows). E3FP writes
a pickle of RDKit bit vectors; the others write an `.npz` with a single float32 array.

## Running one

```bash
# in the environment of that model, see <model>/ENVIRONMENT.md
python baselines/mace/extract_chirality.py \
    --dataset hf:EscheWang/3dcs:chirality \
    --out chirality_mace.npz \
    --device cuda --batch-size 1 --compress --verify
```

The command for each of the seven, with the flags that model needs, is under
[One command per model](#one-command-per-model).

Common to all of them:

- `--dataset` takes the same input specification in every script, resolved by
  `common.py` (`parse_dataset_spec`):

  | `--dataset` | what it reads | needs |
  |---|---|---|
  | `hf:EscheWang/3dcs:chirality` | that Hub dataset and that config | `datasets` |
  | `hf:EscheWang/3dcs`, `EscheWang/3dcs` | the same repository, config `chirality` | `datasets` |
  | `hfdisk:data/hf/chirality`, `data/hf/chirality` | a `save_to_disk` directory of that config | `datasets` |
  | `conformers.pkl` | a pickle of RDKit molecules: a list, or a dict of lists concatenated in insertion order | — |
  | `lmdb:rotation_conformers_0.lmdb` | a rotation shard: one list of `(Mol, energy, torsion)` per key | `lmdb` |

  Rows of the Hugging Face config are read in ascending `offset`, and each row's `offset` is
  checked against the running conformer count, so `mol_blocks` concatenated that way is the
  row order of the published embedding files. `python -m three_dbench download dataset --task
  chirality` writes the `save_to_disk` directory that the fourth form reads.
- `--limit N` stops after N conformers and `--start N` skips the first N, so a slice can be
  run first; `--verify` then compares that slice (see `--verify-rows`).
- `--verify [REFERENCE]` compares the file just written with a reference: with no argument,
  the published embedding of that model, downloaded from `EscheWang/3dcs-embeddings`
  (`pip install huggingface_hub`); otherwise `hub:<path in that repository>` or a local path.
  It prints both checksums, the elementwise differences, the per-row cosine similarity and
  the number of rows outside a few thresholds. It does not change the exit status.
- `--verify-rows` says which rows of the reference the output covers. The default compares a
  short output with the first rows of the reference (`prefix`) and a full output row by row
  (`full`); `--verify-rows 2000:4000` (or `2000+`) compares against that slice, and
  `--verify-rows @rows.npy` against the listed 0-based reference rows. The report then names
  the selection and how many rows it compared.
- Each script prints the versions of every numerically relevant library, its own arguments,
  the SHA-256 of the weights it loaded and the SHA-256 of the file it wrote.
- Rows are never skipped: a conformer that cannot be featurised is an error, not a silent
  shift of the row order.

## One command per model

Every command below reads the published dataset and verifies against the published
embedding. The paths in angle brackets are the checkout, weight file and dictionary that
`<model>/ENVIRONMENT.md` gives the download command and SHA-256 for. Add `--limit 2000` to
any of them for a first run over a slice.

```bash
# E3FP            (CPU only)
python baselines/e3fp/extract_chirality.py \
    --dataset hf:EscheWang/3dcs:chirality \
    --out sampled_chi.pkl --jobs 24 --verify

# MACE
python baselines/mace/extract_chirality.py \
    --dataset hf:EscheWang/3dcs:chirality \
    --out chirality.npz \
    --device cuda --batch-size 1 --compress --verify

# Uni-Mol
python baselines/unimol/extract_chirality.py \
    --dataset hf:EscheWang/3dcs:chirality \
    --unimol-repo <Uni-Mol>/unimol \
    --weights <mol_pre_no_h_220816.pt> \
    --out chirality_unimol.npz \
    --batch-size 256 --device cuda:0 --verify

# Mol-AE
python baselines/molae/extract_chirality.py \
    --dataset hf:EscheWang/3dcs:chirality \
    --weights <checkpoint_7_1000000.pt> \
    --unimol-dir <Uni-Mol>/unimol/unimol \
    --dict <Uni-Mol>/unimol/example_data/molecule/dict.txt \
    --out molae_chirality.npz \
    --batch-size 256 --device cuda:0 --num-workers 8 --verify

# GemNet-Q
python baselines/gemnet/extract_chirality.py \
    --dataset hf:EscheWang/3dcs:chirality \
    --gemnet-repo <gemnet_pytorch> \
    --out gemnet_chirality.npz \
    --batch-size 8 --device cuda --checkpoint-every 8000 --verify

# FMG
python baselines/fmg/extract_chirality.py \
    --dataset hf:EscheWang/3dcs:chirality \
    --fmg-repo <FMG> \
    --checkpoint <model-120qm9_3rd_run.pt> \
    --out chirality_fmg.npz \
    --batch-size 32 --device cuda:0 --verify

# MolSpectra      (architecture only; the checkpoint is supplied by the caller)
python baselines/molspectra/extract_chirality.py \
    --dataset hf:EscheWang/3dcs:chirality \
    --repo <MolSpectra> \
    --checkpoint <denoised-pcqm4mv2.ckpt> \
    --out molspectra_chirality.npz \
    --batch-size 128 --device cuda \
    --arch torchmdnet --hydrogens remove --pool add --verify
```

The output goes straight into the evaluator:

```bash
python -m three_dbench evaluate chirality \
    --dataset-dir data/hf/chirality \
    --embeddings chirality_mace.npz --embedding-key arr_0 \
    --model-name mace --output-dir out/ \
    --distance euclidean --metric-version paper --unsup-kmax n-1
```

## Agreement with the published embeddings

**Which input.** The published embedding files were computed from the source RDKit molecules,
whose coordinates carry full float precision. The public `EscheWang/3dcs` dataset stores the
same geometries as V2000 MOL blocks, which hold four decimals, so a run that starts from the
dataset starts up to 5e-5 A away from the coordinates behind the published file. The first
table below is the comparison from the full-precision molecules, the second is what the
published dataset gives; [Input precision](#input-precision) is why they differ.

### From the full-precision molecules, all 52,391 conformers

Each script was run over all 52,391 conformers and the result compared with the published
file of that model. "Metrics" is the largest absolute difference over the six chirality
metrics (ES-AUC, NN@1-Acc, SCI, SCI_unsup, Hopkins, DBI) computed by `three_dbench evaluate
chirality --distance euclidean --metric-version paper` from the published file and from the
regenerated file.

| Model | Bytes identical | Per-row cosine | max abs diff | mean abs diff | Metrics (max abs diff) |
|---|---|---|---|---|---|
| E3FP | yes, all 52,391 fingerprints | — (identical bits) | — | — | 0 |
| GemNet-Q | no | min 0.99923; 52,390 of 52,391 rows ≥ 0.9999996 | — | 3.3e-04 | 1.6e-04 |
| Uni-Mol | no | min 0.9999999995 | 1.18e-04 | 1.01e-06 | 1.42e-05 |
| Mol-AE | no | min 0.9999997616 | 7.77e-04 | 5.72e-07 | 8.3e-05 |
| MolSpectra | no | mean −0.18 | — | — | 5.5e-02 |
| MACE | no | min 0.9999997634 | 4.90e-03 | 2.42e-08 | 8.9e-07 |
| FMG | no | min 0.99999970 | 6.00e-04 | 6.60e-06 | 6.6e-05 |

Notes on individual rows:

- **E3FP** is computed on the CPU from integer bit operations and is reproducible exactly:
  `baselines/e3fp/extract_chirality.py` run over the full-precision RDKit molecules reproduces
  all 52,391 fingerprints of the published file bit for bit (`identical_fraction 1`,
  `tanimoto_min 1`; e3fp 1.2.7, rdkit 2026.03.6, 214 s with `--jobs 24`). The pickle itself is
  not byte-identical, because the container is rewritten. Evaluating the regenerated file gives
  ES-AUC 0.485935, NN@1-Acc 0.177939, SCI −0.012543 and SCI_unsup 0.033825, equal at six
  decimals to the reference values in `reproduce/table2_chirality/expected.csv` (Hopkins is not
  defined for fingerprints). Starting from the four-decimal MOL blocks of the Hugging Face
  dataset instead, 2,846 of 3,000 sampled fingerprints match.
- **GemNet-Q**: the one row below cosine 0.9999996 (row 49740) is a structure of seven
  disconnected fragments including a lone `[H]` and several radical carbons.
- **MACE**: the largest absolute difference comes from two rows of norm 1,743 and 15,228,
  where the float32 spacing is already O(1e-2); no row has a cosine below 1 − 1e-6.
- **MolSpectra**: the script computes the documented quantity — the per-atom scalar
  representation of the MolSpectra equivariant Transformer summed over the atoms of the
  molecule — but the MolSpectra authors publish no checkpoint, so it has to be supplied by
  the caller. The row above used the public `denoised-pcqm4mv2.ckpt` that MolSpectra
  initialises from; with that checkpoint the output is a MolSpectra-architecture embedding of
  the same shape, not a copy of the published file. The published
  `chirality/molspectra/sampled_mol_feature.npz` is the artifact the paper's MolSpectra
  values were computed from. Its entry in the table above is the largest difference over the
  five metrics of `reproduce/table2_chirality/expected.csv`; DBI was not recorded for that run.
- **FMG**: the `smiles` side array of the output is metadata written by the RDKit build in
  use; 108 of 52,391 entries are written differently by rdkit 2024.9.6 than in the published
  file (`[O][Na]` against `O[Na]`), and all 108 re-canonicalise to the same molecule.

### From the published `mol_blocks`, first 2,000 conformers

Each script over the first 2,000 conformers read from the published `mol_blocks`
(`--dataset hf:EscheWang/3dcs:chirality --limit 2000 --verify`, or the `hfdisk:` form of a
local copy of the same config), compared with the first 2,000 rows of the published file of
that model — what `--verify` prints for a run an outside reader can reproduce from the
released dataset. One A100-80GB, each model in the environment its `ENVIRONMENT.md` builds.

| Model | Per-row cosine | max abs diff | mean abs diff |
|---|---|---|---|
| E3FP | 1,925 of 2,000 fingerprints bit-identical; Tanimoto mean 0.9946, min 0.511 | — | — |
| GemNet-Q | mean 0.99999989, min 0.99999966; 0 rows < 1 − 1e−6 | 4.98e-03 | 3.35e-04 |
| Uni-Mol | mean 0.9999965, min 0.9945; 32 rows < 1 − 1e−6 | 3.85e-01 | 2.44e-04 |
| Mol-AE | mean 0.9999948, min 0.9921; 56 rows < 1 − 1e−6 | 4.36e-01 | 3.16e-04 |
| MolSpectra | mean −0.184 (a different checkpoint, see the note above) | 3.76e+02 | 1.95e+01 |
| MACE | mean 0.9999999995, min 0.9999999906; 0 rows < 1 − 1e−6 | 1.57e-03 | 1.33e-06 |
| FMG | mean 0.9999861, min 0.9946; 9 rows < 0.999 | 7.02e-01 | 3.10e-04 |

Over those 2,000 conformers (163 molecules), the chirality metrics computed from the
regenerated file differ from the ones computed from the same rows of the published file by at
most 1.9e-03 (ES-AUC), 2.4e-03 (NN@1-Acc), 1.8e-02 (DBI), 1.9e-04 (Hopkins) and 2.9e-03
(SCI_unsup), taking the largest value over MACE, Uni-Mol, Mol-AE, GemNet-Q and FMG. A run from
a pickle of the full-precision molecules gives the first table instead.

None of the neural files are byte-identical, and none are expected to be: the forward passes
run in float32 on the GPU, where the reduction order depends on the batch size, the library
build and the device. Two runs of the same script on the same machine measure the size of
that effect — for FMG, two runs with identical settings differ by up to 6e-4 per element, the
same size as the difference from the published file; for MACE, batch size 1 against 16 moves
the output by at most 1.4e-7 on a 128-conformer sample; for GemNet, TF32 on against off moves
it by 2.0e-3.

## Input precision

The `mol_blocks` of the Hugging Face dataset are V2000 MDL molblocks, which store coordinates
with four decimals (at most 5e-5 A from the values in the source RDKit molecules the published
files were computed from). Models that build a neighbour graph with a hard cutoff, that encode
pair distances, or that align a molecule onto its principal axes, can react to that rounding on
a small fraction of conformers. Measured per model, running from the MOL blocks against running
from the full-precision molecules:

| Model | rows the rounding moves | reference |
|---|---|---|
| Mol-AE | 2.8 % of rows below cosine 1 − 1e−6, lowest 0.9921 | [molae/ENVIRONMENT.md](molae/ENVIRONMENT.md#input-precision) |
| MolSpectra | 1.2 % of rows below cosine 0.9999 | [molspectra/ENVIRONMENT.md](molspectra/ENVIRONMENT.md#input-precision) |
| FMG | 0.4 % of rows (PCA axis swaps), lowest cosine 0.9931 | [fmg/ENVIRONMENT.md](fmg/ENVIRONMENT.md#input-precision) |
| Uni-Mol | 0.1 % of rows below cosine 0.9999, lowest 0.9945 | [unimol/ENVIRONMENT.md](unimol/ENVIRONMENT.md#input-precision) |
| E3FP | 2,846 of 3,000 sampled fingerprints bit-identical | [e3fp/ENVIRONMENT.md](e3fp/ENVIRONMENT.md#input-precision) |
| GemNet-Q | no row below cosine 1 − 1e−6; max abs diff 5.1e-04 | [gemnet/ENVIRONMENT.md](gemnet/ENVIRONMENT.md#input-precision) |
| MACE | max abs diff 3.8e-05, per-row cosine at least 0.9999998 | [mace/ENVIRONMENT.md](mace/ENVIRONMENT.md#input-precision) |

ES-AUC, NN@1-Acc, SCI, SCI_unsup and Hopkins then move by less than 5.3e-4 in every case; DBI, a
ratio that is heavy-tailed over molecules, moves more (up to 4.4e-2 for FMG, where a single
molecule of 3,903 accounts for most of the shift). The source molecules are not part of the
release; a run from the dataset is the reproducible path, and the numbers above are its cost.

## Other tracks

The chirality set is what these scripts cover. For the rotation track, the GemNet script also
reads the per-shard LMDB inputs and writes one file per shard in the layout the
`--layout by-shard` reader expects; see `gemnet/ENVIRONMENT.md`. The trajectory (rMD17)
embeddings are published in `EscheWang/3dcs-embeddings` and are not regenerated by anything
here.
