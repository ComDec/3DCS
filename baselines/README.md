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
# the environment of that model, see <model>/ENVIRONMENT.md
python baselines/mace/extract_chirality.py \
    --dataset hf:EscheWang/3dcs:chirality \
    --out chirality_mace.npz \
    --device cuda --batch-size 1 --compress --verify
```

Common to all of them:

- `--dataset` takes the Hugging Face config (`hf:EscheWang/3dcs:chirality`, or a
  `save_to_disk` directory) or a pickle of RDKit molecules with one conformer each.
- Each script prints the versions of every numerically relevant library, its own arguments,
  the SHA-256 of the weights it loaded and the SHA-256 of the file it wrote.
- `--verify [REFERENCE]` compares the file just written with a reference: with no argument,
  the published embedding of that model, downloaded from `EscheWang/3dcs-embeddings`
  (`pip install huggingface_hub`); otherwise `hub:<path in that repository>` or a local path.
  It prints both checksums, the elementwise differences, the per-row cosine similarity and
  the number of rows outside a few thresholds. It does not change the exit status.
- Rows are never skipped: a conformer that cannot be featurised is an error, not a silent
  shift of the row order.

The output goes straight into the evaluator:

```bash
python -m three_dbench evaluate chirality \
    --dataset-dir data/hf/chirality \
    --embeddings chirality_mace.npz --embedding-key arr_0 \
    --model-name mace --output-dir out/ \
    --distance euclidean --metric-version paper --unsup-kmax n-1
```

## Agreement with the published embeddings

Each script was run over all 52,391 conformers and the result compared with the published
file of that model. The numbers below are that comparison. "Metrics" is the largest absolute
difference over the six chirality metrics (ES-AUC, NN@1-Acc, SCI, SCI_unsup, Hopkins, DBI)
computed by `three_dbench evaluate chirality --distance euclidean --metric-version paper`
from the published file and from the regenerated file.

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
  values were computed from.
- **FMG**: the `smiles` side array of the output is metadata written by the RDKit build in
  use; 108 of 52,391 entries are written differently by rdkit 2024.9.6 than in the published
  file (`[O][Na]` against `O[Na]`), and all 108 re-canonicalise to the same molecule.

None of the neural files are byte-identical, and none are expected to be: the forward passes
run in float32 on the GPU, where the reduction order depends on the batch size, the library
build and the device. Two runs of the same script on the same machine measure the size of
that effect — for FMG, two runs with identical settings differ by up to 6e-4 per element, the
same size as the difference from the published file; for MACE, batch size 1 against 16 moves
the output by at most 1.4e-7; for GemNet, TF32 on against off moves it by 2.0e-3.

## Input precision

The `mol_blocks` of the Hugging Face dataset are V2000 MDL molblocks, which store coordinates
with four decimals (at most 5e-5 A from the values in the source RDKit molecules). Models
that build a neighbour graph with a hard cutoff, or that align a molecule onto its principal
axes, can react to that rounding on a small fraction of conformers: measured per model, 0.4 %
of rows for FMG (PCA axis swaps), 1.2 % for MolSpectra, 0.1 % for Uni-Mol, and a maximum
difference of 3.8e-05 for MACE. The per-model `ENVIRONMENT.md` gives the numbers. ES-AUC,
NN@1-Acc, SCI, SCI_unsup and Hopkins then move by less than 5.3e-4 in every case; DBI, a
ratio that is heavy-tailed over molecules, moves more (up to 4.4e-2 for FMG, where a single
molecule of 3,903 accounts for most of the shift).

## Other tracks

The chirality set is what these scripts cover. For the rotation track, the GemNet script also
reads the per-shard LMDB inputs and writes one file per shard in the layout the
`--layout by-shard` reader expects; see `gemnet/ENVIRONMENT.md`. The trajectory (rMD17)
embeddings are published in `EscheWang/3dcs-embeddings` and are not regenerated by anything
here.
