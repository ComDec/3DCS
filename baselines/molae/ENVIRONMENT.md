# Mol-AE — environment

`extract_chirality.py` is a wrapper around the Uni-Mol inference pipeline: it writes the
LMDB that `unimol.tasks.UniMolTask` reads, builds `unimol_base`, loads the Mol-AE
pre-trained checkpoint into it and keeps the `[CLS]` representation (`mol_repr_cls`). No
third-party code and no weights are redistributed here.

## Upstream

| component | source | pin |
|---|---|---|
| Uni-Mol (inference pipeline: `unimol` task, `unimol_infer` loss, `unimol_base` model, dataset classes) | <https://github.com/deepmodeling/Uni-Mol> | commit `90f52c41299a1a582da0f9765e9f87aa21faa16a` |
| Uni-Core (framework: options, tasks, collators, checkpoint loading) | <https://github.com/dptech-corp/Uni-Core> | commit `ace6fae1c8479a9751f2bb1e1d6e4047427bc134` |
| Mol-AE (reference implementation of the pre-training architecture; not needed at inference time) | <https://github.com/yjwtheonly/MolAE> | commit `2992f2d5862e104745dce7f4af3ca58fe366a4b6` |

## Weights and dictionary

| field | value |
|---|---|
| file | `checkpoint_7_1000000.pt` (Mol-AE pre-training) |
| size | 760,699,523 bytes |
| sha256 | `b4ca21a63799976fbf435a1c7275d5ef6e93a854cc7d90955dbaec50ef89b8c0` |
| source | the Google Drive link in the Mol-AE README (<https://github.com/yjwtheonly/MolAE>) |
| `args.arch` / `args.loss` recorded inside | `unimol_MAE_padding` / `unimol_MAE` |
| encoder / decoder | 15 layers, 512 embed, 2048 ffn, 64 heads / 5 layers, 2048 ffn, 64 heads |
| `remove_hydrogen` / `only_polar` / `dict_name` / `max_atoms` | `True` / `0` / `dict.txt` / 256 |
| `num_updates` | 1,000,000 |

The atom dictionary is Uni-Mol's `unimol/example_data/molecule/dict.txt`, sha256
`94135cb9a9198f988de684cb61e2c372882a3bd59b8320effbae704c38057127` (30 symbols; `[MASK]` is
added by the task, giving 31).

### How the encoder is loaded

The checkpoint's parameter names are a superset of Uni-Mol `unimol_base`: `embed_tokens`,
`encoder.*`, `gbf` and `gbf_proj` match one for one, while `decoder.*`, `lm_head.*`,
`dist_head.*` and `pair2coord_proj.*` are extra. Uni-Mol's inference entry point loads with
`load_state_dict(..., strict=False)`, so the Mol-AE encoder is used and the MAE decoder and
pre-training heads are dropped. The script prints the counts: 0 missing, 82 unexpected.

## Install

Built on an A100-80GB, CUDA driver 595.71.05.

```bash
conda create -y -p ./env python=3.10
./env/bin/python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
./env/bin/python -m pip install numpy==1.26.4 lmdb==1.4.1 scipy pandas tqdm \
    ml_collections tensorboardX iopath tokenizers==0.19.1 scikit-learn==1.5.2 rdkit==2025.3.6
git clone https://github.com/dptech-corp/Uni-Core.git && git -C Uni-Core checkout ace6fae
./env/bin/python -m pip install --no-build-isolation ./Uni-Core          # no CUDA extensions
git clone https://github.com/deepmodeling/Uni-Mol.git && git -C Uni-Mol checkout 90f52c4
./env/bin/python -m pip install --no-deps -e ./Uni-Mol/unimol
```

Resolved versions printed by the script at run time: python 3.10.21, numpy 1.26.4,
torch 2.4.1+cu121 (cuda 12.1), rdkit 2025.03.6, lmdb 1.4.1, unicore 0.0.1.

Uni-Core's fused CUDA kernels are optional (`setup.py` sets `DISABLE_CUDA_EXTENSION = True`
unless `--enable-cuda-ext` is passed). They were not built, so `fused_layer_norm`,
`fused_softmax`, `fused_rms_norm`, `fused_multi_tensor` and `fused_rounding` fall back to
their PyTorch implementations.

RDKit is used only to read the input molecules and to canonicalise a per-row SMILES *name*
that never reaches the model. A version new enough to unpickle the released molecule
pickles is required (RDKit pickle version 16.2, i.e. RDKit >= 2024.09).

## Run

```bash
python extract_chirality.py \
  --dataset  chirality_bench_conformers_noised_only_aslist.pkl \
  --weights  checkpoint_7_1000000.pt \
  --unimol-dir /path/to/Uni-Mol/unimol/unimol \
  --dict     /path/to/Uni-Mol/unimol/example_data/molecule/dict.txt \
  --out      molae_chirality.npz \
  --batch-size 256 --device cuda:0 --num-workers 8 --verify
```

`--dataset` also takes `hf:EscheWang/3dcs` with `--hf-config chirality`.

## Settings

`infer.py --task unimol --loss unimol_infer --arch unimol_base --only-polar 0 --conf-size 1
--random-token-prob 0 --leave-unmasked-prob 1.0 --mode infer`, output `mol_repr_cls` (the
512-d `[CLS]` vector), float32, no `--fp16`. `--leave-unmasked-prob 1.0` with
`--random-token-prob 0` makes `MaskPointsDataset` an identity (no mask, no coordinate
noise), and `valid_step` calls `model.eval()`, so the pipeline is deterministic up to
floating-point reduction order.

Row order is preserved end to end: the script writes LMDB records under the ascii keys
`"0" … "N-1"`, `LMDBDataset.__getitem__` looks a record up by `str(idx)`,
`UnicoreDataset.ordered_indices` returns `arange(N)`, and `next_epoch_itr(shuffle=False)`
keeps the batches sequential.

## Cost

84 s for 52,391 conformers on one A100-80GB (~624 conformers/s), under 2 GB of GPU memory at
`--batch-size 256`.
