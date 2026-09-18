# E3FP — environment

`extract_chirality.py` needs only `e3fp` and RDKit. There are no model weights.

## Upstream

| item | source | version |
|---|---|---|
| E3FP | <https://github.com/keiserlab/e3fp> (LGPL-3.0) | `e3fp==1.2.7` on PyPI |

## Install

```bash
conda create -y -p ./env python=3.10
./env/bin/pip install "e3fp==1.2.7" "rdkit==2024.9.6" "numpy<2"
# only for --dataset hf:... / hfdisk:...
./env/bin/pip install datasets
# only for --verify
./env/bin/pip install huggingface_hub
```

`e3fp` pulls in `sdaxen_python_utilities`, `smart_open` and `mmh3`.

## Run

```bash
python extract_chirality.py \
    --dataset chirality_bench_conformers_noised_only_aslist.pkl \
    --out sampled_chi.pkl --jobs 24 --verify
```

`--dataset` also takes `hf:EscheWang/3dcs:chirality` or a `save_to_disk` directory of that
config.

## Fingerprint parameters

```
bits=1024, level=5, radius_multiplier=1.5, stereo=True,
include_disconnected=True, rdkit_invariants=True, first=1, counts=False
```

Hydrogens are kept. These are the parameters of the published
`chirality/e3fp/sampled_chi.pkl`; the `e3fp` defaults (`bits=4096`,
`radius_multiplier=1.718`, `rdkit_invariants=False`) give different bits.

## Agreement with the published file

Run over the full-precision RDKit molecules with the parameters above, all 52,391
fingerprints are bit-identical to the published `chirality/e3fp/sampled_chi.pkl`
(`--verify` reports `identical 52391`, `identical_fraction 1`, `tanimoto_min 1`), with
e3fp 1.2.7 and rdkit 2026.03.6. The pickle itself is not byte-identical, because the
container is rewritten.

## Input precision

The `mol_blocks` of the Hugging Face config store coordinates with four decimals. E3FP bins
interatomic distances, so that rounding can move a shell boundary: over 3,000 sampled
conformers, fingerprints computed from the MOL blocks agree with the published file for 2,846
of 3,000. Start from a pickle of the molecules when the bits have to match exactly.

## Cost

CPU only. 52,391 conformers in 214 s with `--jobs 24` (~250 conformers/s).
