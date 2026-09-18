#!/usr/bin/env python
"""Extract MACE descriptors for the 3DCS chirality benchmark.

What this script computes
-------------------------
For every input conformer it runs a pretrained MACE foundation model, takes the
O(3)-invariant (l = 0) part of the node features of every interaction layer,
concatenates the per-layer invariants, and averages them over the atoms of the
molecule.  With the default model (``mace_mp`` "medium", 128 channels, two
interaction layers) this yields a 256-dimensional vector per conformer:
``[layer-1 invariants (128) | layer-2 invariants (128)]``.

All atoms are used, including hydrogens.  Conformer 0 of every RDKit ``Mol`` is
used.  The output ``.npz`` stores a single float32 array of shape
``(n_conformers, 256)`` under the key ``arr_0``, in the same row order as the
input, matching the layout of the released
``3dcs-embeddings/chirality/mace/chirality.npz``.

Third-party dependencies (not redistributed here)
-------------------------------------------------
  * mace-torch   -- https://github.com/ACEsuit/mace (MIT).  Pinned: 0.3.15
  * MACE-MP-0 "medium" weights, downloaded by ``mace`` itself from
    https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model
    (cached as ``~/.cache/mace/20231203mace128L1_epoch199model``;
    sha256 01bfe22100139f424713cf921144e5509cbe353d67aa9fa1be9c6e1e0ed35845)
  * torch, e3nn, ase, rdkit, numpy -- see requirements.txt

Example
-------
  python extract_chirality.py \
      --dataset hf:EscheWang/3dcs:chirality \
      --out chirality.npz --device cuda --batch-size 1 --compress
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger("extract_chirality")

MODEL_NAME = "mace"
OUTPUT_KEY = "arr_0"


def _common():
    """Load ``baselines/common.py`` (verification helpers) without touching sys.path."""
    import importlib.util
    import os

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common.py")
    spec = importlib.util.spec_from_file_location("baselines_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# input
# --------------------------------------------------------------------------- #
def load_mols(dataset: str):
    """Return a list of RDKit Mol objects with 3D conformers, in benchmark row order.

    ``dataset`` is one of

    * a path to a pickle holding a list of RDKit ``Mol`` objects
      (``chirality_bench_conformers_noised_only_aslist.pkl``), or
    * ``hf:<repo_id>:<config>`` to pull the released Hugging Face dataset
      (``hf:EscheWang/3dcs:chirality``), or
    * ``hfdisk:<path>`` for a ``datasets.save_to_disk`` directory of that config.

    The Hugging Face rows store one ``mol_blocks`` list per stereoisomer plus the
    ``offset`` of its first conformer in the embedding matrix, so concatenating
    ``mol_blocks`` in ascending ``offset`` order reproduces the row order of the
    released embedding files.  MOL blocks carry coordinates with four decimals,
    so geometries read this way differ from the pickle by up to 5e-5 A.
    """
    from rdkit import Chem

    if dataset.startswith(("hf:", "hfdisk:")):
        if dataset.startswith("hfdisk:"):
            from datasets import load_from_disk

            ds = load_from_disk(dataset[len("hfdisk:") :])
        else:
            from datasets import load_dataset

            parts = dataset.split(":")
            if len(parts) != 3:
                raise ValueError("expected hf:<repo_id>:<config>")
            ds = load_dataset(parts[1], parts[2], split="train")
        if "mol_blocks" not in ds.column_names:
            raise KeyError(f"expected a 'mol_blocks' column, got {ds.column_names}")
        order = np.argsort(np.asarray(ds["offset"]))
        mols = []
        for row_idx in order:
            for block in ds[int(row_idx)]["mol_blocks"]:
                mol = Chem.MolFromMolBlock(block, removeHs=False, sanitize=False)
                if mol is None:
                    raise ValueError(f"could not parse a MOL block in row {int(row_idx)}")
                mols.append(mol)
        return mols

    with open(dataset, "rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, list):
        raise TypeError(f"{dataset} does not contain a list of RDKit Mol objects")
    return data


def mol_to_atoms(mol, conf_id: int):
    from ase import Atoms

    if mol.GetNumConformers() <= conf_id:
        raise ValueError(f"molecule has {mol.GetNumConformers()} conformers, need {conf_id + 1}")
    conf = mol.GetConformer(conf_id)
    return Atoms(
        symbols=[atom.GetSymbol() for atom in mol.GetAtoms()],
        positions=np.asarray(conf.GetPositions(), dtype=np.float64),
    )


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def build_calculator(model: str, device: str, dtype: str):
    from mace.calculators import foundations_models

    return foundations_models.mace_mp(model=model, device=device, default_dtype=dtype, return_raw_model=False)


def aggregate(descriptors: np.ndarray, method: str) -> np.ndarray:
    if method == "mean":
        return descriptors.mean(axis=0)
    if method == "sum":
        return descriptors.sum(axis=0)
    raise ValueError(f"unknown aggregation {method}")


def run_single(calc, mols, args) -> np.ndarray:
    """One molecule per forward pass. This is the reference path."""
    out: list[np.ndarray] = []
    n = len(mols)
    for idx, mol in enumerate(mols):
        atoms = mol_to_atoms(mol, args.conf_id)
        desc = calc.get_descriptors(atoms=atoms, invariants_only=not args.full_features, num_layers=args.num_layers)
        desc = np.asarray(desc, dtype=np.float32)
        out.append(np.nan_to_num(aggregate(desc, args.aggregation), copy=False))
        if args.log_every and (idx + 1) % args.log_every == 0:
            LOGGER.info("… %d/%d", idx + 1, n)
    return np.stack(out).astype(np.float32, copy=False)


def run_batched(calc, mols, args) -> np.ndarray:
    """Several molecules per forward pass.

    Faster, but the scatter reductions run in a different order, so results can
    differ from ``run_batched`` in the last float32 digits.  ``--batch-size 1``
    is the reference path.
    """
    import torch
    from e3nn import o3
    from mace import data as mace_data
    from mace.modules.utils import extract_invariant
    from mace.tools import torch_geometric, utils

    model = calc.models[0]
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    r_max = float(model.r_max)
    heads = getattr(model, "heads", ["Default"])
    num_interactions = int(model.num_interactions)
    num_layers = num_interactions if args.num_layers == -1 else args.num_layers
    irreps_out = o3.Irreps(str(model.products[0].linear.irreps_out))
    l_max = irreps_out.lmax
    num_invariant_features = irreps_out.dim // (l_max + 1) ** 2

    configs = []
    for mol in mols:
        atoms = mol_to_atoms(mol, args.conf_id)
        configs.append(mace_data.config_from_atoms(atoms))
    dataset = [mace_data.AtomicData.from_config(c, z_table=z_table, cutoff=r_max, heads=heads) for c in configs]
    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
    )
    out: list[np.ndarray] = []
    done = 0
    for batch in loader:
        batch = batch.to(args.device)
        res = model(batch.to_dict(), compute_force=False, compute_virials=False, compute_stress=False)
        node_feats = res["node_feats"]
        if args.full_features:
            feats = node_feats
        else:
            feats = extract_invariant(
                node_feats, num_layers=num_layers, num_features=num_invariant_features, l_max=l_max
            )
            keep = num_invariant_features * num_layers
            feats = feats[:, :keep]
        idx = batch.batch
        nb = int(idx.max().item()) + 1
        summed = torch.zeros(nb, feats.shape[1], dtype=feats.dtype, device=feats.device)
        summed.index_add_(0, idx, feats)
        if args.aggregation == "mean":
            counts = torch.bincount(idx, minlength=nb).clamp(min=1).unsqueeze(1).to(feats.dtype)
            pooled = summed / counts
        else:
            pooled = summed
        arr = pooled.detach().cpu().numpy().astype(np.float32)
        out.append(np.nan_to_num(arr, copy=False))
        done += arr.shape[0]
        if args.log_every and done % args.log_every < args.batch_size:
            LOGGER.info("… %d/%d", done, len(mols))
    return np.concatenate(out, axis=0).astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--dataset", required=True, help="pickle of RDKit Mols, hf:<repo_id>:<config>, or hfdisk:<save_to_disk dir>"
    )
    p.add_argument("--out", required=True, type=Path, help="output .npz path")
    p.add_argument(
        "--model",
        default="medium",
        help="mace_mp model name: small | medium | large | medium-mpa-0 … (default: medium)",
    )
    p.add_argument(
        "--aggregation",
        choices=("mean", "sum"),
        default="mean",
        help="how per-atom descriptors are pooled (default: mean)",
    )
    p.add_argument(
        "--num-layers", type=int, default=-1, help="number of interaction layers to keep, -1 = all (default: -1)"
    )
    p.add_argument(
        "--full-features",
        action="store_true",
        help="keep the full equivariant node features instead of the invariant part",
    )
    p.add_argument("--conf-id", type=int, default=0, help="conformer id (default: 0)")
    p.add_argument(
        "--batch-size", type=int, default=1, help="molecules per forward pass; 1 is the reference path (default: 1)"
    )
    p.add_argument("--device", default="cuda", help="torch device (default: cuda)")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--limit", type=int, default=None, help="only process the first N molecules")
    p.add_argument(
        "--compress", action="store_true", help="write with np.savez_compressed (the released file is compressed)"
    )
    p.add_argument("--log-every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--verify",
        nargs="?",
        const="published",
        default=None,
        metavar="REFERENCE",
        help="after writing, compare the output with a reference file: 'published' "
        "(the published MACE chirality embedding), hub:<path in "
        "EscheWang/3dcs-embeddings>, or a local path",
    )
    p.add_argument(
        "--verify-key", default=None, help="array key to read from the --verify reference (default: its published key)"
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import ase
    import e3nn
    import mace
    import rdkit
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False

    LOGGER.info("python %s", sys.version.split()[0])
    LOGGER.info(
        "numpy %s | torch %s (cuda %s) | e3nn %s | mace-torch %s | ase %s | rdkit %s",
        np.__version__,
        torch.__version__,
        torch.version.cuda,
        e3nn.__version__,
        mace.__version__,
        ase.__version__,
        rdkit.__version__,
    )
    LOGGER.info("args: %s", json.dumps({k: str(v) for k, v in vars(args).items()}, sort_keys=True))

    mols = load_mols(args.dataset)
    if args.limit is not None:
        mols = mols[: args.limit]
    LOGGER.info("loaded %d molecules from %s", len(mols), args.dataset)

    t0 = time.time()
    calc = build_calculator(args.model, args.device, args.dtype)
    LOGGER.info("model ready in %.1fs", time.time() - t0)

    t0 = time.time()
    arr = run_single(calc, mols, args) if args.batch_size == 1 else run_batched(calc, mols, args)
    LOGGER.info("extracted %s in %.1fs", arr.shape, time.time() - t0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = np.savez_compressed if args.compress else np.savez
    writer(args.out, arr)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    LOGGER.info("wrote %s  (%d bytes)  sha256=%s", args.out, args.out.stat().st_size, digest)

    if args.verify:
        _common().verify(
            args.out,
            model=MODEL_NAME,
            reference=args.verify,
            produced_key=OUTPUT_KEY,
            reference_key=args.verify_key,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
