#!/usr/bin/env python3
"""Extract FMG embeddings for the 3DCS chirality set.

The released ``chirality/fmg/...npz`` embedding file holds, for every conformer of the
3DCS chirality set, a 128-dimensional vector read out of the field-based molecular
generation (FMG) 3D U-Net of

    Dumitrescu et al., "E(3)-equivariant models cannot learn chirality: Field-based
    molecular generation", ICLR 2025.  https://github.com/Dumitrescu-Alexandru/FMG

This script is the extraction wrapper only.  The FMG model code and the FMG QM9
checkpoint are third party and are NOT redistributed here; ``--fmg-repo`` must point at
your own clone of the upstream repository and ``--checkpoint`` at the QM9 checkpoint
obtained from the FMG authors.  See the accompanying README for the exact revision and
checksums that were used.

What the script computes, per conformer:

  1. atom coordinates are mean-centred and rotated onto their PCA axes
     (``utils.align`` of the FMG repo; a reflection is turned into a rotation by
     flipping the third axis, so handedness is preserved);
  2. only the atoms of ``--atom-channels`` (default ``C,O,N,F``) are kept.  Hydrogen is
     dropped, as are any other elements;
  3. the molecule is rendered onto a ``--grid-size``^3 cubic grid of spacing
     ``--resolution`` A, centred at the origin, as a sum of isotropic Gaussians of
     variance ``--gaussian-std``: one channel per kept element, plus one channel per bond
     order (1.0, 2.0, 3.0 and, with ``--explicit-aromatic``, 1.5) with a Gaussian at each
     bond midpoint.  Values are thresholded at 0.1 and min-max normalised per channel
     (``utils.create_gaussian_batch_pdf_values`` of the FMG repo);
  4. the field is passed through the class-conditional 3D U-Net at a fixed diffusion
     timestep (``--timestep``), with no noise added to the field, class label 0 and no
     classifier-free-guidance dropout;
  5. the output of the U-Net's ``final_res_block`` -- a (128, G, G, G) feature map -- is
     averaged over the three spatial axes, giving the 128-d embedding.

Two input forms are accepted and produce rows in the same order:

  * ``--dataset <dir-or-repo-id>`` -- the ``chirality`` config of the ``EscheWang/3dcs``
    Hugging Face dataset, either a ``save_to_disk`` directory or a Hub id.  Conformers
    are taken from the ``mol_blocks`` column, dataset row order, and each row's
    ``offset`` is checked against the running conformer count.
  * ``--dataset <file.pkl>`` -- a pickle holding a list of RDKit molecules with one
    conformer each.

Output: a compressed ``.npz`` with ``embeddings`` (float32, N x 128) and ``smiles``
(isomeric SMILES of each input molecule, as written by the RDKit build in use).

Example
-------
    python extract_chirality.py \
        --fmg-repo /path/to/FMG \
        --checkpoint /path/to/model-120qm9_3rd_run.pt \
        --dataset /path/to/hf/chirality \
        --out chirality_fmg.npz \
        --batch-size 32 --device cuda:0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------------
# defaults -- these are the settings the released chirality embeddings were made with
# ---------------------------------------------------------------------------------
DEFAULT_GRID_SIZE = 24
DEFAULT_RESOLUTION = 0.40
DEFAULT_GAUSSIAN_STD = 0.08
DEFAULT_TIMESTEP = 50
DEFAULT_ATOM_CHANNELS = "C,O,N,F"
DEFAULT_UNET_DIM = 128
DEFAULT_UNET_DIM_MULTS = "1,2,3"
DEFAULT_NUM_CLASSES = 17


@dataclass
class MolRecord:
    coords: np.ndarray
    atoms: np.ndarray
    bonds: np.ndarray  # (n_bonds, 3): begin, end, bond order
    smiles: str


# ---------------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------------
def _bond_order_map():
    from rdkit import Chem

    return {
        Chem.rdchem.BondType.SINGLE: 1.0,
        Chem.rdchem.BondType.DOUBLE: 2.0,
        Chem.rdchem.BondType.TRIPLE: 3.0,
        Chem.rdchem.BondType.AROMATIC: 1.5,
    }


def _record_from_mol(mol, align, allowed, bond_order_map) -> MolRecord | None:
    from rdkit import Chem

    if mol is None or mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer()
    coords = np.array(
        [
            [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
            for i in range(mol.GetNumAtoms())
        ],
        dtype=np.float32,
    )
    coords = align(coords)
    atom_symbols = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])
    valid_mask = np.array([sym in allowed for sym in atom_symbols])
    if not valid_mask.any():
        return None
    coords = coords[valid_mask]
    atoms = atom_symbols[valid_mask]
    index_map = {orig: new for new, orig in enumerate(np.where(valid_mask)[0])}
    bonds = []
    for bond in mol.GetBonds():
        order = bond_order_map.get(bond.GetBondType())
        if order is None:
            continue
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin in index_map and end in index_map:
            bonds.append((index_map[begin], index_map[end], order))
    smiles = Chem.MolToSmiles(Chem.Mol(mol), isomericSmiles=True)
    bond_array = np.array(bonds, dtype=np.float32).reshape(-1, 3) if bonds else np.zeros((0, 3), dtype=np.float32)
    return MolRecord(coords=coords, atoms=atoms, bonds=bond_array, smiles=smiles)


def iter_mols_from_pickle(path: str) -> Iterable:
    with open(path, "rb") as fh:
        mols = pickle.load(fh)
    yield from mols


def iter_mols_from_hf(dataset: str, revision: str | None) -> Iterable:
    from rdkit import Chem

    if os.path.isdir(dataset):
        from datasets import load_from_disk

        ds = load_from_disk(dataset)
    else:
        from datasets import load_dataset

        kw = {"name": "chirality", "split": "train"}
        if revision:
            kw["revision"] = revision
        ds = load_dataset(dataset, **kw)
    seen = 0
    for row in ds:
        offset = row.get("offset")
        if offset is not None and int(offset) != seen:
            raise RuntimeError(
                f"row offset {offset} does not match the running conformer count {seen}; "
                "the dataset is not in its published order"
            )
        for block in row["mol_blocks"]:
            mol = Chem.MolFromMolBlock(block, removeHs=False)
            seen += 1
            yield mol


def load_records(args, align) -> list[MolRecord]:
    allowed = set(s.strip() for s in args.atom_channels.split(",") if s.strip())
    bond_order_map = _bond_order_map()
    if args.dataset.endswith(".pkl") or args.dataset.endswith(".pickle"):
        source = iter_mols_from_pickle(args.dataset)
    else:
        source = iter_mols_from_hf(args.dataset, args.hf_revision)
    records: list[MolRecord] = []
    skipped = 0
    for mol in source:
        if args.limit is not None and len(records) >= args.limit:
            break
        rec = _record_from_mol(mol, align, allowed, bond_order_map)
        if rec is None:
            skipped += 1
            continue
        records.append(rec)
    if skipped:
        print(f"[warn] skipped {skipped} molecules without a conformer or without any kept atom")
    return records


# ---------------------------------------------------------------------------------
# field construction + model
# ---------------------------------------------------------------------------------
class MolFieldDataset:
    """torch Dataset over MolRecords -- returns the Gaussian centres and their channels."""

    def __init__(self, records, atom_channels: Sequence[str], bond_orders: Sequence[float]):
        import torch

        self._torch = torch
        self.records = records
        self.atom_channels = list(atom_channels)
        self.bond_orders = list(bond_orders)
        self.atom_map = {sym: i for i, sym in enumerate(self.atom_channels)}
        self.bond_map = {o: len(self.atom_channels) + i for i, o in enumerate(self.bond_orders)}
        self.total_channels = len(self.atom_channels) + len(self.bond_orders)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        torch = self._torch
        rec = self.records[idx]
        points, ids, counts = [], [], []
        coords, atoms = rec.coords, rec.atoms
        for sym in self.atom_channels:
            mask = atoms == sym
            n = int(mask.sum())
            counts.append(n)
            if n:
                points.append(coords[mask])
                ids.append(np.full(n, self.atom_map[sym], dtype=np.int64))
        bond_counts = []
        if rec.bonds.size:
            for order in self.bond_orders:
                mask = np.isclose(rec.bonds[:, 2], order)
                n = int(mask.sum())
                bond_counts.append(n)
                if n:
                    mids = 0.5 * (coords[rec.bonds[mask, 0].astype(int)] + coords[rec.bonds[mask, 1].astype(int)])
                    points.append(mids)
                    ids.append(np.full(n, self.bond_map[order], dtype=np.int64))
        else:
            bond_counts.extend([0] * len(self.bond_orders))
        if not points:
            points = [coords]
            ids = [np.zeros(len(coords), dtype=np.int64)]
            counts = [len(coords)] + [0] * (len(self.atom_channels) - 1)
            bond_counts = [0] * len(self.bond_orders)
        coords_all = np.concatenate(points, axis=0).astype(np.float32)
        ids_all = np.concatenate(ids, axis=0).astype(np.int64)
        return (
            torch.from_numpy(coords_all),
            torch.from_numpy(ids_all),
            counts + bond_counts,
            rec.smiles,
        )


def make_collate(total_channels: int):
    import torch

    def _collate(batch):
        coord_list, id_list, n_lists, smiles = [], [], [], []
        for sample_idx, (coords, ids, n_list, smi) in enumerate(batch):
            coord_list.append(coords)
            id_list.append(ids + sample_idx * total_channels)
            n_lists.append(n_list)
            smiles.append(smi)
        return torch.cat(coord_list, 0), torch.cat(id_list, 0), n_lists, smiles

    return _collate


def prepare_grid(grid_size: int, resolution: float, device):
    import torch

    half = 0.5 * resolution * (grid_size - 1)
    axis = np.linspace(-half, half, grid_size, dtype=np.float32)
    flat = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    return torch.tensor(flat, dtype=torch.float32, device=device)


def build_encoder(args, total_channels: int, device):
    import torch
    from denoising_diffusion_pytorch.classifier_free_guidance import Unet3D

    dim_mults = tuple(int(m) for m in args.unet_dim_mults.split(",") if m.strip())

    class DiffusionEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.unet = Unet3D(
                dim=args.unet_dim,
                num_classes=args.num_classes,
                cond_drop_prob=0.0,
                dim_mults=dim_mults,
                channels=total_channels,
                legacy_attention=args.legacy_attention,
                add_pe=args.add_pe,
            )
            self._latent = None
            self.unet.final_res_block.register_forward_hook(self._capture)

        def _capture(self, module, inputs, output):
            self._latent = output

        def forward(self, fields, labels, timestep):
            bsz = fields.shape[0]
            t = torch.full((bsz,), float(timestep), dtype=torch.float32, device=fields.device)
            self._latent = None
            self.unet(fields, t, labels, cond_var=None, cond_drop_prob=0.0)
            if self._latent is None:
                raise RuntimeError("failed to capture the final_res_block feature map")
            emb = self._latent.mean(dim=(2, 3, 4))
            self._latent = None
            return emb

    model = DiffusionEncoder().to(device)

    state = torch.load(args.checkpoint, map_location=device)
    if "ema" in state and "online_model.model.init_conv.weight" in state["ema"]:
        weights, prefix = state["ema"], "online_model.model."
    else:
        weights, prefix = state.get("model", state), "module.model."
    unet_state = {k[len(prefix) :]: v for k, v in weights.items() if k.startswith(prefix)}
    if not unet_state:
        raise RuntimeError(f"no weights with prefix {prefix!r} found in {args.checkpoint}")
    missing, unexpected = model.unet.load_state_dict(unet_state, strict=False)
    if missing:
        raise RuntimeError(f"checkpoint is missing {len(missing)} keys, first: {missing[:5]}")
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected keys in the checkpoint, first: {unexpected[:5]}")
    print(f"loaded {len(unet_state)} tensors from {args.checkpoint} (prefix {prefix!r})")
    model.eval()
    return model


# ---------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------
def sha256(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


MODEL_NAME = "fmg"
OUTPUT_KEY = "embeddings"


def _common():
    """Load ``baselines/common.py`` (verification helpers) without touching sys.path."""
    import importlib.util
    import os

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common.py")
    spec = importlib.util.spec_from_file_location("baselines_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Extract FMG embeddings for the 3DCS chirality set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="HF chirality config (save_to_disk dir or Hub id), or a .pkl of RDKit molecules",
    )
    p.add_argument("--hf-revision", default=None, help="pin the Hub revision when --dataset is a Hub id")
    p.add_argument("--out", required=True, help="output .npz")
    p.add_argument(
        "--fmg-repo", required=True, help="clone of https://github.com/Dumitrescu-Alexandru/FMG (prepended to sys.path)"
    )
    p.add_argument("--checkpoint", required=True, help="FMG QM9 3D U-Net checkpoint (.pt)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "-1" else "cpu")
    p.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    p.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION)
    p.add_argument("--gaussian-std", type=float, default=DEFAULT_GAUSSIAN_STD)
    p.add_argument("--timestep", type=int, default=DEFAULT_TIMESTEP)
    p.add_argument("--atom-channels", default=DEFAULT_ATOM_CHANNELS)
    p.add_argument(
        "--no-explicit-aromatic",
        dest="explicit_aromatic",
        action="store_false",
        help="drop the aromatic (order 1.5) bond channel; the released embeddings keep it",
    )
    p.set_defaults(explicit_aromatic=True)
    p.add_argument("--unet-dim", type=int, default=DEFAULT_UNET_DIM)
    p.add_argument("--unet-dim-mults", default=DEFAULT_UNET_DIM_MULTS)
    p.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES)
    p.add_argument("--legacy-attention", action="store_true")
    p.add_argument("--add-pe", action="store_true")
    p.add_argument("--class-label", type=int, default=0, help="class-conditioning label fed to the U-Net")
    p.add_argument("--limit", type=int, default=None, help="only process the first N conformers (debugging)")
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="force deterministic cuDNN kernels and disable TF32; slower, and not the setting "
        "the released file was produced with (library defaults)",
    )
    p.add_argument("--log-interval", type=int, default=5000)
    p.add_argument("--dtype", choices=["float32"], default="float32")
    p.add_argument(
        "--verify",
        nargs="?",
        const="published",
        default=None,
        metavar="REFERENCE",
        help="after writing, compare the output with a reference file: 'published' "
        "(the published FMG chirality embedding), hub:<path in "
        "EscheWang/3dcs-embeddings>, or a local path",
    )
    p.add_argument(
        "--verify-key", default=None, help="array key to read from the --verify reference (default: its published key)"
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    repo = os.path.abspath(os.path.expanduser(args.fmg_repo))
    if not os.path.isdir(repo):
        sys.exit(f"--fmg-repo {repo} is not a directory")
    sys.path.insert(0, repo)

    import torch
    from rdkit import RDLogger, rdBase

    RDLogger.DisableLog("rdApp.*")

    from utils import align, create_gaussian_batch_pdf_values  # FMG upstream

    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "rdkit": rdBase.rdkitVersion,
        "fmg_repo": repo,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print("versions:", json.dumps(versions))
    print("checkpoint sha256:", sha256(args.checkpoint))
    print("settings:", json.dumps({k: v for k, v in vars(args).items()}, default=str))

    torch.manual_seed(0)
    np.random.seed(0)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    print(
        f"cudnn.deterministic={torch.backends.cudnn.deterministic} benchmark={torch.backends.cudnn.benchmark} allow_tf32={torch.backends.cudnn.allow_tf32} matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}"
    )

    device = torch.device(args.device)

    t0 = time.time()
    records = load_records(args, align)
    print(f"loaded {len(records)} conformers in {time.time() - t0:.1f} s")
    if not records:
        sys.exit("no usable conformers")

    atom_channels = [s.strip() for s in args.atom_channels.split(",") if s.strip()]
    bond_orders = [1.0, 2.0, 3.0] + ([1.5] if args.explicit_aromatic else [])
    dataset = MolFieldDataset(records, atom_channels, bond_orders)
    print(f"field channels: {atom_channels} + bond orders {bond_orders} = {dataset.total_channels}")

    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=make_collate(dataset.total_channels),
    )

    model = build_encoder(args, dataset.total_channels, device)
    grid = prepare_grid(args.grid_size, args.resolution, device)

    embeddings, smiles_all, processed = [], [], 0
    t0 = time.time()
    with torch.no_grad():
        for coords, inds, n_lists, smiles in loader:
            coords = coords.to(device, non_blocking=True)
            inds = inds.to(device, non_blocking=True)
            fields = create_gaussian_batch_pdf_values(
                x=grid,
                coords=coords,
                N_list=n_lists,
                std=args.gaussian_std,
                device=device,
                gaussian_indices=inds,
                no_fields=dataset.total_channels,
                grid_shapes=[args.grid_size] * 3,
            )
            labels = torch.full((fields.shape[0],), args.class_label, dtype=torch.long, device=device)
            emb = model(fields, labels, args.timestep)
            embeddings.append(emb.float().cpu().numpy())
            smiles_all.extend(smiles)
            prev, processed = processed, processed + len(smiles)
            if args.log_interval > 0 and (
                processed == len(dataset) or processed // args.log_interval != prev // args.log_interval
            ):
                rate = processed / max(time.time() - t0, 1e-9)
                print(f"  {processed}/{len(dataset)}  {rate:.1f} mol/s", flush=True)
    dt = time.time() - t0
    embeddings = np.concatenate(embeddings, axis=0).astype(np.float32)
    print(f"forward pass: {len(dataset)} conformers in {dt:.1f} s ({len(dataset) / dt:.1f} mol/s)")
    if device.type == "cuda":
        print(
            f"peak GPU memory: {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB allocated, {torch.cuda.max_memory_reserved(device) / 2**30:.2f} GiB reserved"
        )

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(out, embeddings=embeddings, smiles=np.array(smiles_all))
    print(f"wrote {out}  shape={embeddings.shape}  sha256={sha256(out)}")

    if args.verify:
        _common().verify(
            out, model=MODEL_NAME, reference=args.verify, produced_key=OUTPUT_KEY, reference_key=args.verify_key
        )


if __name__ == "__main__":
    main()
