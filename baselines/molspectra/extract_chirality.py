#!/usr/bin/env python
"""Extract MolSpectra-style molecule-level embeddings for the 3DCS chirality set.

The released 3DCS file ``chirality/molspectra/sampled_mol_feature.npz`` holds one
256-d vector per conformer.  In the MolSpectra reference implementation
(https://github.com/AzureLeon1/MolSpectra, an equivariant-Transformer /
TorchMD-NET fork) that quantity is produced inside ``TorchMD_Net.forward`` as::

    x, v, z, pos, batch = self.representation_model(z, pos, batch=batch)
    mol_feature = scatter(x, batch, dim=0, reduce=self.reduce_op)   # reduce_op = "add"

i.e. the per-atom scalar representation of the equivariant Transformer, summed
over the atoms of each molecule.  This script computes exactly that quantity.
It does not vendor any third-party code or weights: point --repo at a local
clone of the upstream MolSpectra repository and --checkpoint at a TorchMD-NET
equivariant-Transformer checkpoint (256-d, 8 layers).

The output depends entirely on which checkpoint is supplied.  The MolSpectra
authors release their QM9S dataset but no pre-trained checkpoint, so the weights
have to be supplied by the caller; with a different checkpoint this script
produces a valid MolSpectra-architecture embedding, not a copy of the released
3DCS file.  --arch records which of the two encoder variants is in use and the
script prints how many checkpoint tensors were missing, so a mismatch is
visible in the log.

Usage
-----
  python extract_chirality.py \
      --dataset  hf:EscheWang/3dcs:chirality \
      --repo     /path/to/MolSpectra \
      --checkpoint /path/to/et-256.ckpt \
      --out      molspectra_chirality.npz \
      --batch-size 128 --device cuda

Outputs an .npz with key ``arr_0`` of shape (n_conformers, 256), float32, rows
in the order of the input dataset.
"""

import argparse
import hashlib
import os
import sys
import time

MODEL_NAME = "molspectra"


def _common():
    """Load ``baselines/common.py`` (verification helpers) without touching sys.path."""
    import importlib.util
    import os

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common.py")
    spec = importlib.util.spec_from_file_location("baselines_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help=_common().DATASET_SPEC_HELP)
    p.add_argument("--repo", required=True, help="Local clone of https://github.com/AzureLeon1/MolSpectra")
    p.add_argument("--checkpoint", required=True, help="TorchMD-NET equivariant-Transformer checkpoint (.ckpt).")
    p.add_argument("--out", required=True, help="Output .npz path.")
    p.add_argument("--batch-size", type=int, default=128, help="Molecules per forward pass (default: 128).")
    p.add_argument("--device", default="cuda", help="cuda | cuda:N | cpu (default: cuda).")
    p.add_argument("--key", default="arr_0", help="npz key to write (default: arr_0).")
    # --- recipe knobs -------------------------------------------------------
    p.add_argument(
        "--hydrogens",
        choices=["remove", "keep"],
        default="remove",
        help="Whether explicit hydrogens are fed to the encoder. "
        "remove (default) drops every atom with Z == 1, i.e. the encoder "
        "sees heavy atoms only; keep feeds the molecule as stored.",
    )
    p.add_argument(
        "--pool",
        choices=["add", "mean", "max"],
        default="add",
        help="Pooling of per-atom features (default: add, = MolSpectra reduce_op).",
    )
    p.add_argument(
        "--arch",
        choices=["molspectra", "torchmdnet"],
        default="molspectra",
        help="molspectra: MolSpectra's ET, with the per-layer x_norms / "
        "vec_norms it adds (use_dataset_md17=False). "
        "torchmdnet: those extra norms disabled (use_dataset_md17=True), "
        "which is the plain upstream TorchMD-NET / denoising ET.",
    )
    p.add_argument(
        "--layernorm-on-vec",
        default="whitened",
        choices=["whitened", "none"],
        help="Equivariant output layer norm (default: whitened).",
    )
    p.add_argument("--normalize", action="store_true", help="L2-normalise each output row (off by default).")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--verify",
        nargs="?",
        const="published",
        default=None,
        metavar="REFERENCE",
        help="after writing, compare the output with a reference file: 'published' "
        "(the published MolSpectra chirality embedding), hub:<path in "
        "EscheWang/3dcs-embeddings>, or a local path",
    )
    p.add_argument(
        "--verify-key", default=None, help="array key to read from the --verify reference (default: its published key)"
    )
    p.add_argument("--verify-rows", default=None, metavar="ROWS", help=_common().ROW_SELECTION_HELP)
    p.add_argument("--limit", type=int, default=None, help="only process the first N conformers")
    p.add_argument("--start", type=int, default=0, help="skip the first N conformers")
    return p.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_molecules(spec, *, limit=None, start=0):
    """Return a flat list of RDKit Mol in the canonical 3DCS row order.

    ``spec`` takes the ``--dataset`` syntax shared by every script in ``baselines/``
    (see ``baselines/common.py``): ``hf:<repo_id>[:<config>]``, ``hfdisk:<dir>`` or a plain
    ``save_to_disk`` directory, a bare Hub dataset id, a pickle of RDKit molecules (a list,
    or a dict of lists flattened in insertion order), or ``lmdb:<file>``.  MolSpectra reads
    atomic numbers and coordinates only.
    """
    return _common().load_conformers(spec, limit=limit, start=start)


def main():
    args = parse_args()

    import numpy as np
    import torch
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    sys.path.insert(0, os.path.abspath(args.repo))
    import rdkit
    import torch_cluster
    import torch_geometric
    from torch_scatter import scatter
    from torchmdnet.models.torchmd_et import TorchMD_ET

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)  # scatter-add has no deterministic kernel

    print("# ---- versions ----", flush=True)
    print(f"# python            {sys.version.split()[0]}")
    print(f"# torch             {torch.__version__} (cuda {torch.version.cuda})")
    print(f"# torch_scatter     {getattr(__import__('torch_scatter'), '__version__', '?')}")
    print(f"# torch_cluster     {torch_cluster.__version__}")
    print(f"# torch_geometric   {torch_geometric.__version__}")
    print(f"# numpy             {np.__version__}")
    print(f"# rdkit             {rdkit.__version__}")
    print(f"# repo              {os.path.abspath(args.repo)}")
    print(f"# checkpoint        {args.checkpoint}")
    print(f"# checkpoint sha256 {sha256(args.checkpoint)}")
    print(f"# dataset           {args.dataset}")
    if os.path.isfile(args.dataset):
        print(f"# dataset sha256    {sha256(args.dataset)}")
    print(
        f"# recipe            arch={args.arch} hydrogens={args.hydrogens} "
        f"pool={args.pool} layernorm_on_vec={args.layernorm_on_vec} "
        f"normalize={args.normalize} dtype={args.dtype}"
    )

    # ---- model ------------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    hp = dict(ckpt.get("hyper_parameters", {}))

    def h(k, default):
        v = hp.get(k, default)
        return default if v is None else v

    model = TorchMD_ET(
        hidden_channels=h("embedding_dimension", 256),
        num_layers=h("num_layers", 8),
        num_rbf=h("num_rbf", 64),
        rbf_type=h("rbf_type", "expnorm"),
        trainable_rbf=h("trainable_rbf", False),
        activation=h("activation", "silu"),
        attn_activation=h("attn_activation", "silu"),
        neighbor_embedding=h("neighbor_embedding", True),
        num_heads=h("num_heads", 8),
        distance_influence=h("distance_influence", "both"),
        cutoff_lower=h("cutoff_lower", 0.0),
        cutoff_upper=h("cutoff_upper", 5.0),
        max_z=h("max_z", 100),
        max_num_neighbors=h("max_num_neighbors", 32),
        layernorm_on_vec=None if args.layernorm_on_vec == "none" else args.layernorm_on_vec,
        use_dataset_md17=(args.arch == "torchmdnet"),
    )

    # the checkpoint stores the whole TorchMD_Net; keep the representation trunk only
    prefix = "model.representation_model."
    sd = {k[len(prefix) :]: v for k, v in ckpt["state_dict"].items() if k.startswith(prefix)}
    if not sd:
        prefix = "representation_model."
        sd = {k[len(prefix) :]: v for k, v in ckpt["state_dict"].items() if k.startswith(prefix)}
    ret = model.load_state_dict(sd, strict=False)
    print(f"# loaded {len(sd)} tensors; missing={len(ret.missing_keys)} unexpected={len(ret.unexpected_keys)}")
    if ret.missing_keys:
        print(f"#   missing (left at init): {sorted(set(k.split('.')[0] for k in ret.missing_keys))}")
    if ret.unexpected_keys:
        print(f"#   unexpected: {ret.unexpected_keys[:8]}")
    if ret.missing_keys and args.arch == "molspectra":
        print(
            "# WARNING: this checkpoint has no weights for MolSpectra's per-layer "
            "x_norms / vec_norms, so they stay at LayerNorm init and the encoder is not "
            "the one the checkpoint was trained as. Pass --arch torchmdnet for an "
            "upstream TorchMD-NET checkpoint; --arch molspectra needs a MolSpectra one.",
            flush=True,
        )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    model = model.to(device=device, dtype=dtype).eval()

    # ---- data -------------------------------------------------------------
    mols = load_molecules(args.dataset, limit=args.limit, start=args.start)
    n = len(mols)
    print(f"# conformers        {n}", flush=True)

    zs, poss = [], []
    for mol in mols:
        z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
        xyz = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
        if args.hydrogens == "remove":
            keep = z != 1
            z, xyz = z[keep], xyz[keep]
        zs.append(z)
        poss.append(xyz)
    n_atoms_fed = int(sum(len(z) for z in zs))
    print(f"# atoms fed         {n_atoms_fed} ({n_atoms_fed / max(len(mols), 1):.2f} per conformer)")

    out = np.zeros((n, model.hidden_channels), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            stop = min(start + args.batch_size, n)
            zb = np.concatenate(zs[start:stop])
            pb = np.concatenate(poss[start:stop], axis=0)
            bb = np.concatenate([np.full(len(zs[i]), i - start, dtype=np.int64) for i in range(start, stop)])
            z = torch.from_numpy(zb).to(device)
            pos = torch.from_numpy(pb).to(device=device, dtype=dtype)
            batch = torch.from_numpy(bb).to(device)
            x, _v, _z, _pos, _batch = model(z, pos, batch)
            feat = scatter(x, batch, dim=0, dim_size=stop - start, reduce=args.pool)
            if args.normalize:
                feat = feat / feat.norm(dim=1, keepdim=True).clamp_min(1e-12)
            out[start:stop] = feat.float().cpu().numpy()
            if (start // args.batch_size) % 50 == 0:
                print(f"#   {stop}/{n}  {time.time() - t0:.1f}s", flush=True)
    dt = time.time() - t0
    print(f"# forward wall time {dt:.1f}s ({n / dt:.1f} conformers/s)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, **{args.key: out})
    print(f"# wrote {args.out}  shape={out.shape}  sha256={sha256(args.out)}")

    if args.verify:
        _common().verify(
            args.out,
            model=MODEL_NAME,
            reference=args.verify,
            produced_key=args.key,
            reference_key=args.verify_key,
            rows=args.verify_rows or (f"{args.start}+" if args.start else None),
        )


if __name__ == "__main__":
    main()
