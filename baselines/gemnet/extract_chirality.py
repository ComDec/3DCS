#!/usr/bin/env python3
"""Extract GemNet-Q graph embeddings for the 3DCS chirality conformer set.

What this computes
------------------
For every conformer, the molecule is reduced to its heavy atoms
(``rdkit.Chem.RemoveAllHs``), turned into a GemNet graph (5 A edge cutoff,
10 A quadruplet interaction cutoff), and pushed through the pretrained GemNet-Q
backbone. The per-atom representation ``h`` produced by the last interaction
block is averaged over the atoms of each molecule, giving one 128-d vector per
conformer. The result is written to a ``.npz`` whose single array is keyed
``gemnet``, in the row order of the input dataset.

Third-party code and weights
----------------------------
The GemNet architecture and the GemNet-Q weights are not part of this
repository. Fetch them from the upstream project and point ``--gemnet-repo`` at
the checkout:

    git clone https://github.com/TUM-DAML/gemnet_pytorch.git
    git -C gemnet_pytorch checkout a0164f74217155232d39c35f0bb2c016bd3f44da

    # pretrained/GemNet-Q/model.pth   sha256 d51e00af18cf9dd0097f0bc251386fdd539cfe82fdeb2790d0a0f9a241f169cd
    # pretrained/scaling_factors.json sha256 f9c855d929b8774b003c20246733faa571befe9d999bda2025fe7618cc100635

This script does not modify that checkout and contains no code derived from it.
The graph (edges, triplets, quadruplets) is built by upstream's own
``gemnet.training.data_container.DataContainer``, and ``h`` is read with a
forward hook on the last interaction block, so upstream's ``gemnet.py`` runs
exactly as published. Upstream is licensed under the Hippocratic License 2.0;
check that it permits your use before running it.

Environment (see README.md for the exact install commands)
----------------------------------------------------------
python 3.10, torch 2.1.0+cu121, torch_scatter 2.1.2, numpy 1.24.4, scipy 1.10.1,
sympy 1.12, numba 0.58.1, rdkit 2025.03.5. ``torch_scatter`` is optional: without it the
pure-PyTorch fallback in ``compat/torch_scatter.py`` is used, which agrees with
the compiled package to 7.4e-06 on this dataset.

Examples
--------
    # from the released HuggingFace dataset
    python extract_chirality.py --gemnet-repo ./gemnet_pytorch \
        --dataset hf:EscheWang/3dcs:chirality \
        --out gemnet_chirality.npz --batch-size 32 --device cuda

    # from a local pickle holding a list of RDKit Mol objects
    python extract_chirality.py --gemnet-repo ./gemnet_pytorch \
        --dataset chirality_bench_conformers_noised_only_aslist.pkl \
        --out gemnet_chirality.npz
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

MODEL_NAME = "gemnet"


def _common():
    """Load ``baselines/common.py`` (verification helpers) without touching sys.path."""
    import importlib.util
    import os

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common.py")
    spec = importlib.util.spec_from_file_location("baselines_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Extract GemNet-Q chirality embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--gemnet-repo",
        default="gemnet_pytorch",
        help="Checkout of github.com/TUM-DAML/gemnet_pytorch (commit a0164f7).",
    )
    p.add_argument(
        "--pretrained-dir",
        default=None,
        help="Directory with model.pth + model_kwargs.json "
        "(default: <gemnet-repo>/pretrained/GemNet-Q). scaling_factors.json is "
        "resolved relative to its parent, as upstream does.",
    )
    p.add_argument("--dataset", required=True, help=_common().DATASET_SPEC_HELP)
    p.add_argument("--out", required=True, help="Output .npz path.")
    p.add_argument("--key", default="gemnet", help="Array key inside the .npz (default: gemnet).")
    p.add_argument("--batch-size", type=int, default=32, help="Conformers per forward pass.")
    p.add_argument("--device", default="cuda", help="cuda, cuda:0 or cpu.")
    p.add_argument(
        "--pooling",
        default="mean",
        choices=["mean", "add"],
        help="Per-molecule reduction of the final atom features h.",
    )
    p.add_argument(
        "--hydrogens",
        default="all",
        choices=["all", "remove", "keep"],
        help="all = Chem.RemoveAllHs (heavy atoms only); remove = Chem.RemoveHs, which "
        "keeps the hydrogens RDKit treats as stereo-defining; keep = no removal.",
    )
    p.add_argument("--cutoff", type=float, default=5.0, help="GemNet edge cutoff in Angstrom.")
    p.add_argument("--int-cutoff", type=float, default=10.0, help="GemNet quadruplet interaction cutoff in Angstrom.")
    p.add_argument("--triplets-only", action="store_true", help="Disable quadruplet interactions.")
    p.add_argument(
        "--round-coords",
        type=int,
        default=0,
        help="Round coordinates to N decimals before featurising (0 = off). The MDL mol "
        "blocks of the released dataset carry 4 decimals; this makes a run on a "
        "full-precision pickle comparable with one on the dataset.",
    )
    p.add_argument(
        "--max-atoms", type=int, default=0, help="Truncate conformers to this many atoms (0 = no truncation)."
    )
    p.add_argument("--fp16", action="store_true", help="Run the forward pass under autocast(float16).")
    p.add_argument(
        "--tf32",
        action="store_true",
        help="Allow TensorFloat-32 matmuls on Ampere+ GPUs. Off by default: it costs "
        "about three mantissa bits and makes the output machine-dependent.",
    )
    p.add_argument("--limit", type=int, default=0, help="Process only the first N conformers.")
    p.add_argument("--start", type=int, default=0, help="Skip the first N conformers.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-threads", type=int, default=8)
    p.add_argument("--log-interval", type=int, default=5000, help="Progress line every N conformers.")
    p.add_argument(
        "--oom-retries",
        type=int,
        default=3,
        help="On CUDA OOM, wait and retry the batch, halving it each time down to 1 "
        "conformer. A few large molecules need far more memory than the median, "
        "and a shared GPU can be busy. 0 disables.",
    )
    p.add_argument("--oom-wait", type=float, default=30.0, help="Seconds to wait before each OOM retry.")
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Write the partial output every N conformers (0 = only at the end). The "
        "partial file is the same .npz with the rows computed so far.",
    )
    p.add_argument(
        "--work-dir",
        default=None,
        help="Where to put the temporary .npz that upstream's DataContainer reads (default: next to --out).",
    )
    p.add_argument(
        "--verify",
        nargs="?",
        const="published",
        default=None,
        metavar="REFERENCE",
        help="after writing, compare the output with a reference file: 'published' "
        "(the published GemNet chirality embedding), hub:<path in "
        "EscheWang/3dcs-embeddings>, or a local path",
    )
    p.add_argument(
        "--verify-key", default=None, help="array key to read from the --verify reference (default: its published key)"
    )
    p.add_argument("--verify-rows", default=None, metavar="ROWS", help=_common().ROW_SELECTION_HELP)
    return p.parse_args(argv)


# ------------------------------------------------------------------- environment
def ensure_imports(gemnet_repo: Path):
    """Put the upstream checkout on sys.path and make its two hard edges work:
    torch_scatter (falls back to compat/) and the numpy aliases later releases removed."""
    notes = []
    repo = gemnet_repo.resolve()
    if not (repo / "gemnet" / "model" / "gemnet.py").exists():
        raise SystemExit(f"--gemnet-repo {repo} does not look like a gemnet_pytorch checkout")
    # Upstream ships gemnet/ without __init__.py files; add them if they are missing.
    for sub in ("gemnet", "gemnet/model", "gemnet/model/layers", "gemnet/training"):
        init = repo / sub / "__init__.py"
        if (repo / sub).is_dir() and not init.exists():
            init.touch()
            notes.append(f"created {init.relative_to(repo)}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import torch_scatter  # noqa: F401

        notes.append("torch_scatter: installed package")
    except ImportError:
        sys.path.append(str(HERE / "compat"))
        import torch_scatter  # noqa: F401

        notes.append("torch_scatter: compat shim")
    import math

    # Upstream targets numpy < 1.24 and uses names later numpy releases removed:
    # np.bool (data_container.py) and np.math.factorial (basis_utils.py). Restore the
    # aliases here rather than editing the upstream checkout.
    restored = []
    for name, value in (("bool", bool), ("int", int), ("float", float), ("object", object)):
        if not hasattr(np, name):
            setattr(np, name, value)
            restored.append(f"np.{name}")
    if not hasattr(np, "math"):
        np.math = math
        restored.append("np.math")
    if restored:
        notes.append("restored removed numpy aliases for upstream: " + ", ".join(restored))
    return notes


def set_deterministic(seed: int, tf32: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32


def print_versions():
    import numba
    import rdkit
    import scipy
    import sympy
    import torch
    import torch_scatter

    print("[versions] python        ", sys.version.split()[0])
    print("[versions] numpy         ", np.__version__)
    print("[versions] scipy         ", scipy.__version__)
    print("[versions] sympy         ", sympy.__version__)
    print("[versions] numba         ", numba.__version__)
    print("[versions] rdkit         ", rdkit.__version__)
    print("[versions] torch         ", torch.__version__, "cuda", torch.version.cuda)
    print(
        "[versions] torch_scatter ", getattr(torch_scatter, "__version__", "compat shim"), "at", torch_scatter.__file__
    )


# ------------------------------------------------------------------------- data
def load_molecules(spec: str, *, limit: int | None = None, start: int = 0):
    """Return the conformers as RDKit Mol objects, in dataset row order.

    ``spec`` takes the ``--dataset`` syntax shared by every script in ``baselines/``
    (see ``baselines/common.py``): ``hf:<repo_id>[:<config>]``, ``hfdisk:<dir>`` or a plain
    ``save_to_disk`` directory, a bare Hub dataset id, a pickle of RDKit molecules, or
    ``lmdb:<file>`` for a rotation shard.  GemNet reads atomic numbers and coordinates only.
    """
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    return _common().load_conformers(spec, limit=limit, start=start)


def mol_to_arrays(mol, hydrogens: str, max_atoms: int, round_coords: int = 0):
    from rdkit import Chem

    if mol is None:
        return None
    mol = Chem.Mol(mol)
    if hydrogens == "all":
        mol = Chem.RemoveAllHs(mol)
    elif hydrogens == "remove":
        mol = Chem.RemoveHs(mol)
    if mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer(0)
    idx = [a.GetIdx() for a in mol.GetAtoms()]
    Z = np.array([mol.GetAtomWithIdx(i).GetAtomicNum() for i in idx], dtype=np.int64)
    R = np.array(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in idx],
        dtype=np.float32,
    )
    if round_coords:
        R = np.round(R.astype(np.float64), round_coords).astype(np.float32)
    if max_atoms and len(Z) > max_atoms:
        Z, R = Z[:max_atoms], R[:max_atoms]
    return {"Z": Z, "R": R}


def build_container(confs, work_dir: Path, cutoff: float, int_cutoff: float, triplets_only: bool):
    """Hand the conformers to upstream's own DataContainer, which builds the edge, triplet
    and quadruplet index sets. It reads an .npz off disk, so write one first."""
    from gemnet.training.data_container import DataContainer

    work_dir.mkdir(parents=True, exist_ok=True)
    N = np.array([c["Z"].size for c in confs], dtype=np.int64)
    Z = np.concatenate([c["Z"] for c in confs]).astype(np.int64)
    R = np.concatenate([c["R"] for c in confs]).astype(np.float32)
    path = work_dir / f"_gemnet_inputs_{len(confs)}_{os.getpid()}.npz"
    np.savez(path, N=N, Z=Z, R=R, E=np.zeros((len(confs), 1), dtype=np.float32), F=np.zeros_like(R))
    return DataContainer(str(path), cutoff, int_cutoff, triplets_only), path


# ------------------------------------------------------------------------ model
def load_model(pretrained_dir: Path, device, triplets_only: bool):
    import torch
    from gemnet.model.gemnet import GemNet

    with open(pretrained_dir / "model_kwargs.json") as fh:
        kwargs = json.load(fh)
    kwargs["triplets_only"] = triplets_only
    if "scale_file" in kwargs:
        kwargs["scale_file"] = str(pretrained_dir.parent / kwargs["scale_file"])
    model = GemNet(**kwargs).to(device)
    state = torch.load(pretrained_dir / "model.pth", map_location=device)
    if isinstance(state, dict):
        state = state.get("state_dict", state.get("model", state))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[load] missing keys:", missing)
    if unexpected:
        print("[load] unexpected keys:", unexpected)
    model.eval()
    return model


class _StopForward(Exception):
    """Raised from the hook once h has been captured, so the unused energy/force tail of
    the upstream forward (an autograd.grad call) never runs."""


class AtomFeatureReader:
    """Capture h, the atom representation leaving the last interaction block.

    Upstream's own `forward` pools exactly this tensor when it builds a graph
    embedding, so hooking it needs no change to the upstream file.
    """

    def __init__(self, model):
        self.h = None
        self._handle = model.int_blocks[-1].register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        self.h = output[0]
        raise _StopForward

    def __call__(self, model, inputs, pooling: str):
        import torch
        from torch_scatter import scatter

        self.h = None
        try:
            model(inputs)
        except _StopForward:
            pass
        if self.h is None:
            raise RuntimeError("the forward hook did not fire; check the gemnet_pytorch commit")
        batch_seg = inputs["batch_seg"]
        n_mol = int(torch.max(batch_seg)) + 1
        return scatter(self.h, batch_seg, dim=0, dim_size=n_mol, reduce=pooling)

    def close(self):
        self._handle.remove()


def _forward(model, reader, container, idx, device, use_cuda, args):
    import torch

    inputs = container[idx]
    inputs = {k: v.to(device, non_blocking=use_cuda) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16, enabled=args.fp16 and use_cuda):
            emb = reader(model, inputs, args.pooling)
    return emb.float().detach().cpu().numpy()


def run_batch(model, reader, container, idx, device, use_cuda, args):
    """One forward pass, retrying on CUDA OOM with progressively smaller sub-batches.

    Splitting a batch does not change the result: the graph is block-diagonal, so each
    conformer's embedding depends only on its own atoms.
    """
    import torch

    try:
        return _forward(model, reader, container, idx, device, use_cuda, args)
    except torch.cuda.OutOfMemoryError:
        if not use_cuda or args.oom_retries <= 0 or len(idx) == 1:
            raise
    size = max(1, len(idx) // 2)
    for attempt in range(1, args.oom_retries + 1):
        torch.cuda.empty_cache()
        print(
            f"[oom] rows {idx[0]}..{idx[-1]}: retry {attempt}/{args.oom_retries} "
            f"in sub-batches of {size} after {args.oom_wait:.0f}s",
            flush=True,
        )
        time.sleep(args.oom_wait)
        try:
            parts = [
                _forward(model, reader, container, idx[i : i + size], device, use_cuda, args)
                for i in range(0, len(idx), size)
            ]
            return np.concatenate(parts, axis=0)
        except torch.cuda.OutOfMemoryError:
            if size == 1:
                continue
            size = max(1, size // 2)
    raise RuntimeError(
        f"rows {idx[0]}..{idx[-1]} still OOM after {args.oom_retries} retries; "
        f"lower --batch-size or wait for the GPU to free up"
    )


# ------------------------------------------------------------------------- main
def main(argv=None):
    args = parse_args(argv)
    repo = Path(args.gemnet_repo)
    notes = ensure_imports(repo)
    import torch

    torch.set_num_threads(args.num_threads)
    set_deterministic(args.seed, tf32=args.tf32)
    for note in notes:
        print("[compat]", note)
    print_versions()
    print("[args]", json.dumps(vars(args)))

    pdir = Path(args.pretrained_dir) if args.pretrained_dir else repo / "pretrained" / "GemNet-Q"
    mols = load_molecules(args.dataset, limit=args.limit or None, start=args.start)
    lo = args.start
    hi = lo + len(mols)
    print(f"[data] {len(mols)} conformers (rows {lo}..{hi - 1}) from {args.dataset}")

    confs = [mol_to_arrays(m, args.hydrogens, args.max_atoms, args.round_coords) for m in mols]
    bad = [lo + i for i, c in enumerate(confs) if c is None]
    if bad:
        raise RuntimeError(f"conformer(s) {bad} could not be featurised; refusing to silently shift the row order")
    work = Path(args.work_dir) if args.work_dir else Path(args.out).resolve().parent
    container, npz_path = build_container(confs, work, args.cutoff, args.int_cutoff, args.triplets_only)
    print(
        f"[graph] upstream DataContainer over {len(container)} conformers "
        f"({sum(c['Z'].size for c in confs)} atoms), cutoff={args.cutoff} int_cutoff={args.int_cutoff}"
    )

    use_cuda = "cuda" in args.device and torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")
    model = load_model(pdir, device, args.triplets_only)
    reader = AtomFeatureReader(model)

    out = None
    t0 = time.time()
    done = 0
    next_log = 0
    next_ckpt = args.checkpoint_every
    for start in range(0, len(confs), args.batch_size):
        idx = np.arange(start, min(start + args.batch_size, len(confs)))
        emb = run_batch(model, reader, container, idx, device, use_cuda, args)
        if out is None:
            out = np.empty((len(confs), emb.shape[1]), dtype=np.float32)
        out[start : start + len(idx)] = emb
        done += len(idx)
        if done >= next_log:
            print(f"[run] {done}/{len(confs)}  {done / max(time.time() - t0, 1e-9):.1f} conf/s", flush=True)
            next_log += args.log_interval
        if args.checkpoint_every and done >= next_ckpt:
            np.savez(args.out, **{args.key: out[:done]})
            print(f"[ckpt] wrote {done} rows to {args.out}", flush=True)
            next_ckpt += args.checkpoint_every
    reader.close()
    npz_path.unlink(missing_ok=True)
    dt = time.time() - t0
    print(f"[run] done {done} conformers in {dt:.1f}s ({done / dt:.1f} conf/s)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **{args.key: out})
    print(f"[out] {args.out}  key={args.key}  shape={out.shape}  dtype={out.dtype}")
    print(f"[out] sha256={_common().sha256_file(args.out)}")

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
