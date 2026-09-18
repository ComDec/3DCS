#!/usr/bin/env python3
"""Extract Uni-Mol molecule embeddings for the 3DCS chirality benchmark.

What this computes
------------------
For every conformer of the benchmark, in dataset order, the 512-d representation
of the ``[CLS]`` token produced by the pretrained Uni-Mol molecular encoder
(``unimol_base``, checkpoint ``mol_pre_no_h_220816.pt``, hydrogens removed, one
conformer per input, no masking and no coordinate noise).  The output is a
``.npz`` holding a single ``(n_conformers, 512)`` float32 array under the key
``arr_0``.

The Uni-Mol data pipeline is used unmodified: this script builds the LMDB that
``unimol.tasks.UniMolTask`` expects, parses the same command line that
``unimol/infer.py`` parses, and then runs the encoder forward pass itself so that
only the ``[CLS]`` row is kept (``infer.py`` additionally materialises the
per-atom and per-pair representations, which for 52k conformers is hundreds of
gigabytes of intermediate pickle).

Requirements (third-party code and weights are NOT shipped with this script)
---------------------------------------------------------------------------
  * Uni-Core   https://github.com/dptech-corp/Uni-Core
  * Uni-Mol    https://github.com/deepmodeling/Uni-Mol  (the ``unimol/`` project)
  * weights    mol_pre_no_h_220816.pt, from the Uni-Mol release assets
  * dict.txt   Uni-Mol/unimol/example_data/molecule/dict.txt (shipped with Uni-Mol)
See README_unimol.md for exact versions and install commands.

Usage
-----
  python extract_chirality.py \
      --dataset hf:EscheWang/3dcs:chirality \
      --unimol-repo /path/to/Uni-Mol/unimol \
      --weights /path/to/mol_pre_no_h_220816.pt \
      --out chirality_unimol.npz

``--dataset`` takes the syntax shared by every script in ``baselines/``:
``hf:EscheWang/3dcs:chirality``, ``hfdisk:<dir>`` or a plain ``save_to_disk``
directory, a bare Hub dataset id, or a pickle holding RDKit molecules (a list, or
a dict whose values are lists; dict values are concatenated in insertion order).
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

CONF_SIZE = 1  # one conformer per LMDB record; TTADataset takes index 0
MAX_ATOMS = 256  # Uni-Mol default; the chirality set has <= 112 heavy atoms
DEFAULT_BATCH_SIZE = 256


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def load_molecules(spec: str, *, limit: int | None = None, start: int = 0):
    """Return the flat, dataset-ordered list of RDKit molecules.

    ``spec`` takes the ``--dataset`` syntax shared by every script in ``baselines/``
    (see ``baselines/common.py``): ``hf:<repo_id>[:<config>]``, ``hfdisk:<dir>`` or a plain
    ``save_to_disk`` directory, a bare Hub dataset id, a pickle of RDKit molecules, or
    ``lmdb:<file>``.  Uni-Mol reads element symbols and coordinates only.
    """
    mols = _common().load_conformers(spec, limit=limit, start=start)
    print(f"[data] {len(mols)} conformers from {spec}")
    return mols


def build_lmdb(mols, lmdb_path: Path) -> int:
    """Write the records ``unimol.data.LMDBDataset`` reads, keyed by row index.

    Record: {"atoms": [element symbols, hydrogens included],
             "coordinates": [float32 (n_atoms, 3) array],
             "smi": canonical SMILES (a name; it never enters the model input)}
    Hydrogens are stripped later by the Uni-Mol pipeline (``--only-polar 0``).
    """
    import lmdb
    from rdkit import Chem

    env = lmdb.open(
        str(lmdb_path),
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )
    txn = env.begin(write=True)
    for i, mol in enumerate(mols):
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        coords = [c.GetPositions().astype(np.float32) for c in mol.GetConformers()]
        if not coords:
            raise ValueError(f"molecule {i} has no conformer")
        try:
            smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
        except Exception:
            smi = Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False), canonical=False)
        txn.put(f"{i}".encode("ascii"), pickle.dumps({"atoms": atoms, "coordinates": coords, "smi": smi}, protocol=-1))
        if (i + 1) % 20000 == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    return len(mols)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def unimol_cli(
    data_dir: Path, subset: str, user_dir: Path, weights: Path, batch_size: int, num_workers: int, seed: int
):
    """The Uni-Mol inference command line, as a list of arguments.

    ``--only-polar 0`` removes every hydrogen (the checkpoint is the no-hydrogen
    model); ``--conf-size 1`` takes the single stored conformer;
    ``--random-token-prob 0`` together with ``--leave-unmasked-prob 1.0`` turns
    Uni-Mol's masking into a no-op, so no token is replaced and no coordinate
    noise is added.
    """
    return [
        str(data_dir),
        "--user-dir",
        str(user_dir),
        "--valid-subset",
        subset,
        "--results-path",
        str(data_dir),
        "--num-workers",
        str(num_workers),
        "--ddp-backend=c10d",
        "--batch-size",
        str(batch_size),
        "--task",
        "unimol",
        "--loss",
        "unimol_infer",
        "--arch",
        "unimol_base",
        "--path",
        str(weights),
        "--only-polar",
        "0",
        "--dict-name",
        "dict.txt",
        "--conf-size",
        str(CONF_SIZE),
        "--max-atoms",
        str(MAX_ATOMS),
        "--log-interval",
        "50",
        "--log-format",
        "simple",
        "--random-token-prob",
        "0",
        "--leave-unmasked-prob",
        "1.0",
        "--mode",
        "infer",
        "--seed",
        str(seed),
    ]


def run_encoder(argv, subset: str, device: str):
    import torch
    from unicore import checkpoint_utils, options, tasks, utils

    parser = options.get_validation_parser()
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser, input_args=argv)

    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    missing, _ = model.load_state_dict(state["model"], strict=False)
    if missing:
        print(f"[model] parameters not in the checkpoint: {sorted(missing)}")
    use_cuda = device != "cpu"
    if use_cuda:
        torch.cuda.set_device(torch.device(device).index or 0)
        model.cuda()
    model.eval()

    task.load_dataset(subset, combine=False, epoch=1)
    dataset = task.dataset(subset)
    itr = task.get_batch_iterator(
        dataset=dataset,
        batch_size=args.batch_size,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=args.required_batch_size_multiple,
        seed=args.seed,
        num_shards=1,
        shard_id=0,
        num_workers=args.num_workers,
        data_buffer_size=args.data_buffer_size,
    ).next_epoch_itr(shuffle=False)

    chunks, n_done, t0 = [], 0, time.time()
    with torch.no_grad():
        for sample in itr:
            if len(sample) == 0:
                continue
            if use_cuda:
                sample = utils.move_to_cuda(sample)
            # This is the tensor UniMolInferLoss records as "mol_repr_cls".
            encoder_rep, _ = model(**sample["net_input"], features_only=True)
            chunks.append(encoder_rep[:, 0, :].float().cpu().numpy())
            n_done += chunks[-1].shape[0]
            if len(chunks) % 50 == 0:
                print(f"[infer] {n_done} conformers, {time.time() - t0:.0f}s", flush=True)
    out = np.concatenate(chunks, axis=0).astype(np.float32)
    print(f"[infer] {out.shape[0]} conformers in {time.time() - t0:.0f}s")
    return out


# --------------------------------------------------------------------------- #
MODEL_NAME = "unimol"
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def print_versions(weights: Path):
    import lmdb
    import rdkit
    import torch
    import unicore

    print(f"[env] python      {sys.version.split()[0]}")
    print(f"[env] torch       {torch.__version__} (cuda {torch.version.cuda})")
    print("[env] unicore     {}".format(getattr(unicore, "__version__", "unknown")))
    print(f"[env] numpy       {np.__version__}")
    print(f"[env] rdkit       {rdkit.__version__}")
    print(f"[env] lmdb        {lmdb.__version__}")
    if torch.cuda.is_available():
        print(f"[env] gpu         {torch.cuda.get_device_name(0)}")
    print(f"[env] weights     {weights.name} sha256={sha256(weights)}")


def main() -> int:
    common = _common()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help=common.DATASET_SPEC_HELP)
    ap.add_argument("--out", required=True, help="output .npz (single array under key arr_0)")
    ap.add_argument("--unimol-repo", required=True, help="the unimol/ project directory of a Uni-Mol checkout")
    ap.add_argument("--weights", required=True, help="mol_pre_no_h_220816.pt")
    ap.add_argument("--dict", default=None, help="dict.txt (default: <unimol-repo>/example_data/molecule/dict.txt)")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--device", default="cuda:0", help="cuda:N or cpu")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Uni-Core seed; the inference pipeline is deterministic, so it does not "
        "change the output (masking is disabled and one conformer is stored)",
    )
    ap.add_argument("--limit", type=int, default=None, help="only process the first N conformers")
    ap.add_argument("--start", type=int, default=0, help="skip the first N conformers")
    ap.add_argument("--work-dir", default=None, help="scratch directory for the LMDB")
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument(
        "--verify",
        nargs="?",
        const="published",
        default=None,
        metavar="REFERENCE",
        help="after writing, compare the output with a reference file: 'published' "
        "(the published Uni-Mol chirality embedding), hub:<path in "
        "EscheWang/3dcs-embeddings>, or a local path",
    )
    ap.add_argument(
        "--verify-key", default=None, help="array key to read from the --verify reference (default: its published key)"
    )
    ap.add_argument("--verify-rows", default=None, metavar="ROWS", help=common.ROW_SELECTION_HELP)
    a = ap.parse_args()

    repo = Path(a.unimol_repo).resolve()
    weights = Path(a.weights).resolve()
    dict_txt = Path(a.dict).resolve() if a.dict else repo / "example_data" / "molecule" / "dict.txt"
    for p, what in ((repo / "unimol", "Uni-Mol package"), (weights, "weights"), (dict_txt, "dict.txt")):
        if not p.exists():
            sys.exit(f"{what} not found: {p}")
    sys.path.insert(0, str(repo))

    print_versions(weights)
    print(f"[env] dict.txt    {dict_txt} sha256={sha256(dict_txt)}")

    work = Path(a.work_dir).resolve() if a.work_dir else Path(tempfile.mkdtemp(prefix="unimol_chi_"))
    data_dir = work / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(dict_txt, data_dir / "dict.txt")
    subset = "mols"

    try:
        mols = load_molecules(a.dataset, limit=a.limit, start=a.start)
        n = build_lmdb(mols, data_dir / (subset + ".lmdb"))
        del mols
        lmdb_name = data_dir / (subset + ".lmdb")
        print(f"[data] wrote {n} records to {lmdb_name}")

        argv = unimol_cli(data_dir, subset, repo / "unimol", weights, a.batch_size, a.num_workers, a.seed)
        print("[run ] " + " ".join(argv))
        emb = run_encoder(argv, subset, a.device)
        if emb.shape[0] != n:
            sys.exit(f"got {emb.shape[0]} rows for {n} inputs")

        out = Path(a.out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, emb)
        print(f"[out ] {out}  shape={emb.shape} dtype={emb.dtype} key=arr_0")
        print(f"[out ] sha256={sha256(out)}")

        if a.verify:
            common.verify(
                out,
                model=MODEL_NAME,
                reference=a.verify,
                produced_key=OUTPUT_KEY,
                reference_key=a.verify_key,
                rows=a.verify_rows or (f"{a.start}+" if a.start else None),
            )
    finally:
        if not a.keep_work and a.work_dir is None:
            shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
