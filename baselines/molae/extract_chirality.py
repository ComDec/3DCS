#!/usr/bin/env python3
"""Extract Mol-AE (MolAE) molecule-level embeddings for the 3DCS chirality benchmark.

What this script computes
-------------------------
For every input conformer it returns the 512-d encoder representation of the
``[CLS]`` token of the Mol-AE encoder, evaluated with the Uni-Mol inference
pipeline (``--task unimol --loss unimol_infer --arch unimol_base --mode infer``).

The Mol-AE pre-trained checkpoint (``arch: unimol_MAE_padding``) contains a
15-layer / 512-d encoder whose parameter names and shapes are identical to
Uni-Mol ``unimol_base`` plus an extra 5-layer MAE decoder.  The decoder is not
used at inference time: the checkpoint is loaded with ``strict=False`` into a
``unimol_base`` model, so only the encoder / embedding / GBF parameters are
consumed.  The script prints the missing / unexpected key counts so the load is
auditable.

Pre-processing (inherited unchanged from upstream Uni-Mol):
  * all hydrogens removed (``--only-polar 0`` -> ``remove_hydrogen=True``)
  * molecules cropped to at most ``--max-atoms`` heavy atoms
  * coordinates centred (mean subtracted); no rotation, no scaling
  * ``--random-token-prob 0 --leave-unmasked-prob 1.0`` makes ``MaskPointsDataset``
    a deterministic identity: no token is replaced and no coordinate noise is added
  * ``[CLS]`` prepended, ``[SEP]`` appended, both with coordinate (0, 0, 0)
  * ``--conf-size 1``: one conformer per input record, no test-time augmentation

Output: an ``.npz`` written with ``numpy.savez(out, embeddings)``, i.e. a single
array under the key ``arr_0`` of shape ``(n_conformers, 512)``, float32, in the
same order as the input records.

Third-party code and weights are NOT redistributed with this script.  See
``ENVIRONMENT.md`` for the upstream repositories, commits and the checkpoint the
script expects.

Example
-------
    python extract_chirality.py \
        --dataset hf:EscheWang/3dcs:chirality \
        --weights checkpoint_7_1000000.pt \
        --unimol-dir /path/to/Uni-Mol/unimol/unimol \
        --dict /path/to/Uni-Mol/unimol/example_data/molecule/dict.txt \
        --out molae_chirality.npz \
        --batch-size 256 --device cuda:0
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import random
import shutil
import sys
import tempfile
import time

import numpy as np


# --------------------------------------------------------------------------- #
# input handling
# --------------------------------------------------------------------------- #
def _common():
    """Load ``baselines/common.py`` (input and verification helpers) without touching sys.path."""
    import importlib.util

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common.py")
    spec = importlib.util.spec_from_file_location("baselines_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_molecules(
    spec: str,
    hf_split: str = "train",
    hf_config: str = "chirality",
    *,
    limit: int | None = None,
    start: int = 0,
):
    """Return a list of RDKit molecules, in benchmark row order.

    ``spec`` takes the ``--dataset`` syntax shared by every script in ``baselines/``
    (see ``baselines/common.py``): ``hf:<repo_id>[:<config>]``, ``hfdisk:<dir>`` or a plain
    ``save_to_disk`` directory, a bare Hub dataset id, a pickle of RDKit molecules (a list,
    or a dict of lists flattened in insertion order), or ``lmdb:<file>``.

    The ``chirality`` config of ``EscheWang/3dcs`` stores one ``mol_blocks`` list per
    stereoisomer -- one MDL MOL block per conformer -- and the ``offset`` of the first of
    them, so reading ``mol_blocks`` in ascending ``offset`` gives the 52,391 conformers in
    the row order of the published embedding file.  Mol-AE reads element symbols and
    coordinates only.
    """
    return _common().load_conformers(spec, hf_config=hf_config, hf_split=hf_split, limit=limit, start=start)


def build_lmdb(mols, lmdb_path: str) -> int:
    """Write the molecules in the Uni-Mol inference LMDB format.

    Each record is ``{"atoms": [symbol, ...], "coordinates": [array(n, 3), ...],
    "smi": canonical_smiles}`` stored under the ascii key ``str(index)``.
    ``LMDBDataset.__getitem__`` looks records up by ``str(idx)``, so the row order
    of the output is exactly the order of ``mols``.
    """
    import lmdb
    from rdkit import Chem

    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )
    txn = env.begin(write=True)
    n = 0
    for idx, mol in enumerate(mols):
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        coordinates = [conf.GetPositions().astype(np.float32) for conf in mol.GetConformers()]
        if not coordinates:
            raise ValueError(f"molecule {idx} has no conformer")
        # `smi` is only a per-row name carried through to the logging output; it
        # never enters the model input, so a molecule RDKit refuses to
        # re-sanitise still yields the correct embedding.
        try:
            smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
        except Exception:
            try:
                smi = Chem.MolToSmiles(mol)
            except Exception:
                smi = f"row{idx}"
        txn.put(
            f"{idx}".encode("ascii"),
            pickle.dumps({"atoms": atoms, "coordinates": coordinates, "smi": smi}, protocol=-1),
        )
        n += 1
        if n % 20000 == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    return n


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
def extract(args) -> np.ndarray:
    import torch
    from unicore import checkpoint_utils, options, tasks, utils

    parser = options.get_validation_parser()
    options.add_model_args(parser)
    argv = [
        args.data_dir,
        "--user-dir",
        args.unimol_dir,
        "--valid-subset",
        args.subset,
        "--results-path",
        args.data_dir,
        "--num-workers",
        str(args.num_workers),
        "--ddp-backend",
        "c10d",
        "--batch-size",
        str(args.batch_size),
        "--task",
        "unimol",
        "--loss",
        "unimol_infer",
        "--arch",
        "unimol_base",
        "--path",
        args.weights,
        "--only-polar",
        "0",
        "--dict-name",
        os.path.basename(args.dict),
        "--conf-size",
        str(args.conf_size),
        "--max-atoms",
        str(args.max_atoms),
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
        str(args.seed),
    ]
    model_args = options.parse_args_and_arch(parser, argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.tf32 is False:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    device = torch.device(args.device)
    state = checkpoint_utils.load_checkpoint_to_cpu(model_args.path)
    ckpt_arch = getattr(state.get("args", None), "arch", None)
    print(f"[info] checkpoint arch recorded in file: {ckpt_arch}")

    task = tasks.setup_task(model_args)
    model = task.build_model(model_args)
    incompatible = model.load_state_dict(state["model"], strict=False)
    print(
        f"[info] load_state_dict(strict=False): "
        f"{len(incompatible.missing_keys)} missing, "
        f"{len(incompatible.unexpected_keys)} unexpected"
    )
    if incompatible.missing_keys:
        print(f"[info] missing (sample): {incompatible.missing_keys[:8]}")
    if incompatible.unexpected_keys:
        prefixes = sorted({k.split(".")[0] for k in incompatible.unexpected_keys})
        print(f"[info] unexpected key prefixes (unused): {prefixes}")
    model = model.to(device)
    if args.fp16:
        model = model.half()
    model.eval()

    task.load_dataset(args.subset, combine=False, epoch=1)
    dataset = task.dataset(args.subset)
    itr = task.get_batch_iterator(
        dataset=dataset,
        batch_size=model_args.batch_size,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=model_args.required_batch_size_multiple,
        seed=model_args.seed,
        num_shards=1,
        shard_id=0,
        num_workers=model_args.num_workers,
        data_buffer_size=model_args.data_buffer_size,
    ).next_epoch_itr(shuffle=False)

    reps = []
    names = []
    t0 = time.time()
    n_done = 0
    with torch.no_grad():
        for sample in itr:
            if len(sample) == 0:
                continue
            sample = utils.move_to_cuda(sample) if device.type == "cuda" else sample
            encoder_rep, _ = model(**sample["net_input"], features_only=True)
            if args.pooling == "cls":
                pooled = encoder_rep[:, 0, :]
            else:
                tokens = sample["net_input"]["src_tokens"]
                pad = task.dictionary.pad()
                bos = task.dictionary.bos()
                eos = task.dictionary.eos()
                mask = (tokens.ne(pad) & tokens.ne(bos) & tokens.ne(eos)).unsqueeze(-1).to(encoder_rep.dtype)
                if args.pooling == "mean":
                    pooled = (encoder_rep * mask).sum(1) / mask.sum(1).clamp(min=1)
                elif args.pooling == "sum":
                    pooled = (encoder_rep * mask).sum(1)
                else:
                    raise ValueError(args.pooling)
            reps.append(pooled.float().data.cpu().numpy())
            names.extend(sample["target"]["smi_name"])
            n_done += pooled.shape[0]
            if n_done % (args.batch_size * 20) == 0:
                print(f"[info] {n_done} conformers, {time.time() - t0:.1f}s", flush=True)

    embeddings = np.concatenate(reps, axis=0).astype(np.float32)
    print(
        f"[info] done: {embeddings.shape} in {time.time() - t0:.1f}s "
        f"({embeddings.shape[0] / max(time.time() - t0, 1e-9):.1f} conf/s)"
    )
    if args.names_out:
        np.save(args.names_out, np.array(names, dtype=object), allow_pickle=True)
    return embeddings


# --------------------------------------------------------------------------- #
MODEL_NAME = "molae"
OUTPUT_KEY = "arr_0"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    common = _common()
    ap = argparse.ArgumentParser(description="Mol-AE [CLS] embedding extraction for the 3DCS chirality set")
    ap.add_argument("--dataset", required=True, help=common.DATASET_SPEC_HELP)
    ap.add_argument("--hf-config", default="chirality", help="config to read when --dataset names a Hub dataset")
    ap.add_argument("--hf-split", default="train")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N conformers")
    ap.add_argument("--start", type=int, default=0, help="skip the first N conformers")
    ap.add_argument("--out", required=True, help="output .npz (key arr_0)")
    ap.add_argument("--weights", required=True, help="Mol-AE pre-trained checkpoint")
    ap.add_argument(
        "--unimol-dir",
        required=True,
        help="path to the Uni-Mol 'unimol' python package (used as --user-dir)",
    )
    ap.add_argument("--dict", required=True, help="path to Uni-Mol dict.txt")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--conf-size", type=int, default=1)
    ap.add_argument("--max-atoms", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pooling", default="cls", choices=["cls", "mean", "sum"])
    ap.add_argument("--fp16", action="store_true", help="run the model in half precision")
    ap.add_argument(
        "--tf32",
        action="store_true",
        help="allow TF32 matmuls (off by default for reproducibility)",
    )
    ap.add_argument(
        "--work-dir",
        default=None,
        help="where the intermediate LMDB is written (default: a temp dir)",
    )
    ap.add_argument("--keep-work-dir", action="store_true")
    ap.add_argument("--names-out", default=None, help="optional .npy of per-row SMILES")
    ap.add_argument(
        "--verify",
        nargs="?",
        const="published",
        default=None,
        metavar="REFERENCE",
        help="after writing, compare the output with a reference file: 'published' "
        "(the published Mol-AE chirality embedding), hub:<path in "
        "EscheWang/3dcs-embeddings>, or a local path",
    )
    ap.add_argument(
        "--verify-key", default=None, help="array key to read from the --verify reference (default: its published key)"
    )
    ap.add_argument("--verify-rows", default=None, metavar="ROWS", help=common.ROW_SELECTION_HELP)
    args = ap.parse_args()

    work_dir = args.work_dir or tempfile.mkdtemp(prefix="molae_extract_")
    os.makedirs(work_dir, exist_ok=True)
    args.data_dir = work_dir
    args.subset = "molae_input"

    shutil.copyfile(args.dict, os.path.join(work_dir, os.path.basename(args.dict)))

    lmdb_path = os.path.join(work_dir, args.subset + ".lmdb")
    if not os.path.exists(lmdb_path):
        mols = load_molecules(args.dataset, args.hf_split, args.hf_config, limit=args.limit, start=args.start)
        n = build_lmdb(mols, lmdb_path)
        print(f"[info] wrote {n} records to {lmdb_path}")
    else:
        print(f"[info] reusing existing LMDB {lmdb_path}")

    # versions
    import lmdb as lmdb_mod
    import rdkit
    import torch
    import unicore

    print(f"[versions] python      {sys.version.split()[0]}")
    print(f"[versions] numpy       {np.__version__}")
    print(f"[versions] torch       {torch.__version__} (cuda {torch.version.cuda})")
    print(f"[versions] rdkit       {rdkit.__version__}")
    print(f"[versions] lmdb        {lmdb_mod.__version__}")
    print(f"[versions] unicore     {unicore.__version__}")
    print(f"[versions] weights     {os.path.basename(args.weights)}")
    print(f"[versions] weights sha {sha256(args.weights)}")
    print(f"[versions] dict sha    {sha256(args.dict)}")

    embeddings = extract(args)
    np.savez(args.out, embeddings)
    print(f"[info] wrote {args.out} shape={embeddings.shape} dtype={embeddings.dtype}")
    print(f"[info] output sha256 {sha256(args.out)}")

    if args.verify:
        common.verify(
            args.out,
            model=MODEL_NAME,
            reference=args.verify,
            produced_key=OUTPUT_KEY,
            reference_key=args.verify_key,
            rows=args.verify_rows or (f"{args.start}+" if args.start else None),
        )

    if not args.keep_work_dir and args.work_dir is None:
        shutil.rmtree(work_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
