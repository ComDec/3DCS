#!/usr/bin/env python
"""Extract E3FP fingerprints for the 3DCS chirality benchmark.

What this script computes
-------------------------
For every input conformer it computes one E3FP (Extended 3-Dimensional FingerPrint)
with the parameters below and converts it to an RDKit ``ExplicitBitVect``:

    bits=1024, level=5, radius_multiplier=1.5, stereo=True,
    include_disconnected=True, rdkit_invariants=True, first=1, counts=False

Hydrogens are kept: the conformer is passed to ``e3fp`` exactly as it is stored in the
dataset.  These are not the ``e3fp`` defaults (which are ``bits=4096``,
``radius_multiplier=1.718``, ``rdkit_invariants=False``).

The output is a pickle holding ``{"e3fp": [ExplicitBitVect, ...]}``, one entry per
conformer in benchmark row order -- the layout of the published
``chirality/e3fp/sampled_chi.pkl``, which the chirality evaluator reads with
``--embedding-key e3fp`` and scores with the Tanimoto distance.

Third-party dependencies (not redistributed here)
-------------------------------------------------
  * e3fp     -- https://github.com/keiserlab/e3fp (LGPL-3.0).  Pinned: 1.2.7
  * rdkit, numpy, smart_open, sdaxen_python_utilities (pulled in by e3fp)

Example
-------
  python extract_chirality.py \\
      --dataset hf:EscheWang/3dcs:chirality \\
      --out sampled_chi.pkl --jobs 8 --verify
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import pickle
import sys
import time
from collections.abc import Sequence

MODEL_NAME = "e3fp"
OUTPUT_KEY = "e3fp"

#: The fingerprint parameters of the published file.
FPRINT_PARAMS = {
    "bits": 1024,
    "level": 5,
    "radius_multiplier": 1.5,
    "stereo": True,
    "include_disconnected": True,
    "rdkit_invariants": True,
    "first": 1,
    "counts": False,
}

_PARAMS: dict = {}


def _common():
    """Load ``baselines/common.py`` (input and verification helpers) without touching sys.path."""
    import importlib.util
    import os

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common.py")
    spec = importlib.util.spec_from_file_location("baselines_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_molecules(spec: str, *, limit: int | None = None, start: int = 0):
    """Return the input conformers as RDKit molecules, in benchmark row order.

    ``spec`` takes the ``--dataset`` syntax shared by every script in ``baselines/``
    (see ``baselines/common.py``).  E3FP reads bond orders and stereochemistry, so a MOL
    block RDKit refuses to sanitise is an error rather than a row read unsanitised.
    """
    return _common().load_conformers(spec, limit=limit, start=start, sanitize_fallback=False)


def fingerprint(mol):
    """E3FP of conformer 0 of ``mol`` as an RDKit ``ExplicitBitVect``."""
    from e3fp.pipeline import fprints_from_mol

    if not mol.HasProp("_Name"):
        mol.SetProp("_Name", "mol")  # e3fp names the fingerprint after the molecule
    return fprints_from_mol(mol, fprint_params=_PARAMS or FPRINT_PARAMS)[0].to_rdkit()


def _quiet_e3fp() -> None:
    """e3fp logs one INFO line per molecule on the root logger; keep the run log readable."""
    logging.getLogger().setLevel(logging.WARNING)


def _init_worker(params: dict) -> None:
    global _PARAMS
    _PARAMS = params
    _quiet_e3fp()


def run(mols, params: dict, jobs: int, log_every: int) -> list:
    """Fingerprint every molecule, preserving the input order."""
    global _PARAMS
    _PARAMS = params
    n = len(mols)
    started = time.time()
    if jobs <= 1:
        out = []
        for i, mol in enumerate(mols):
            out.append(fingerprint(mol))
            if log_every and (i + 1) % log_every == 0:
                print(f"  {i + 1}/{n}  {(i + 1) / (time.time() - started):.1f} conf/s", flush=True)
        return out

    import multiprocessing as mp

    with mp.Pool(jobs, initializer=_init_worker, initargs=(params,)) as pool:
        out = []
        for i, fp in enumerate(pool.imap(fingerprint, mols, chunksize=64)):
            out.append(fp)
            if log_every and (i + 1) % log_every == 0:
                print(f"  {i + 1}/{n}  {(i + 1) / (time.time() - started):.1f} conf/s", flush=True)
    return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    common = _common()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help=common.DATASET_SPEC_HELP)
    p.add_argument("--out", required=True, help="output .pkl")
    p.add_argument("--key", default=OUTPUT_KEY, help="dict key written into the pickle (default: e3fp)")
    p.add_argument("--bits", type=int, default=FPRINT_PARAMS["bits"])
    p.add_argument("--level", type=int, default=FPRINT_PARAMS["level"])
    p.add_argument("--radius-multiplier", type=float, default=FPRINT_PARAMS["radius_multiplier"])
    p.add_argument("--first", type=int, default=FPRINT_PARAMS["first"])
    p.add_argument(
        "--no-stereo",
        dest="stereo",
        action="store_false",
        help="drop the stereochemical atom invariants (the published file uses stereo=True)",
    )
    p.add_argument("--no-include-disconnected", dest="include_disconnected", action="store_false")
    p.add_argument("--no-rdkit-invariants", dest="rdkit_invariants", action="store_false")
    p.add_argument("--counts", action="store_true", help="count fingerprint instead of a bit vector")
    p.add_argument("--jobs", type=int, default=1, help="worker processes (default: 1)")
    p.add_argument("--limit", type=int, default=None, help="only process the first N conformers")
    p.add_argument("--start", type=int, default=0, help="skip the first N conformers")
    p.add_argument("--log-every", type=int, default=5000)
    p.add_argument(
        "--verify",
        nargs="?",
        const="published",
        default=None,
        metavar="REFERENCE",
        help="after writing, compare the output with a reference file: 'published' "
        "(the published E3FP chirality fingerprints), hub:<path in "
        "EscheWang/3dcs-embeddings>, or a local path",
    )
    p.add_argument(
        "--verify-key", default=None, help="dict key to read from the --verify reference (default: its published key)"
    )
    p.add_argument("--verify-rows", default=None, metavar="ROWS", help=common.ROW_SELECTION_HELP)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    common = _common()
    common.print_versions(("e3fp", "rdkit"))
    _quiet_e3fp()

    params = {
        "bits": args.bits,
        "level": args.level,
        "radius_multiplier": args.radius_multiplier,
        "stereo": args.stereo,
        "include_disconnected": args.include_disconnected,
        "rdkit_invariants": args.rdkit_invariants,
        "first": args.first,
        "counts": args.counts,
    }
    print(f"[params] {params}")

    mols = load_molecules(args.dataset, limit=args.limit, start=args.start)
    print(f"[data] {len(mols)} conformers from {args.dataset}", flush=True)

    started = time.time()
    fprints = run(mols, params, args.jobs, args.log_every)
    print(f"[run] {len(fprints)} fingerprints in {time.time() - started:.1f}s")

    with open(args.out, "wb") as handle:
        pickle.dump({args.key: fprints}, handle)
    with open(args.out, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    print(f"[out] {args.out}  key={args.key}  n={len(fprints)}  sha256={digest}")

    if args.verify:
        common.verify(
            args.out,
            model=MODEL_NAME,
            reference=args.verify,
            produced_key=args.key,
            reference_key=args.verify_key,
            rows=args.verify_rows or (f"{args.start}+" if args.start else None),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
