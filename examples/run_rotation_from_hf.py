"""Download the rotation dataset from HF (if needed) and evaluate per-shard embeddings.

Requires: pip install -e .

The published GemNet rotation embeddings (``EscheWang/3dcs-embeddings``, ``rotation/gemnet``) hold
one file per shard, ``rotation_conformers_{shard}.npz``; rows of shard ``s`` are aligned with the
per-shard ``offset`` column of the HF rotation config. Example::

    python -m three_dbench download dataset --task rotation
    python -m three_dbench download embeddings --task rotation --models gemnet
    python examples/run_rotation_from_hf.py --shards 0 --max-keys 200 --n-jobs 4

``--flat-cache`` additionally writes the selected shards as one ``.npy`` in dataset row order
(``numpy.lib.format.open_memmap``) and evaluates that flat array instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from three_dbench.benchmarks import evaluate_rotation_embeddings
from three_dbench.benchmarks.rotation import ShardedEmbeddings, build_rotation_index, write_flat_embeddings
from three_dbench.embeddings import EmbeddingArray

ROOT = Path.cwd()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rotation evaluation from HF datasets")
    parser.add_argument("--repo-id", type=str, default="EscheWang/3dcs", help="HF dataset repo")
    parser.add_argument("--config", type=str, default="rotation", help="Dataset config name")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data" / "hf" / "rotation")
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=ROOT / "data" / "embeddings" / "rotation" / "gemnet",
        help="Directory with one embedding file per shard (e.g. rotation_conformers_{shard}.npz)",
    )
    parser.add_argument("--embedding-key", type=str, default=None, help="NPZ key (default: auto)")
    parser.add_argument("--shard-file-pattern", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="gemnet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "rotation" / "gemnet")
    parser.add_argument("--shards", type=int, nargs="*", default=None, help="Shard IDs to evaluate")
    parser.add_argument("--max-keys", type=int, default=None, help="Evaluate at most N molecules")
    parser.add_argument("--molecule-list", type=Path, default=None, help="File with one key per line")
    parser.add_argument("--sample-ratio", type=float, default=None)
    parser.add_argument("--sample-seed", type=int, default=2027)
    parser.add_argument("--metric-version", type=str, choices=["paper", "v2"], default="paper")
    parser.add_argument("--metrics", type=str, nargs="*", default=["cosine", "euclidean"])
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--flat-cache", type=Path, default=None, help="Directory for an optional flat .npy cache")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import load_dataset, load_from_disk

    if not args.dataset_dir.exists():
        ds = load_dataset(args.repo_id, name=args.config, split="train")
        args.dataset_dir.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(args.dataset_dir))
        print(f"Saved HF dataset to {args.dataset_dir}")
    else:
        print(f"Using cached dataset at {args.dataset_dir}")
    ds = load_from_disk(str(args.dataset_dir))

    sharded = ShardedEmbeddings.from_directory(
        args.embeddings_dir, key=args.embedding_key, pattern=args.shard_file_pattern
    )
    shards = args.shards if args.shards is not None else sharded.shards
    common = dict(
        output_dir=args.output_dir,
        model_name=args.model_name,
        metrics=args.metrics,
        metric_version=args.metric_version,
        molecule_list=args.molecule_list,
        sample_ratio=args.sample_ratio,
        sample_seed=args.sample_seed,
        max_keys=args.max_keys,
        n_jobs=args.n_jobs,
        progress=True,
    )

    if args.flat_cache is None:
        evaluate_rotation_embeddings(dataset=ds, embeddings_by_shard=sharded, shards=shards, **common)
    else:
        # Flat layout: restrict the dataset to the requested shards first, so that the flat
        # array covers exactly the rows of the evaluated dataset, in dataset row order.
        shard_set = set(shards)
        subset = ds.filter(lambda s: s in shard_set, input_columns=["shard"])
        index = build_rotation_index(subset)
        tag = "-".join(str(s) for s in sorted(shard_set))
        cache = args.flat_cache / f"rotation_{args.model_name}_{args.embedding_key or 'auto'}_shards{tag}.npy"
        if cache.exists():
            import numpy as np

            flat = np.load(cache, mmap_mode="r")
            print(f"Using flat cache {cache}")
        else:
            flat = write_flat_embeddings(index, sharded, cache)
            print(f"Wrote flat cache {cache} {flat.shape}")
        if flat.shape[0] != int(index["n_conformers"].sum()):
            raise ValueError(f"Flat cache {cache} does not match the dataset; delete it and rerun.")
        evaluate_rotation_embeddings(dataset=subset, embeddings=EmbeddingArray(array=flat, kind="vector"), **common)
    print(f"Report saved to {args.output_dir}")


if __name__ == "__main__":
    main()
