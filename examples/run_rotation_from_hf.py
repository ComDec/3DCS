"""Download rotation dataset from HF and run evaluation.

Requires: pip install -e .
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
from datasets import load_dataset

from three_dbench.benchmarks import evaluate_rotation_embeddings
from three_dbench.embeddings import EmbeddingArray

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rotation evaluation from HF datasets")
    parser.add_argument("--repo-id", type=str, required=True, help="HF dataset repo, e.g. EscheWang/3dcs")
    parser.add_argument("--config", type=str, default="rotation", help="Dataset config name")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data" / "hf" / "rotation")
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=ROOT / "data" / "rotation" / "results" / "gemnet",
        help="Directory containing rotation_conformers_*.npz",
    )
    parser.add_argument("--embedding-key", type=str, default="gemnet")
    parser.add_argument("--model-name", type=str, default="gemnet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "rotation" / "gemnet")
    parser.add_argument(
        "--shards",
        type=int,
        nargs="*",
        default=None,
        help="Optional shard IDs to evaluate (e.g. --shards 0 1 2).",
    )
    parser.add_argument(
        "--max-keys",
        type=int,
        default=None,
        help="Optional cap on number of dataset keys for a quick test.",
    )
    return parser.parse_args()


def _load_embedding_npz(path: Path, key: str) -> np.ndarray:
    with np.load(path) as data:
        if key in data.files:
            return data[key]
        if "arr_0" in data.files:
            return data["arr_0"]
        return data[data.files[0]]


def _concat_embeddings(emb_dir: Path, shards: Iterable[int], key: str, cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        return np.load(cache_path, mmap_mode="r")

    arrays = []
    total = 0
    dim = None
    dtype = None
    for shard in shards:
        path = emb_dir / f"rotation_conformers_{shard}.npz"
        arr = _load_embedding_npz(path, key)
        arrays.append(arr)
        total += arr.shape[0]
        if dim is None:
            dim = arr.shape[1]
            dtype = arr.dtype

    if dim is None or dtype is None:
        raise ValueError("No embeddings loaded.")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(cache_path, mode="w+", dtype=dtype, shape=(total, dim))
    cursor = 0
    for arr in arrays:
        mm[cursor : cursor + arr.shape[0]] = arr
        cursor += arr.shape[0]
    mm.flush()
    return np.load(cache_path, mmap_mode="r")


def _reindex_offsets(rows: List[dict]) -> List[dict]:
    cursor = 0
    out = []
    for row in rows:
        row = dict(row)
        row["offset"] = int(cursor)
        cursor += int(row["n_conformers"])
        out.append(row)
    return out


def main() -> None:
    args = parse_args()

    if not args.dataset_dir.exists():
        ds = load_dataset(args.repo_id, name=args.config, split="train")
        args.dataset_dir.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(args.dataset_dir))
        print(f"Saved HF dataset to {args.dataset_dir}")
    else:
        print(f"Using cached dataset at {args.dataset_dir}")

    from datasets import load_from_disk, Dataset

    ds = load_from_disk(str(args.dataset_dir))
    if args.shards:
        shard_set = set(args.shards)
        ds = ds.filter(lambda row: row["shard"] in shard_set)
        rows = ds.to_list()
        rows = _reindex_offsets(rows)
        ds = Dataset.from_list(rows)

    tmp_dir = args.dataset_dir.parent / "rotation_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    shards = args.shards or list(range(16))
    cache_path = tmp_dir / f"rotation_{args.model_name}_all.npy"
    embeddings = _concat_embeddings(args.embeddings_dir, shards, args.embedding_key, cache_path)

    if args.max_keys is not None:
        ds = ds.select(range(min(args.max_keys, len(ds))))
        rows = _reindex_offsets(ds.to_list())
        ds = Dataset.from_list(rows)

    local_dir = tmp_dir / ("rotation_subset" if args.shards or args.max_keys else "rotation_full")
    ds.save_to_disk(str(local_dir))

    emb = EmbeddingArray(array=embeddings, kind="vector")

    total = int(sum(ds["n_conformers"]))
    if total > embeddings.shape[0]:
        raise ValueError(f"Embedding length {embeddings.shape[0]} < total conformers {total}")

    evaluate_rotation_embeddings(
        dataset_dir=local_dir,
        embeddings=emb,
        output_dir=args.output_dir,
        model_name=args.model_name,
    )
    print(f"Report saved to {args.output_dir}")


if __name__ == "__main__":
    main()
