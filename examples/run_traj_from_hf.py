"""Download trajectory energies from HF and run evaluation.

Requires: pip install -e .
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from three_dbench.benchmarks import evaluate_trajectory_embeddings
from three_dbench.embeddings import load_embeddings_dir

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trajectory evaluation from HF datasets")
    parser.add_argument("--repo-id", type=str, required=True, help="HF dataset repo, e.g. EscheWang/3dcs")
    parser.add_argument("--config", type=str, default="traj_energies", help="Dataset config name")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data" / "hf" / "traj" / "energies")
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=ROOT / "data" / "traj" / "results" / "unimol",
        help="Directory of rmd17_*.npz files",
    )
    parser.add_argument("--embedding-key", type=str, default="arr_0")
    parser.add_argument("--model-name", type=str, default="unimol")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "traj" / "unimol")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--window", type=int, default=2000)
    parser.add_argument("--metric-embed", type=str, default="cosine")
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--random-seed", type=int, default=2025)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dataset_dir.exists():
        ds = load_dataset(args.repo_id, name=args.config, split="train")
        args.dataset_dir.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(args.dataset_dir))
        print(f"Saved HF dataset to {args.dataset_dir}")
    else:
        print(f"Using cached dataset at {args.dataset_dir}")

    if not args.embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {args.embeddings_dir}")

    emb_dict = load_embeddings_dir(args.embeddings_dir, file_glob="rmd17_*.npz", key=args.embedding_key)
    evaluate_trajectory_embeddings(
        dataset_dir=args.dataset_dir,
        embeddings_by_mol=emb_dict,
        output_dir=args.output_dir,
        model_name=args.model_name,
        n_samples=args.n_samples,
        window=args.window,
        metric_embed=args.metric_embed,
        block_size=args.block_size,
        random_seed=args.random_seed,
    )
    print(f"Report saved to {args.output_dir}")


if __name__ == "__main__":
    main()
