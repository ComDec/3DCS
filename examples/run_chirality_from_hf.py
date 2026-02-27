"""Download chirality dataset from HF and run evaluation.

Requires: pip install -e .
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from three_dbench.benchmarks import evaluate_chirality_embeddings
from three_dbench.embeddings import load_embeddings

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run chirality evaluation from HF datasets")
    parser.add_argument("--repo-id", type=str, required=True, help="HF dataset repo, e.g. EscheWang/3dcs")
    parser.add_argument("--config", type=str, default="chirality", help="Dataset config name")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data" / "hf" / "chirality")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=ROOT / "data" / "chirality" / "unimol" / "1.npz",
        help="Embedding file path",
    )
    parser.add_argument("--embedding-key", type=str, default="arr_0")
    parser.add_argument("--model-name", type=str, default="unimol")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "chirality" / "unimol")
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

    if not args.embeddings.exists():
        raise FileNotFoundError(f"Embeddings not found: {args.embeddings}")

    embeddings = load_embeddings(args.embeddings, key=args.embedding_key)
    evaluate_chirality_embeddings(
        dataset_dir=args.dataset_dir,
        embeddings=embeddings,
        output_dir=args.output_dir,
        model_name=args.model_name,
    )
    print(f"Report saved to {args.output_dir}")


if __name__ == "__main__":
    main()
