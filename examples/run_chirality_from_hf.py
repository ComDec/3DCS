"""Download the chirality dataset from HF and evaluate one embedding file.

Requires: pip install -e .

Example (published UniMol embedding, Table 2 protocol)::

    python examples/run_chirality_from_hf.py \
        --embeddings data/embeddings/chirality/unimol/1.npz --embedding-key arr_0 --model-name unimol
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from three_dbench.benchmarks.chirality import evaluate_chirality_embeddings, load_chirality_embeddings

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run chirality evaluation from HF datasets")
    parser.add_argument("--repo-id", type=str, default="EscheWang/3dcs", help="HF dataset repo")
    parser.add_argument("--config", type=str, default="chirality", help="Dataset config name")
    parser.add_argument("--revision", type=str, default=None, help="Optional HF revision (commit hash)")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data" / "hf" / "chirality")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=ROOT / "data" / "embeddings" / "chirality" / "unimol" / "1.npz",
        help="Embedding file: one row per conformer in dataset order, or a pickled fingerprint list/dict",
    )
    parser.add_argument("--embedding-key", type=str, default="arr_0", help="NPZ key or pickle dict entry (e3fp)")
    parser.add_argument("--model-name", type=str, default="unimol")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--distance", choices=["euclidean", "cosine"], default="euclidean")
    parser.add_argument("--metric-version", choices=["paper", "v2"], default="paper")
    parser.add_argument("--unsup-kmax", type=int, default=None, help="Best-k silhouette cap (default n-1)")
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dataset_dir.exists():
        ds = load_dataset(args.repo_id, name=args.config, split="train", revision=args.revision)
        args.dataset_dir.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(args.dataset_dir))
        print(f"Saved HF dataset to {args.dataset_dir}")
    else:
        print(f"Using cached dataset at {args.dataset_dir}")

    if not args.embeddings.exists():
        raise FileNotFoundError(f"Embeddings not found: {args.embeddings}")

    output_dir = (
        args.output_dir or ROOT / "results" / "chirality" / f"{args.model_name}_{args.distance}_{args.metric_version}"
    )
    embeddings = load_chirality_embeddings(args.embeddings, key=args.embedding_key)
    _, summary = evaluate_chirality_embeddings(
        dataset_dir=args.dataset_dir,
        embeddings=embeddings,
        output_dir=output_dir,
        model_name=args.model_name,
        distance=args.distance,
        metric_version=args.metric_version,
        unsup_kmax=args.unsup_kmax,
        n_jobs=args.n_jobs,
    )
    for k in ("ESA_AUC_mean", "NN1_acc_mean", "hopkins_mean", "sil_sup_mean", "sil_unsup_mean"):
        print(f"{k}: {summary[k]:.6f}")
    print(f"Report saved to {output_dir}")


if __name__ == "__main__":
    main()
