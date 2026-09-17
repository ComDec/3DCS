"""Download trajectory energies from HF and run the energy (trajectory) evaluation.

Defaults reproduce the published protocol (legacy windows, paper metric definitions, float64
energies). Embeddings for the paper baselines are in the ``EscheWang/3dcs-embeddings`` dataset under
``traj/<model>/rmd17_<mol>.{npz,pkl}``; see ``reproduce/energy_tables_3_6_7/run.sh``.

Requires: pip install -e .
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from three_dbench.benchmarks import evaluate_trajectory_embeddings
from three_dbench.traj.io import load_traj_embeddings

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trajectory evaluation from HF datasets")
    parser.add_argument("--repo-id", type=str, default="EscheWang/3dcs", help="HF dataset repo")
    parser.add_argument("--config", type=str, default="traj_energies", help="Dataset config name")
    parser.add_argument("--revision", type=str, default=None, help="HF dataset revision (default: main)")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data" / "hf" / "traj" / "energies")
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=ROOT / "data" / "traj" / "results" / "unimol",
        help="Directory of rmd17_*.npz (vectors) or rmd17_*.pkl (RDKit fingerprints) files",
    )
    parser.add_argument("--embedding-key", type=str, default=None, help="NPZ key (default: arr_0 / only array)")
    parser.add_argument("--model-name", type=str, default="unimol")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "traj" / "unimol")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--window", type=int, default=2000)
    parser.add_argument("--metric-embed", type=str, default=None, choices=["cosine", "euclidean", "tanimoto"])
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--random-seed", type=int, default=2025)
    parser.add_argument("--window-scheme", choices=["legacy", "shared"], default="legacy")
    parser.add_argument("--metric-version", choices=["paper", "v2"], default="paper")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--molecules", type=str, nargs="*", default=None)
    parser.add_argument("--energy-precision-check", choices=["error", "warn", "ignore"], default="error")
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

    if not args.embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {args.embeddings_dir}")

    emb_dict = load_traj_embeddings(args.embeddings_dir, key=args.embedding_key)
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
        window_scheme=args.window_scheme,
        metric_version=args.metric_version,
        n_jobs=args.n_jobs,
        molecules=args.molecules,
        energy_precision_check=args.energy_precision_check,
    )
    print(f"Report saved to {args.output_dir}")


if __name__ == "__main__":
    main()
