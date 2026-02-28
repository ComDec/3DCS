"""Command-line interface for 3DBench."""

from __future__ import annotations

import argparse
from pathlib import Path

from three_dbench.embeddings import load_embeddings, load_embeddings_dict, load_embeddings_dir
from three_dbench.utils.paths import DATA_ROOT, RESULTS_ROOT


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3DBench CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser("convert", help="Convert raw datasets to Hugging Face format")
    convert.add_argument("task", choices=["chirality", "rotation", "traj"], help="Dataset to convert")
    convert.add_argument("--output-dir", type=Path, default=None, help="Destination directory")
    convert.add_argument("--input-pkl", type=Path, default=None, help="Chirality pickle path")
    convert.add_argument("--lmdb-root", type=Path, default=None, help="Rotation LMDB root")
    convert.add_argument("--mol-pkl-dir", type=Path, default=None, help="Trajectory mol pickle directory")
    convert.add_argument("--energy-dir", type=Path, default=None, help="Trajectory energy NPZ directory")
    convert.add_argument("--no-mol-blocks", action="store_true", help="Skip MolBlock storage")
    convert.add_argument("--shards", type=int, nargs="*", default=None, help="Rotation shard IDs to convert")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate embeddings against HF datasets")
    evaluate.add_argument("task", choices=["chirality", "rotation", "traj"], help="Benchmark to run")
    evaluate.add_argument("--dataset-dir", type=Path, required=True, help="HF dataset directory")
    evaluate.add_argument("--embeddings", type=Path, required=True, help="Embeddings file or directory")
    evaluate.add_argument("--embedding-key", type=str, default=None, help="Key for NPZ or pickle dict")
    evaluate.add_argument("--model-name", type=str, default="custom", help="Model name for reports")
    evaluate.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    evaluate.add_argument("--metrics", type=str, nargs="*", default=None, help="Rotation distance metrics")
    evaluate.add_argument(
        "--layout", type=str, choices=["flat", "by-key"], default="flat", help="Rotation embedding layout"
    )
    evaluate.add_argument("--n-samples", type=int, default=100, help="Trajectory samples per molecule")
    evaluate.add_argument("--window", type=int, default=2000, help="Trajectory window size")
    evaluate.add_argument("--metric-embed", type=str, default="cosine", help="Trajectory distance metric")
    evaluate.add_argument("--block-size", type=int, default=4096, help="Trajectory distance block size")
    evaluate.add_argument("--random-seed", type=int, default=2025, help="Trajectory random seed")
    evaluate.add_argument("--per-mol-min-n", type=int, default=2, help="Chirality minimum conformers per molecule")
    evaluate.add_argument("--max-molecules", type=int, default=None, help="Chirality max molecules for testing")
    evaluate.add_argument("--do-unsup-when-single-en", action="store_true", help="Chirality unsupervised metrics")
    return parser.parse_args()


def _convert_dataset(args: argparse.Namespace) -> None:
    include_mols = not args.no_mol_blocks
    if args.task == "chirality":
        from three_dbench.datasets.chirality import convert_chirality_pkl_to_hf

        input_pkl = args.input_pkl or (DATA_ROOT / "chirality" / "chirality_bench_conformers_noised_only.pkl")
        output_dir = args.output_dir or (DATA_ROOT / "hf" / "chirality")
        convert_chirality_pkl_to_hf(input_pkl, output_dir, include_mol_blocks=include_mols)
        print(f"Saved chirality HF dataset to {output_dir}")
    elif args.task == "rotation":
        from three_dbench.datasets.rotation import convert_rotation_lmdb_to_hf

        lmdb_root = args.lmdb_root or (DATA_ROOT / "rotation" / "results")
        output_dir = args.output_dir or (DATA_ROOT / "hf" / "rotation")
        convert_rotation_lmdb_to_hf(
            lmdb_root,
            output_dir,
            shards=args.shards,
            include_mol_blocks=include_mols,
        )
        print(f"Saved rotation HF dataset to {output_dir}")
    elif args.task == "traj":
        from three_dbench.datasets.traj import convert_traj_energy_npz_to_hf, convert_traj_frames_pkl_to_hf

        mol_pkl_dir = args.mol_pkl_dir or (DATA_ROOT / "traj" / "mol_pkl")
        energy_dir = args.energy_dir or (DATA_ROOT / "traj" / "npz_data")
        output_dir = args.output_dir or (DATA_ROOT / "hf" / "traj")
        frames_dir = output_dir / "frames"
        energies_dir = output_dir / "energies"
        convert_traj_frames_pkl_to_hf(mol_pkl_dir, frames_dir, include_mol_blocks=include_mols)
        convert_traj_energy_npz_to_hf(energy_dir, energies_dir)
        print(f"Saved trajectory HF datasets to {output_dir}")


def _evaluate_embeddings(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = RESULTS_ROOT / args.task / args.model_name

    if args.task == "chirality":
        from three_dbench.benchmarks import evaluate_chirality_embeddings

        embeddings = load_embeddings(args.embeddings, key=args.embedding_key)
        evaluate_chirality_embeddings(
            dataset_dir=args.dataset_dir,
            embeddings=embeddings,
            output_dir=output_dir,
            model_name=args.model_name,
            per_mol_min_n=args.per_mol_min_n,
            do_unsup_when_single_en=args.do_unsup_when_single_en,
            max_molecules=args.max_molecules,
        )
        print(f"Chirality report saved to {output_dir}")
        return

    if args.task == "rotation":
        from three_dbench.benchmarks import evaluate_rotation_embeddings

        metrics = args.metrics or ["cosine", "euclidean"]
        if args.layout == "by-key":
            emb_dict = load_embeddings_dict(args.embeddings, key=args.embedding_key)
            evaluate_rotation_embeddings(
                dataset_dir=args.dataset_dir,
                embeddings_by_key=emb_dict,
                output_dir=output_dir,
                model_name=args.model_name,
                metrics=metrics,
            )
        else:
            embeddings = load_embeddings(args.embeddings, key=args.embedding_key)
            evaluate_rotation_embeddings(
                dataset_dir=args.dataset_dir,
                embeddings=embeddings,
                output_dir=output_dir,
                model_name=args.model_name,
                metrics=metrics,
            )
        print(f"Rotation report saved to {output_dir}")
        return

    if args.task == "traj":
        from three_dbench.benchmarks import evaluate_trajectory_embeddings

        if args.embeddings.is_dir():
            emb_dict = load_embeddings_dir(args.embeddings, file_glob="rmd17_*.npz", key=args.embedding_key)
        else:
            emb_dict = load_embeddings_dict(args.embeddings, key=args.embedding_key)
        evaluate_trajectory_embeddings(
            dataset_dir=args.dataset_dir,
            embeddings_by_mol=emb_dict,
            output_dir=output_dir,
            model_name=args.model_name,
            n_samples=args.n_samples,
            window=args.window,
            metric_embed=args.metric_embed,
            block_size=args.block_size,
            random_seed=args.random_seed,
        )
        print(f"Trajectory report saved to {output_dir}")
        return


def main() -> None:
    args = _parse_args()
    if args.command == "convert":
        _convert_dataset(args)
    elif args.command == "evaluate":
        _evaluate_embeddings(args)


if __name__ == "__main__":
    main()
