"""Command-line interface for 3DBench."""

from __future__ import annotations

import argparse
from pathlib import Path

from three_dbench.embeddings import load_embeddings, load_embeddings_dict
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
        "--layout",
        type=str,
        choices=["flat", "by-shard", "by-key"],
        default=None,
        help="Rotation embedding layout (default: by-shard for a directory, flat for a file)",
    )
    evaluate.add_argument(
        "--offset-mode",
        type=str,
        choices=["auto", "per-shard", "global"],
        default="auto",
        help="Rotation: how the dataset 'offset' column is interpreted",
    )
    evaluate.add_argument("--shard-file-pattern", type=str, default=None, help="Rotation by-shard file pattern")
    evaluate.add_argument("--shards", type=int, nargs="*", default=None, help="Rotation shard IDs to evaluate")
    evaluate.add_argument("--molecule-list", type=Path, default=None, help="Rotation: file with one key per line")
    evaluate.add_argument("--sample-ratio", type=float, default=None, help="Rotation: fraction of molecules per shard")
    evaluate.add_argument("--sample-seed", type=int, default=2027, help="Rotation: seed for --sample-ratio")
    evaluate.add_argument("--min-conformers", type=int, default=2, help="Rotation: skip molecules with fewer")
    evaluate.add_argument("--max-keys", type=int, default=None, help="Rotation: evaluate at most N molecules")
    evaluate.add_argument(
        "--metric-version",
        type=str,
        choices=["paper", "v2"],
        default="paper",
        help="All tasks: 'paper' reproduces the published numbers (default), 'v2' uses the corrected "
        "definitions documented in docs/METRICS.md",
    )
    evaluate.add_argument("--lie-k", type=int, default=None, help="Rotation: override k of LIE@k")
    evaluate.add_argument(
        "--lie-self",
        type=str,
        choices=["include", "exclude"],
        default=None,
        help="Rotation: include the conformer itself in its LIE neighbourhood",
    )
    evaluate.add_argument(
        "--as-variant",
        type=str,
        default=None,
        choices=["mean_delta_circular", "median_delta_circular", "median_halfdelta_circular", "median_dz_linear"],
        help="Rotation: override the angular smoothness definition",
    )
    evaluate.add_argument("--extra-metrics", action="store_true", help="Rotation: also dCor, Mantel, stress, triplets")
    evaluate.add_argument(
        "--replicate-offset-drift",
        action="store_true",
        help="Rotation (by-shard): also report metrics with the embedding-cursor drift of the published full run",
    )
    evaluate.add_argument("--n-jobs", type=int, default=1, help="All tasks: worker processes (-1 = all CPUs)")
    evaluate.add_argument("--n-samples", type=int, default=100, help="Trajectory samples per molecule")
    evaluate.add_argument("--window", type=int, default=2000, help="Trajectory window size")
    evaluate.add_argument(
        "--metric-embed",
        type=str,
        choices=["cosine", "euclidean", "tanimoto"],
        default=None,
        help="Trajectory distance (default: tanimoto for fingerprint pickles, cosine for vectors)",
    )
    evaluate.add_argument("--block-size", type=int, default=4096, help="Trajectory distance block size")
    evaluate.add_argument("--random-seed", type=int, default=2025, help="Trajectory random seed")
    evaluate.add_argument(
        "--window-scheme",
        choices=["legacy", "shared"],
        default="legacy",
        help="Trajectory windows: legacy = per-molecule reseeding as in the paper (default); shared = 0.1.0 CLI",
    )
    evaluate.add_argument("--molecules", type=str, nargs="*", default=None, help="Trajectory molecule subset")
    evaluate.add_argument(
        "--energy-precision-check",
        choices=["error", "warn", "ignore"],
        default="error",
        help="Action when trajectory energies look quantized (e.g. float32-cast)",
    )
    evaluate.add_argument(
        "--time-ordered", action="store_true", help="Trajectory frames are time-ordered (v2 TS/Smoothness)"
    )
    evaluate.add_argument(
        "--legacy-traj-len", type=int, default=100_000, help="Trajectory length assumed by the legacy window scheme"
    )
    evaluate.add_argument("--per-mol-min-n", type=int, default=2, help="Chirality minimum conformers per molecule")
    evaluate.add_argument("--max-molecules", type=int, default=None, help="Chirality max molecules for testing")
    evaluate.add_argument("--do-unsup-when-single-en", action="store_true", help="Chirality unsupervised metrics")
    # --- chirality options (defaults reproduce the published Table 2; see docs/metrics/chirality.md) ---
    evaluate.add_argument(
        "--distance",
        choices=["euclidean", "cosine"],
        default="euclidean",
        help="Chirality: distance for continuous embeddings (default euclidean = published Table 2); "
        "RDKit fingerprints always use Tanimoto",
    )
    evaluate.add_argument(
        "--unsup-kmax",
        type=_parse_unsup_kmax,
        default=None,
        help="Chirality: largest k for the best-k silhouette (SCI_unsup); 'n-1' (default, published run) or an int",
    )

    download = subparsers.add_parser("download", help="Download datasets or published embeddings from Hugging Face")
    download.add_argument("what", choices=["dataset", "embeddings"], help="What to download")
    download.add_argument(
        "--task",
        required=True,
        choices=["chirality", "traj", "rotation", "chirality_legacy_15218", "all"],
        help="Benchmark task",
    )
    download.add_argument("--models", type=str, nargs="*", default=None, help="Embeddings: model directories")
    download.add_argument("--out", type=Path, default=None, help="Output root (default: data/hf or data/embeddings)")
    download.add_argument("--repo-id", type=str, default=None, help="Override the HF repo id")
    download.add_argument("--revision", type=str, default=None, help="HF revision (branch, tag or commit)")
    download.add_argument("--results", action="store_true", help="Embeddings: also fetch results/<task>/ files")
    download.add_argument("--no-verify", action="store_true", help="Embeddings: skip SHA-256 verification")
    download.add_argument("--dry-run", action="store_true", help="Embeddings: list files without downloading")
    download.add_argument("--overwrite", action="store_true", help="Dataset: replace an existing save_to_disk dir")
    return parser.parse_args()


def _parse_unsup_kmax(value: str):
    if value.lower() in {"n-1", "none", "all"}:
        return None
    try:
        k = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--unsup-kmax must be 'n-1' or an integer >= 2, got {value!r}") from exc
    if k < 2:
        raise argparse.ArgumentTypeError(f"--unsup-kmax must be >= 2, got {k}")
    return k


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
        from three_dbench.benchmarks.chirality import evaluate_chirality_embeddings, load_chirality_embeddings

        embeddings = load_chirality_embeddings(args.embeddings, key=args.embedding_key)
        _, summary = evaluate_chirality_embeddings(
            dataset_dir=args.dataset_dir,
            embeddings=embeddings,
            output_dir=output_dir,
            model_name=args.model_name,
            per_mol_min_n=args.per_mol_min_n,
            do_unsup_when_single_en=args.do_unsup_when_single_en,
            unsup_kmax=args.unsup_kmax,
            max_molecules=args.max_molecules,
            distance=args.distance,
            metric_version=args.metric_version,
            n_jobs=args.n_jobs,
        )
        print(
            "ES-AUC {ESA_AUC_mean:.6f}  NN@1-Acc {NN1_acc_mean:.6f}  Hopkins {hopkins_mean:.6f}  "
            "SCI {sil_sup_mean:.6f}  SCI_unsup {sil_unsup_mean:.6f}".format(**summary)
        )
        print(f"Chirality report saved to {output_dir}")
        return

    if args.task == "rotation":
        from three_dbench.benchmarks import evaluate_rotation_embeddings
        from three_dbench.benchmarks.rotation import ShardedEmbeddings

        metrics = args.metrics or ["cosine", "euclidean"]
        layout = args.layout or ("by-shard" if args.embeddings.is_dir() else "flat")
        common = {
            "dataset_dir": args.dataset_dir,
            "output_dir": output_dir,
            "model_name": args.model_name,
            "metrics": metrics,
            "metric_version": args.metric_version,
            "lie_k": args.lie_k,
            "lie_include_self": None if args.lie_self is None else args.lie_self == "include",
            "as_variant": args.as_variant,
            "extra_metrics": True if args.extra_metrics else None,
            "offset_mode": args.offset_mode,
            "shards": args.shards,
            "molecule_list": args.molecule_list,
            "sample_ratio": args.sample_ratio,
            "sample_seed": args.sample_seed,
            "min_conformers": args.min_conformers,
            "max_keys": args.max_keys,
            "n_jobs": args.n_jobs,
            "progress": True,
            "replicate_offset_drift": args.replicate_offset_drift,
        }
        if layout == "by-key":
            emb_dict = load_embeddings_dict(args.embeddings, key=args.embedding_key)
            evaluate_rotation_embeddings(embeddings_by_key=emb_dict, **common)
        elif layout == "by-shard":
            sharded = ShardedEmbeddings.from_directory(
                args.embeddings, key=args.embedding_key, pattern=args.shard_file_pattern
            )
            evaluate_rotation_embeddings(embeddings_by_shard=sharded, **common)
        else:
            embeddings = load_embeddings(args.embeddings, key=args.embedding_key)
            evaluate_rotation_embeddings(embeddings=embeddings, **common)
        print(f"Rotation report saved to {output_dir}")
        return

    if args.task == "traj":
        from three_dbench.benchmarks import evaluate_trajectory_embeddings
        from three_dbench.traj.io import load_traj_embeddings

        emb_dict = load_traj_embeddings(args.embeddings, key=args.embedding_key)
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
            window_scheme=args.window_scheme,
            metric_version=args.metric_version,
            n_jobs=args.n_jobs,
            molecules=args.molecules,
            energy_precision_check=args.energy_precision_check,
            time_ordered=args.time_ordered,
            legacy_traj_len=args.legacy_traj_len,
        )
        print(f"Trajectory report saved to {output_dir}")
        return


def _download(args: argparse.Namespace) -> None:
    from three_dbench import download

    if args.what == "dataset":
        if args.task == "chirality_legacy_15218":
            raise SystemExit("The 15,218-conformer chirality set is only published as embeddings.")
        download.download_dataset(
            args.task,
            args.out or (DATA_ROOT / "hf"),
            repo_id=args.repo_id or download.DATASET_REPO_ID,
            revision=args.revision,
            overwrite=args.overwrite,
        )
    else:
        download.download_embeddings(
            args.task,
            args.out or (DATA_ROOT / "embeddings"),
            models=args.models,
            include_results=args.results,
            repo_id=args.repo_id or download.EMBEDDINGS_REPO_ID,
            revision=args.revision,
            verify=not args.no_verify,
            dry_run=args.dry_run,
        )


def main() -> None:
    args = _parse_args()
    if args.command == "convert":
        _convert_dataset(args)
    elif args.command == "evaluate":
        _evaluate_embeddings(args)
    elif args.command == "download":
        _download(args)


if __name__ == "__main__":
    main()
