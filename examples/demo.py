"""Self-contained demo: download HF data and evaluate with bundled GemNet fixtures.

This script demonstrates the full evaluation pipeline using small example embeddings
included in the repository. No external embedding generation is needed.

Usage:
    pip install -e .
    python examples/demo.py chirality
    python examples/demo.py trajectory
    python examples/demo.py all
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parent / "fixtures"
DEMO_CACHE = Path(__file__).resolve().parent / ".demo_cache"

HF_REPO = "EscheWang/3dcs"


def _download_or_load(config: str, cache_dir: Path):
    """Download HF dataset if not cached locally."""
    from datasets import load_dataset, load_from_disk

    if cache_dir.exists():
        print(f"  Using cached dataset at {cache_dir}")
        return load_from_disk(str(cache_dir))

    print(f"  Downloading {HF_REPO} config={config} ...")
    ds = load_dataset(HF_REPO, name=config, split="train")
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(cache_dir))
    print(f"  Saved to {cache_dir}")
    return ds


def demo_chirality(output_dir: Path | None = None) -> None:
    """Run chirality evaluation with bundled GemNet demo embeddings."""
    from three_dbench.chirality.evaluation import evaluate_en_separation_from_counts

    print("\n=== Chirality Demo ===")

    # Load fixture
    fixture_path = FIXTURES / "chirality_gemnet_demo.npz"
    with np.load(fixture_path) as data:
        embeddings = data["gemnet"]
    n_emb = embeddings.shape[0]
    print(f"  Loaded fixture: {embeddings.shape} from {fixture_path.name}")

    # Download dataset
    ds = _download_or_load("chirality", DEMO_CACHE / "chirality")

    # Subset dataset to match fixture size
    key_to_counts: OrderedDict[str, int] = OrderedDict()
    cumsum = 0
    for row in ds:
        n = int(row["n_conformers"])
        if cumsum + n > n_emb:
            break
        key_to_counts[row["key"]] = n
        cumsum += n

    print(f"  Using {len(key_to_counts)} keys ({cumsum} conformers)")
    embeddings = embeddings[:cumsum]

    # Evaluate
    rows, summary = evaluate_en_separation_from_counts(
        key_to_counts,
        embeddings,
        per_mol_min_n=2,
    )

    # Print summary
    print(f"\n  Results ({len(rows)} molecules evaluated):")
    for key in ["ESA_AUC_mean", "NN1_acc_mean", "sil_sup_mean", "DBI_mean"]:
        if key in summary:
            print(f"    {key}: {summary[key]:.4f}")

    if output_dir is not None:
        import json

        import pandas as pd

        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "chirality_per_molecule.json").open("w") as f:
            json.dump(rows, f, indent=2)
        pd.DataFrame([summary]).to_csv(output_dir / "chirality_summary.csv", index=False)
        print(f"  Saved to {output_dir}")

    print("  Chirality demo complete.")


def demo_trajectory(output_dir: Path | None = None) -> None:
    """Run trajectory evaluation with bundled GemNet demo embeddings."""
    from three_dbench.traj.evaluation import (
        compute_energy_metrics_from_condensed,
        pairwise_distances_from_embeddings_large,
    )

    print("\n=== Trajectory Demo ===")

    # Load fixtures
    demo_mols = ["rmd17_aspirin", "rmd17_ethanol"]
    emb_dict = {}
    for mol in demo_mols:
        fixture_path = FIXTURES / f"traj_gemnet_{mol}.npz"
        with np.load(fixture_path) as data:
            emb_dict[mol] = data["gemnet"]
        print(f"  Loaded fixture: {mol} {emb_dict[mol].shape}")

    # Download dataset
    ds = _download_or_load("traj_energies", DEMO_CACHE / "traj_energies")

    # Evaluate
    rng = np.random.default_rng(2025)
    n_samples = 3
    window = 2000
    all_metrics: dict[str, list[dict]] = {}

    for row in ds:
        mol_type = row["mol_type"]
        if mol_type not in emb_dict:
            continue

        energies = np.asarray(row["energies"], dtype=np.float64)
        embeddings = emb_dict[mol_type]
        n_frames = min(len(energies), embeddings.shape[0])
        energies = energies[:n_frames]
        embeddings = embeddings[:n_frames]

        max_start = n_frames - window
        if max_start <= 0:
            continue
        starts = rng.integers(0, max_start, size=min(n_samples, max_start), endpoint=False)

        mol_metrics: list[dict] = []
        for start in starts:
            end = int(start + window)
            Z = embeddings[start:end]
            D = pairwise_distances_from_embeddings_large(
                Z, metric="cosine", block_size=4096, out_mode="condensed", progress=False
            )
            m = compute_energy_metrics_from_condensed(D, energies[start:end], compute_ejs_auc=False)
            mol_metrics.append(m)

        all_metrics[mol_type] = mol_metrics
        means = {k: np.mean([m[k] for m in mol_metrics]) for k in mol_metrics[0]}
        print(f"\n  {mol_type} (mean over {len(mol_metrics)} windows):")
        for key in ["spearman", "kendall", "iso_R2", "dCor"]:
            if key in means:
                print(f"    {key}: {means[key]:.4f}")

    if output_dir is not None:
        import pandas as pd

        output_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for mol_type, metrics_list in all_metrics.items():
            for m in metrics_list:
                for k, v in m.items():
                    rows.append({"mol_type": mol_type, "metric": k, "value": float(v)})
        pd.DataFrame(rows).to_csv(output_dir / "trajectory_details.csv", index=False)
        print(f"\n  Saved to {output_dir}")

    print("  Trajectory demo complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run evaluation demos with bundled GemNet fixtures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python examples/demo.py chirality\n  python examples/demo.py all",
    )
    parser.add_argument(
        "task",
        choices=["chirality", "trajectory", "all"],
        nargs="?",
        default="all",
        help="Which demo to run (default: all)",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Save results to this directory")
    args = parser.parse_args()

    if args.task in ("chirality", "all"):
        out = args.output_dir / "chirality" if args.output_dir else None
        demo_chirality(output_dir=out)
    if args.task in ("trajectory", "all"):
        out = args.output_dir / "trajectory" if args.output_dir else None
        demo_trajectory(output_dir=out)

    print("\nDone.")


if __name__ == "__main__":
    main()
