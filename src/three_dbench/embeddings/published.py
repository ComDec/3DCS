"""Layout of the published baseline embeddings (Hugging Face ``EscheWang/3dcs-embeddings``).

Files are stored with their original bytes and array keys; only directory names are
normalised. ``python -m three_dbench download embeddings`` places them under
``<out>/<path>`` (default ``data/embeddings``), so the paths below are relative to that root.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

EMBEDDINGS_REPO_ID = "EscheWang/3dcs-embeddings"
DATASET_REPO_ID = "EscheWang/3dcs"
TASKS = ("chirality", "traj", "rotation")
MODELS = ("e3fp", "gemnet", "molae", "molspectra", "unimol", "fmg", "mace")
RMD17_MOLECULES = (
    "aspirin",
    "azobenzene",
    "benzene",
    "ethanol",
    "malonaldehyde",
    "naphthalene",
    "paracetamol",
    "salicylic",
    "toluene",
    "uracil",
)


@dataclass(frozen=True)
class PublishedEmbedding:
    """Where a published embedding lives and how to read it."""

    task: str
    model: str
    path: str  # file (chirality), directory (traj / rotation)
    key: str | None  # array key (npz) or dict entry (pkl); None for pickled lists
    layout: str  # "flat" | "per-molecule" | "by-shard"
    file_pattern: str | None = None
    shards: tuple[int, ...] | None = None
    notes: str = ""

    def local_path(self, root: str | Path = "data/embeddings") -> Path:
        return Path(root) / self.path


_ALL_SHARDS = tuple(range(16))

PUBLISHED_EMBEDDINGS: dict[tuple[str, str], PublishedEmbedding] = {
    # Table 2 set: 52,391 conformers in the row order of the HF chirality config.
    ("chirality", "e3fp"): PublishedEmbedding("chirality", "e3fp", "chirality/e3fp/sampled_chi.pkl", "e3fp", "flat"),
    ("chirality", "gemnet"): PublishedEmbedding(
        "chirality", "gemnet", "chirality/gemnet/sampled_feature.npz", "gemnet", "flat"
    ),
    ("chirality", "molae"): PublishedEmbedding("chirality", "molae", "chirality/molae/1.npz", "arr_0", "flat"),
    ("chirality", "molspectra"): PublishedEmbedding(
        "chirality", "molspectra", "chirality/molspectra/sampled_mol_feature.npz", "arr_0", "flat"
    ),
    ("chirality", "unimol"): PublishedEmbedding("chirality", "unimol", "chirality/unimol/1.npz", "arr_0", "flat"),
    ("chirality", "fmg"): PublishedEmbedding(
        "chirality",
        "fmg",
        "chirality/fmg/chirality_bench_conformers_noised_only_aslist_embed.npz",
        "embeddings",
        "flat",
    ),
    ("chirality", "mace"): PublishedEmbedding("chirality", "mace", "chirality/mace/chirality.npz", "arr_0", "flat"),
    # rMD17: one file per molecule, rmd17_<molecule>.{npz,pkl}.
    ("traj", "e3fp"): PublishedEmbedding(
        "traj", "e3fp", "traj/e3fp", None, "per-molecule", "rmd17_{molecule}.pkl", notes="pickled list of bit vectors"
    ),
    ("traj", "gemnet"): PublishedEmbedding(
        "traj", "gemnet", "traj/gemnet", "gemnet", "per-molecule", "rmd17_{molecule}.npz"
    ),
    ("traj", "molae"): PublishedEmbedding(
        "traj", "molae", "traj/molae", "arr_0", "per-molecule", "rmd17_{molecule}.npz"
    ),
    ("traj", "molspectra"): PublishedEmbedding(
        "traj", "molspectra", "traj/molspectra", "arr_0", "per-molecule", "rmd17_{molecule}.npz"
    ),
    ("traj", "unimol"): PublishedEmbedding(
        "traj", "unimol", "traj/unimol", "arr_0", "per-molecule", "rmd17_{molecule}.npz"
    ),
    ("traj", "fmg"): PublishedEmbedding(
        "traj", "fmg", "traj/fmg", "embeddings", "per-molecule", "rmd17_{molecule}.npz"
    ),
    ("traj", "mace"): PublishedEmbedding("traj", "mace", "traj/mace", "arr_0", "per-molecule", "rmd17_{molecule}.npz"),
    # Rotation: file index = shard id, rows ordered by the per-shard offset of the HF rotation config.
    ("rotation", "gemnet"): PublishedEmbedding(
        "rotation", "gemnet", "rotation/gemnet", "gemnet", "by-shard", "rotation_conformers_{shard}.npz", _ALL_SHARDS
    ),
    ("rotation", "fmg"): PublishedEmbedding(
        "rotation",
        "fmg",
        "rotation/fmg",
        "embeddings",
        "by-shard",
        "rot_mol_list_{shard}_embed.npz",
        (0,),
        notes="shard 0 only; not used in the paper",
    ),
    ("rotation", "mace"): PublishedEmbedding(
        "rotation",
        "mace",
        "rotation/mace",
        "arr_0",
        "by-shard",
        "rot{shard}.npz",
        (0,),
        notes="shard 0 only; not used in the paper",
    ),
}


def published_embedding(task: str, model: str) -> PublishedEmbedding:
    """Look up a published embedding; raises KeyError with the available combinations."""
    try:
        return PUBLISHED_EMBEDDINGS[(task, model)]
    except KeyError:
        avail = sorted(f"{t}/{m}" for t, m in PUBLISHED_EMBEDDINGS)
        raise KeyError(f"No published embedding for task={task!r}, model={model!r}. Available: {avail}") from None
