"""Download the 3DCS datasets and the published baseline embeddings from Hugging Face.

Datasets come from the ``EscheWang/3dcs`` configs and are written with ``save_to_disk`` to the
paths the evaluators expect::

    data/hf/chirality        (config chirality)
    data/hf/traj/energies    (config traj_energies)
    data/hf/traj/frames      (config traj_frames)
    data/hf/rotation         (config rotation, ~7.5 GB)

Embeddings come from ``EscheWang/3dcs-embeddings``. Files are listed in its ``manifest.csv``
(``path,task,model,size_bytes,sha256,...``), downloaded to ``<out>/<path>`` and verified
against the manifest SHA-256.
"""

from __future__ import annotations

import csv
import hashlib
import io
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from three_dbench.embeddings.published import DATASET_REPO_ID, EMBEDDINGS_REPO_ID

DATASET_TASK_CONFIGS: dict[str, dict[str, str]] = {
    "chirality": {"chirality": "chirality"},
    "traj": {"traj_energies": "traj/energies", "traj_frames": "traj/frames"},
    "rotation": {"rotation": "rotation"},
}
EMBEDDING_TASKS = ("chirality", "traj", "rotation", "chirality_legacy_15218")
MANIFEST_NAME = "manifest.csv"


class ChecksumError(RuntimeError):
    """A downloaded file does not match the SHA-256 in the manifest."""


@dataclass
class ManifestEntry:
    path: str
    task: str
    model: str
    size_bytes: int | None
    sha256: str
    row: dict


def sha256_file(path: Path, chunk_size: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def download_dataset(
    task: str,
    out: Path = Path("data/hf"),
    *,
    repo_id: str = DATASET_REPO_ID,
    revision: str | None = None,
    configs: Sequence[str] | None = None,
    overwrite: bool = False,
    load_dataset_fn: Callable | None = None,
) -> dict[str, Path]:
    """Download the dataset configs of ``task`` and ``save_to_disk`` them under ``out``.

    Returns ``{config: saved_path}``. Existing directories are kept unless ``overwrite``.
    """
    tasks = list(DATASET_TASK_CONFIGS) if task == "all" else [task]
    for t in tasks:
        if t not in DATASET_TASK_CONFIGS:
            raise ValueError(f"Unknown dataset task {t!r}; choose from {sorted(DATASET_TASK_CONFIGS)} or 'all'")
    if load_dataset_fn is None:
        from datasets import load_dataset as load_dataset_fn
    saved: dict[str, Path] = {}
    for t in tasks:
        for config, rel in DATASET_TASK_CONFIGS[t].items():
            if configs is not None and config not in configs:
                continue
            target = Path(out) / rel
            if target.exists() and any(target.iterdir()) and not overwrite:
                print(f"[download] {config}: {target} exists, skipping (use --overwrite to replace)")
                saved[config] = target
                continue
            print(f"[download] {repo_id} config={config} -> {target}")
            kwargs = {"name": config, "split": "train"}
            if revision is not None:
                kwargs["revision"] = revision
            ds = load_dataset_fn(repo_id, **kwargs)
            if target.exists() and overwrite:
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(target))
            saved[config] = target
    return saved


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def parse_manifest(text: str) -> list[ManifestEntry]:
    reader = csv.DictReader(io.StringIO(text))
    missing = {"path", "sha256"} - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"manifest.csv lacks columns {sorted(missing)}")
    entries = []
    for row in reader:
        path = row["path"].strip()
        parts = path.split("/")
        size = row.get("size_bytes")
        entries.append(
            ManifestEntry(
                path=path,
                task=(row.get("task") or parts[0]).strip(),
                model=(row.get("model") or (parts[1] if len(parts) > 2 else "")).strip(),
                size_bytes=int(size) if size not in (None, "") else None,
                sha256=(row.get("sha256") or "").strip().lower(),
                row=row,
            )
        )
    return entries


def _normalise_model(name: str) -> str:
    return name.strip().lower()


def select_manifest_entries(
    entries: Iterable[ManifestEntry],
    *,
    task: str,
    models: Sequence[str] | None = None,
    include_results: bool = False,
) -> list[ManifestEntry]:
    """Filter manifest entries by path prefix (``<task>/<model>/...``, ``results/<task>/...``)."""
    tasks = list(EMBEDDING_TASKS) if task == "all" else [task]
    wanted_models = {_normalise_model(m) for m in models} if models else None
    out = []
    for e in entries:
        parts = e.path.split("/")
        top = parts[0]
        if top in tasks:
            model = parts[1] if len(parts) > 2 else ""
            if wanted_models is None or _normalise_model(model) in wanted_models:
                out.append(e)
        elif include_results and top == "results" and len(parts) > 1 and (task == "all" or parts[1] in tasks):
            out.append(e)
    return out


def download_embeddings(
    task: str,
    out: Path = Path("data/embeddings"),
    *,
    models: Sequence[str] | None = None,
    include_results: bool = False,
    repo_id: str = EMBEDDINGS_REPO_ID,
    revision: str | None = None,
    verify: bool = True,
    dry_run: bool = False,
    hf_hub_download_fn: Callable | None = None,
) -> list[Path]:
    """Download published embeddings listed in the repo manifest and verify their SHA-256.

    Files already present with the manifest checksum are not downloaded again.
    Raises :class:`ChecksumError` if a downloaded file does not match.
    """
    if task != "all" and task not in EMBEDDING_TASKS:
        raise ValueError(f"Unknown embeddings task {task!r}; choose from {EMBEDDING_TASKS} or 'all'")
    if hf_hub_download_fn is None:
        from huggingface_hub import hf_hub_download as hf_hub_download_fn
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(
        hf_hub_download_fn(
            repo_id=repo_id, filename=MANIFEST_NAME, repo_type="dataset", revision=revision, local_dir=str(out)
        )
    )
    entries = parse_manifest(manifest_path.read_text(encoding="utf-8"))
    selected = select_manifest_entries(entries, task=task, models=models, include_results=include_results)
    if not selected:
        raise ValueError(f"No manifest entries for task={task!r}, models={models!r}")
    total = sum(e.size_bytes or 0 for e in selected)
    print(f"[download] {len(selected)} files, {total / 1e9:.2f} GB from {repo_id}")
    paths = []
    for e in selected:
        target = out / e.path
        if dry_run:
            print(f"  {e.path} ({(e.size_bytes or 0) / 1e6:.1f} MB)")
            continue
        if target.exists() and verify and e.sha256 and sha256_file(target) == e.sha256:
            print(f"  ok (cached) {e.path}")
            paths.append(target)
            continue
        local = Path(
            hf_hub_download_fn(
                repo_id=repo_id, filename=e.path, repo_type="dataset", revision=revision, local_dir=str(out)
            )
        )
        if local.resolve() != target.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local, target)
        if verify and e.sha256:
            digest = sha256_file(target)
            if digest != e.sha256:
                raise ChecksumError(f"{e.path}: sha256 {digest} != manifest {e.sha256}")
            print(f"  ok {e.path}")
        else:
            print(f"  downloaded {e.path} (not verified)")
        paths.append(target)
    return paths


__all__ = [
    "ChecksumError",
    "DATASET_TASK_CONFIGS",
    "EMBEDDING_TASKS",
    "ManifestEntry",
    "download_dataset",
    "download_embeddings",
    "parse_manifest",
    "select_manifest_entries",
    "sha256_file",
]
