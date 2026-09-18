"""Offline tests of the download helper (Hugging Face calls are mocked)."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest

from three_dbench import download


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def fake_repo(tmp_path):
    """A local directory that mimics EscheWang/3dcs-embeddings."""
    repo = tmp_path / "repo"
    files = {
        "chirality/gemnet/sampled_feature.npz": lambda p: np.savez(p, gemnet=np.ones((3, 2), np.float32)),
        "chirality/molae/1.npz": lambda p: np.savez(p, arr_0=np.zeros((3, 4), np.float32)),
        "rotation/gemnet/rotation_conformers_0.npz": lambda p: np.savez(p, gemnet=np.ones((5, 2), np.float32)),
        "rotation/gemnet/rotation_conformers_1.npz": lambda p: np.savez(p, gemnet=np.ones((4, 2), np.float32)),
        "results/rotation/all_metric.csv": lambda p: p.write_text("a,b\n1,2\n"),
        "results/chirality/chirality_metrics_summary.csv": lambda p: p.write_text("x\n1\n"),
    }
    rows = ["path,task,model,size_bytes,sha256,format,keys,shape,dtype"]
    for rel, make in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        make(p)
        parts = rel.split("/")
        task = parts[0] if parts[0] != "results" else "results"
        model = parts[1] if parts[0] != "results" else ""
        rows.append(f"{rel},{task},{model},{p.stat().st_size},{_sha(p)},npz,,,")
    (repo / "manifest.csv").write_text("\n".join(rows) + "\n")
    return repo


def _fake_hf_hub_download(repo: Path, calls: list):
    def _dl(*, repo_id, filename, repo_type, revision, local_dir):
        assert repo_type == "dataset"
        calls.append(filename)
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / filename, target)
        return str(target)

    return _dl


def test_download_embeddings_selects_and_verifies(fake_repo, tmp_path):
    calls: list = []
    out = tmp_path / "emb"
    paths = download.download_embeddings(
        "rotation", out, models=["gemnet"], hf_hub_download_fn=_fake_hf_hub_download(fake_repo, calls)
    )
    assert sorted(p.relative_to(out).as_posix() for p in paths) == [
        "rotation/gemnet/rotation_conformers_0.npz",
        "rotation/gemnet/rotation_conformers_1.npz",
    ]
    assert calls[0] == "manifest.csv"
    # second call: files are cached and verified, only the manifest is fetched again
    calls.clear()
    download.download_embeddings(
        "rotation", out, models=["gemnet"], hf_hub_download_fn=_fake_hf_hub_download(fake_repo, calls)
    )
    assert calls == ["manifest.csv"]


def test_download_embeddings_results_and_models(fake_repo, tmp_path):
    calls: list = []
    paths = download.download_embeddings(
        "chirality",
        tmp_path / "emb",
        models=["molae"],
        include_results=True,
        hf_hub_download_fn=_fake_hf_hub_download(fake_repo, calls),
    )
    rels = sorted(p.relative_to(tmp_path / "emb").as_posix() for p in paths)
    assert rels == ["chirality/molae/1.npz", "results/chirality/chirality_metrics_summary.csv"]


def test_download_embeddings_checksum_mismatch(fake_repo, tmp_path):
    manifest = (fake_repo / "manifest.csv").read_text().splitlines()
    manifest = [manifest[0]] + [
        line.replace(line.split(",")[4], "0" * 64) if line.startswith("chirality/molae") else line
        for line in manifest[1:]
    ]
    (fake_repo / "manifest.csv").write_text("\n".join(manifest) + "\n")
    with pytest.raises(download.ChecksumError):
        download.download_embeddings(
            "chirality", tmp_path / "emb", models=["molae"], hf_hub_download_fn=_fake_hf_hub_download(fake_repo, [])
        )


def test_download_embeddings_unknown_selection(fake_repo, tmp_path):
    with pytest.raises(ValueError, match="No manifest entries"):
        download.download_embeddings("traj", tmp_path / "emb", hf_hub_download_fn=_fake_hf_hub_download(fake_repo, []))
    with pytest.raises(ValueError):
        download.download_embeddings("nope", tmp_path / "emb", hf_hub_download_fn=_fake_hf_hub_download(fake_repo, []))


def test_download_dataset_paths(tmp_path):
    datasets = pytest.importorskip("datasets")
    calls = []

    def fake_load_dataset(repo_id, name, split, **kwargs):
        calls.append((repo_id, name, split))
        return datasets.Dataset.from_dict({"config": [name]})

    saved = download.download_dataset("traj", tmp_path / "hf", load_dataset_fn=fake_load_dataset)
    assert calls == [("EscheWang/3dcs", "traj_energies", "train"), ("EscheWang/3dcs", "traj_frames", "train")]
    assert saved["traj_energies"] == tmp_path / "hf" / "traj" / "energies"
    assert datasets.load_from_disk(str(saved["traj_frames"]))["config"] == ["traj_frames"]
    # existing directories are kept
    calls.clear()
    download.download_dataset("traj", tmp_path / "hf", load_dataset_fn=fake_load_dataset)
    assert calls == []
    with pytest.raises(ValueError):
        download.download_dataset("nope", tmp_path / "hf", load_dataset_fn=fake_load_dataset)


def test_parse_manifest_infers_task_model():
    text = "path,sha256\nchirality/fmg/x.npz,abc\nresults/traj/traj_e.csv,def\n"
    entries = download.parse_manifest(text)
    assert (entries[0].task, entries[0].model) == ("chirality", "fmg")
    sel = download.select_manifest_entries(entries, task="traj", include_results=True)
    assert [e.path for e in sel] == ["results/traj/traj_e.csv"]
