"""CLI help, import side effects, path resolution and embedding IO helpers."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import three_dbench
from three_dbench.embeddings import load_embeddings, published_embedding, select_array_key

# Subprocesses run from a temporary directory: make them import the same package as this process.
_PKG_PARENT = str(Path(three_dbench.__file__).resolve().parents[1])


def _env(**extra):
    env = {k: v for k, v in os.environ.items() if k != "THREE_DBENCH_HOME"}
    env["PYTHONPATH"] = os.pathsep.join([_PKG_PARENT, env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env.update(extra)
    return env


def _run(args, **kw):
    return subprocess.run([sys.executable, "-m", "three_dbench", *args], capture_output=True, text=True, **kw)


def test_cli_download_help():
    res = _run(["download", "--help"])
    assert res.returncode == 0
    for token in ("dataset", "embeddings", "--task", "--models", "--out", "--revision"):
        assert token in res.stdout


def test_cli_evaluate_rotation_flags():
    res = _run(["evaluate", "--help"])
    assert res.returncode == 0
    for token in (
        "by-shard",
        "--molecule-list",
        "--sample-ratio",
        "--sample-seed",
        "--metric-version",
        "--lie-k",
        "--as-variant",
    ):
        assert token in res.stdout


def test_import_has_no_side_effects(tmp_path):
    env = _env(THREE_DBENCH_HOME=str(tmp_path / "home"))
    code = (
        "import three_dbench.benchmarks, three_dbench.rotation.evaluation as r; "
        "from three_dbench.utils import paths; print(paths.DATA_ROOT)"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == str((tmp_path / "home" / "data").resolve())
    assert not (tmp_path / "home").exists()
    assert not (tmp_path / "results").exists()


def test_paths_default_to_cwd(tmp_path):
    env = _env()
    res = subprocess.run(
        [sys.executable, "-c", "from three_dbench.utils.paths import RESULTS_ROOT; print(RESULTS_ROOT)"],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
    )
    assert Path(res.stdout.strip()) == Path(tmp_path).resolve() / "results"


def test_select_array_key_skips_strings(tmp_path):
    p = tmp_path / "fmg.npz"
    np.savez(p, smiles=np.array(["C", "CC"]), embeddings=np.ones((2, 3), np.float32))
    with np.load(p) as data:
        assert select_array_key(data, None) == "embeddings"
    p2 = tmp_path / "other.npz"
    np.savez(p2, names=np.array(["a", "b"]), vectors=np.ones((2, 3)))
    with np.load(p2) as data:
        assert select_array_key(data, None) == "vectors"
        with pytest.raises(KeyError):
            select_array_key(data, "missing")


def test_load_embeddings_from_directory(tmp_path):
    d = tmp_path / "chirality" / "gemnet"
    d.mkdir(parents=True)
    np.savez(d / "sampled_feature.npz", gemnet=np.arange(6, dtype=np.float32).reshape(3, 2))
    emb = load_embeddings(d)
    assert emb.kind == "vector" and emb.array.shape == (3, 2)


def test_published_registry():
    spec = published_embedding("rotation", "gemnet")
    assert spec.layout == "by-shard" and spec.key == "gemnet" and len(spec.shards) == 16
    assert published_embedding("chirality", "fmg").key == "embeddings"
    with pytest.raises(KeyError):
        published_embedding("rotation", "unimol")
