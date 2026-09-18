"""float64 energy handling and the quantization detector."""

from __future__ import annotations

import numpy as np
import pytest
from traj_helpers import save_energy_dataset, synthetic_embeddings, synthetic_energies

from three_dbench.datasets.traj import (
    QuantizedEnergiesError,
    check_energy_precision,
    convert_traj_energy_npz_to_hf,
    detect_quantized_energies,
    load_traj_energy_dataset,
)


def test_float64_energies_not_flagged():
    report = detect_quantized_energies(synthetic_energies(5000, seed=0), mol_type="rmd17_x")
    assert not report.quantized
    assert report.unique_fraction == 1.0
    assert not report.float32_representable


def test_float32_cast_of_absolute_energies_flagged():
    energies = synthetic_energies(5000, seed=1).astype(np.float32).astype(np.float64)
    report = detect_quantized_energies(energies, mol_type="rmd17_x")
    assert report.quantized
    assert report.float32_representable
    assert "float32" in report.reason


def test_few_distinct_values_flagged():
    energies = np.repeat(np.arange(40, dtype=np.float64) * 0.123456789, 100)
    report = detect_quantized_energies(energies)
    assert report.quantized
    assert "distinct" in report.reason


def test_relative_float32_energies_not_flagged():
    rng = np.random.default_rng(3)
    energies = (5.0 * rng.standard_normal(5000)).astype(np.float32).astype(np.float64)
    assert not detect_quantized_energies(energies).quantized


def test_check_energy_precision_policies():
    bad = detect_quantized_energies(synthetic_energies(2000, seed=2).astype(np.float32), mol_type="rmd17_bad")
    good = detect_quantized_energies(synthetic_energies(2000, seed=2), mol_type="rmd17_good")
    with pytest.raises(QuantizedEnergiesError, match="rmd17_bad"):
        check_energy_precision([good, bad], policy="error")
    with pytest.warns(UserWarning, match="quantized"):
        flagged = check_energy_precision([good, bad], policy="warn")
    assert [r.mol_type for r in flagged] == ["rmd17_bad"]
    assert check_energy_precision([good], policy="error") == []
    assert len(check_energy_precision([bad], policy="ignore")) == 1
    with pytest.raises(ValueError):
        check_energy_precision([good], policy="strict")


def test_converter_keeps_float64_exactly(tmp_path):
    src = tmp_path / "npz"
    src.mkdir()
    energies = {"rmd17_a": synthetic_energies(300, seed=4), "rmd17_b": synthetic_energies(250, seed=5)}
    for mol, e in energies.items():
        np.savez(src / f"{mol}.npz", energies=e, coords=np.zeros((e.size, 3, 3)))
    out = tmp_path / "hf"
    convert_traj_energy_npz_to_hf(src, out)
    ds = load_traj_energy_dataset(out)
    for row in ds:
        loaded = np.asarray(row["energies"], dtype=np.float64)
        assert row["n_frames"] == loaded.size
        np.testing.assert_array_equal(loaded, energies[row["mol_type"]])
        assert not np.array_equal(loaded, loaded.astype(np.float32))


def test_evaluate_rejects_quantized_energies(tmp_path):
    from three_dbench.benchmarks import evaluate_trajectory_embeddings

    e64 = synthetic_energies(200, seed=6)
    dataset_dir = save_energy_dataset(tmp_path / "ds", {"rmd17_a": e64.astype(np.float32)})
    emb = {"rmd17_a": synthetic_embeddings(e64, 8, seed=7)}
    kwargs = dict(
        dataset_dir=dataset_dir,
        embeddings_by_mol=emb,
        n_samples=1,
        window=40,
        legacy_traj_len=200,
        verbose=False,
    )
    with pytest.raises(QuantizedEnergiesError):
        evaluate_trajectory_embeddings(**kwargs)
    _, summary = evaluate_trajectory_embeddings(**kwargs, energy_precision_check="ignore")
    assert not summary.empty
