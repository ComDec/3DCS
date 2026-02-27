"""Tests for chirality evaluation logic."""
from __future__ import annotations

import numpy as np
import pytest

from three_dbench.chirality.evaluation import (
    build_flat_index_from_counts,
    evaluate_en_separation_from_counts,
    parse_key_en,
)


def test_parse_key_en_standard():
    mol, en = parse_key_en("CHEMBL100259::en3_A5:S;A7:S")
    assert mol == "CHEMBL100259"
    assert en == "3"


def test_parse_key_en_simple():
    mol, en = parse_key_en("MOL1::en0")
    assert mol == "MOL1"
    assert en == "0"


def test_parse_key_en_fallback():
    mol, en = parse_key_en("some_unknown_key")
    assert mol == "some_unknown_key"
    assert en == "0"


def test_build_flat_index_from_counts():
    counts = {"MOL1::en0": 3, "MOL1::en1": 2, "MOL2::en0": 4}
    flat = build_flat_index_from_counts(counts)
    assert sum(flat.counts) == 9
    assert len(flat.mol_to_indices) == 2


def test_evaluate_basic(chirality_key_counts, rng):
    total = sum(chirality_key_counts.values())
    embeddings = rng.standard_normal((total, 6))
    rows, summary = evaluate_en_separation_from_counts(
        chirality_key_counts,
        embeddings,
        per_mol_min_n=2,
    )
    assert len(rows) == 2
    assert summary["n_molecules"] == 2
    assert "ESA_AUC_mean" in summary
    assert "NN1_acc_mean" in summary


def test_single_en_skip():
    """A molecule with only one enantiomer should be skipped or marked."""
    counts = {"MOL1::en0": 5}
    embeddings = np.random.default_rng(0).standard_normal((5, 4))
    rows, summary = evaluate_en_separation_from_counts(counts, embeddings, per_mol_min_n=2)
    assert len(rows) == 1
    assert rows[0]["mode"] == "skip_single_en"


def test_well_separated_high_auc(rng):
    """Two well-separated enantiomers should produce high ESA AUC."""
    counts = {
        "MOL::en0": 20,
        "MOL::en1": 20,
    }
    # Create embeddings with clear separation
    emb0 = rng.standard_normal((20, 8)) + 10
    emb1 = rng.standard_normal((20, 8)) - 10
    embeddings = np.vstack([emb0, emb1]).astype(np.float32)

    rows, summary = evaluate_en_separation_from_counts(counts, embeddings, per_mol_min_n=2)
    assert summary["ESA_AUC_mean"] > 0.95
    assert summary["NN1_acc_mean"] > 0.95
