"""Tests for molecule serialization utilities."""

from __future__ import annotations

import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402

from three_dbench.datasets.serialization import (  # noqa: E402
    blocks_to_mols,
    mol_from_block,
    mol_to_block,
    mols_to_blocks,
)


def test_molblock_roundtrip():
    mol = Chem.MolFromSmiles("CCO")
    block = mol_to_block(mol)
    mol2 = mol_from_block(block, sanitize=False)
    assert mol2 is not None
    assert mol2.GetNumAtoms() == mol.GetNumAtoms()


def test_batch_roundtrip():
    mols = [Chem.MolFromSmiles(s) for s in ["CCO", "c1ccccc1", "CC(=O)O"]]
    blocks = mols_to_blocks(mols)
    assert len(blocks) == 3
    mols2 = blocks_to_mols(blocks, sanitize=False)
    assert len(mols2) == 3
    for m1, m2 in zip(mols, mols2):
        assert m1.GetNumAtoms() == m2.GetNumAtoms()


def test_invalid_block_raises():
    with pytest.raises(ValueError, match="Failed to parse"):
        mol_from_block("not a valid molblock")
