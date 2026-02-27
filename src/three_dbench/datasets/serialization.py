"""Serialization helpers for RDKit molecules."""
from __future__ import annotations

from typing import Iterable, List

from rdkit import Chem


def mol_to_block(mol: Chem.Mol) -> str:
    """Convert an RDKit Mol into a MolBlock string with 3D coordinates."""
    return Chem.MolToMolBlock(mol)


def mol_from_block(block: str, *, sanitize: bool = False) -> Chem.Mol:
    """Reconstruct an RDKit Mol from a MolBlock string."""
    mol = Chem.MolFromMolBlock(block, removeHs=False, sanitize=sanitize)
    if mol is None:
        raise ValueError("Failed to parse MolBlock into an RDKit Mol.")
    return mol


def mols_to_blocks(mols: Iterable[Chem.Mol]) -> List[str]:
    """Convert an iterable of molecules into MolBlock strings."""
    return [mol_to_block(m) for m in mols]


def blocks_to_mols(blocks: Iterable[str], *, sanitize: bool = False) -> List[Chem.Mol]:
    """Convert MolBlock strings into RDKit molecules."""
    return [mol_from_block(b, sanitize=sanitize) for b in blocks]
