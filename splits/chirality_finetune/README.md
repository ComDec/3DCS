# Chirality fine-tuning split

The train / validation / test split used for the chirality fine-tuning experiments. It is an
8:1:1 split of the chirality set by **Bemis–Murcko scaffold of the parent molecule**: every parent
molecule, together with all of its enumerated stereoisomers and all of their conformers, falls
entirely inside one of the three splits, and no scaffold is shared between splits.

The files index the published dataset `EscheWang/3dcs`, config `chirality` (fields
`key` / `mol_id` / `en_id` / `n_conformers` / `offset` / `mol_blocks`).

| File | Data rows | Content |
|---|---|---|
| `train.csv` | 11,927 | one row per stereoisomer entry |
| `valid.csv` | 1,496 | one row per stereoisomer entry |
| `test.csv` | 1,480 | one row per stereoisomer entry |

Each file has the header `key,mol_id,en_id`, and the three columns are copied verbatim from the
corresponding row of the `chirality` config, so `key` can be used as a join key directly. Rows are
sorted by `mol_id`, then by `en_id`. A `key` looks like

```
CHEMBL1255901::en1_A3:S;A8:S
```

i.e. `<mol_id>::en<en_id>_<atom index>:<R|S>;...`, where the suffix lists the CIP label assigned to
each stereocentre of that stereoisomer.

## Counts

| | train | valid | test | total |
|---|---|---|---|---|
| Entries (`key`) | 11,927 | 1,496 | 1,480 | 14,903 |
| Parent molecules (`mol_id`) | 3,072 | 431 | 400 | 3,903 |
| … of which have ≥ 2 stereoisomers | 3,024 | 421 | 397 | 3,842 |
| Conformers (sum of `n_conformers`) | 42,389 | 5,071 | 4,931 | 52,391 |
| Distinct Murcko scaffolds | 1,995 | 195 | 271 | 2,461 |

Ratios: 80.0 / 10.0 / 9.9 % by entry, 78.7 / 11.0 / 10.2 % by parent molecule,
80.9 / 9.7 / 9.4 % by conformer.

Both molecule counts are given because they are used in different places. 3,072 / 431 / 400 is the
number of distinct `mol_id` values in each file. 3,024 / 421 / 397 is the number of parent molecules
that contribute at least two stereoisomers, which is the subset that supports a within-molecule
stereoisomer comparison; the remaining 48 / 10 / 3 molecules contribute a single stereoisomer entry
each.

The union of the three files is the whole `chirality` config: all 14,903 rows and all 52,391
conformers are assigned to exactly one split, with no row left over and no row in two splits.

## Verification

All numbers below were recomputed from `EscheWang/3dcs` (config `chirality`) with RDKit 2026.03.6
rather than carried over from the original run.

| Check | Result |
|---|---|
| Every row of the three files resolves to a row of the `chirality` config | 14,903 / 14,903 matched, 0 unmatched |
| `mol_id` and `en_id` agree with the values stored in the dataset | 0 mismatches over 14,903 rows |
| Union of the three splits vs. the full `chirality` config | set-equal; 0 rows in only one of the two |
| A `key` appears in two splits | 0 for all three pairs |
| A `mol_id` appears in two splits | 0 for all three pairs |
| A Murcko scaffold appears in two splits | 0 for all three pairs |
| Conformer counts (`n_conformers`) sum to the per-split totals above | agrees |

Scaffolds were computed from the `mol_blocks` field: for each `mol_id`, the first conformer of its
first stereoisomer was parsed with `Chem.MolFromMolBlock(..., removeHs=True)` and reduced with
`MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)`. All 3,903 molecules parsed
and produced a scaffold. Ignoring chirality is the stricter test here, since it merges scaffolds
that differ only in stereochemistry and can therefore only create collisions, not hide them.

125 of the molecules are acyclic and so reduce to the empty Murcko scaffold. All 125 are in
`valid.csv`, which is why `valid` has 431 molecules but only 195 distinct scaffolds. The empty
scaffold is confined to a single split, so it does not cause any train/valid/test scaffold sharing;
the scaffold-overlap counts above are 0 whether or not the empty scaffold is included.

## Use

```python
import pandas as pd
from datasets import load_dataset

ds = load_dataset("EscheWang/3dcs", "chirality", split="train")

keys = set(pd.read_csv("splits/chirality_finetune/test.csv")["key"])
test = ds.filter(lambda batch: [k in keys for k in batch["key"]], batched=True)

print(test.num_rows, sum(test["n_conformers"]))  # 1480 4931
```

`mol_id` and `en_id` are included as separate columns so that the same files can be used to group
by parent molecule (for example, to build stereoisomer pairs within a molecule) without having to
parse `key`.

## Scope

These files are the data split only. The fine-tuning code and the fine-tuned checkpoints are not
part of this repository.

## SHA-256

```
160658b1e57f5797a0102f6cde3668111261775dee94f1984f72f8e2207ce5cf  train.csv
6928e3bbf3b15358e5e347791a42c1d67ad0de07c0bfa0416e3c792505b0a15c  valid.csv
fe36850cf81bf67ae4d9eba2c9ed3365d5ca4ae464651295c459cba3b976867f  test.csv
```
