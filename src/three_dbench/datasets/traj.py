"""Hugging Face dataset conversion for the trajectory benchmark."""

from __future__ import annotations

import pickle
import warnings
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import datasets
import numpy as np

from .serialization import mol_to_block

ENERGY_PRECISION_POLICIES = ("error", "warn", "ignore")

# A dataset is flagged as quantized when fewer than this fraction of its frames carry a distinct energy ...
_MIN_UNIQUE_FRACTION = 0.5
# ... or when every value is float32-representable and the float32 step at the energy magnitude is
# at least this fraction of the energy inter-quartile range.
_MAX_FLOAT32_STEP_OVER_IQR = 1e-4
_MIN_FRAMES_FOR_CHECK = 50


class QuantizedEnergiesError(ValueError):
    """Raised when trajectory energies look quantized (e.g. a float32 cast of absolute energies)."""


@dataclass(frozen=True)
class EnergyPrecisionReport:
    """Summary of the numerical resolution of one molecule's energies."""

    mol_type: str
    n_frames: int
    n_unique: int
    unique_fraction: float
    float32_representable: bool
    float32_step: float
    iqr: float
    quantized: bool
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def detect_quantized_energies(energies: Iterable[float], mol_type: str = "") -> EnergyPrecisionReport:
    """Heuristically detect quantized (low-resolution) energies.

    rMD17 stores absolute total energies of about -4e5 kcal/mol. Casting them to float32 leaves a
    resolution of 1/64-1/256 kcal/mol and only ~1-3% distinct values per molecule, which silently
    changes CKA, Smoothness and KS. Energies are flagged when

    * fewer than 50% of the frames carry a distinct value, or
    * every value is exactly float32-representable *and* the float32 step at the largest magnitude
      is at least 1e-4 of the inter-quartile range of the energies.

    Relative energies stored as float32 (small magnitude, fine float32 step) are not flagged.
    """
    arr = np.asarray(energies, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n < _MIN_FRAMES_FOR_CHECK:
        return EnergyPrecisionReport(mol_type, n, n, 1.0, False, float("nan"), float("nan"), False, "too few frames")

    n_unique = int(np.unique(arr).size)
    unique_fraction = n_unique / n
    f32_repr = bool(np.array_equal(arr, arr.astype(np.float32).astype(np.float64)))
    max_abs = float(np.max(np.abs(arr)))
    step32 = float(np.spacing(np.float32(max_abs)))
    q75, q25 = np.percentile(arr, [75.0, 25.0])
    iqr = float(q75 - q25)

    reasons = []
    if unique_fraction < _MIN_UNIQUE_FRACTION:
        reasons.append(f"only {n_unique} distinct values for {n} frames ({unique_fraction:.2%})")
    if f32_repr and (iqr <= 0.0 or step32 >= _MAX_FLOAT32_STEP_OVER_IQR * iqr):
        reasons.append(
            f"all values are float32-representable; float32 step {step32:.3g} at |E|~{max_abs:.3g} "
            f"is >= {_MAX_FLOAT32_STEP_OVER_IQR:g} x IQR ({iqr:.3g})"
        )
    quantized = bool(reasons)
    return EnergyPrecisionReport(
        mol_type=mol_type,
        n_frames=n,
        n_unique=n_unique,
        unique_fraction=float(unique_fraction),
        float32_representable=f32_repr,
        float32_step=step32,
        iqr=iqr,
        quantized=quantized,
        reason="; ".join(reasons),
    )


def check_energy_precision(
    reports: Iterable[EnergyPrecisionReport], policy: str = "error"
) -> list[EnergyPrecisionReport]:
    """Apply ``policy`` ("error", "warn", "ignore") to the quantized reports and return them."""
    if policy not in ENERGY_PRECISION_POLICIES:
        raise ValueError(f"policy must be one of {ENERGY_PRECISION_POLICIES}, got {policy!r}")
    flagged = [r for r in reports if r.quantized]
    if not flagged or policy == "ignore":
        return flagged
    lines = "\n".join(f"  - {r.mol_type}: {r.reason}" for r in flagged)
    msg = (
        "Trajectory energies look quantized (for example, a float32 cast of absolute energies):\n"
        f"{lines}\n"
        "Published energy metrics (CKA, Smoothness, KS) were computed with float64 rMD17 energies and are not "
        "reproducible from quantized values. Re-download the float64 energies (EscheWang/3dcs, config "
        "'traj_energies', current revision) or convert the original rMD17 npz files with "
        "`python -m three_dbench convert traj`. Use --energy-precision-check warn|ignore to proceed anyway."
    )
    if policy == "error":
        raise QuantizedEnergiesError(msg)
    warnings.warn(msg, stacklevel=2)
    return flagged


def convert_traj_frames_pkl_to_hf(
    mol_pkl_dir: Path,
    output_dir: Path,
    *,
    include_mol_blocks: bool = True,
) -> datasets.Dataset:
    """Convert trajectory conformer pickles into a Hugging Face dataset."""
    pkl_paths = sorted(mol_pkl_dir.glob("rmd17_*.pkl"))

    def _iter_rows():
        for pkl_path in pkl_paths:
            mol_type = pkl_path.stem
            with pkl_path.open("rb") as f:
                mol_list = pickle.load(f)
            for idx, mol in enumerate(mol_list):
                yield {
                    "mol_type": mol_type,
                    "frame_idx": int(idx),
                    "mol_block": mol_to_block(mol) if include_mol_blocks else "",
                }

    features = datasets.Features(
        {
            "mol_type": datasets.Value("string"),
            "frame_idx": datasets.Value("int32"),
            "mol_block": datasets.Value("string"),
        }
    )
    ds = datasets.Dataset.from_generator(_iter_rows, features=features)
    ds.save_to_disk(str(output_dir))
    return ds


def convert_traj_energy_npz_to_hf(energy_dir: Path, output_dir: Path) -> datasets.Dataset:
    """Convert trajectory energies into a Hugging Face dataset.

    Energies are stored as float64 without any cast, so the dataset holds exactly the values of the
    source npz files (rMD17: absolute total energies in kcal/mol).
    """
    rows = []
    for npz_path in sorted(energy_dir.glob("rmd17_*.npz")):
        mol_type = npz_path.stem
        with np.load(npz_path) as data:
            energies = np.asarray(data["energies"], dtype=np.float64).reshape(-1)
        report = detect_quantized_energies(energies, mol_type=mol_type)
        if report.quantized:
            warnings.warn(f"Source energies for {mol_type} look quantized: {report.reason}", stacklevel=2)
        rows.append(
            {
                "mol_type": mol_type,
                "n_frames": int(energies.shape[0]),
                "energies": energies.tolist(),
            }
        )
    ds = datasets.Dataset.from_list(rows)
    ds.save_to_disk(str(output_dir))
    return ds


def load_traj_frames_dataset(dataset_dir: Path) -> datasets.Dataset:
    """Load the trajectory frames dataset from disk."""
    return datasets.load_from_disk(str(dataset_dir))


def load_traj_energy_dataset(dataset_dir: Path) -> datasets.Dataset:
    """Load the trajectory energies dataset from disk."""
    return datasets.load_from_disk(str(dataset_dir))
