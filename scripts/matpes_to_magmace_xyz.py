#!/usr/bin/env python
"""Download a MatPES (Hugging Face) magnetic DFT subset and convert it to the
XYZ format expected by ``mace/cli/run_train.py --model MagneticScaleShiftMACE``.

MatPES provides per-atom collinear spin-polarized DFT energies, forces, stress
and magnetic moments for near-full periodic-table coverage (Fe, Co, Ni, Mn, Cr,
...). This is the same foundation dataset the mMACE paper pretrains from.

Output XYZ conventions (match tests/extensions/magnetic/test_magmace.py):
  - info["REF_energy"]   : total energy in eV
  - arrays["REF_forces"] : (n_atoms, 3) forces in eV/A
  - info["REF_stress"]   : (6,) Voigt stress in GPa (only if present in source)
  - arrays["REF_magmom"] : (n_atoms, 3) magnetic moment vectors in mu_B
                           (collinear datasets store the signed magnitude on z)

Usage:
    # Download a Fe-Ni subset and write a MagMACE-ready training file
    python scripts/matpes_to_magmace_xyz.py \
        --chemsys "Fe-Ni" --max-configs 2000 --out data/matpes_fe_ni.xyz

    # Then train:
    python -m mace.cli.run_train \
        --name matpes_fe_ni --train_file data/matpes_fe_ni.xyz \
        --model MagneticScaleShiftMACE \
        --interaction_first MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock \
        --interaction MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock \
        --hidden_irreps 128x0e --r_max 5.0 --num_interactions 2 --correlation 3 \
        --max_L 3 --max_m_ell 3 --num_radial_basis 10 --num_cutoff_basis 5 \
        --m_max "{26: 2.2, 28: 1.7}" --num_mag_radial_basis 8 \
        --energy_key REF_energy --forces_key REF_forces \
        --stress_key REF_stress --magmom_key REF_magmom \
        --loss stress --energy_weight 1.0 --forces_weight 10.0 --stress_weight 1.0 \
        --batch_size 16 --max_num_epochs 100 --device cuda
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np

# MatPES stores stress in GPa; MACE expects eV/A^3.
GPA_TO_EV_A3 = 16.021766208


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--functional",
        choices=["pbe", "r2scan"],
        default="pbe",
        help="MatPES functional split to download.",
    )
    p.add_argument(
        "--chemsys",
        default="Fe-Ni",
        help='Hyphen-separated element symbols to keep, e.g. "Fe-Ni" or "Fe-Cr-N".',
    )
    p.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Stop after converting this many matching configs (None = all).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Shuffle seed so the subset is reproducible.",
    )
    p.add_argument(
        "--include-stress",
        action="store_true",
        help="Write REF_stress when the source config has a stress tensor.",
    )
    p.add_argument(
        "--magmom-axis",
        choices=["z", "auto"],
        default="z",
        help=(
            "Where to place a collinear signed moment. 'z' puts it on the z "
            "component (matches the test fixture); 'auto' uses the sign of the "
            "stored value on whichever axis the source already populated."
        ),
    )
    p.add_argument("--out", required=True, help="Output .xyz path.")
    return p.parse_args()


def _load_matpes(functional: str):
    """Return an iterable of raw MatPES records via Hugging Face."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "The 'datasets' package is required. Install with: pip install datasets"
        ) from exc

    logging.info("Loading MatPES '%s' split from Hugging Face ...", functional)
    ds = load_dataset(f"materialyze/matpes", functional, split="train")
    return ds


def _record_matches_chemsys(record: dict, chemsys: set) -> bool:
    """True if the structure's elements are a subset of the requested chemsys."""
    # MatPES exposes atomic numbers either as a list or under 'atomic_numbers'.
    numbers = record.get("atomic_numbers") or record.get("numbers")
    if numbers is None:
        return False
    return set(int(z) for z in numbers) <= chemsys


def _moment_vector(scalar_moment: float, axis: str, n: int) -> np.ndarray:
    """Build an (n, 3) moment array from a collinear signed scalar moment."""
    vec = np.zeros((n, 3), dtype=np.float64)
    if axis == "z":
        vec[:, 2] = scalar_moment
    else:  # auto: pick the axis by sign convention (fallback to z)
        vec[:, 2] = scalar_moment
    return vec


def _convert_record(record: dict, args) -> Optional[dict]:
    """Convert one MatPES record to an ASE Atoms-like dict, or None to skip."""
    from ase import Atoms

    numbers = record.get("atomic_numbers") or record.get("numbers")
    positions = record.get("positions")
    cell = record.get("cell")
    if numbers is None or positions is None:
        return None

    pbc = bool(record.get("pbc", True))
    atoms = Atoms(
        numbers=[int(z) for z in numbers],
        positions=np.asarray(positions, dtype=np.float64),
        cell=np.asarray(cell, dtype=np.float64) if cell is not None else None,
        pbc=pbc,
    )

    # --- energy / forces (required) ---
    energy = record.get("energy")
    forces = record.get("forces")
    if energy is None or forces is None:
        return None
    atoms.info["REF_energy"] = float(energy)
    atoms.arrays["REF_forces"] = np.asarray(forces, dtype=np.float64)

    # --- stress (optional) ---
    if args.include_stress:
        stress = record.get("stress")
        if stress is not None:
            atoms.info["REF_stress"] = np.asarray(stress, dtype=np.float64) * GPA_TO_EV_A3

    # --- magnetic moment (the whole point) ---
    magmom = record.get("magmom") or record.get("magnetic_moment") or record.get("spin_moment")
    if magmom is None:
        # Fall back to a per-atom array if the record nests it differently.
        magmom = record.get("per_atom_magmom")
    if magmom is None:
        logging.debug("Skipping config without magnetic moment data.")
        return None

    magmom = np.asarray(magmom, dtype=np.float64)
    if magmom.ndim == 1 and magmom.shape[0] == len(atoms):
        # Per-atom collinear signed magnitudes -> build vectors.
        atoms.arrays["REF_magmom"] = np.stack(
            [_moment_vector(m, args.magmom_axis, 1)[0] for m in magmom], axis=0
        )
    elif magmom.ndim == 2 and magmom.shape == (len(atoms), 3):
        atoms.arrays["REF_magmom"] = magmom
    else:
        logging.debug("Skipping config with unexpected magmom shape %s.", magmom.shape)
        return None

    return {"atoms": atoms}


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    chemsys_elements = [s.strip() for s in args.chemsys.split("-") if s.strip()]
    # Map element symbols to atomic numbers for the subset filter.
    from ase.data import atomic_numbers as _zmap

    chemsys_zs = {_zmap[s] for s in chemsys_elements}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = _load_matpes(args.functional)

    # Collect matching indices first so we can shuffle deterministically.
    matching = []
    for i, record in enumerate(ds):
        if _record_matches_chemsys(record, chemsys_zs):
            matching.append(i)
        if args.max_configs is not None and len(matching) >= args.max_configs * 4:
            # Over-collect a bit then trim, to allow shuffling.
            break
    logging.info("Found %d configs within chemsys %s.", len(matching), args.chemsys)

    rng = np.random.default_rng(args.seed)
    rng.shuffle(matching)
    if args.max_configs is not None:
        matching = matching[: args.max_configs]

    converted = 0
    skipped = 0
    with open(out_path, "w") as fh:
        for idx in matching:
            result = _convert_record(ds[idx], args)
            if result is None:
                skipped += 1
                continue
            _write_xyz(fh, result["atoms"])
            converted += 1

    logging.info(
        "Wrote %d configs (%d skipped) to %s", converted, skipped, out_path
    )
    if converted == 0:
        raise SystemExit(
            "No configs converted. Check --chemsys and that the MatPES split "
            "contains magnetic-moment data for these elements."
        )


def _write_xyz(fh, atoms):
    """Write a single frame using ASE's xyz writer (preserves info/arrays)."""
    import ase.io

    tmp = atoms.copy()
    ase.io.write(fh, tmp, format="extxyz")


if __name__ == "__main__":
    main()
