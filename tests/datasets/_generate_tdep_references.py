"""Build the frozen TDEP reference fixtures under ``tests/datasets/tdep_*``.

Run by hand when the TDEP outputs change; the tests only read the checked-in
``reference.npz`` and never call TDEP.

    micromamba run -n gkmx python tests/datasets/_generate_tdep_references.py

Source: the patched-TDEP sweep in ``benchmark_codes/gvm_sweep`` (which writes
``velocity_offdiagonal`` / ``n_invariant_operations``) plus the force constants
from the 240905 project. One material per space group present there -- 221
Pm-3m, 225 Fm-3m, 166 R-3m -- chosen for coverage:

  KI_bcc  2 atoms,  6 modes, |LG| 2..48   cheapest, highest symmetry
  Rb2O    3 atoms,  9 modes, |LG| 2..48   antifluorite, the only 3-atom cell
  KPTe2   4 atoms, 12 modes, |LG| 1..12   only low-symmetry case, only |LG| = 1

|LG| = 1 matters: there the little-group average must be bit-identical to no
average, which is the control that stops the whole comparison passing vacuously.

Units are converted here, once, so the tests never carry conversion factors:
frequencies rad/s -> THz, velocities m/s -> A/fs. Lifetimes (s) and heat
capacities (J/K) are stored as TDEP writes them, since kappa is rebuilt from
them in SI and compared against TDEP's own W/mK.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from ase.io import read, write

BOHR = 1.8897261246257702
SWEEP = Path("/home/sure/Projects/gkmx_project/benchmark_codes/gvm_sweep")
DATA = Path("/home/sure/data/240905_MLGK_16_materials")
SUB = "lattice_expansion/lattice_expansion_300/MD_04/tdep"
HERE = Path(__file__).resolve().parent

MATERIALS = ("KI_bcc", "Rb2O", "KPTe2")


def _kappa_block(lines, header):
    """The kxx, kyy, kzz triple under a ``# <header>`` line of outfile.thermal_conductivity."""
    for i, line in enumerate(lines):
        if line.strip().lstrip("# ").startswith(header):
            return np.array([float(x) for x in lines[i + 2].split()][:3])
    raise KeyError(f"{header!r} not found in outfile.thermal_conductivity")


def build(name):
    src, out = SWEEP / name, HERE / f"tdep_{name}"
    out.mkdir(exist_ok=True)

    uc = read(src / "infile.ucposcar", format="vasp")
    sc = read(src / "infile.ssposcar", format="vasp")
    write(out / "geometry.in.primitive", uc, format="aims")
    write(out / "geometry.in.supercell", sc, format="aims")
    fc = (DATA / name / SUB / "FORCE_CONSTANTS_tdep").read_bytes()
    (out / "FORCE_CONSTANTS_tdep").write_bytes(fc)

    with h5py.File(src / "outfile.thermal_conductivity_grid.hdf5", "r") as f:
        q_ir_bohr = np.array(f["qpoints_irreducible"])
        q_full_bohr = np.array(f["qpoints"])
        freq = np.array(f["frequencies"])
        gv = np.array(f["group_velocities"])
        v_off = np.array(f["velocity_offdiagonal"])
        nlg = np.array(f["n_invariant_operations"]).ravel().astype(int)
        eig = np.array(f["eigenvectors_re"]) + 1j * np.array(f["eigenvectors_im"])
        tau = np.array(f["lifetimes"])
        lw = np.array(f["linewidths"])
        cv = np.array(f["harmonic_heat_capacity"])

    # TDEP writes q Cartesian in Bohr^-1 against a 2pi-free reciprocal lattice.
    A = np.array(uc.cell)

    def to_frac(x):
        return (x * BOHR) @ A.T

    q_ir, q_full = to_frac(q_ir_bohr), to_frac(q_full_bohr)

    # eigenvectors are stored on the full grid; keep only the irreducible q, in
    # the order of qpoints_irreducible, so the tests need no index gymnastics.
    M = round(len(q_full) ** (1 / 3))

    def grid(x):
        # (x + 1e-8) % 1.0, never x - floor(x): naive wrap sends -1e-16 to 0.999
        return np.rint(((x + 1e-8) % 1.0) * M).astype(int) % M

    key = {tuple(k): i for i, k in enumerate(grid(q_full))}
    idx = np.array([key[tuple(k)] for k in grid(q_ir)])

    # kappa_C is assembled on the irreducible wedge with TDEP's own weights,
    # w = (|G| / |LG|) / N_q, which sum to 1 exactly. Frequencies and linewidths
    # go in as rad/s: the coherence kernel (G1+G2)/((G1+G2)^2 + (w1-w2)^2) mixes
    # them, so both must share units.
    import spglib
    ops = spglib.get_symmetry(
        (A, uc.get_scaled_positions(), uc.get_atomic_numbers()), symprec=1e-5)
    weights = (len(ops["rotations"]) / nlg) / len(q_full)

    L = (src / "outfile.thermal_conductivity").read_text().splitlines()
    np.savez_compressed(
        out / "reference.npz",
        q_ir=q_ir, q_full=q_full,
        frequencies_THz=freq / (2 * np.pi * 1e12),
        group_velocities=gv / 1e5,                 # m/s -> A/fs
        eigenvectors_ir=eig[idx],                  # TDEP convention, as stored
        velocity_offdiagonal=v_off / 1e5,          # m/s -> A/fs
        n_invariant_operations=nlg,
        lifetimes_s=tau, heat_capacity_JK=cv,
        weights_ir=weights,
        frequencies_ir_rad_s=freq[idx],
        linewidths_ir_rad_s=lw[idx],
        heat_capacity_ir_JK=cv[idx],
        kappa_sma=_kappa_block(L, "Single mode approximation"),
        kappa_collective=_kappa_block(L, "Collective contribution"),
        kappa_coherent=_kappa_block(L, "Off diagonal"),
        kappa_total=_kappa_block(L, "Total thermal conductivity"),
    )
    kb = sum(p.stat().st_size for p in out.iterdir()) / 1024
    print(f"{name:8s} {len(uc):2d} atoms {3*len(uc):3d} modes  "
          f"{len(q_ir):3d} irreducible / {len(q_full)} q  |LG| "
          f"{sorted(set(nlg.tolist()))}  {kb:6.0f} KB")


if __name__ == "__main__":
    for m in MATERIALS:
        build(m)
