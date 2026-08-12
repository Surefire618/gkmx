"""Convert TDEP's native output into the frozen ``reference.npz`` the tests read.

Run by hand; the tests never call TDEP and never parse its output.

    micromamba run -n gkmx python tests/datasets/_generate_tdep_references.py

Two files per fixture, both shipped unmodified as TDEP wrote them:

    outfile.thermal_conductivity            kappa per channel, full tensor
    outfile.thermal_conductivity_grid.hdf5  mode-resolved arrays

They are the provenance. Anyone can open them, see TDEP's own dataset names, and
check what gkmx is being asked to reproduce. This script reads those two files
*from the fixture directory itself* and writes ``reference.npz`` beside them, so
the conversion is reproducible in place with no external run directory.

WHAT THE CONVERSION DOES, and nothing else:

  q-points     TDEP writes them Cartesian in Bohr^-1 against a 2pi-free
               reciprocal lattice; stored fractional, ``(q * BOHR) @ A.T``.
  frequencies  rad/s -> THz for `frequencies_THz`. The `*_ir_rad_s` arrays keep
               rad/s: the coherence kernel mixes frequencies with linewidths, so
               those two must share units.
  velocities   m/s -> A/fs, both the diagonal and `velocity_offdiagonal`. The
               latter is stored COMPLEX, assembled from TDEP's
               `velocity_offdiagonal` and `velocity_offdiagonal_im`. The
               imaginary part is not decoration: after the invariant-bilinear fix
               it enters `buf_velsq`, so a real-only reference cannot check
               kappa_C, and `|v_ss'|^2` is only phase-invariant with it.
  lifetimes,   stored exactly as TDEP writes them (s, J/K); kappa is rebuilt
  heat cap.    from them in SI and compared against TDEP's own W/mK.
  kappa        the six independent components kxx kyy kzz kxy kxz kyz are
               expanded into the full symmetric (3, 3). Stored as a tensor, not
               a diagonal, so the fixtures can assert the off-diagonal -- it is
               exactly 0 for every channel on all three cells, which is a real
               constraint on gkmx and was unavailable while only a (3,) diagonal
               was kept.
  irreducible  `eigenvectors_ir` and the other `*_ir` arrays are the full-grid
  wedge        arrays indexed at `qpoints_irreducible`, in that order, so the
               tests need no index gymnastics. `weights_ir` is TDEP's own
               w = (|G| / |LG|) / N_q, which sums to 1 exactly.

HOW THE NATIVE FILES WERE PRODUCED: `dev_scripts/run_tdep_fixture_refs.sh`, one
run per material with `benchmark_codes/tdep_build` -- the diagnostic patch that
emits `qpoints_irreducible` / `velocity_offdiagonal` / `n_invariant_operations`,
plus the coherence fix (the invariant bilinear in `kappa.f90`). Settings taken
verbatim from the original sweep: 300 K, 6^3, thirdorder, no isotope, Adaptive
Gaussian, sigma 1.0, 200 iterations, tol 1e-5.

Materials -- one per space group, chosen for coverage:

  KI_bcc  2 atoms,  6 modes, |LG| 2..48   cheapest, highest symmetry
  Rb2O    3 atoms,  9 modes, |LG| 2..48   antifluorite, the only 3-atom cell
  KPTe2   4 atoms, 12 modes, |LG| 1..12   only low-symmetry case, only |LG| = 1

|LG| = 1 matters: there the little-group average must be bit-identical to no
average, which is the control that stops the whole comparison passing vacuously.

One caveat worth knowing before regenerating: `kappa_collective` comes from the
iterative BTE, whose solution depends on where the 1e-5 convergence test lands.
Rerunning the same input moved KPTe2's by 1.9e-03 relative (and `kappa_total`,
which contains it, by 2.7e-05). Every other channel reproduced bit-identically.
Do not tighten a tolerance to chase that; it is the solver, not gkmx.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import spglib
from ase.io import read

# TDEP writes q Cartesian in Bohr^-1 against a 2pi-free reciprocal lattice.
BOHR = 1.8897261246257702
HERE = Path(__file__).resolve().parent

MATERIALS = ("KI_bcc", "Rb2O", "KPTe2")

CHANNELS = {
    "sma": "Single mode approximation",
    "collective": "Collective contribution",
    "coherent": "Off diagonal",
    "total": "Total thermal conductivity",
}


def read_kappa_tensors(path):
    """``{channel: (3, 3)}`` from ``outfile.thermal_conductivity``.

    Each channel is a comment header followed by ``kxx kyy kzz kxy kxz kyz``.
    kappa is symmetric, so those six fix the tensor.
    """
    lines = Path(path).read_text().splitlines()
    out = {}
    for key, header in CHANNELS.items():
        for i, line in enumerate(lines):
            if line.strip().lstrip("# ").startswith(header):
                xx, yy, zz, xy, xz, yz = [float(x) for x in lines[i + 2].split()][:6]
                out[key] = np.array([[xx, xy, xz],
                                     [xy, yy, yz],
                                     [xz, yz, zz]])
                break
    missing = set(CHANNELS) - set(out)
    if missing:
        raise KeyError(f"{path}: no block for {sorted(missing)}")
    return out


def build(name):
    out = HERE / f"tdep_{name}"
    uc = read(out / "geometry.in.primitive", format="aims")

    with h5py.File(out / "outfile.thermal_conductivity_grid.hdf5", "r") as f:
        q_ir_bohr = np.array(f["qpoints_irreducible"])
        q_full_bohr = np.array(f["qpoints"])
        freq = np.array(f["frequencies"])
        gv = np.array(f["group_velocities"])
        v_off = (np.array(f["velocity_offdiagonal"])
                 + 1j * np.array(f["velocity_offdiagonal_im"]))
        nlg = np.array(f["n_invariant_operations"]).ravel().astype(int)
        eig = np.array(f["eigenvectors_re"]) + 1j * np.array(f["eigenvectors_im"])
        tau = np.array(f["lifetimes"])
        lw = np.array(f["linewidths"])
        cv = np.array(f["harmonic_heat_capacity"])

    A = np.array(uc.cell)

    def to_frac(x):
        return (x * BOHR) @ A.T

    q_ir, q_full = to_frac(q_ir_bohr), to_frac(q_full_bohr)

    # Eigenvectors are stored on the full grid; keep only the irreducible q, in
    # the order of qpoints_irreducible, so the tests need no index gymnastics.
    M = round(len(q_full) ** (1 / 3))

    def grid(x):
        # (x + 1e-8) % 1.0, never x - floor(x): naive wrap sends -1e-16 to 0.999
        return np.rint(((x + 1e-8) % 1.0) * M).astype(int) % M

    key = {tuple(k): i for i, k in enumerate(grid(q_full))}
    idx = np.array([key[tuple(k)] for k in grid(q_ir)])

    ops = spglib.get_symmetry(
        (A, uc.get_scaled_positions(), uc.get_atomic_numbers()), symprec=1e-5)
    weights = (len(ops["rotations"]) / nlg) / len(q_full)

    kappa = read_kappa_tensors(out / "outfile.thermal_conductivity")

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
        kappa_sma=kappa["sma"],
        kappa_collective=kappa["collective"],
        kappa_coherent=kappa["coherent"],
        kappa_total=kappa["total"],
    )
    off = max(float(np.abs(v - np.diag(np.diagonal(v))).max()) for v in kappa.values())
    kb = sum(p.stat().st_size for p in out.iterdir()) / 1024
    print(f"{name:8s} {len(uc):2d} atoms {3*len(uc):3d} modes  "
          f"{len(q_ir):3d} irreducible / {len(q_full)} q  |LG| "
          f"{sorted(set(nlg.tolist()))}  max|kappa off-diag| {off:.1e}  {kb:6.0f} KB")


if __name__ == "__main__":
    for m in MATERIALS:
        build(m)
