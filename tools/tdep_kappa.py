"""kappa_BTE and kappa_QHGK from a TDEP run: TDEP's lifetimes + cv, gkmx's harmonics.

    GKMX_MASSES=TDEP python tools/tdep_kappa.py RUN_DIR [...] [--extrapolate]

Reads `infile.ucposcar`/`infile.ssposcar` (or the aims-format pair the in-repo
fixtures ship), `FORCE_CONSTANTS_tdep`, and `outfile.thermal_conductivity_grid
.hdf5` (or its older name). `outfile.thermal_conductivity` is optional and used
only as a reference. `--extrapolate` calls the Green-Kubo pipeline's own
`get_interpolation_data`, so the kappa(1/nq) -> kappa(inf) method cannot drift
from the pipeline's.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
from ase.io import read

from gkmx import DynamicalMatrix, keys
from gkmx import _constants as C
from gkmx import masses as _masses
from gkmx._log import talk, warn
from gkmx.brillouin import get_q_grid
from gkmx.interpolation import get_interpolation_data
from gkmx.io import parse_force_constants
from gkmx.kappa import get_harmonic_cv, get_kappa_BTE, get_kappa_QHGK, symmetrize_kappa
from gkmx.phonon import CONVENTIONS

BOHR = 1.8897261246257702      # Bohr^-1 -> A^-1, TDEP's 2pi-free q convention
EV = 1.602176634e-19           # J per eV

GRID_FILES = (
    "outfile.thermal_conductivity_grid.hdf5",
    "outfile.grid_thermal_conductivity.hdf5",   # older TDEP name
)
GEOMETRY_FILES = (
    ("infile.ucposcar", "infile.ssposcar", "vasp"),
    ("geometry.in.primitive", "geometry.in.supercell", "aims"),
)


def _talk(message):
    talk(message, prefix="tdep_kappa")


def _warn(message):
    warn(message, prefix="tdep_kappa")


def get_q_mesh(q_frac, max_n=512, tol=1e-6):
    """Snap q onto its own regular mesh, one axis at a time.

    Converting TDEP's Bohr^-1 q back through the lattice leaves a ~1e-9 residual,
    which exceeds the 1e-9 wrap tolerance in `interpolate_to_grid` and
    `get_unit_grid_extended`: a coordinate meant to be 0 wraps to 0.999999999, is
    never replicated to the +1 face, and `griddata` then NaNs whole zone faces of
    every dense mesh.
    """
    q_frac = np.array(q_frac, dtype=float)

    for axis in range(3):
        mesh_n = None
        for n in range(1, max_n + 1):
            scaled = q_frac[:, axis] * n
            if np.abs(scaled - np.rint(scaled)).max() < tol:
                mesh_n = n
                break

        if mesh_n is None:
            _warn(f"q-axis {axis} is not a regular mesh; leaving it as read")
            continue

        q_frac[:, axis] = np.rint(q_frac[:, axis] * mesh_n) / mesh_n

    return q_frac


def load_thermal_conductivity(path, temperature, tol=1.0):
    """``(K, channel)`` from `outfile.thermal_conductivity`, or ``(None, why)``.

    Current TDEP writes labelled blocks of ``kxx kyy kzz kxy kxz kyz``; the SMA
    block is the one this tool rebuilds, since the grid lifetimes are RTA. Older
    builds wrote bare rows ``T kxx kyy kzz kxz kyz kxy kzx kzy kyx`` -- kxz
    *before* kxy -- holding the iterative kappa, so a gap against it is the
    collective correction. Misreading that order gives an asymmetric tensor, so
    it is checked and rejected rather than returned wrong.
    """
    path = Path(path)
    if not path.exists():
        return None, "absent"

    labelled_blocks = {}
    legacy_rows = []
    current_label = None

    for line in path.read_text().splitlines():
        line = line.strip()

        if line.startswith("#"):
            comment = line.lstrip("# ").strip()
            is_channel_name = (comment
                               and not comment[0].isdigit()
                               and ":" not in comment
                               and not comment.startswith("k"))
            if is_channel_name:
                current_label = comment
            continue

        try:
            values = [float(x) for x in line.split()]
        except ValueError:
            continue

        if len(values) == 6 and current_label is not None:
            labelled_blocks.setdefault(current_label, values)
        elif len(values) >= 10:
            legacy_rows.append(values)

    if labelled_blocks:
        return _read_labelled_block(labelled_blocks)
    if legacy_rows:
        return _read_legacy_row(np.atleast_2d(legacy_rows), temperature, tol)
    return None, "unparsable"


def _read_labelled_block(blocks):
    """The SMA block of a current-format file, as a symmetric (3, 3)."""
    name = None
    for label in blocks:
        if label.startswith("Single mode"):
            name = label
            break

    if name is None:
        return None, f"no SMA block (have {list(blocks)})"

    kxx, kyy, kzz, kxy, kxz, kyz = blocks[name]
    K = np.array([[kxx, kxy, kxz],
                  [kxy, kyy, kyz],
                  [kxz, kyz, kzz]])
    return K, "SMA"


def _read_legacy_row(rows, temperature, tol):
    """The row at `temperature` of a legacy multi-temperature file."""
    hits = np.flatnonzero(np.abs(rows[:, 0] - temperature) < tol)
    if hits.size == 0:
        return None, f"no row within {tol} K of {temperature}"

    row = rows[hits[0]]
    kxx, kyy, kzz = row[1], row[2], row[3]
    kxz, kyz, kxy = row[4], row[5], row[6]
    kzx, kzy, kyx = row[7], row[8], row[9]
    K = np.array([[kxx, kxy, kxz],
                  [kyx, kyy, kyz],
                  [kzx, kzy, kzz]])

    if np.abs(K - K.T).max() > 1e-3 * np.abs(K).max():
        return None, "legacy columns give an asymmetric tensor"
    return K, "iterative (legacy format; includes the collective correction)"


def _find_input_files(run_dir):
    """``(primitive_path, supercell_path, ase_format, grid_path)`` for a run."""
    geometry = None
    for primitive_name, supercell_name, ase_format in GEOMETRY_FILES:
        primitive = run_dir / primitive_name
        supercell = run_dir / supercell_name
        if primitive.exists() and supercell.exists():
            geometry = (primitive, supercell, ase_format)
            break

    grid = None
    for grid_name in GRID_FILES:
        if (run_dir / grid_name).exists():
            grid = run_dir / grid_name
            break

    if geometry is None:
        raise FileNotFoundError(
            f"{run_dir}: no geometry pair; looked for {GEOMETRY_FILES}")
    if grid is None:
        raise FileNotFoundError(
            f"{run_dir}: no grid file; looked for {GRID_FILES}")

    return (*geometry, grid)


def load(run_dir, grid_file=None):
    """Everything one TDEP run supplies, converted to gkmx units."""
    run_dir = Path(run_dir)
    primitive_path, supercell_path, ase_format, grid_path = _find_input_files(run_dir)
    if grid_file is not None:
        grid_path = Path(grid_file)

    primitive = read(primitive_path, format=ase_format)
    supercell = read(supercell_path, format=ase_format)
    force_constants = np.asarray(
        parse_force_constants(str(run_dir / "FORCE_CONSTANTS_tdep"), two_dim=False))

    with h5py.File(grid_path, "r") as f:
        temperature = float(np.asarray(f.attrs["temperature"]).ravel()[0])
        q_bohr = np.asarray(f["qpoints"])
        frequencies_rad_s = np.asarray(f["frequencies"])
        tau_s = np.asarray(f["lifetimes"])
        cv_JK = np.asarray(f["harmonic_heat_capacity"])
        v_ms = np.asarray(f["group_velocities"])

    q_frac = get_q_mesh((q_bohr * BOHR) @ np.array(primitive.cell).T)
    if np.linalg.norm(q_frac[0]) > 1e-9:
        raise ValueError(
            f"{run_dir}: the first q-point is not Gamma. The acoustic w_inv cutoff "
            "is a batch order statistic and changes silently without Gamma in the "
            "batch.")

    kappa_ref, ref_channel = load_thermal_conductivity(
        run_dir / "outfile.thermal_conductivity", temperature)
    if kappa_ref is None:
        _warn(f"{run_dir.name}: no reference kappa ({ref_channel}); "
              "kappa_BTE will have no external check this run")

    # gkmx's kappa kernels carry no volume and no 1/Nq -- the whole normalization
    # lives in cv (gkmx.kappa). TDEP sums (1/(V nq)) cv v v tau, so both factors
    # go here, along with J -> eV.
    cv_gkmx = cv_JK / (EV * primitive.get_volume() * len(q_frac))

    return SimpleNamespace(
        name=run_dir.name,
        temperature=temperature,
        primitive=primitive,
        supercell=supercell,
        force_constants=force_constants,
        q_frac=q_frac,
        nq=len(q_frac),
        tau_s=tau_s,
        tau_fs=tau_s * 1e15,
        cv_JK=cv_JK,
        cv=cv_gkmx,
        v_ms=v_ms,
        w_gkmx=frequencies_rad_s / (2 * np.pi * 1e12) / C.omega_to_THz,
        kappa_ref=kappa_ref,
        kappa_ref_channel=ref_channel,
    )


def kappa(run, convention="TDEP", check_symmetry=False):
    """Both transport channels on TDEP's own q-grid.

    Returns the `DynamicalMatrix` alongside the results, retargeted onto that
    grid so `extrapolate` can train on it directly.
    """
    dmx = DynamicalMatrix(
        force_constants=run.force_constants, primitive=run.primitive,
        supercell=run.supercell, with_group_velocity_matrices=True,
        backend="numpy", precision="fp64", convention=convention)

    solution = dmx.get_solution(run.q_frac, with_group_velocity_matrices=True)

    # Retarget the object onto TDEP's grid. Its accessors are read-only
    # properties over two private fields -- `_q_grid` backs `q_points`, and
    # `_solution` backs `w_qs`, `w2_qs`, `w_inv_qs` and `v_qsa_cartesian` -- which
    # is everything `get_interpolation_data` reads. `get_mesh_and_solution` and
    # the rest of the object are untouched.
    dmx._q_grid = get_q_grid(run.q_frac, run.primitive)
    dmx._solution = solution

    w_qs = np.asarray(solution.w_qs)
    # Checked negative frequencies
    if (w_qs < -1e-6 * np.abs(w_qs).max()).any():
        raise ValueError(f"{run.name}: imaginary modes in the solve; "
                         "kappa_QHGK would be NaN everywhere")

    v_qsa = np.asarray(solution.v_qsa_cartesian)
    v_qssa = np.ascontiguousarray(np.asarray(solution.v_qssa_cartesian))

    kappa_bte = np.asarray(
        get_kappa_BTE(v_qsa, run.tau_fs, run.cv).data)
    kappa_qhgk = np.asarray(
        get_kappa_QHGK(v_qssa=v_qssa, tau_qs=run.tau_fs, w_qs=w_qs,
                       w_inv_qs=np.asarray(solution.w_inv_qs),
                       cv_qs=run.cv).data)

    # Split against the diagonal of the SAME operator. Subtracting kappa_BTE
    # instead would mix the coherence with a velocity-convention difference and
    # can come out negative.
    v_diagonal = np.einsum("qssa->qsa", v_qssa).real
    kappa_particle = np.asarray(
        get_kappa_BTE(v_diagonal, run.tau_fs, run.cv).data)
    kappa_coherence = kappa_qhgk - kappa_particle

    # Independent unit chain: the same kappa assembled directly in SI. A wrong
    # J/K -> eV/(K A^3) constant would just rescale kappa and still look plausible.
    kappa_si = np.einsum("qs,qsa,qsb,qs->ab", run.cv_JK, run.v_ms, run.v_ms,
                         run.tau_s) / (run.primitive.get_volume() * 1e-30 * run.nq)

    frequencies_gkmx = np.sort(np.abs(w_qs), axis=1) * C.omega_to_THz
    frequencies_tdep = np.sort(np.abs(run.w_gkmx), axis=1) * C.omega_to_THz

    results = {
        "name": run.name,
        "temperature": run.temperature,
        "nq": run.nq,
        "convention": convention,
        "mass_table": _masses.get_default(),
        "kappa_BTE": kappa_bte.tolist(),
        "kappa_QHGK": kappa_qhgk.tolist(),
        "kappa_particle": kappa_particle.tolist(),
        "kappa_coherence": kappa_coherence.tolist(),
        "coherent_share": float(np.trace(kappa_coherence) / np.trace(kappa_qhgk)),
        "kappa_ref_channel": run.kappa_ref_channel,
        "max_dfreq_THz": float(np.abs(frequencies_gkmx - frequencies_tdep).max()),
        "unit_chain_rel": float(np.abs(kappa_bte - kappa_si).max()
                                / max(np.abs(kappa_si).max(), 1e-30)),
    }

    _talk(f"=== {run.name}  T={run.temperature:g} K  nq={run.nq}  "
          f"convention={convention}  masses={_masses.get_default()} ===")
    _talk(f"  frequencies vs TDEP: {results['max_dfreq_THz']:.2e} THz"
          f"   unit chain: {results['unit_chain_rel']:.2e}")
    _talk(f"  kappa_BTE       diag = {np.diagonal(kappa_bte).round(4)}")
    if run.kappa_ref is not None:
        results["kappa_TDEP_ref"] = run.kappa_ref.tolist()
        _talk(f"  TDEP reference  diag = {np.diagonal(run.kappa_ref).round(4)}"
              f"   [{run.kappa_ref_channel}]")
    _talk(f"  kappa_QHGK      diag = {np.diagonal(kappa_qhgk).round(4)}")
    _talk(f"  kappa_coherence diag = {np.diagonal(kappa_coherence).round(4)}"
          f"   ({results['coherent_share']:.1%} of kappa_QHGK)")

    if check_symmetry:
        _report_symmetry(results, run, kappa_bte, kappa_qhgk, kappa_coherence)

    return results, dmx


def _report_symmetry(results, run, kappa_bte, kappa_qhgk, kappa_coherence):
    """Check if kappa equal its own symmetrized form.
    """
    channels = (("BTE", kappa_bte),
                ("QHGK", kappa_qhgk),
                ("coherence", kappa_coherence))

    for label, K in channels:
        residual = np.abs(K - symmetrize_kappa(K, run.primitive)).max()
        violation = float(residual / max(np.abs(K).max(), 1e-30))
        results[f"symmetry_violation_{label}"] = violation
        flag = "  <-- VIOLATION" if violation > 1e-3 else ""
        _talk(f"  point-group violation, kappa_{label}: {violation:.2e}{flag}")


def extrapolate(run, dmx, nq_max=20):
    """kappa(1/nq) -> kappa(inf) through the GK pipeline's own routine.
    Use per-mode quantum harmonic heat capacity.
    """
    volume = run.primitive.get_volume()

    # quantum=True, not gkmx's classical default: TDEP populates modes with
    # Bose-Einstein statistics and computed its lifetimes that way, so a classical
    # cv here would not reproduce its kappa.
    def heat_capacity(solution):
        return get_harmonic_cv(solution.w_qs, run.temperature,
                               quantum=True) / (volume * run.nq)

    data = get_interpolation_data(dmx, run.tau_fs, heat_capacity, nq_max=nq_max,
                                  quasi_harmonic_greenkubo=True)
    if not data:
        return {"error": "interpolation failed; q-sampling too sparse"}

    def value(key):
        """Some entries come back bare, some as ``(dims, array)`` for xarray."""
        entry = data[key]
        if isinstance(entry, tuple):
            entry = entry[1]
        return np.asarray(entry)

    return {
        "kappa_ha_QHGK": value(keys.kappa_ha_QHGK).tolist(),
        "kappa_sweep_BTE": value(keys.interpolation_kappa_array).tolist(),
        "kappa_sweep_QHGK": value(keys.interpolation_kappa_array_QHGK).tolist(),
        "correction_ab": value(keys.interpolation_correction_ab).tolist(),
        "correction_factor": float(value(keys.interpolation_correction_factor)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--convention", choices=CONVENTIONS, default="TDEP")
    parser.add_argument("--grid-file", default=None,
                        help="override the grid hdf5 (sweeps ship several)")
    parser.add_argument("--extrapolate", action="store_true")
    parser.add_argument("--nq-max", type=int, default=20,
                        help="v_ss' is O(nq^3 ns^2); lower it for many-atom cells")
    parser.add_argument("--check-symmetry", action="store_true")
    parser.add_argument("-o", "--out", default=None)
    args = parser.parse_args(argv)

    if _masses.get_default() != "TDEP":
        _warn(f"GKMX_MASSES is {_masses.get_default()!r}, not 'TDEP'; frequency "
              "agreement will floor at ~1e-5 relative from the mass table alone")

    results = []
    for run_dir in args.run_dirs:
        try:
            run = load(run_dir, grid_file=args.grid_file)
            result, dmx = kappa(run, args.convention,
                                check_symmetry=args.check_symmetry)
            if args.extrapolate:
                result["extrapolation"] = extrapolate(run, dmx, args.nq_max)
        except Exception as exc:  # one bad run must not kill a sweep
            import traceback
            traceback.print_exc()
            result = {"name": Path(run_dir).name,
                      "error": f"{type(exc).__name__}: {exc}"}
        results.append(result)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))
        _talk(f"wrote {args.out}")

    return results


if __name__ == "__main__":
    sys.exit(1 if any("error" in r for r in main()) else 0)
