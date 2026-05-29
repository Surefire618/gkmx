"""Strict side-by-side validation of ``gkmx.Phonon`` against a live
phonopy import on diamond Si (Fd-3m, ``N_p=2``, ``N_sc=16``).

phonopy is imported at test time (``pytest.importorskip``) so failures
can be diagnosed without regenerating the static phonopy reference
fixtures used in ``test_phonopy_reference.py``.

Three grids stress different code paths:
  - ``commensurate_2x2x2``: 8 high-symmetry points (every point either
    degenerate or on a BZ face).
  - ``mp_4x4x4``: 64-point Monkhorst-Pack (mix of interior/boundary,
    degenerate/non-degenerate).
  - ``random_32``: 32 random points, almost surely interior and
    non-degenerate (no BZ-face / no exact eigenvalue collisions).

A q-point is **interior** iff no non-zero primitive reciprocal-lattice
vector ``G`` satisfies ``|q − G| = |q|`` (i.e. q is not on a BZ face).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

phonopy = pytest.importorskip("phonopy")

from gkmx import _constants as C
from gkmx.phonon import Phonon

from .._tolerances import TOL_FP64, TOL_FP64_DERIVED

_DEFAULT_PHONOPY_TEST_DIR = Path(__file__).resolve().parents[3] / "phonopy" / "test"
PHONOPY_TEST_DIR = Path(os.environ.get("GKMX_PHONOPY_TEST_DIR",
                                       str(_DEFAULT_PHONOPY_TEST_DIR)))

DEGEN_TOL_THZ = 1e-6
# Boundary V_ab carries a touch more rounding (projector + gauge
# averaging in `_perturb_D`); 1e-9 is the empirical Si ceiling, 3
# orders tighter than TOL_FP64_DERIVED.
TOL_BOUNDARY_V_AB = 1e-9


def _grids():
    axis2 = np.array([0.0, 0.5])
    commensurate = np.array(np.meshgrid(axis2, axis2, axis2,
                                         indexing="ij")).reshape(3, -1).T
    axis4 = (np.arange(4) - 2) / 4
    mp4 = np.array(np.meshgrid(axis4, axis4, axis4,
                                indexing="ij")).reshape(3, -1).T
    random32 = np.random.default_rng(42).uniform(-0.5, 0.5, size=(32, 3))
    return {"commensurate_2x2x2": commensurate,
            "mp_4x4x4": mp4,
            "random_32": random32}


GRIDS = _grids()


@pytest.fixture(scope="module")
def si():
    yaml_path = PHONOPY_TEST_DIR / "phonopy_params_Si.yaml"
    if not yaml_path.exists():
        pytest.skip(f"phonopy Si yaml not found at {yaml_path}")
    ph = phonopy.load(str(yaml_path), is_nac=False, symmetrize_fc=False)
    # Build ASE Atoms via cartesian positions (`scaled_positions=` triggers
    # an ASE-internal `np.array(..., copy=False)` deprecation on numpy 2.x).
    prim = Atoms(symbols=ph.primitive.symbols,
                 positions=np.asarray(ph.primitive.positions),
                 cell=np.asarray(ph.primitive.cell),
                 masses=ph.primitive.masses, pbc=True)
    sc = Atoms(symbols=ph.supercell.symbols,
               positions=np.asarray(ph.supercell.positions),
               cell=np.asarray(ph.supercell.cell),
               masses=ph.supercell.masses, pbc=True)
    fc = np.asarray(ph.force_constants, dtype=np.float64)
    if fc.shape[0] == fc.shape[1]:
        fc = fc[ph.primitive.p2s_map]
    factor = (float(ph.unit_conversion_factor) / C.omega_to_THz) ** 2

    recip = 2 * np.pi * np.linalg.inv(prim.cell.array.T)
    g_int = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1)
                       for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)])
    return {"ph": ph, "prim": prim, "sc": sc, "fc": fc, "factor": factor,
            "recip": recip, "g_cart": g_int @ recip}


def _is_boundary(q_frac, recip, g_cart, atol=1e-9):
    """True if q is on a BZ face: ∃ g ≠ 0 with |q − g| = |q|."""
    q_cart = q_frac @ recip
    out = np.zeros(len(q_cart), dtype=bool)
    for i, qc in enumerate(q_cart):
        out[i] = bool(np.any(np.abs(
            np.linalg.norm(qc - g_cart, axis=1) - np.linalg.norm(qc)
        ) < atol))
    return out


def _run_phonopy(ph, q):
    ph.run_qpoints(q, with_eigenvectors=True, with_group_velocities=True)
    qd = ph.get_qpoints_dict()
    return (np.asarray(qd["frequencies"]),
            np.asarray(qd["eigenvectors"]),
            np.asarray(qd["group_velocities"]))


def _run_gkmx(si, q):
    sol = Phonon(si["fc"], si["prim"], si["sc"], backend="numpy",
                 factor=si["factor"]).solve(q, with_velocities=True)
    return (
        np.asarray(sol.w_qs) * C.omega_to_THz,                     # f (Nq, Ns)
        np.moveaxis(np.asarray(sol.e_qsi), -1, -2),                # e column layout (Nq, Ns, Ns)
        np.asarray(sol.v_qsa_cartesian) * 1e3,                     # v (Nq, Ns, 3) THz·Å
    )


@pytest.mark.parametrize("grid_name", list(GRIDS))
def test_frequencies_match_phonopy_on_every_q(si, grid_name):
    """gkmx and phonopy diagonalize the same D(q) — sorted frequencies
    must agree at fp64 on every q-point, regardless of grid."""
    q = GRIDS[grid_name]
    f_ph, _, _ = _run_phonopy(si["ph"], q)
    f_gx, _, _ = _run_gkmx(si, q)
    max_abs = float(np.abs(np.sort(f_ph, axis=-1)
                           - np.sort(f_gx, axis=-1)).max())
    scale = max(float(np.abs(f_ph).max()), 1e-12)
    assert max_abs / scale < TOL_FP64


@pytest.mark.parametrize("grid_name", ["mp_4x4x4", "random_32"])
def test_eigenvectors_velocities_and_V_ab_match_interior_nondegenerate(si, grid_name):
    """At interior, fully non-degenerate q-points (no BZ face, no
    eigenvalue collisions), the eigenvector basis is unique up to an
    overall phase per band. All three quantities must therefore match
    phonopy at fp64:
      - per-mode eigenvector overlap is diagonal with unit magnitude
      - per-mode ``v_sa`` matches element-wise
      - ``V_ab(q) = Σ_s v_sa·v_sb`` matches (a redundant check that
        catches a regression which corrupted v_sa in a phase-cancelling
        way).
    """
    q = GRIDS[grid_name]
    bdry = _is_boundary(q, si["recip"], si["g_cart"])
    f_ph, e_ph, v_ph = _run_phonopy(si["ph"], q)
    _, e_gx, v_gx = _run_gkmx(si, q)

    eligible = [qi for qi in range(len(q))
                if (not bdry[qi])
                and np.all(np.diff(np.sort(f_ph[qi])) > DEGEN_TOL_THZ)]
    assert eligible, f"{grid_name}: no interior non-degenerate q-points"

    max_diag_dev, max_off, max_dv_rel = 0.0, 0.0, 0.0
    for qi in eligible:
        M = np.abs(e_ph[qi].conj().T @ e_gx[qi])
        max_diag_dev = max(max_diag_dev, float(np.abs(np.diag(M) - 1.0).max()))
        off = M - np.diag(np.diag(M))
        max_off = max(max_off, float(off.max()))
        scale = max(float(np.abs(v_ph[qi]).max()), 1e-12)
        max_dv_rel = max(max_dv_rel,
                         float(np.abs(v_ph[qi] - v_gx[qi]).max()) / scale)
    assert max_diag_dev < TOL_FP64_DERIVED
    assert max_off < TOL_FP64_DERIVED
    assert max_dv_rel < TOL_FP64

    V_ph = np.einsum("qsa,qsb->qab", v_ph[eligible], v_ph[eligible])
    V_gx = np.einsum("qsa,qsb->qab", v_gx[eligible], v_gx[eligible])
    scale = max(float(np.abs(V_ph).max()), 1e-30)
    assert float(np.abs(V_ph - V_gx).max()) / scale < TOL_FP64


@pytest.mark.parametrize("grid_name", ["commensurate_2x2x2", "mp_4x4x4"])
def test_degenerate_subspace_projectors_match_interior(si, grid_name):
    """At interior degenerate q-points, phonopy applies ``_perturb_D``
    while gkmx uses a different basis. Per-mode comparison is then
    gauge-dependent; the projector ``P = Σ_{s∈grp} |e_s⟩⟨e_s|`` is
    gauge-invariant and must still match at fp64-derived precision."""
    q = GRIDS[grid_name]
    bdry = _is_boundary(q, si["recip"], si["g_cart"])
    interior = np.where(~bdry)[0]
    assert len(interior) > 0
    f_ph, e_ph, _ = _run_phonopy(si["ph"], q)
    _, e_gx, _ = _run_gkmx(si, q)

    any_deg, max_dP = False, 0.0
    for qi in interior:
        w = f_ph[qi]
        start = 0
        for si_idx in range(1, len(w) + 1):
            if si_idx == len(w) or abs(w[si_idx] - w[start]) > DEGEN_TOL_THZ:
                if si_idx - start > 1:
                    any_deg = True
                    P_ph = e_ph[qi, :, start:si_idx] @ e_ph[qi, :, start:si_idx].conj().T
                    P_gx = e_gx[qi, :, start:si_idx] @ e_gx[qi, :, start:si_idx].conj().T
                    max_dP = max(max_dP, float(np.abs(P_ph - P_gx).max()))
                start = si_idx
    assert any_deg, f"{grid_name}: no interior degenerate q-points"
    assert max_dP < TOL_FP64_DERIVED


@pytest.mark.parametrize("grid_name", ["commensurate_2x2x2", "mp_4x4x4"])
def test_boundary_V_ab_matches_phonopy(si, grid_name):
    """At BZ-boundary q-points, per-band ``v_sa`` can differ between
    phonopy and gkmx on degenerate modes (gauge). The
    rotation-invariant ``V_ab`` must still match — this is what the
    2026-04-16 ``_perturb_D`` / ``_symmetrize_group_velocity`` port
    fixed; a regression there would reintroduce the cubic-symmetry
    breaking at X points (the historical 39.4/36.0/20.6 instead of
    uniform 42.7 on Si X)."""
    q = GRIDS[grid_name]
    bdry = _is_boundary(q, si["recip"], si["g_cart"])
    assert bdry.any()
    _, _, v_ph = _run_phonopy(si["ph"], q[bdry])
    _, _, v_gx = _run_gkmx(si, q[bdry])
    V_ph = np.einsum("qsa,qsb->qab", v_ph, v_ph)
    V_gx = np.einsum("qsa,qsb->qab", v_gx, v_gx)
    max_abs = float(np.abs(V_ph - V_gx).max())
    assert max_abs < TOL_BOUNDARY_V_AB, (
        f"{grid_name}: max boundary |ΔV_ab| = {max_abs:.3e}"
    )
