"""Tests for ``gkmx.Phonon`` (standalone phonon backend)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from ase import io as ase_io

from gkmx.io import parse_force_constants
from gkmx.lattice_points import get_commensurate_q_points
from gkmx.phonon import Phonon, SolutionWithGVM

from .._tolerances import TOL_FP64

DATA_DIR = Path(__file__).parent.parent / "datasets" / "KI_B2_MLIP"


@pytest.fixture(scope="module")
def setup():
    """FC + primitive/supercell + commensurate q-grid + reference DMX."""
    prim = ase_io.read(str(DATA_DIR / "geometry.in.primitive"), format="aims")
    sc = ase_io.read(str(DATA_DIR / "geometry.in.supercell"), format="aims")
    fc = np.asarray(parse_force_constants(
        str(DATA_DIR / "FORCE_CONSTANTS_tdep"), two_dim=False))
    q = np.asarray(get_commensurate_q_points(prim.cell, sc.cell, fractional=True))
    ref = xr.open_dataset(str(DATA_DIR / "DynamicalMatrix.nc"),
                          engine="h5netcdf").load()
    return prim, sc, fc, q, ref


def test_solve_shapes_invariants_and_reference(setup):
    """One shot through the default solver path:
      - returns a SolutionWithGVM when ``with_group_velocity_matrices=True``
      - every output field has the documented q-first shape
      - D(q) is Hermitian to fp64 (the explicit Hermitization works)
      - eigenvectors are orthonormal to fp64
      - sorted-per-q frequencies match the checked-in reference
      - v_qsa is real."""
    prim, sc, fc, q, ref = setup
    Nq, Ns = len(q), 3 * len(prim)

    sol = Phonon(fc, prim, sc).solve(q, with_group_velocity_matrices=True)
    assert isinstance(sol, SolutionWithGVM)
    assert sol.w_qs.shape == (Nq, Ns)
    assert sol.w_inv_qs.shape == (Nq, Ns)
    assert sol.w2_qs.shape == (Nq, Ns)
    assert sol.v_qsa_cartesian.shape == (Nq, Ns, 3)
    assert sol.v_qssa_cartesian.shape == (Nq, Ns, Ns, 3)
    assert sol.e_qsi.shape == (Nq, Ns, Ns)
    assert sol.D_qij.shape == (Nq, Ns, Ns)
    assert np.isrealobj(sol.v_qsa_cartesian)

    D = sol.D_qij
    asym = float(np.abs(D - np.conj(np.swapaxes(D, -1, -2))).max())
    assert asym / max(float(np.abs(D).max()), 1e-30) < TOL_FP64

    e = sol.e_qsi
    overlap = np.einsum("qsi,qti->qst", np.conj(e), e)
    assert np.abs(overlap - np.eye(Ns)[None]).max() < TOL_FP64

    q_ref = np.asarray(ref["q_points"].data)
    sol_at_ref = Phonon(fc, prim, sc).solve(q_ref, with_velocities=False)
    w_gkmx = np.sort(np.abs(np.asarray(sol_at_ref.w_qs)), axis=-1)
    w_ref = np.sort(np.abs(np.asarray(ref["w_qs"].data)), axis=-1)
    assert w_gkmx.shape == w_ref.shape
    scale = max(float(np.abs(w_ref).max()), 1e-30)
    assert np.abs(w_gkmx - w_ref).max() / scale < TOL_FP64


def test_solve_fp32_preserves_dtype(setup):
    """``precision="fp32"`` must propagate to every Solution field."""
    prim, sc, fc, q, _ = setup
    sol = Phonon(fc, prim, sc, precision="fp32").solve(q)
    assert sol.w_qs.dtype == np.float32
    assert sol.w_inv_qs.dtype == np.float32
    assert sol.w2_qs.dtype == np.float32
    assert sol.v_qsa_cartesian.dtype == np.float32
    assert sol.e_qsi.dtype == np.complex64
    assert sol.D_qij.dtype == np.complex64


def test_solve_without_velocities(setup):
    """``with_velocities=False`` short-circuits the dD/dq construction."""
    prim, sc, fc, q, _ = setup
    sol = Phonon(fc, prim, sc).solve(q, with_velocities=False)
    assert sol.v_qsa_cartesian is None


def test_factor_scales_freqs_by_sqrt(setup):
    """D is linear in FC; eigenvalues scale linearly with ``factor``;
    ``w = sqrt(|ev|)`` therefore scales as ``sqrt(factor)``. The
    ``factor=1`` default must be bit-exact with no kwarg."""
    prim, sc, fc, q, _ = setup
    sol_default = Phonon(fc, prim, sc).solve(q)
    sol_one = Phonon(fc, prim, sc, factor=1.0).solve(q)
    np.testing.assert_array_equal(sol_default.w_qs, sol_one.w_qs)

    sol_k = Phonon(fc, prim, sc, factor=4.0).solve(q)
    mask = np.abs(sol_default.w_qs) > 1e-6
    np.testing.assert_allclose(
        sol_k.w_qs[mask] / sol_default.w_qs[mask],
        np.sqrt(4.0), rtol=TOL_FP64, atol=0,
    )


def test_unknown_backend_and_bad_p2s_map_raise(setup):
    """Two input-validation paths in ``Phonon.__init__``."""
    prim, sc, fc, _, _ = setup
    with pytest.raises(ValueError, match="Unknown backend"):
        Phonon(fc, prim, sc, backend="cuda")
    with pytest.raises(ValueError, match="p2s_map must have shape"):
        Phonon(fc, prim, sc, p2s_map=np.array([0, 1, 2]))


def test_p2s_map_is_noop_for_tdep(setup):
    """For TDEP/FHI-aims inputs prim and sc atoms already align with
    the implicit ``p2s_map``; passing the explicit map must be a no-op
    on the frequencies."""
    from gkmx.lattice_points import get_s2p_map
    prim, sc, fc, q, _ = setup
    s2p = np.asarray(get_s2p_map(prim, sc))
    p2s = np.array([int(np.where(s2p == i)[0][0]) for i in range(len(prim))])
    sol_a = Phonon(fc, prim, sc).solve(q)
    sol_b = Phonon(fc, prim, sc, p2s_map=p2s).solve(q)
    np.testing.assert_allclose(sol_a.w_qs, sol_b.w_qs, rtol=TOL_FP64, atol=0)
