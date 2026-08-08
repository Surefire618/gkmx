"""Tests for ``gkmx.DynamicalMatrix``."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from ase import io as ase_io

from gkmx.dynamical_matrix import DynamicalMatrix
from gkmx.io import parse_force_constants
from gkmx.phonon import _symmetrize_v_site

from .._tolerances import TOL_FP64, TOL_FP64_DERIVED

DATA_DIR = Path(__file__).parent.parent / "datasets" / "KI_B2_MLIP"


@pytest.fixture(scope="module")
def setup():
    """Primitive, supercell, FC, and the reference DynamicalMatrix.nc."""
    prim = ase_io.read(str(DATA_DIR / "geometry.in.primitive"), format="aims")
    sc = ase_io.read(str(DATA_DIR / "geometry.in.supercell"), format="aims")
    fc = np.asarray(parse_force_constants(
        str(DATA_DIR / "FORCE_CONSTANTS_tdep"), two_dim=False))
    ref = xr.open_dataset(str(DATA_DIR / "DynamicalMatrix.nc"),
                          engine="h5netcdf").load()
    return prim, sc, fc, ref


@pytest.fixture(scope="module")
def dmx(setup):
    prim, sc, fc, _ = setup
    # the checked-in DynamicalMatrix.nc reference predates ASR enforcement
    return DynamicalMatrix(force_constants=fc, primitive=prim, supercell=sc,
                           enforce_translational_invariance=False,
                           with_group_velocity_matrices=True)


def test_shapes_invariants_and_reference(dmx, setup):
    """Single end-to-end check on the numpy backend:
        - every cached array has the documented q-first shape
        - remapped FC is symmetric (constructed via 0.5*(M + M.T))
        - sorted frequencies match the cached reference at fp64
        - sorted Σ|v|² per q matches the cached reference
        - spectral resolution: ``Σ_qs w² |e⟩⟨e| = D``"""
    prim, sc, _, ref = setup
    Np, Na, Ns = len(prim), len(sc), 3 * len(prim)
    Nq = len(dmx.q_points)

    assert dmx.fc_phonopy.shape == (Np, Na, 3, 3)
    assert dmx.w_qs.shape == (Nq, Ns)
    assert dmx.w_inv_qs.shape == (Nq, Ns)
    assert dmx.v_qsa_cartesian.shape == (Nq, Ns, 3)
    assert dmx.e_qsI.shape == (Nq, Ns, 3 * Na)
    assert dmx.remapped.shape == (3 * Na, 3 * Na)
    assert dmx.solution.v_qssa_cartesian.shape == (Nq, Ns, Ns, 3)

    R = dmx.remapped
    assert np.abs(R - R.T).max() / max(np.abs(R).max(), 1e-30) < TOL_FP64

    w_ref = np.sort(np.asarray(ref["w_qs"].data).flatten())
    assert np.abs(np.sort(dmx.w_qs.flatten()) - w_ref).max() \
        / max(np.abs(w_ref).max(), 1e-30) < TOL_FP64

    v2_ref = np.sort((np.asarray(ref["v_qsa"].data) ** 2).sum(axis=(1, 2)))
    v2_new = np.sort((dmx.v_qsa_cartesian ** 2).sum(axis=(1, 2)))
    assert np.abs(v2_new - v2_ref).max() \
        / max(np.abs(v2_ref).max(), 1e-30) < TOL_FP64_DERIVED

    nI = dmx.e_qsI.shape[-1]
    weighted = (dmx.e_qsI * dmx.w2_qs[:, :, None]).reshape(-1, nI)
    eH = dmx.e_qsI.reshape(-1, nI).conj()
    D_recon = (weighted.T @ eH).real
    M = np.repeat(sc.get_masses(), 3)
    D_fc = dmx.remapped * M[:, None] ** -0.5 * M[None, :] ** -0.5
    assert np.abs(D_recon - D_fc).max() \
        / max(np.abs(D_fc).max(), 1e-30) < TOL_FP64_DERIVED


def test_hdf5_round_trip(dmx, tmp_path):
    """Save then load: w_qs and v magnitudes match at fp64; the lazy
    remapped FC is rebuilt identically post-load."""
    path = tmp_path / "dmx.nc"
    dmx.to_hdf5(str(path), include_D_qij=True,
                include_group_velocity_matrices=True)
    other = DynamicalMatrix.from_hdf5(str(path))

    np.testing.assert_allclose(
        np.sort(dmx.w_qs.flatten()), np.sort(other.w_qs.flatten()),
        rtol=TOL_FP64, atol=0)
    np.testing.assert_allclose(
        np.sort(np.linalg.norm(dmx.v_qsa_cartesian, axis=-1).flatten()),
        np.sort(np.linalg.norm(other.v_qsa_cartesian, axis=-1).flatten()),
        rtol=TOL_FP64, atol=0)
    rel = np.abs(dmx.remapped - other.remapped).max() \
        / max(np.abs(dmx.remapped).max(), 1e-30)
    assert rel < TOL_FP64


def test_get_mesh_and_solution(dmx):
    """``get_mesh_and_solution`` returns a (q_grid, Solution) pair on a
    user-requested BZ mesh, sized correctly for the reduced wedge."""
    grid, sol = dmx.get_mesh_and_solution([4, 4, 4], reduced=True)
    Ns = dmx.w_qs.shape[1]
    Nq = len(grid.points)
    assert sol.w_qs.shape == (Nq, Ns)
    assert sol.v_qsa_cartesian.shape == (Nq, Ns, 3)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_jax_and_numpy_agree_with_reference(backend, setup):
    """Both backends diagonalize the same D(q); sorted frequencies and
    sorted Σ|v|² match the cached reference (which was baked with the
    numpy backend) at fp64. Cross-validates the JAX kernel against the
    numpy one without a separate per-q comparison (q-grid ordering is
    free)."""
    prim, sc, fc, ref = setup
    d = DynamicalMatrix(force_constants=fc, primitive=prim, supercell=sc,
                        enforce_translational_invariance=False,
                        with_group_velocity_matrices=True, backend=backend)
    w_ref = np.sort(np.asarray(ref["w_qs"].data).flatten())
    assert np.abs(np.sort(d.w_qs.flatten()) - w_ref).max() \
        / max(np.abs(w_ref).max(), 1e-30) < TOL_FP64

    v2_ref = np.sort((np.asarray(ref["v_qsa"].data) ** 2).sum(axis=(1, 2)))
    v2 = np.sort((d.v_qsa_cartesian ** 2).sum(axis=(1, 2)))
    assert np.abs(v2 - v2_ref).max() \
        / max(np.abs(v2_ref).max(), 1e-30) < TOL_FP64_DERIVED

    # `diag(v_qssa)` is no longer `v_qsa`: `v_qsa` carries the site-symmetrization
    # (phonopy-canonical, and correct for the diagonal) while `v_qssa` does not,
    # because the Cartesian-only average sets Gamma(R, q) = 1 and corrupts the
    # velocity operator off the diagonal. The two are still tied exactly -- one
    # is the site average of the other -- which pins both paths at once.
    diag = np.einsum("qssa->qsa", d.solution.v_qssa_cartesian).real
    recip = np.linalg.inv(np.asarray(prim.cell))
    sym = _symmetrize_v_site(diag, d.q_points, d._phonon._symm_rots_frac, recip)
    live = np.abs(d.v_qsa_cartesian).sum(axis=-1) > 0    # v_qsa zeroes sub-threshold modes
    assert np.abs(sym[live] - np.asarray(d.v_qsa_cartesian)[live]).max() \
        / max(np.abs(d.v_qsa_cartesian).max(), 1e-30) < TOL_FP64
