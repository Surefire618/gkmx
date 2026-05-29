"""Tests for ``gkmx.brillouin``."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from ase import io as ase_io

from gkmx.brillouin import get_bz_mesh, get_q_grid, get_symmetrized_array
from gkmx.lattice_points import get_commensurate_q_points

from .._tolerances import TOL_FP64

DATA_DIR = Path(__file__).parent.parent / "datasets" / "KI_B2_MLIP"


@pytest.fixture(scope="module")
def q_grid():
    prim = ase_io.read(str(DATA_DIR / "geometry.in.primitive"), format="aims")
    sc = ase_io.read(str(DATA_DIR / "geometry.in.supercell"), format="aims")
    q_frac = get_commensurate_q_points(prim.cell, sc.cell, fractional=True)
    return get_q_grid(q_frac, primitive=prim), prim, sc, q_frac


def test_get_q_grid_pm3m_invariants(q_grid):
    """Cubic Pm-3m KI on a 4×4×4 grid: spglib detects the 48-op point
    group and folds 64 q-points down to ≤ 16 (empirical wedge is 8).
    Pins the irreducible-reduction factor, the canonical map2full,
    weight conservation, and the q ↔ q_cart round-trip in one place."""
    grid, prim, _, q_frac = q_grid
    n_ir = len(grid.ir.points)
    assert len(grid.points) == len(q_frac)
    assert n_ir <= 16, f"Pm-3m reduced to {n_ir} > 16 — spglib regression"
    assert grid.ir.weights.sum() == len(grid.points)
    # map2full is canonical: np.unique-inverse recovers it.
    _, inv = np.unique(grid.ir.map2full, return_inverse=True)
    assert np.array_equal(inv, grid.ir.map2full)
    # q ↔ Cartesian round-trip.
    q_cart = prim.cell.reciprocal().cartesian_positions(grid.points)
    np.testing.assert_allclose(q_cart, grid.points_cartesian, rtol=TOL_FP64, atol=0)


def test_get_bz_mesh_full_and_reduced():
    """Full mesh has nq³ points in [-0.5, 0.5); the reduced wedge has
    fewer points and its weights re-sum to the full mesh count."""
    prim = ase_io.read(str(DATA_DIR / "geometry.in.primitive"), format="aims")
    full = get_bz_mesh([4, 4, 4], atoms=prim, reduced=False)
    ir = get_bz_mesh([4, 4, 4], atoms=prim, reduced=True)
    assert len(full.points) == 64
    assert np.all(full.points >= -0.5 - 1e-9)
    assert np.all(full.points < 0.5 + 1e-9)
    assert len(ir.points) < 64
    assert ir.weights.sum() == 64


def test_get_symmetrized_array_collapses_equivalent_qs(q_grid):
    """Symmetrization must (a) make every star of equivalent q-points
    carry the same value, (b) preserve the mean (the linear-average
    contract), (c) leave constants unchanged."""
    grid, *_ = q_grid
    rng = np.random.RandomState(42)
    Nq = len(grid.points)
    arr = xr.DataArray(rng.randn(6, Nq), dims=("s", "q"), name="test")
    sym = get_symmetrized_array(arr, grid.map2ir, grid.ir.map2full)
    sym_data = np.asarray(sym)

    for ir_idx in range(len(grid.ir.points)):
        mask = grid.ir.map2full == ir_idx
        if mask.sum() > 1:
            stars = sym_data[:, mask]
            assert np.allclose(stars - stars[:, :1], 0)

    np.testing.assert_allclose(
        np.asarray(arr).mean(axis=1), sym_data.mean(axis=1),
        rtol=TOL_FP64, atol=0,
    )

    constant = xr.DataArray(np.full((6, Nq), 3.14), dims=("s", "q"), name="c")
    assert np.allclose(
        get_symmetrized_array(constant, grid.map2ir, grid.ir.map2full),
        constant,
    )
