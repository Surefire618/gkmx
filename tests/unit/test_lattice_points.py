"""Tests for ``gkmx.lattice_points`` and ``gkmx.mic``."""

from pathlib import Path

import numpy as np
import pytest
from ase import io as ase_io

from gkmx.lattice_points import (
    get_commensurate_q_points,
    get_lattice_points,
    get_unit_grid_extended,
    map_I_to_iL,
)
from gkmx.mic import fold as mic_fold
from gkmx.mic import _IMAGE_SHIFTS, is_orthogonal

from .._tolerances import TOL_FP64

DATASETS = Path(__file__).parent.parent / "datasets"


def _read(name):
    d = DATASETS / name
    return (ase_io.read(str(d / "geometry.in.primitive"), format="aims"),
            ase_io.read(str(d / "geometry.in.supercell"), format="aims"))


@pytest.mark.parametrize("material", ["KI_B2_MLIP", "GaN", "Ga2O3"])
def test_lattice_points_and_commensurate_q(material):
    """For each crystal class (cubic / hex / monoclinic):
        - ``|lattice_points|`` matches ``|det(supercell_matrix)|``
        - sorted lattice points start at the origin
        - all interior points sit in supercell-frac ``[-0.5, 0.5)``
        - ``|commensurate q|`` equals ``N_sc / N_p``
        - Γ is in the q-grid
        - ``exp(2πi q·L) = 1`` for every q on the supercell lattice."""
    prim, sc = _read(material)
    lps = get_lattice_points(prim.cell, sc.cell, extended=False, sort=True)
    M = np.rint(
        np.linalg.solve(np.asarray(prim.cell).T, np.asarray(sc.cell).T).T
    ).astype(int)
    assert len(lps) == int(round(abs(np.linalg.det(M))))
    assert np.linalg.norm(lps[0]) < 1e-10

    from ase.cell import Cell
    frac = Cell(sc.cell).scaled_positions(lps)
    assert np.all(frac > -0.5 - 1e-5) and np.all(frac < 0.5 - 1e-5)

    q = get_commensurate_q_points(prim.cell, sc.cell, fractional=True)
    assert len(q) == len(sc) // len(prim)
    assert np.min(np.linalg.norm(q, axis=1)) < 1e-10
    qM = q @ M.T
    np.testing.assert_allclose(qM, np.round(qM), atol=TOL_FP64)


def test_map_I_to_iL_round_trip_and_species():
    """``I2iL`` covers every supercell atom; the inverse map round-trips;
    every primitive atom is mapped exactly ``N_sc/N_p`` times; and the
    chemical species at each I matches the primitive atom it points to."""
    prim, sc = _read("KI_B2_MLIP")
    I2iL, iL2I = map_I_to_iL(prim, sc)
    Np, Nsc = len(prim), len(sc)
    assert I2iL.shape == (Nsc, 2)
    assert np.all((I2iL[:, 0] >= 0) & (I2iL[:, 0] < Np))
    _, counts = np.unique(I2iL[:, 0], return_counts=True)
    assert np.all(counts == Nsc // Np)
    # round-trip
    for II in range(Nsc):
        i, L = I2iL[II]
        assert iL2I[i, L] == II
    # species preservation
    prim_syms = prim.get_chemical_symbols()
    sc_syms = sc.get_chemical_symbols()
    for II in range(Nsc):
        assert sc_syms[II] == prim_syms[I2iL[II, 0]]


def test_unit_grid_extended():
    """Wraps to ``[0, 1)`` and produces at least as many points as input."""
    prim, sc = _read("KI_B2_MLIP")
    q = get_commensurate_q_points(prim.cell, sc.cell, fractional=True)
    r = get_unit_grid_extended(q)
    assert np.all(r.points >= -1e-9) and np.all(r.points < 1 + 1e-9)
    assert len(r.points_extended) >= len(q)
    assert len(r.map2extended) == len(r.points_extended)
    assert np.all(r.map2extended < len(q))


# ---------------------------------------------------------------------------
# MIC fold
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cells():
    return {
        "cubic": np.eye(3) * 4.50,
        "ortho": np.diag([3.0, 5.0, 7.0]),
        "hex":   _read("GaN")[1].cell.array,
        "mono":  _read("Ga2O3")[1].cell.array,
    }


@pytest.mark.parametrize("name", ["cubic", "ortho", "hex", "mono"])
def test_mic_fold_matches_ase(cells, name):
    """The MIC fold must agree with ASE's reference on every cell class.
    Uses random fractional disps in ``[-1.5, 1.5]`` to exercise wrap-around."""
    from ase.geometry import find_mic
    cell = cells[name]
    rng = np.random.default_rng(int.from_bytes(name.encode(), "little"))
    disp = (rng.uniform(-1.5, 1.5, size=(2000, 3)) @ cell)
    ref = find_mic(disp, np.asarray(cell))[0]
    out = mic_fold(disp, cell)
    np.testing.assert_allclose(out, ref, atol=1e-10, rtol=0)


def test_mic_fold_dtype_and_zero(cells):
    """fp32 in → fp32 out; zero in → exact zero out."""
    cell = cells["cubic"].astype(np.float32)
    disp = np.zeros((4, 8, 3), dtype=np.float32)
    out = mic_fold(disp, cell)
    assert out.shape == disp.shape and out.dtype == np.float32
    np.testing.assert_allclose(out, 0.0, atol=1e-14)


def test_mic_two_tier_matches_dense_search(cells):
    """The numpy two-tier path must equal the dense 27-image search on
    slanted cells where some rows enter the search tier (defect hops)
    and others stay in the safe fast tier."""
    def dense(disp, cell):
        inv = np.linalg.inv(cell)
        r = disp @ inv - np.round(disp @ inv)
        shifts = _IMAGE_SHIFTS.astype(disp.dtype)
        cand = (r[..., None, :] - shifts) @ cell
        best = (cand * cand).sum(-1).argmin(-1)
        return (r - shifts[best]) @ cell

    rng = np.random.default_rng(123)
    for name in ("hex", "mono"):
        cell = np.asarray(cells[name])
        disp = np.concatenate([
            rng.uniform(-0.05, 0.05, size=(1500, 3)) @ cell,  # safe tier
            rng.uniform(-1.5,  1.5,  size=(500, 3))  @ cell,  # hop tier
        ])
        np.testing.assert_array_equal(
            mic_fold(disp, cell, search=True), dense(disp, cell)
        )


def test_is_orthogonal(cells):
    """Cubic / ortho yes; hex / mono no; near-cubic with float noise yes;
    1°-skewed cubic no."""
    assert is_orthogonal(cells["cubic"])
    assert is_orthogonal(cells["ortho"])
    assert not is_orthogonal(cells["hex"])
    assert not is_orthogonal(cells["mono"])

    rng = np.random.default_rng(0)
    near_cubic = cells["cubic"] + rng.standard_normal((3, 3)) * 1e-9
    assert is_orthogonal(near_cubic)

    skewed = cells["cubic"].copy()
    skewed[1, 0] = cells["cubic"][0, 0] * 0.0175  # ~1° shear
    assert not is_orthogonal(skewed)
