"""Tests for ``gkmx.lattice_points`` and ``gkmx.mic``."""

from pathlib import Path

import numpy as np
import pytest
from ase import io as ase_io

from gkmx.lattice_points import (
    get_commensurate_q_points,
    get_lattice_points,
    get_smallest_vectors,
    get_unit_grid_extended,
    map_I_to_iL,
)
from gkmx.mic import _IMAGE_SHIFTS, is_orthogonal
from gkmx.mic import fold as mic_fold

from .._tolerances import TOL_FP64, TOL_FP64_DERIVED

DATASETS = Path(__file__).parent.parent / "datasets"

# alpha is the skewed case: P = [[-3,1,2],[-1,3,-2],[-1,-1,-1]], det 24. Its
# supercell passes mic._is_pairwise_reduced, so a Gauss/Lagrange check does not
# flag it; only a brute-force image search does.
GA2O3_PHASES = ["alpha", "beta", "kappa"]


def _read(name):
    d = DATASETS / name
    return (ase_io.read(str(d / "geometry.in.primitive"), format="aims"),
            ase_io.read(str(d / "geometry.in.supercell"), format="aims"))


def _read_phase(phase):
    d = DATASETS / "Ga2O3"
    return (ase_io.read(str(d / f"geometry.in.primitive.{phase}"), format="aims"),
            ase_io.read(str(d / f"geometry.in.supercell.{phase}"), format="aims"))


def _brute_smallest_vectors(primitive, supercell, radius=3, tol=1e-5, block=40):
    """Reference ``(svec_frac, multi)`` from an exhaustive image search.

    Same contract as ``get_smallest_vectors`` but with no assumption about how
    far the nearest image can be: every shift in ``[-radius, radius]^3`` of the
    supercell lattice is tried. Blocked over the supercell axis to bound peak
    memory at ``block * N_p * (2*radius+1)^3``.
    """
    A_prim = np.asarray(primitive.cell, dtype=np.float64)
    A_sc = np.asarray(supercell.cell, dtype=np.float64)
    r_base = (np.asarray(supercell.positions)[:, None, :]
              - np.asarray(primitive.positions)[None, :, :])

    rng = np.arange(-radius, radius + 1, dtype=np.float64)
    ns = np.stack(np.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)
    shifts = ns @ A_sc

    N_sc, N_p = r_base.shape[:2]
    keep = []
    multi = np.zeros((N_sc, N_p), dtype=np.int32)
    for k0 in range(0, N_sc, block):
        cand = r_base[k0:k0 + block, :, None, :] + shifts[None, None, :, :]
        dist = np.linalg.norm(cand, axis=-1)
        mask = dist < dist.min(axis=-1, keepdims=True) + tol
        multi[k0:k0 + block] = mask.sum(axis=-1)
        keep.append((cand, mask))

    V_max = int(multi.max())
    svec_cart = np.zeros((N_sc, N_p, V_max, 3), dtype=np.float64)
    for b, (cand, mask) in enumerate(keep):
        for k in range(cand.shape[0]):
            for i in range(N_p):
                sel = np.where(mask[k, i])[0]
                svec_cart[b * block + k, i, : len(sel)] = cand[k, i, sel]
    return svec_cart @ np.linalg.inv(A_prim), multi


def _avg_bloch_phase(svec_frac, multi, q_frac):
    """The multi-image-averaged phase ``_solve_kernel`` builds, ``(N_q, N_sc, N_p)``."""
    mask = np.arange(svec_frac.shape[2])[None, None, :] < multi[:, :, None]
    phase = np.exp(2j * np.pi * np.einsum("qa,kiva->qkiv", q_frac, svec_frac))
    phase = phase * mask[None, :, :, :]
    return phase.sum(axis=-1) / np.maximum(multi, 1)[None, :, :]


def _noncommensurate_q(primitive, supercell, n=16, seed=0):
    """Random interior q, none of them commensurate with ``supercell``."""
    P = np.round(np.asarray(supercell.cell)
                 @ np.linalg.inv(np.asarray(primitive.cell))).astype(int)
    q = np.random.default_rng(seed).uniform(-0.5, 0.5, size=(n, 3))
    x = q @ P.T
    assert not np.any(np.all(np.abs(x - np.rint(x)) < 1e-8, axis=1))
    return q


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
    assert len(lps) == round(abs(np.linalg.det(M)))
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


@pytest.mark.parametrize("phase", GA2O3_PHASES)
def test_smallest_vectors_are_minimal(phase):
    """Every returned image must be a genuine nearest image.

    The ``{-1,0,1}^3`` window is only sufficient once the separation has been
    folded into a reduced cell. Without the fold, a pair sitting near the far
    corner of a skewed supercell needs a shift of 2 or more, and the search
    silently returns a longer image: alpha picks 11.767 A where the true
    nearest is 3.715 A."""
    primitive, supercell = _read_phase(phase)
    svec, multi = get_smallest_vectors(primitive, supercell)
    ref_svec, ref_multi = _brute_smallest_vectors(primitive, supercell)

    A_prim = np.asarray(primitive.cell)
    d = np.linalg.norm(svec[:, :, 0, :] @ A_prim, axis=-1)
    d_ref = np.linalg.norm(ref_svec[:, :, 0, :] @ A_prim, axis=-1)

    n_bad = int((d > d_ref + TOL_FP64_DERIVED).sum())
    assert n_bad == 0, (
        f"{phase}: {n_bad} of {d.size} pairs got a non-minimal image; "
        f"worst is {(d - d_ref).max():.4f} A longer than the true nearest")
    np.testing.assert_array_equal(multi, ref_multi)


@pytest.mark.parametrize("phase", GA2O3_PHASES)
def test_smallest_vectors_bloch_phase_at_noncommensurate_q(phase):
    """The averaged Bloch phase entering ``D(q)`` must match the reference.

    This is the physical consequence of the geometry test above: a wrong image
    carries a wrong ``exp(2 pi i q.R)``. It must be checked off the commensurate
    set — see ``test_commensurate_q_cannot_detect_wrong_images``."""
    primitive, supercell = _read_phase(phase)
    q = _noncommensurate_q(primitive, supercell)

    got = _avg_bloch_phase(*get_smallest_vectors(primitive, supercell), q)
    ref = _avg_bloch_phase(*_brute_smallest_vectors(primitive, supercell), q)
    np.testing.assert_allclose(got, ref, atol=TOL_FP64_DERIVED, rtol=0)


def test_commensurate_q_cannot_detect_wrong_images():
    """Positive control for the test above: prove non-commensurate q is required.

    Displace one image by a supercell lattice vector — the exact error mode of
    a too-small search window. At commensurate q, ``q.(R - R')`` is an integer,
    so the phase is unchanged and any check there passes on broken vectors. Off
    the commensurate set the same error is plainly visible."""
    primitive, supercell = _read_phase("alpha")
    svec, multi = _brute_smallest_vectors(primitive, supercell)

    wrong = svec.copy()
    L_sc = np.asarray(supercell.cell)[0] @ np.linalg.inv(np.asarray(primitive.cell))
    wrong[0, 0, 0] += L_sc

    q_comm = get_commensurate_q_points(primitive.cell, supercell.cell, fractional=True)
    np.testing.assert_allclose(
        _avg_bloch_phase(wrong, multi, q_comm),
        _avg_bloch_phase(svec, multi, q_comm),
        atol=TOL_FP64_DERIVED, rtol=0,
        err_msg="commensurate q should be blind to a whole-supercell shift")

    q_free = _noncommensurate_q(primitive, supercell)
    got = _avg_bloch_phase(wrong, multi, q_free)
    ref = _avg_bloch_phase(svec, multi, q_free)
    assert np.abs(got - ref).max() > 0.1, (
        "non-commensurate q failed to see a whole-supercell shift; "
        "this test has no resolving power")
