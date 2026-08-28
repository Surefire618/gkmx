"""Tests for ``gkmx.Phonon`` (standalone phonon backend)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from ase import io as ase_io

from gkmx.io import parse_force_constants
from gkmx.lattice_points import get_commensurate_q_points
from gkmx.phonon import (
    CONVENTIONS,
    Phonon,
    SolutionWithGVM,
    degenerate_sets,
    translational_invariance,
)
from gkmx.space_group import refine_geometry, space_group_invariance

from .._tolerances import TOL_FP32_BLOCK_POWER, TOL_FP64

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


@pytest.fixture(scope="module")
def tdep_setup():
    """tdep_KI_bcc: TDEP-fitted constants, so multiplets survive away from TRIM.

    KI_B2_MLIP cannot constrain the averaging: its multiplets sit at TRIM,
    where v = 0 by symmetry. These TDEP constants give 58 multiplets, 56 of
    them with a nonzero averaged velocity.
    """
    d = Path(__file__).parent.parent / "datasets" / "tdep_KI_bcc"
    prim = ase_io.read(str(d / "geometry.in.primitive"), format="aims")
    sc = ase_io.read(str(d / "geometry.in.supercell"), format="aims")
    fc = np.asarray(parse_force_constants(
        str(d / "FORCE_CONSTANTS_tdep"), two_dim=False))
    q = np.asarray(get_commensurate_q_points(prim.cell, sc.cell, fractional=True))
    return prim, sc, fc, q


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

    # the checked-in DynamicalMatrix.nc reference is baked with the library
    # defaults (ASR + space-group projection, TDEP convention)
    sol = Phonon(fc, prim, sc).solve(
        q, with_group_velocity_matrices=True)
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
    # library defaults (ASR + space-group projection), matching the bake of
    # the checked-in DynamicalMatrix.nc reference (regenerated 2026-08-17)
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


def _block_row_power(v, w):
    """Per-(q, degenerate-block) row power ``sum_{s in B, t, a} |v|^2``.

    Invariant under per-mode phases and intra-block rotations on both
    sides, so it is comparable across precisions (whose eigh bases differ
    inside multiplets), yet q- and block-resolved, so a localized error
    cannot hide behind global compensation the way the banned scalar
    ``sum|v_off|^2`` allows."""
    Nq, Ns = w.shape
    blmap = {}
    for qj, i0, i1 in degenerate_sets(w):
        blmap.setdefault(qj, []).append((i0, i1))
    out = []
    for qi in range(Nq):
        bounds, pos = [], 0
        for i0, i1 in blmap.get(qi, []):
            bounds += [(s, s + 1) for s in range(pos, i0)] + [(i0, i1)]
            pos = i1
        bounds += [(s, s + 1) for s in range(pos, Ns)]
        out.append(np.array([np.sum(np.abs(v[qi, i0:i1]) ** 2)
                             for i0, i1 in bounds]))
    return out


def test_fp32_velocity_operator_matches_fp64(setup):
    """fp32 vs fp64 ``v_qssa`` per convention on the Gamma-containing grid,
    compared block-row-wise. The only in-suite detector for the fp32
    acoustic-gate floor, the pre-average band mask, and the frequency-block
    G projection: with any of them reverted the TDEP mode average smears
    1/(2 sqrt(w w')) Gamma-acoustic garbage into live modes (5-38x
    inflation before the gate-floor fix). Measured 6.4e-7..7.5e-7."""
    prim, sc, fc, q, _ = setup
    for c in CONVENTIONS:
        rp = {}
        for p in ("fp32", "fp64"):
            sol = Phonon(fc, prim, sc, precision=p, convention=c).solve(
                q, with_velocities=True, with_group_velocity_matrices=True)
            v = np.asarray(sol.v_qssa_cartesian, dtype=np.complex128)
            w = np.abs(np.asarray(sol.w_qs, dtype=np.float64))
            rp[p] = _block_row_power(v, w)
        scale = max(b.max() for b in rp["fp64"])
        worst = max(np.abs(a - b).max()
                    for a, b in zip(rp["fp32"], rp["fp64"]))
        assert worst / scale < TOL_FP32_BLOCK_POWER, (
            f"{c}: per-(q,block) row power fp32 vs fp64 rel = "
            f"{worst/scale:.2e}")


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


def test_convention_is_validated(setup):
    """`convention` selects the whole upstream convention, not one knob.

    "TDEP" gives every member of a multiplet the subspace mean, averages the
    mode indices of v_qssa with Gamma(R, q), and carries the per-atom Bloch
    phase; "PHONO3PY" rotates the degenerate basis to diagonalise
    dD/dq . probe and applies the Cartesian site average to v_qsa, leaving
    v_qssa branch-aligned but un-averaged (phono3py 3a706b1f); "RAW" applies
    nothing at all; "WIGNER" rotates each multiplet so the bare operator's
    v^x is diagonal. TDEP, RAW and WIGNER satisfy diag(v_qssa) == v_qsa;
    PHONO3PY does not, because its v_qsa is site-averaged while its v_qssa is
    branch-aligned -- two different operators."""
    prim, sc, fc, _, _ = setup
    assert set(CONVENTIONS) == {"TDEP", "PHONO3PY", "RAW", "WIGNER"}

    for convention in CONVENTIONS:
        ph = Phonon(force_constants=fc, primitive=prim, supercell=sc,
                    convention=convention)
        assert ph.convention == convention

    with pytest.raises(ValueError, match="convention"):
        Phonon(force_constants=fc, primitive=prim, supercell=sc, convention="bogus")


def test_tdep_method_gives_one_velocity_per_multiplet(tdep_setup):
    """TDEP assigns the subspace mean, so a multiplet carries a single velocity.

    That mean is a trace, hence basis-independent.
    """
    prim, sc, fc, q = tdep_setup
    sol = Phonon(force_constants=fc, primitive=prim, supercell=sc,
                 precision="fp64", convention="TDEP").solve(q, with_velocities=True)
    w = np.abs(np.asarray(sol.w_qs))
    v = np.asarray(sol.v_qsa_cartesian)

    blocks = list(degenerate_sets(w))
    assert blocks, "fixture has no degenerate multiplets; this test is vacuous"

    for qi, start, stop in blocks:
        blk = v[qi, start:stop]
        assert np.allclose(blk, blk[0], atol=1e-12), (
            f"tdep left different velocities in the multiplet at q={qi}: {blk}")


def test_convention_agrees_on_frequencies(setup):
    """Frequencies are the eigenvalues of D(q), which neither half of the
    convention touches: the Bloch map is a similarity transform and the multiplet
    treatment acts only on the velocity."""
    prim, sc, fc, q, _ = setup
    w = {c: np.asarray(Phonon(force_constants=fc, primitive=prim, supercell=sc,
                              precision="fp64", convention=c).solve(q).w_qs)
         for c in CONVENTIONS}
    scale = np.abs(w["TDEP"]).max()
    rel = max(np.abs(w[c] - w["TDEP"]).max() / scale for c in CONVENTIONS)
    assert rel < TOL_FP64, f"convention moved the frequencies by {rel:.2e}"


def test_phono3py_is_raw_plus_branch_alignment(setup):
    """PHONO3PY differs from RAW by a unitary inside each degenerate block.

    Both now carry the same unsymmetrized operator, so the total weight is
    invariant; only alignment moves weight onto the diagonal kappa_BTE reads.
    """
    prim, sc, fc, q, _ = setup
    v = {}
    for c in ("PHONO3PY", "RAW"):
        ph = Phonon(force_constants=fc, primitive=prim, supercell=sc,
                    precision="fp64", convention=c)
        sol = ph.solve(q, with_velocities=True,
                       with_group_velocity_matrices=True)
        v[c] = np.asarray(sol.v_qssa_cartesian)

    tot = {c: float((np.abs(x) ** 2).sum()) for c, x in v.items()}
    assert abs(tot["PHONO3PY"] - tot["RAW"]) / max(tot["RAW"], 1e-30) < TOL_FP64, (
        f"alignment is not unitary: total operator weight moved "
        f"{tot['PHONO3PY']:.6e} vs {tot['RAW']:.6e}")

    share = {c: float((np.abs(np.einsum("qssa->qsa", x)) ** 2).sum()) / max(tot[c], 1e-30)
             for c, x in v.items()}
    assert share["PHONO3PY"] > share["RAW"], (
        f"alignment did not concentrate weight on the diagonal "
        f"(PHONO3PY {share['PHONO3PY']:.4f} vs RAW {share['RAW']:.4f})")


def test_convention_changes_the_off_diagonal_velocity(setup):
    """``convention`` reaches ``v_ss'`` off the diagonal, where the Bloch gauge acts.

    The switch would be a silent no-op if it did not, and every comparison drawn
    between the two conventions elsewhere would be vacuous.
    """
    prim, sc, fc, q, _ = setup
    v = {c: np.asarray(
            Phonon(force_constants=fc, primitive=prim, supercell=sc,
                   precision="fp64", convention=c).solve(
                q, with_velocities=True,
                with_group_velocity_matrices=True).v_qssa_cartesian)
         for c in CONVENTIONS}

    off = ~np.eye(v["PHONO3PY"].shape[1], dtype=bool)
    moved = (np.abs(v["PHONO3PY"] - v["TDEP"])[:, off].max()
             / np.abs(v["PHONO3PY"]).max())
    assert moved > 1e-3, (
        f"convention left v_ss' unchanged off the diagonal (max {moved:.2e}); "
        f"the switch is a no-op")


def test_velocity_operator_is_hermitian(setup):
    """``v_ss' = conj(v_s's)``, inherited from dD/dq being Hermitian.

    A property of the solver, not of any kappa kernel: with a 1.5 % anti-Hermitian
    part injected here, both gauge invariances in test_kappa_gauge.py still hold,
    because the QHGK trace is invariant whether or not v is Hermitian.
    """
    prim, sc, fc, q, _ = setup
    for c in CONVENTIONS:
        v = np.asarray(Phonon(force_constants=fc, primitive=prim, supercell=sc,
                              precision="fp64", convention=c).solve(
            q, with_velocities=True,
            with_group_velocity_matrices=True).v_qssa_cartesian)
        rel = np.abs(v - np.conj(np.swapaxes(v, 1, 2))).max() / np.abs(v).max()
        assert rel < TOL_FP64, f"{c}: v_ss' is not Hermitian, rel={rel:.2e}"


def test_space_group_invariance_properties():
    """The FC projector: projector, ASR-preserving, G-invariant, coset-exact.

    All on a ~2 % noise-injected FC -- mandatory, because TDEP-fitted FCs are
    already symmetric and every assertion below would pass with the projector
    stubbed to ``return fc``. tdep_Ga2O3_kappa: non-symmorphic (Pna2_1),
    32 supercell ops over 4 rotations, so the coset restriction is exercised.
    """
    d = Path(__file__).parent.parent / "datasets" / "tdep_Ga2O3_kappa"
    prim = ase_io.read(str(d / "geometry.in.primitive"), format="aims")
    sc = ase_io.read(str(d / "geometry.in.supercell"), format="aims")
    fc = np.load(d / "force_constants.npz")["force_constants"].astype(float)

    rng = np.random.default_rng(20260817)
    noisy = fc + 0.02 * np.abs(fc).max() * rng.standard_normal(fc.shape)
    noisy, _ = translational_invariance(noisy, prim, sc)

    projected, removed = space_group_invariance(noisy, prim, sc)
    assert removed > 1e-3, "noise injection failed; every check below is vacuous"

    # projector: a second application changes nothing
    _, second = space_group_invariance(projected, prim, sc)
    assert second < 1e-14

    # ASR preserved exactly (row sums map to rotated row sums)
    assert np.abs(projected.sum(axis=1)).max() \
        / max(np.abs(projected).max(), 1e-30) < 1e-13

    # positive control: the noisy input itself fails by orders of magnitude
    assert removed / max(second, 1e-30) > 1e10


def test_refine_geometry_properties():
    """Noise-injected controls for the geometry projection, mirroring
    ``test_space_group_invariance_properties``: a Wyckoff coordinate pushed
    1e-6 A off its site is restored, the projection is idempotent, the wrap
    branch of every atom is preserved (a lattice-vector shift broke the
    KPTe2 anchors), boundary atoms land exactly on integers so downstream
    hard wraps are deterministic, and an incommensurate pair raises."""
    d = Path(__file__).parent.parent / "datasets" / "tdep_KI_bcc"
    prim = ase_io.read(str(d / "geometry.in.primitive"), format="aims")
    sc = ase_io.read(str(d / "geometry.in.supercell"), format="aims")

    noisy = sc.copy()
    noisy.positions = noisy.positions + 0.0
    noisy.positions[5] += [1e-6, -1e-6, 1e-6]
    _, s2, res = refine_geometry(prim, noisy)
    assert res > 5e-7, "residual does not reflect the injected displacement"
    assert np.abs(s2.positions[5] - sc.positions[5]).max() < 1e-8, \
        "injected 1e-6 A displacement not projected out"
    assert np.abs(s2.positions - sc.positions).max() < 1e-7

    # idempotency on already-symmetric input (one pass is exact there; a
    # noisy input also moves the detected origin, so ITS convergence is
    # geometric rather than one-shot)
    pc, scn, _ = refine_geometry(prim, sc)
    pc2, scn2, res2 = refine_geometry(pc, scn)
    assert res2 < 1e-12
    assert np.abs(scn2.positions - scn.positions).max() < 1e-12
    assert np.abs(np.asarray(pc2.cell) - np.asarray(pc.cell)).max() < 1e-12

    # wrap-branch preservation: no atom moved by a lattice vector
    dfrac = (s2.positions - sc.positions) @ np.linalg.inv(np.asarray(sc.cell))
    assert np.abs(dfrac).max() < 0.5

    # boundary determinism: near-integer fractional coordinates are exact
    f = s2.positions @ np.linalg.inv(np.asarray(s2.cell))
    near = np.abs(f - np.rint(f)) < 1e-9
    assert np.all(f[near] == np.rint(f)[near])

    bad = prim.copy()
    bad.set_cell(np.asarray(prim.cell) * 1.017, scale_atoms=True)
    with pytest.raises(ValueError, match="integer multiple"):
        refine_geometry(bad, sc)
