"""Gauge invariance of ``gkmx.kappa.get_kappa_QHGK``.

An eigenvector basis of ``D(q)`` is not unique. Two freedoms leave it a valid
eigenbasis of the same matrix with the same eigenvalues:

  * a per-mode phase ``e_s -> exp(i th_s) e_s``, under which
    ``v_ss' -> exp(i(th_s' - th_s)) v_ss'``;
  * a unitary inside a degenerate multiplet, ``V -> U^dag V U`` with ``U`` block
    diagonal on the blocks of equal frequency.

Neither is observable, so ``kappa_QHGK`` must not move under either.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase.io import read

from gkmx import phonon as _phonon_mod
from gkmx.io import parse_force_constants
from gkmx.kappa import get_kappa_QHGK, symmetrize_kappa
from gkmx.phonon import CONVENTIONS, Phonon, degenerate_sets

from .._tolerances import TOL_FP64

DATA_DIR = Path(__file__).parent.parent / "datasets" / "tdep_KI_bcc"
NSEED = 8


@pytest.fixture(scope="module")
def tdep_reference():
    """Everything the fixture directory supplies, read once.

    TDEP's q_full (216 points) rather than the commensurate grid: it is the grid
    its lifetimes and heat capacities are tabulated on, and it carries 96
    degenerate multiplets against 58, so the intra-multiplet rotation has more to
    act on.

    TDEP's real weights rather than constants. Uniform tau and cv make the
    assertions algebraically automatic: the kernel weight is then the same for
    every pair inside a degenerate block, kappa_C collapses to a fixed factor
    times the block trace, and nothing involving the weights can fail. TDEP's
    weights hold that property honestly instead -- spread inside a multiplet is
    0.0e+00 for tau (`scattering.f90:267` averages it over the multiplet) and
    1.9e-15 for cv -- so a reference that broke it would show up here.

    cv goes in as TDEP writes it (J/K per mode), so kappa is not in W/mK. Every
    assertion is relative, so it does not matter.
    """
    ref = np.load(DATA_DIR / "reference.npz")
    return SimpleNamespace(
        primitive=read(DATA_DIR / "geometry.in.primitive", format="aims"),
        supercell=read(DATA_DIR / "geometry.in.supercell", format="aims"),
        force_constants=np.asarray(parse_force_constants(
            str(DATA_DIR / "FORCE_CONSTANTS_tdep"), two_dim=False)),
        q=ref["q_full"],
        tau=ref["lifetimes_s"] * 1e15,          # s -> fs, what qhgk_tau_eff wants
        cv=ref["heat_capacity_JK"],
    )


def _solve_random_phase(phonon, q, phase_seed=None):
    """Solve ``phonon`` at ``q``, optionally in a random per-mode gauge.

    ``phase_seed`` rotates the eigenvectors where they are produced, so the whole
    downstream path -- degenerate handling, Bloch map, little-group average,
    velocity construction -- runs in the rotated basis. ``e_s -> ph_s e_s`` sends
    ``M[a,s,t] = <e_s|dD^a|e_t>`` to ``conj(ph_s) ph_t M[a,s,t]``.
    """
    if phase_seed is None:
        sol = phonon.solve(q, with_velocities=True,
                           with_group_velocity_matrices=True)
    else:
        original = _phonon_mod._numpy_solve

        def gauged(*args, **kwargs):
            w2, e, M, D = original(*args, **kwargs)
            rng = np.random.default_rng(phase_seed)
            g = np.exp(2j * np.pi * rng.random(e.shape[:2]))
            return (w2, g[:, :, None] * e,
                    M * (np.conj(g)[:, None, :, None] * g[:, None, None, :]), D)

        _phonon_mod._numpy_solve = gauged
        try:
            sol = phonon.solve(q, with_velocities=True,
                               with_group_velocity_matrices=True)
        finally:
            _phonon_mod._numpy_solve = original

    return (np.asarray(sol.v_qssa_cartesian), np.asarray(sol.w_qs),
            np.asarray(sol.w_inv_qs))


@pytest.fixture(scope="module", params=CONVENTIONS, ids=list(CONVENTIONS))
def solution(request, tdep_reference):
    """Velocity operator and TDEP's weights, per convention.

    Gauge invariance is a property of the kernel, not of either convention, so it
    has to hold under both.
    """
    d = tdep_reference
    phonon = Phonon(force_constants=d.force_constants, primitive=d.primitive,
                    supercell=d.supercell, backend="numpy", precision="fp64",
                    convention=request.param)
    v, w, w_inv = _solve_random_phase(phonon, d.q)
    return SimpleNamespace(phonon=phonon, data=d, v=v, w=w, w_inv=w_inv)


def _rel(K, K0):
    return np.abs(K - K0).max() / np.abs(K0).max()


def _block_gauge(v, w, rng):
    """V -> U^dag V U with U a random unitary inside each degenerate multiplet."""
    out = np.array(v, copy=True)
    for iq in range(v.shape[0]):
        U = np.eye(v.shape[1], dtype=complex)
        for _qi, s0, s1 in degenerate_sets(np.abs(w)[iq][None, :]):
            m = s1 - s0
            A = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
            Q, R = np.linalg.qr(A)
            U[s0:s1, s0:s1] = Q * (np.diag(R) / np.abs(np.diag(R)))[None, :]
        out[iq] = np.einsum("si,sta,tj->ija", np.conj(U), v[iq], U, optimize=True)
    return out


def test_qhgk_invariant_under_per_mode_phase(solution):
    """A per-mode phase applied at the eigensolver cannot move kappa_QHGK."""
    v, w, w_inv = solution.v, solution.w, solution.w_inv
    tau, cv = solution.data.tau, solution.data.cv
    base = np.asarray(get_kappa_QHGK(v, tau, w, w_inv, cv).data)
    for seed in range(NSEED):
        vg, wg, wg_inv = _solve_random_phase(solution.phonon, solution.data.q, phase_seed=seed)
        np.testing.assert_allclose(wg, w, atol=TOL_FP64,
                                   err_msg="the gauge moved the frequencies")
        moved = _rel(np.asarray(get_kappa_QHGK(vg, tau, wg, wg_inv, cv).data), base)
        assert moved < TOL_FP64, (
            f"seed {seed}: kappa_QHGK moved {moved:.2e} under a phase gauge")


def test_qhgk_invariant_under_intra_multiplet_rotation(solution):
    """A unitary inside a degenerate multiplet cannot move kappa_QHGK."""
    v, w, w_inv = solution.v, solution.w, solution.w_inv
    tau, cv = solution.data.tau, solution.data.cv

    # Guard against vacuity: there must be multiplets, and the rotation must
    # actually move the operator. Without both, this test asserts nothing.
    ndeg = sum(s1 - s0 for _qi, s0, s1 in degenerate_sets(np.abs(w)))
    assert ndeg > 0, "no degenerate multiplets: the rotation is the identity"
    moved = max(
        np.abs(_block_gauge(v, w, np.random.default_rng(s)) - v).max()
        for s in range(NSEED))
    assert moved / np.abs(v).max() > 1e-6, (
        f"the block rotation left v unchanged (max move {moved:.2e}); "
        f"the invariance assertion below would be vacuous")

    # TOL_FP64, not TOL_FP64_DERIVED: the multiplets here are degenerate to
    # 1.9e-16, so tau_eff is exactly constant over each block and the deviation
    # is machine precision (measured 8.1e-16 PHONOPY / 1.8e-15 TDEP). At 1e-6 a
    # genuine 1e-5 violation would pass.
    base = np.asarray(get_kappa_QHGK(v, tau, w, w_inv, cv).data)
    for seed in range(NSEED):
        rng = np.random.default_rng(seed)
        got = _rel(np.asarray(get_kappa_QHGK(_block_gauge(v, w, rng), tau, w, w_inv, cv).data), base)
        assert got < TOL_FP64, (
            f"seed {seed}: kappa_QHGK moved {got:.2e} under an intra-multiplet "
            f"rotation")
