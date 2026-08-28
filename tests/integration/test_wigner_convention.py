"""The WIGNER convention: velocity-diagonal degenerate eigenbases.

Simoncelli PRX 12, 041011 (2022) / Fugallo 2013: in each degenerate
multiplet the eigenvectors are chosen so the bare velocity operator is
diagonal along x (residual x-degenerate subgroups broken by y, then z), so
coherences come only from cross-multiplet pairs. Contract under test:
intra-multiplet off-diagonals of v^x vanish; diag(v_qssa) == v_qsa; both
solve paths agree; the rotation is a unitary (invariants match RAW); the
label split equals the block split on the xx component.
"""

from __future__ import annotations

import numpy as np
import pytest

from gkmx import DynamicalMatrix
from gkmx.greenkubo import get_kappa
from gkmx.kappa import get_kappa_QHGK, get_kappa_QHGK_block_split
from gkmx.phonon import degenerate_sets

from .._tolerances import TOL_FP64


@pytest.fixture(scope="module")
def prepared(tiny_trajectory, tiny_fc_file):
    traj = tiny_trajectory.copy(deep=True)
    get_kappa(traj, fc_file=tiny_fc_file, interpolate=False, backend="numpy",
              precision="fp64", convention="RAW")
    return traj


@pytest.fixture(scope="module")
def dmx_pair(prepared):
    return {conv: DynamicalMatrix.from_dataset(
        prepared, with_group_velocity_matrices=True, precision="fp64",
        convention=conv) for conv in ("WIGNER", "RAW")}


def _blocks(sol):
    return list(degenerate_sets(np.asarray(sol.w_qs)))


def test_wigner_velocity_diagonal_in_multiplets(dmx_pair):
    """The defining property: v^x has no intra-multiplet off-diagonals
    (and within x-degenerate subgroups, neither do v^y then v^z)."""
    sol = dmx_pair["WIGNER"].solution
    v = np.asarray(sol.v_qssa_cartesian)
    scale = np.abs(v).max()
    worst = 0.0
    for qi, i0, i1 in _blocks(sol):
        blk = v[qi, i0:i1, i0:i1, 0]
        off = blk - np.diag(np.diag(blk))
        worst = max(worst, np.abs(off).max() / scale)
    assert worst < 1e-9, f"intra-multiplet v^x off-diagonal at {worst:.2e}"


def test_wigner_diag_identity(dmx_pair):
    """Copy-paste guard: the WIGNER branch must keep `_align_degenerate_
    branches`'s third return (the PHONO3PY branch one line above discards
    it). RAW satisfies this identity too -- the test does not discriminate
    WIGNER from RAW, only from that specific mistake."""
    sol = dmx_pair["WIGNER"].solution
    dg = np.real(np.einsum("qssa->qsa", np.asarray(sol.v_qssa_cartesian)))
    scale = max(np.abs(np.asarray(sol.v_qsa_cartesian)).max(), 1e-12)
    err = np.abs(dg - np.asarray(sol.v_qsa_cartesian)).max() / scale
    assert err < TOL_FP64, f"diag identity broken at {err:.2e}"


def test_wigner_solve_paths_agree(prepared):
    """The velocities-only path must run the operator pipeline (like TDEP):
    a flattened shortcut would return the unrotated diagonal."""
    full = DynamicalMatrix.from_dataset(
        prepared, with_group_velocity_matrices=True, precision="fp64",
        convention="WIGNER")
    vel_only = DynamicalMatrix.from_dataset(
        prepared, with_group_velocity_matrices=False, precision="fp64",
        convention="WIGNER")
    for name in ("v_qsa_cartesian", "e_qsi"):
        a = np.asarray(getattr(full.solution, name))
        b = np.asarray(getattr(vel_only.solution, name))
        assert np.array_equal(a, b), f"solve paths differ on {name}"


def test_wigner_is_a_unitary_rotation(dmx_pair):
    """Frequencies, eigenvector orthonormality, singleton velocities, and
    per-block operator traces all match RAW: only the intra-multiplet basis
    moved. Invariant, not a discriminator -- passes if the branch is
    deleted; the defining-property tests above carry the coverage."""
    w_sol, r_sol = (dmx_pair[c].solution for c in ("WIGNER", "RAW"))
    assert np.array_equal(np.asarray(w_sol.w_qs), np.asarray(r_sol.w_qs))

    e = np.asarray(w_sol.e_qsi)
    gram = np.einsum("qsi,qti->qst", e.conj(), e)
    eye = np.eye(e.shape[1])[None]
    assert np.abs(gram - eye).max() < 1e-10

    vw = np.asarray(w_sol.v_qssa_cartesian)
    vr = np.asarray(r_sol.v_qssa_cartesian)
    scale = max(np.abs(vr).max(), 1e-12)
    in_block = np.zeros(vw.shape[:2], dtype=bool)
    for qi, i0, i1 in _blocks(w_sol):
        in_block[qi, i0:i1] = True
        tr_w = np.einsum("ssa->a", vw[qi, i0:i1, i0:i1])
        tr_r = np.einsum("ssa->a", vr[qi, i0:i1, i0:i1])
        assert np.abs(tr_w - tr_r).max() / scale < 1e-10
    singles = ~in_block
    dg_w = np.real(np.einsum("qssa->qsa", vw))[singles]
    dg_r = np.real(np.einsum("qssa->qsa", vr))[singles]
    assert np.abs(dg_w - dg_r).max() / scale < 1e-10


def test_wigner_label_split_equals_block_split_on_x(dmx_pair):
    """The PRX statement: with the multiplets velocity-diagonal along x, the
    naive s != s' (label) coherence equals the cross-multiplet (block)
    coherence on the xx component -- the populations conductivity along the
    diagonalized direction is recovered without any block bookkeeping."""
    sol = dmx_pair["WIGNER"].solution
    w = np.asarray(sol.w_qs)
    tau = np.ones_like(w)
    kw = dict(tau_qs=tau, w_qs=w, w_inv_qs=np.asarray(sol.w_inv_qs))
    v = np.asarray(sol.v_qssa_cartesian)

    full = np.asarray(get_kappa_QHGK(v_qssa=v, **kw))
    ns = v.shape[1]
    diag_only = v * np.eye(ns, dtype=bool)[None, :, :, None]
    label_P = np.asarray(get_kappa_QHGK(v_qssa=diag_only, **kw))
    block_P = np.asarray(get_kappa_QHGK_block_split(v, **kw)[0])

    scale = max(abs(full[0, 0]), 1e-12)
    gap = abs(label_P[0, 0] - block_P[0, 0]) / scale
    assert gap < 1e-9, f"xx label/block coherence gap {gap:.2e}"

    # x only: the components do not commute, so yy is NOT closed -- pin the
    # limitation so it stays documented rather than folklore (9.7e-3 here,
    # same as RAW).
    gap_yy = abs(label_P[1, 1] - block_P[1, 1]) / max(abs(full[1, 1]), 1e-12)
    assert gap_yy > 1e-6, "yy unexpectedly closed; scope of WIGNER changed"


def test_wigner_per_mode_v_is_precision_stable(prepared):
    """The highest-resolving-power property: inside multiplets RAW's
    per-mode diagonals are eigh gauge noise and reshuffle wholesale between
    fp32 and fp64 (3e-1), while WIGNER pins them to the operator's branch
    eigenvalues (measured 5e-7)."""
    def v(conv, prec):
        return np.asarray(DynamicalMatrix.from_dataset(
            prepared, with_group_velocity_matrices=True, precision=prec,
            convention=conv).solution.v_qsa_cartesian)

    v64, v32 = v("WIGNER", "fp64"), v("WIGNER", "fp32")
    scale = max(np.abs(v64).max(), 1e-12)
    drift = np.abs(v64 - v32).max() / scale
    assert drift < 1e-5, f"WIGNER per-mode v fp32/fp64 drift {drift:.2e}"

    r64, r32 = v("RAW", "fp64"), v("RAW", "fp32")
    control = np.abs(r64 - r32).max() / scale
    assert control > 1e-2, (
        f"positive control lost resolving power (RAW drift {control:.2e})")
