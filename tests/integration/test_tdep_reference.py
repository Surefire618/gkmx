"""gkmx must reproduce TDEP, from the same force constants, on three space groups.

One fixture per space group in the 240905 set -- 221 Pm-3m (KI_bcc), 225 Fm-3m
(Rb2O), 166 R-3m (KPTe2) -- spanning 6/9/12 modes and every little-group size
from 1 to 48. Frozen TDEP output lives in each ``reference.npz``; nothing here
runs TDEP. Regenerate with ``tests/datasets/_generate_tdep_references.py``.

What is compared, and why it splits that way:

* **frequencies, group velocity** are convention-free, so they compare directly.
* **the velocity operator** is not. gkmx and TDEP use different Bloch
  conventions, so ``v_ss'`` must be mapped through `gkmx.tdep` first, then
  averaged over the little group with ``Gamma(R, q)`` acting on the mode indices
  (`brillouin.symmetrize_v_qssa`). Skipping either leaves an O(1) error.
* **kappa** is rebuilt from TDEP's own lifetimes and heat capacities, so it tests
  the SMA and QHGK assembly rather than the lifetimes.

These run under TDEP's mass table. TDEP computes the IUPAC atomic weight as the
abundance-weighted mean over isotopes; ASE quotes the rounded published value,
and ``dw/w = dm/2m`` floors the agreement at ~1e-05 for O, Cl, Se, Te, N. The
`tdep_masses` fixture sets it process-wide, as ``GKMX_MASSES=TDEP`` would.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import io as ase_io

from gkmx import _constants as C
from gkmx import masses
from gkmx.io import parse_force_constants
from gkmx.phonon import Phonon, _numpy_solve, _rotate_degenerate_subspaces, symmetrize_v_qssa
from gkmx.tdep import to_tdep_convention

from .._tolerances import TOL_FP64_DERIVED

DATA = Path(__file__).parent.parent / "datasets"

# (name, space group). KPTe2 is the only one reaching |LG| = 1, where the
# little-group average must be the identity -- the control that keeps the
# comparison from passing vacuously.
FIXTURES = [("tdep_KI_bcc", 221), ("tdep_Rb2O", 225), ("tdep_KPTe2", 166)]

# Reproduction floor. The residual is TDEP's Fortran vs numpy accumulation, not
# physics: with ASE masses instead these all sit at ~1e-05, four orders worse.
TOL_TDEP = 1e-7


@pytest.fixture(autouse=True)
def tdep_masses():
    """Run the whole comparison on TDEP's mass table, then restore."""
    original = masses.get_default()
    masses.set_default("TDEP")
    yield
    masses.set_default(original)


@pytest.fixture(scope="module", params=FIXTURES, ids=[f"{n}_SG{s}" for n, s in FIXTURES])
def fixture(request):
    name, space_group = request.param
    d = DATA / name
    if not (d / "reference.npz").exists():
        pytest.skip(f"TDEP fixture missing: {d}")
    ref = np.load(d / "reference.npz")
    prim = ase_io.read(str(d / "geometry.in.primitive"), format="aims")
    sc = ase_io.read(str(d / "geometry.in.supercell"), format="aims")
    fc = np.asarray(parse_force_constants(str(d / "FORCE_CONSTANTS_tdep"),
                                          two_dim=False))
    return name, space_group, ref, prim, sc, fc


def _solve(prim, sc, fc, q):
    """gkmx solution at `q`, plus the raw (D, dD/dq) the convention map needs."""
    ph = Phonon(force_constants=fc, primitive=prim, supercell=sc,
                backend="numpy", precision="fp64")
    sol = ph.solve(q, with_velocities=True, with_group_velocity_matrices=True)
    w2, e, M, D = _numpy_solve(
        ph._fc_mw, ph._j_of_k, ph._svec_frac, ph._multi_mask,
        np.asarray(q, float), ph._pcell_cart, ph._N_p,
        dtype_real=np.float64, dtype_complex=np.complex128)
    e, M = _rotate_degenerate_subspaces(w2, e, M)
    U = np.swapaxes(e, -1, -2)
    dD = np.einsum("qij,qajk,qlk->qail", U, M, np.conj(U), optimize=True)
    return sol, D, dD


def test_frequencies_and_group_velocity(fixture):
    """Both are convention-free, so they compare with no transform at all.

    Sorted per q: gkmx and TDEP need not order degenerate modes the same way.
    """
    _, _, ref, prim, sc, fc = fixture
    q = ref["q_full"]
    sol, _, _ = _solve(prim, sc, fc, q)

    f_gkmx = np.sort(np.abs(np.asarray(sol.w_qs)) * C.omega_to_THz, axis=1)
    f_tdep = np.sort(ref["frequencies_THz"], axis=1)
    rel = np.abs(f_gkmx - f_tdep).max() / np.abs(f_tdep).max()
    assert rel < TOL_TDEP, f"frequencies: rel={rel:.2e}"

    v_gkmx = np.sort(np.linalg.norm(np.asarray(sol.v_qsa_cartesian), axis=-1), axis=1)
    v_tdep = np.sort(np.linalg.norm(ref["group_velocities"], axis=-1), axis=1)
    rel = np.abs(v_gkmx - v_tdep).max() / np.abs(v_tdep).max()
    assert rel < TOL_TDEP, f"group velocity: rel={rel:.2e}"


def test_velocity_operator(fixture):
    """``v_ss'`` after the Bloch map and the little-group average.

    TDEP's stored eigenvectors are used rather than gkmx's: their per-mode phases
    are arbitrary but do reach ``Re(v_ss')``, which is what TDEP writes. The
    physics that survives that gauge -- ``sum |v_off|^2`` -- is checked too.
    """
    name, _, ref, prim, sc, fc = fixture
    q = ref["q_ir"]
    sol, D, dD = _solve(prim, sc, fc, q)
    w_inv = np.asarray(sol.w_inv_qs)
    frac = prim.get_scaled_positions()
    ns = D.shape[-1]

    raw = np.zeros((len(q), ns, ns, 3), dtype=complex)
    e_tdep = np.zeros((len(q), ns, ns), dtype=complex)
    for iq in range(len(q)):
        _, dD_T = to_tdep_convention(D[iq], dD[iq], q[iq], frac, prim.positions)
        U_T = np.swapaxes(ref["eigenvectors_ir"][iq], -1, -2)
        e_tdep[iq] = np.swapaxes(U_T, -1, -2)
        s = np.sqrt(w_inv[iq])
        M = np.einsum("ji,ajk,kl->ail", np.conj(U_T), dD_T, U_T, optimize=True)
        raw[iq] = np.moveaxis(M * (0.5 * s[:, None] * s[None, :])[None], 0, -1) \
            * C.gv_to_AA_fs

    v = symmetrize_v_qssa(raw, e_tdep, q, prim).real
    target = ref["velocity_offdiagonal"]
    rel = np.abs(v - target).max() / np.abs(target).max()
    assert rel < TOL_TDEP, f"{name}: v_ss' rel={rel:.2e}"

    off = ~np.eye(ns, dtype=bool)
    ratio = (np.abs(v[:, off]) ** 2).sum() / (np.abs(target[:, off]) ** 2).sum()
    assert abs(ratio - 1.0) < 1e-5, f"{name}: sum|v_off|^2 ratio={ratio:.6f}"


def _averaged_and_raw(ref, prim, sc, fc, which):
    """(averaged, raw) velocity operator at the q-points selected by `which`."""
    q = ref["q_ir"][which]
    sol, D, dD = _solve(prim, sc, fc, q)
    ns = D.shape[-1]
    raw = np.zeros((len(q), ns, ns, 3), dtype=complex)
    for iq in range(len(q)):
        _, dD_T = to_tdep_convention(D[iq], dD[iq], q[iq],
                                     prim.get_scaled_positions(), prim.positions)
        raw[iq] = np.moveaxis(dD_T, 0, -1)
    return symmetrize_v_qssa(raw, np.asarray(sol.e_qsi), q, prim), raw


def test_little_group_average_does_nothing_where_trivial(fixture):
    """At |LG| = 1 there is nothing to average, so v_ss' must come back unchanged.

    Not bit-identical: the little group still contains the identity operation,
    and Gamma(E, q) is applied as a matrix product like any other, so the result
    carries ordinary rounding.
    """
    name, _, ref, prim, sc, fc = fixture
    trivial = np.flatnonzero(ref["n_invariant_operations"] == 1)
    if trivial.size == 0:
        pytest.skip(f"{name} has no |LG| = 1 q-point")

    averaged, raw = _averaged_and_raw(ref, prim, sc, fc, trivial)
    rel = np.abs(averaged - raw).max() / np.abs(raw).max()
    assert rel < TOL_FP64_DERIVED, f"{name}: average moved v_ss' at |LG| = 1, rel={rel:.2e}"


def test_little_group_average_does_something_where_not(fixture):
    """The complement of the test above: at |LG| > 1 the average must bite.

    Without this, a broken convention transform and a broken average could
    cancel and the reproduction would still pass. The average is what kills the
    inter-irrep blocks, so it changes v_ss' by tens of percent.
    """
    name, _, ref, prim, sc, fc = fixture
    nontrivial = np.flatnonzero(ref["n_invariant_operations"] > 1)
    assert nontrivial.size, f"{name} has no |LG| > 1 q-point; this fixture cannot test the average"

    averaged, raw = _averaged_and_raw(ref, prim, sc, fc, nontrivial)
    changed = np.abs(averaged - raw).max() / np.abs(raw).max()
    assert changed > 0.05, f"{name}: average barely moved v_ss' ({changed:.2e}) at |LG| > 1"


def test_sma_kappa_from_tdep_lifetimes(fixture):
    """kappa_SMA = (1/V N_q) sum cv v_a v_b tau, from TDEP's own tau and cv.

    Tests the SMA assembly and the unit chain, not the lifetimes. Compared
    componentwise: KPTe2 is R-3m, where kxx = kyy != kzz, and a diagonal mean
    would hide a 3.8 % error in the anisotropy.
    """
    name, _, ref, prim, sc, fc = fixture
    q = ref["q_full"]
    sol, _, _ = _solve(prim, sc, fc, q)

    v = np.asarray(sol.v_qsa_cartesian) * 1e5          # A/fs -> m/s
    tau, cv = ref["lifetimes_s"], ref["heat_capacity_JK"]
    volume = prim.get_volume() * 1e-30                 # A^3 -> m^3
    kappa = np.einsum("qs,qsa,qsb,qs->ab", cv, v, v, tau) / (volume * len(q))

    target = ref["kappa_sma"]
    got = np.array([kappa[0, 0], kappa[1, 1], kappa[2, 2]])
    rel = np.abs(got - target).max() / np.abs(target).max()
    assert rel < 1e-4, (
        f"{name}: kappa_SMA (kxx,kyy,kzz) = {got} vs TDEP {target}, rel={rel:.2e}")


def test_coherent_kappa_from_tdep_linewidths(fixture):
    """kappa_C from gkmx's velocity operator and TDEP's own linewidths.

    TDEP's kernel (`kappa.f90:296-315`), summed over the irreducible wedge with
    its weights ``w = (|G|/|LG|)/N_q``::

        f0  = (cv_j + cv_k) / 2
        ker = (G_j + G_k) / ((G_j + G_k)^2 + (w_j - w_k)^2)
        kappa_C = (1/V) sum_q w_q sum_{j != k} v_jk (x) v_jk ker f0

    Compared as the trace. TDEP averages ``v (x) v`` over the symmetry-equivalent
    q before accumulating, and that rotation preserves
    ``sum_a (R v)^a (R v)^a = |v|^2`` -- so the trace is reproducible without
    those operations, while the individual Cartesian components are not. It is
    also the phase-invariant
    combination, which matters because ``Re(v_ss')`` is gauge-dependent.

    This is the one test that exercises the velocity operator end-to-end: an
    error anywhere in the Bloch map or the little-group average lands here.
    """
    name, _, ref, prim, sc, fc = fixture
    q = ref["q_ir"]
    sol, D, dD = _solve(prim, sc, fc, q)
    w_inv = np.asarray(sol.w_inv_qs)
    frac = prim.get_scaled_positions()
    ns = D.shape[-1]

    raw = np.zeros((len(q), ns, ns, 3), dtype=complex)
    e_tdep = np.zeros((len(q), ns, ns), dtype=complex)
    for iq in range(len(q)):
        _, dD_T = to_tdep_convention(D[iq], dD[iq], q[iq], frac, prim.positions)
        U_T = np.swapaxes(ref["eigenvectors_ir"][iq], -1, -2)
        e_tdep[iq] = np.swapaxes(U_T, -1, -2)
        s = np.sqrt(w_inv[iq])
        M = np.einsum("ji,ajk,kl->ail", np.conj(U_T), dD_T, U_T, optimize=True)
        raw[iq] = np.moveaxis(M * (0.5 * s[:, None] * s[None, :])[None], 0, -1) \
            * C.gv_to_AA_fs
    v = symmetrize_v_qssa(raw, e_tdep, q, prim).real * 1e5      # A/fs -> m/s

    omega = ref["frequencies_ir_rad_s"]
    gamma = ref["linewidths_ir_rad_s"]
    cv = ref["heat_capacity_ir_JK"]
    weights = ref["weights_ir"]
    volume = prim.get_volume() * 1e-30
    # TDEP drops a mode from the pair sum if either its frequency or its
    # linewidth is below lo_freqtol -- the acoustic modes at Gamma, and the rare
    # mode with no computed linewidth.
    tol = 1e-8 * omega.max()

    total = 0.0
    for iq in range(len(q)):
        o, g, c = omega[iq], gamma[iq], cv[iq]
        live = (o > tol) & (g > tol)
        pair = live[:, None] & live[None, :] & ~np.eye(ns, dtype=bool)
        sum_g = g[:, None] + g[None, :]
        delta = o[:, None] - o[None, :]
        denom = sum_g ** 2 + delta ** 2
        kernel = np.where(pair, sum_g / np.where(denom > 0, denom, 1.0), 0.0)
        f0 = 0.5 * (c[:, None] + c[None, :])
        total += weights[iq] * (((v[iq] ** 2).sum(axis=-1)) * kernel * f0).sum()

    got = total / volume / 3.0
    target = ref["kappa_coherent"].mean()
    rel = abs(got - target) / abs(target)
    assert rel < 1e-4, (
        f"{name}: Tr(kappa_C)/3 = {got:.6f} vs TDEP {target:.6f} W/mK, rel={rel:.2e}")
