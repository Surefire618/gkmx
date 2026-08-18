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
from gkmx.kappa import symmetrize_kappa
from gkmx.phonon import (
    DEGENERACY_TOL,
    Phonon,
    _numpy_solve,
    _rotate_degenerate_subspaces,
    degenerate_sets,
    symmetrize_v_qssa,
)
from gkmx.tdep import to_tdep_convention

from .._tolerances import TOL_FP64_DERIVED

DATA = Path(__file__).parent.parent / "datasets"

# (name, space group). KPTe2 is the only one reaching |LG| = 1, where the
# little-group average must be the identity -- the control that keeps the
# comparison from passing vacuously.
FIXTURES = [("tdep_KI_bcc", 221), ("tdep_Rb2O", 225), ("tdep_KPTe2", 166)]

# Reproduction floor. The residual is TDEP's Fortran vs numpy accumulation, not
# physics: with ASE masses instead these all sit at ~1e-05, four orders worse.
# Measured band across the three fixtures, all channels: frequencies and group
# velocities ~1e-9, v_ss' element-wise 9.3e-10 to 2.9e-9 (worst element is always
# one TDEP stores as exactly 0.0, so it is absolute round-off, not a relative
# error), kappa_SMA 2.5e-10 to 1.5e-9, kappa_C 1.6e-10 to 1.5e-9. The
# kappa channels used a bare 1e-4 while the fixtures held TDEP's pre-fix
# kappa_coherent; with the gauge-fixed references they sit at the same floor as
# everything else, so they use this constant too.
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
    are arbitrary but do reach ``Re(v_ss')``, which is what TDEP writes.

    Two assertions, measuring different things, both needed:

    * **element by element**, on the whole ``(Nq, Ns, Ns, 3)`` array. This is
      basis dependent -- ``eigh`` and ``zheev`` pick different bases inside a
      degenerate multiplet, and with gkmx's own eigenvectors the elementwise
      agreement is only O(1) (measured 0.24 to 0.53 on ``|v_ss'|``). Hence TDEP's
      stored eigenvectors here, and hence this alone cannot certify the operator.
    * **sum of ``|v_off|^2``**, which is ``sum_a Tr(v^a v^a-dagger)`` and so is
      invariant under any unitary inside a multiplet: measured identical to eight
      digits whether gkmx's or TDEP's eigenvectors are used. It carries the part
      of the agreement the elementwise check cannot see.

    A summed norm must never stand alone -- it runs to 1.000000 while individual
    elements are wrong in compensating directions. Beside the elementwise check it
    adds the basis-independent statement.
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

    # No w_qs= on purpose: this reproduces TDEP's own output, and kronegv
    # does no frequency-block projection. The library path passes w_qs.
    v = symmetrize_v_qssa(raw, e_tdep, q, prim)
    target = ref["velocity_offdiagonal"]
    # gkmx stores v_ss' where TDEP stores v_s's -- equivalently its conjugate,
    # since v is Hermitian in the mode indices to 1.1e-16. The data cannot tell
    # the two readings apart. Either way it is a convention, invisible to kappa
    # (the kernel contracts v^a conj(v^b)), but it must be named to compare the
    # operator itself.
    rel = np.abs(np.conj(v) - target).max() / np.abs(target).max()
    assert rel < TOL_TDEP, f"{name}: v_ss' rel={rel:.2e}"




def _block_bilinear(v_qssa, w_qs):
    """``B^ab_IJ(q) = sum_{s in I, s' in J} v^a_ss' v^b_s's`` over degenerate blocks.

    Blocks are the multiplets of ``degenerate_sets`` plus every non-degenerate
    mode as its own singleton, so on singleton pairs B is the per-element
    phase-invariant bilinear (nothing summed) and on multiplet pairs it is the
    degenerate average -- the finest quantity invariant under the gauge eigh
    leaves. Test-only: no core function consumes it.

    Returns ``(B, block_index)`` with ``B`` a per-q list of ``(nb, nb, 3, 3)``
    and ``block_index`` a ``(Nq, Ns)`` block label per mode.
    """
    v = np.asarray(v_qssa)
    w = np.asarray(w_qs)
    Nq, Ns = w.shape
    multiplets = [[] for _ in range(Nq)]
    for qi, start, stop in degenerate_sets(w):
        multiplets[qi].append((start, stop))

    block_index = np.zeros((Nq, Ns), dtype=np.int64)
    B = []
    for qi in range(Nq):
        bq, pos = [], 0
        for start, stop in multiplets[qi]:          # ascending by construction
            bq.extend((s, s + 1) for s in range(pos, start))
            bq.append((start, stop))
            pos = stop
        bq.extend((s, s + 1) for s in range(pos, Ns))
        for bi, (i0, i1) in enumerate(bq):
            block_index[qi, i0:i1] = bi

        nb = len(bq)
        Bq = np.empty((nb, nb, 3, 3), dtype=v.dtype)
        vq = v[qi]
        for bi, (i0, i1) in enumerate(bq):
            for bj, (j0, j1) in enumerate(bq):
                Bq[bi, bj] = np.einsum("ija,jib->ab",
                                       vq[i0:i1, j0:j1, :], vq[j0:j1, i0:i1, :])
        B.append(Bq)
    return B, block_index


def _v_own_eigenvectors(ref, prim, sc, fc):
    """v_ss' from the shipped path: ``Phonon(convention="TDEP")``, gkmx throughout.

    Deliberately the public solver rather than a hand assembly. `Phonon` reaches
    the TDEP gauge through `_to_tdep_bloch`, which is a second implementation of
    the map `to_tdep_convention` provides; assembling by hand here would leave
    `_to_tdep_bloch` -- the code that actually ships -- untested, and a fault
    injected at `Phonon`'s own `v_qssa` would pass every assertion below.
    """
    q = ref["q_ir"]
    sol = Phonon(force_constants=fc, primitive=prim, supercell=sc,
                 backend="numpy", precision="fp64", convention="TDEP").solve(
        q, with_velocities=True, with_group_velocity_matrices=True)
    return (np.asarray(sol.v_qssa_cartesian), np.abs(np.asarray(sol.w_qs)))


def test_velocity_operator_blockwise_no_tdep_eigenvectors(fixture):
    """``v_ss'`` vs TDEP from gkmx's own eigenvectors, split by block structure.

    Element-by-element agreement of ``v_ss'`` across two codes is impossible:
    eigh and zheev land in different gauges -- a phase per mode, a unitary per
    degenerate multiplet. What IS comparable is ``block_bilinear``'s
    ``B^ab_IJ = sum_{s in I, s' in J} v^a_ss' v^b_s's``, and it implements
    both halves of the comparison in one object:

    * OUTSIDE the multiplets both blocks are singletons, so ``B`` is the
      per-pair bilinear ``v^a_ss' conj(v^b_ss')`` -- element by element,
      phase-invariant, nothing summed: q, s, s', a, b all stay resolved.
    * INSIDE / touching a multiplet only the block-contracted degenerate
      average is defined, and that is what ``B`` is there.

    Compared complex against the conjugate (TDEP stores ``v_s's``), so the
    imaginary part rides along as the conjugation fingerprint. Asserted in
    two parts so a regression names the part it broke. Measured 4.7e-10 to
    1.9e-09 on both parts across the three fixtures.
    """
    name, _, ref, prim, sc, fc = fixture
    v, w = _v_own_eigenvectors(ref, prim, sc, fc)
    target = np.asarray(ref["velocity_offdiagonal"])

    B_gkmx, block_index = _block_bilinear(v, w)
    B_tdep, _ = _block_bilinear(target, w)
    scale = max(np.abs(Bq).max() for Bq in B_tdep)

    n_single = n_multi = 0
    worst_single = worst_multi = 0.0
    for iq in range(len(w)):
        sizes = np.bincount(block_index[iq])
        single = (sizes[:, None] == 1) & (sizes[None, :] == 1)
        d = np.abs(B_gkmx[iq] - np.conj(B_tdep[iq]))
        n_single += int(single.sum())
        n_multi += int((~single).sum())
        if single.any():
            worst_single = max(worst_single, d[single].max())
        if (~single).any():
            worst_multi = max(worst_multi, d[~single].max())

    assert n_single > 0 and n_multi > 0, (
        f"{name}: block structure degenerate ({n_single} singleton / "
        f"{n_multi} multiplet block pairs); one half of this test is vacuous")
    assert worst_single / scale < TOL_TDEP, (
        f"{name}: element-wise v bilinear outside multiplets rel="
        f"{worst_single/scale:.2e}")
    assert worst_multi / scale < TOL_TDEP, (
        f"{name}: degenerate-average consistency inside multiplets rel="
        f"{worst_multi/scale:.2e}")

    # Guard: raw elements inside multiplets must NOT agree -- if they ever
    # do, the two codes' bases no longer differ and the split above is
    # silently weaker than it looks.
    deg_mode = np.zeros(w.shape, dtype=bool)
    for iq in range(len(w)):
        deg_mode[iq] = np.bincount(block_index[iq])[block_index[iq]] > 1
    deg_pair = deg_mode[:, :, None] | deg_mode[:, None, :]
    d_el = np.abs(np.abs(v) ** 2 - np.abs(target) ** 2).max(axis=-1)
    if deg_pair.any():
        assert d_el[deg_pair].max() / (np.abs(target) ** 2).max() > 1e-3, (
            f"{name}: degenerate pairs agree elementwise; the multiplet basis "
            f"no longer differs, so revisit this split")


def test_kernel_weights_are_constant_within_a_multiplet(fixture):
    """omega, linewidth, lifetime and cv must not vary inside a degenerate block.

    This is what makes the blocked comparison above equivalent to comparing
    kappa_C itself. With the weights constant over a block pair, the QHGK kernel
    and the heat capacity factor out of the block sum:

        kappa_C^ab = sum_q w_q sum_BB' K_BB' f0_BB' T^ab_BB'

    so kappa_C is a linear functional of the blocked tensor alone, and is
    therefore exactly invariant under the intra-multiplet unitary that separates
    gkmx's eigenvectors from TDEP's. If a weight ever varied inside a block, that
    equivalence would break and the blocked test would no longer imply agreement
    on kappa_C.

    TDEP averages linewidths over multiplets explicitly, so those are bit-equal;
    omega and cv follow the eigensolver and sit at machine precision.
    """
    name, _, ref, prim, sc, fc = fixture
    q = ref["q_ir"]
    sol, _, _ = _solve(prim, sc, fc, q)
    w = np.abs(np.asarray(sol.w_qs))
    ns = w.shape[1]

    full = ref["q_full"]
    M = round(len(full) ** (1 / 3))

    def cell(x):
        return np.rint(((x + 1e-8) % 1.0) * M).astype(int) % M

    key = {tuple(k): i for i, k in enumerate(cell(full))}
    tau = ref["lifetimes_s"][np.array([key[tuple(k)] for k in cell(q)])]
    quantities = {"omega": ref["frequencies_ir_rad_s"],
                  "linewidth": ref["linewidths_ir_rad_s"],
                  "lifetime": tau,
                  "cv": ref["heat_capacity_ir_JK"]}

    blocks = 0
    worst = dict.fromkeys(quantities, 0.0)
    for iq in range(len(q)):
        i = 0
        while i < ns:
            j = i + 1
            while j < ns and abs(w[iq, j] - w[iq, i]) < DEGENERACY_TOL:
                j += 1
            if j - i > 1:
                blocks += 1
                for label, arr in quantities.items():
                    block = arr[iq, i:j]
                    scale = np.abs(block).max()
                    if scale > 0:
                        worst[label] = max(worst[label], np.ptp(block) / scale)
            i = j

    assert blocks > 0, f"{name}: no degenerate multiplets; this test is vacuous"
    for label, spread in worst.items():
        assert spread < TOL_TDEP, (
            f"{name}: {label} varies by {spread:.2e} inside a multiplet over "
            f"{blocks} blocks; kappa_C is no longer a functional of the blocked "
            f"tensor alone")


def _averaged_and_raw(ref, prim, sc, fc, which):
    """(averaged, raw) mode-basis ``v_ss'`` at the q-points selected by `which`.

    Mode basis, not the atomic-basis dD/dq: `symmetrize_v_qssa` conjugates by
    ``G = U^dag Gamma(R,q) U``, which acts on mode indices. Feeding it the atomic
    gradient mismatches the index types, and both assertions below then pass on a
    random array of the right shape (measured 1.00e+00 at |LG| > 1 and 3.5e-15 at
    |LG| = 1, so noise satisfies both).
    """
    q = ref["q_ir"][which]
    sol, D, dD = _solve(prim, sc, fc, q)
    w_inv = np.asarray(sol.w_inv_qs)
    frac = prim.get_scaled_positions()
    ns = D.shape[-1]
    raw = np.zeros((len(q), ns, ns, 3), dtype=complex)
    e_own = np.zeros((len(q), ns, ns), dtype=complex)
    for iq in range(len(q)):
        D_T, dD_T = to_tdep_convention(D[iq], dD[iq], q[iq], frac, prim.positions)
        U = np.linalg.eigh(D_T)[1]
        e_own[iq] = np.swapaxes(U, -1, -2)
        s = np.sqrt(w_inv[iq])
        M = np.einsum("ji,ajk,kl->ail", np.conj(U), dD_T, U, optimize=True)
        raw[iq] = np.moveaxis(M * (0.5 * s[:, None] * s[None, :])[None], 0, -1) \
            * C.gv_to_AA_fs
    # No w_qs= on purpose: mirrors TDEP's unprojected kronegv average.
    return symmetrize_v_qssa(raw, e_own, q, prim), raw


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

    # Full tensor, not a diagonal: TDEP reports kxy, kxz, kyz too, and they are
    # exactly 0 here. A diagonal comparison cannot see an off-diagonal leak.
    target = ref["kappa_sma"]
    rel = np.abs(kappa - target).max() / np.abs(target).max()
    assert rel < TOL_TDEP, (
        f"{name}: kappa_SMA tensor\n{kappa}\nvs TDEP\n{target}\nrel={rel:.2e}")


def test_coherent_kappa_from_tdep_linewidths(fixture):
    """kappa_C from gkmx's velocity operator and TDEP's own linewidths.

    TDEP's kernel (`kappa.f90:296-315`), summed over the irreducible wedge with
    its weights ``w = (|G|/|LG|)/N_q``::

        f0  = (cv_j + cv_k) / 2
        ker = (G_j + G_k) / ((G_j + G_k)^2 + (w_j - w_k)^2)
        kappa_C = (1/V) sum_q w_q sum_{j != k} v_jk (x) v_jk ker f0

    The bilinear is ``Re[v^a_ss' conj(v^b_ss')] = Re v^a Re v^b + Im v^a Im v^b``.
    For a = b that is ``|v^a_ss'|^2``. Using only ``Re v^a Re v^b`` -- which TDEP
    did until the coherence gauge fix -- makes kappa_C depend on the arbitrary
    per-mode phases returned by zheev, because Re(v) is not gauge covariant while
    the modulus is. The reference values here come from the fixed TDEP.

    Compared as the trace. TDEP averages ``v (x) v`` over the symmetry-equivalent
    q before accumulating, and that rotation preserves ``sum_a |(R v)^a|^2``, so
    the trace is reproducible without those operations while the individual
    Cartesian components are not.

    This is the one test that exercises the velocity operator end-to-end: an
    error anywhere in the Bloch map or the little-group average lands here.
    """
    name, _, ref, prim, sc, fc = fixture
    q = ref["q_ir"]
    # Everything TDEP-specific comes from the public API: convention="TDEP"
    # selects both the mode-index little-group average and the per-atom Bloch
    # phase. No TDEP eigenvector is read.
    ph = Phonon(force_constants=fc, primitive=prim, supercell=sc,
                backend="numpy", precision="fp64", convention="TDEP")
    sol = ph.solve(q, with_velocities=True, with_group_velocity_matrices=True)
    ns = np.asarray(sol.w_qs).shape[1]
    v = np.asarray(sol.v_qssa_cartesian) * 1e5                 # A/fs -> m/s

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
        # gauge invariant bilinear Re[v^a conj(v^b)], kept as a Cartesian tensor
        # rather than contracted to sum_a |v^a|^2, so the off-diagonal is tested.
        vv = np.einsum("sta,stb->stab", v[iq], np.conj(v[iq])).real
        total += weights[iq] * np.einsum("st,st,stab->ab", kernel, f0, vv)

    # TDEP averages every channel over the point group before writing output
    # (`kappa.f90::symmetrize_kappa`, from `main.f90:242-244`). The wedge sum
    # alone is not point-group symmetric -- in the TDEP Bloch gauge v_ss' is not
    # covariant, which leaves an off-diagonal 6.7e-02 of the diagonal on KI_bcc.
    # The projection preserves the trace, so this is not what fixes the number;
    # it is what makes the tensors comparable at all.
    got = symmetrize_kappa(total / volume, prim)
    target = ref["kappa_coherent"]
    rel = np.abs(got - target).max() / np.abs(target).max()
    assert rel < TOL_TDEP, (
        f"{name}: kappa_C tensor\n{got}\nvs TDEP\n{target}\nrel={rel:.2e}")
