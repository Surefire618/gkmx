"""Batched phonon solver with numpy / jax backends."""

import collections

import numpy as np
import spglib

from . import _constants as C
from . import masses as _masses
from ._log import warn
from .brillouin import little_group
from .lattice_points import get_p2s_map, get_s2p_map, get_smallest_vectors
from .space_group import space_group_invariance

ASR_TOL = 1e-4    # warn above this relative sum-rule residual


def translational_invariance(fc, primitive, supercell,
                             asr_tol=ASR_TOL, tol=1e-5):
    """Impose the acoustic sum rule on ``fc``; returns ``(fc, rel_residual)``.

        sum_B Phi[i][B] = 0:    Phi[i][I(i, R=0)] -= sum_B Phi[i][B]

    The correction sits on the origin block, so it enters D(q) with phase 1
    at every q: Gamma is repaired exactly and nothing else moves.

    Args:
        fc: ``(N_p, N_sc, 3, 3)`` force constants, not mass-weighted.
        primitive, supercell: ASE Atoms; locate the origin block.
        asr_tol: warn when the residual exceeds this fraction of ``max|Phi|``.
        tol: position tolerance for the origin-block lookup.
    """
    fc = np.asarray(fc)
    residual = fc.sum(axis=1)
    abs_residual = float(np.abs(residual).max())
    scale = float(np.abs(fc).max())
    rel_residual = abs_residual / scale if scale > 0 else 0.0

    out = fc.copy()
    for p, I in enumerate(get_p2s_map(primitive, supercell, tol=tol)):
        out[p, I] -= residual[p]

    if rel_residual > asr_tol:
        warn(f"acoustic sum rule violated: residual {abs_residual:.3e} eV/A^2 "
             f"({rel_residual:.2e} of max|Phi|); removing it from the origin block "
             f"(does not fix degeneracies away from Gamma).", prefix="gkmx.phonon")
    return out, rel_residual




Solution = collections.namedtuple(
    "Solution",
    ("w_qs", "w_inv_qs", "w2_qs", "v_qsa_cartesian", "e_qsi", "D_qij"),
)
"""Phonon eigensolution at a batch of q-points (q-first index layout).

Fields:
    w_qs:             ``(Nq, Ns)`` real, frequencies ``sign(ev) * sqrt(|ev|)``
                      in ASE-native units (no ``/(2π)``). Soft / imaginary
                      modes are negative; near-zero acoustic Gamma modes
                      are zeroed.
    w_inv_qs:         ``(Nq, Ns)`` real, ``1/w_qs`` with the near-zero
                      modes set to 0 (safe to multiply against).
    w2_qs:            ``(Nq, Ns)`` real, ``sign(w_qs) * w_qs**2`` — i.e. the
                      eigenvalues of D(q) with the sign preserved.
    v_qsa_cartesian:  ``(Nq, Ns, 3)`` real, per-mode group velocity in Å/fs.
    e_qsi:            ``(Nq, Ns, Ns)`` complex, row-eigenvectors of D(q)
                      in the primitive basis (``e[q, s, i]`` = i-th basis
                      component of the s-th mode at q).
    D_qij:            ``(Nq, Ns, Ns)`` complex, mass-weighted dynamical
                      matrix D(q).
"""

SolutionWithGVM = collections.namedtuple(
    "SolutionWithGVM",
    ("w_qs", "w_inv_qs", "w2_qs", "v_qsa_cartesian",
     "v_qssa_cartesian", "e_qsi", "D_qij"),
)
"""``Solution`` plus the QHGK off-diagonal group-velocity matrix.

Adds one field on top of ``Solution``:
    v_qssa_cartesian: ``(Nq, Ns, Ns, 3)`` complex, off-diagonal group-
        velocity matrix in Å/fs. Diagonal entries equal
        ``v_qsa_cartesian`` (real part). Used by ``get_kappa_QHGK``.
"""


def _solve_kernel(xp, eigh_fn, fc_mw, j_of_k, svec_frac, multi_mask,
                  q_frac, pcell_cart, N_p, dtype_complex):
    """D(q), dD/dq, and both in the mode basis, on a q-grid.

    Smallest-vectors Bloch convention (matches phonopy on non-primitive
    supercells; `lattice_points.get_smallest_vectors`):

        D_{ia,jb}(q) = sum_{K in j} Phi~_{iK}^{ab} (1/N_iK) sum_v exp(2 pi i q . s_iKv)
        D e_s        = w_s^2 e_s
        dD/dq_a      = sum_K Phi~ (1/N_iK) sum_v 2 pi i s_v^a exp(2 pi i q . s_v)
        M_a          = e^dag (dD/dq_a) e

    ``s`` enters the derivative Cartesian, so the 2 pi survives and cancels
    the 1/2pi in ``gv_to_AA_fs``.

    Args:
        xp, eigh_fn: array namespace (numpy / jax.numpy) + its Hermitian solver.
        fc_mw: ``(N_p, N_sc, 3, 3)`` mass-weighted FC, eV/(A^2 amu).
        j_of_k: ``(N_sc,)`` primitive index of each supercell atom.
        svec_frac, multi_mask: ``(N_sc, N_p, max_multi, 3)`` smallest vectors
            (primitive-fractional) + occupancy; interior-only lattice points
            are wrong for the 30-58 % of pairs with multi > 1.
        q_frac: ``(Nq, 3)`` fractional q, 2pi-free.
        pcell_cart: primitive lattice rows [A].
        N_p: primitive atom count (``Ns = 3 N_p``).
        dtype_complex: from the precision switch.

    Returns:
        ``w2 (Nq, Ns)`` (negative = soft mode), ``e (Nq, Ns, Ns)`` row
        eigenvectors, ``M (Nq, 3, Ns, Ns)`` with
        ``v_s^a = Re M[q,a,s,s] / (2 w_s)``, ``D_q (Nq, Ns, Ns)`` Hermitized.
    """
    two_pi_j = xp.asarray(2j * np.pi, dtype=dtype_complex)

    phase_all = xp.exp(
        two_pi_j * xp.einsum("qa,kiva->qkiv", q_frac, svec_frac).astype(dtype_complex)
    )
    phase_all = phase_all * multi_mask[None, :, :, :].astype(dtype_complex)
    multi_count = multi_mask.sum(axis=-1)
    multi_safe = xp.where(multi_count > 0, multi_count, 1.0)
    avg_phase = phase_all.sum(axis=-1) / multi_safe[None, :, :]

    K_to_j = (j_of_k[:, None] == xp.arange(N_p)[None, :]).astype(dtype_complex)

    D_q = xp.einsum(
        "qki,ikab,kj->qiajb",
        avg_phase, fc_mw.astype(dtype_complex), K_to_j,
        optimize=True,
    )
    Ns = 3 * N_p
    D_q = D_q.reshape(-1, Ns, Ns)
    # Hermitize before eigh — fp rounding can bleed ~eps asymmetry.
    D_q = 0.5 * (D_q + xp.conj(xp.swapaxes(D_q, -1, -2)))

    w2, e = eigh_fn(D_q)
    e = xp.swapaxes(e, -1, -2)

    svec_cart = xp.einsum("kiva,ab->kivb", svec_frac, pcell_cart)
    dphase_all = (
        two_pi_j * phase_all[:, :, :, :, None]
        * svec_cart[None, :, :, :, :].astype(dtype_complex)
    )
    dphase_avg = dphase_all.sum(axis=3) / multi_safe[None, :, :, None]

    dD_dq = xp.einsum(
        "qkia,ikbc,kj->qaibjc",
        dphase_avg, fc_mw.astype(dtype_complex), K_to_j,
        optimize=True,
    )
    dD_dq = dD_dq.reshape(-1, 3, Ns, Ns)
    # Matches phonopy's derivative_dynmat.c:109-123: the pair-wise multi-
    # averaged formula leaves a ~1e-3 non-Hermitian residual that drifts
    # the off-diagonal v_qssa up to 4e-2 relative if not Hermitized.
    dD_dq = 0.5 * (dD_dq + xp.conj(xp.swapaxes(dD_dq, -1, -2)))

    M = xp.einsum("qjn,qanm,qkm->qajk", xp.conj(e), dD_dq, e, optimize=True)

    return w2, e, M, D_q


def _numpy_solve(fc_mw, j_of_k, svec_frac, multi_mask, q_frac, pcell_cart, N_p,
                 *, dtype_real, dtype_complex):
    return _solve_kernel(
        np, np.linalg.eigh,
        fc_mw, j_of_k, svec_frac, multi_mask, q_frac, pcell_cart, N_p,
        dtype_complex,
    )


_JIT_CACHE: dict = {}


def _jax_solve(fc_mw, j_of_k, svec_frac, multi_mask, q_frac, pcell_cart, N_p,
               *, dtype_real, dtype_complex):
    import jax
    import jax.numpy as jnp

    if dtype_real == np.float64 and not jax.config.read("jax_enable_x64"):
        jax.config.update("jax_enable_x64", True)

    jnp_real = jnp.float32 if dtype_real == np.float32 else jnp.float64
    jnp_complex = jnp.complex64 if dtype_real == np.float32 else jnp.complex128

    shape_key = (fc_mw.shape, svec_frac.shape, q_frac.shape, jnp_real)
    if shape_key not in _JIT_CACHE:
        def _kernel(fc_mw, j_of_k, svec_frac, multi_mask, q_frac, pcell_cart):
            return _solve_kernel(
                jnp, jnp.linalg.eigh,
                fc_mw, j_of_k, svec_frac, multi_mask, q_frac, pcell_cart, N_p,
                jnp_complex,
            )
        _JIT_CACHE[shape_key] = jax.jit(_kernel)

    w2, e, M, D_q = _JIT_CACHE[shape_key](
        jnp.asarray(fc_mw, dtype=jnp_real),
        jnp.asarray(j_of_k, dtype=jnp.int64),
        jnp.asarray(svec_frac, dtype=jnp_real),
        jnp.asarray(multi_mask, dtype=jnp_real),
        jnp.asarray(q_frac, dtype=jnp_real),
        jnp.asarray(pcell_cart, dtype=jnp_real),
    )
    return (
        np.asarray(w2, dtype=dtype_real),
        np.asarray(e, dtype=dtype_complex),
        np.asarray(M, dtype=dtype_complex),
        np.asarray(D_q, dtype=dtype_complex),
    )


def _symmetrize_v_site(v_qsa, q_frac, rots_frac, recip_lattice, tol=1e-5):
    """Phonopy-style site-symmetrize per-mode group velocities."""
    real_dtype = v_qsa.real.dtype
    recip_lattice = recip_lattice.astype(real_dtype)
    recip_inv = np.linalg.inv(recip_lattice)
    r_cart = np.einsum("ij,ojk,kl->oil",
                       recip_lattice, rots_frac.astype(real_dtype), recip_inv)

    q_in_BZ = q_frac - np.rint(q_frac)
    rq = np.einsum("oij,qj->oqi", rots_frac, q_in_BZ)
    mask = np.all(np.abs(rq - q_in_BZ[None, :, :]) < tol, axis=-1)
    count = mask.sum(axis=0)
    count_safe = np.where(count > 0, count, 1).astype(real_dtype)

    out = np.einsum("oq,oba,qsa->qsb",
                    mask.astype(v_qsa.dtype), r_cart.astype(v_qsa.dtype), v_qsa,
                    optimize=True)
    out /= count_safe[:, None, None]
    if (count == 0).any():
        empty = count == 0
        out[empty] = v_qsa[empty]
    return out


# Degenerate perturbation theory: the physical multiplet basis diagonalizes
# dD/dq . probe for a generic direction; the probe matches phonopy's
# (group_velocity.py:157) so v_qssa is comparable to theirs.
_PERTURB_PROBE = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)


DEGENERACY_TOL = 1e-4 / C.omega_to_THz   # matches phonopy degenerate_sets cutoff

# A convention decides the eigenvector basis inside degenerate multiplets,
# v_qsa there, and v_qssa off the diagonal; frequencies and masks are
# identical in all three, and diag(v_qssa) == v_qsa holds in all three.
# RAW: eigh basis and native formulas, nothing applied.
# TDEP: TDEP gauge (D_T = P D* P^dag, gkmx/tdep.py), Gamma(R, q) mode
#   average + branch alignment; reproduces TDEP's stored gv, velocity
#   operator and kappa_C (1.000000 on the trio fixtures).
# PHONO3PY: multiplets rotated to diagonalize dD/dq . probe, Cartesian
#   site average on v_qsa and on the whole v_qssa (phono3py-Kubo,
#   group_velocity_matrix.py); reproduces phonopy's gv bit-level, at the
#   cost of suppressed inter-band coherence.
CONVENTIONS = ("TDEP", "PHONO3PY", "RAW")


def degenerate_sets(w_qs, tol=DEGENERACY_TOL):
    """Yield ``(q, start, stop)`` mode ranges holding 2+ degenerate modes.

    Frequencies must be ascending within each q, as eigh returns them. Shared
    by the basis rotation and by the multiplet averaging in greenkubo so the
    two cannot disagree about what counts as degenerate.
    """
    w = np.abs(np.asarray(w_qs))
    Nq, Ns = w.shape
    for qi in range(Nq):
        start = 0
        for si in range(1, Ns + 1):
            if si == Ns or abs(w[qi, si] - w[qi, start]) > tol:
                if si - start > 1:
                    yield qi, start, si
                start = si


def _align_degenerate_branches(w2, e_qsi, v_qssa, v_qsa):
    """Fix the eigh gauge inside each degenerate block ``B``.

    Diagonalize the averaged velocity block ``V_B^a = v_qssa[q, B, B, a]``
    with one unitary ``Q`` (components ``a`` in decreasing-structure order,
    later ones acting inside the earlier ones' degenerate subgroups), then

        e_B            <-  Q^T e_B
        v_qssa[q,B,:]  <-  Q^dag v_qssa[q,B,:]
        v_qssa[q,:,B]  <-  v_qssa[q,:,B] Q
        v_qsa[q,B]     <-  diag(Q^dag V_B Q)

    so eigenvectors, operator and per-mode velocities describe the same
    physical branches (a TRS band-sticking pair keeps its +-c slopes,
    not their flattened mean).

    Scalar blocks (``V_B^a = c_a 1``): ``Q = 1``, ``v_qsa <- c`` (trace
    mean). Singletons keep the raw Hellmann-Feynman ``v_qsa``, which the
    blockmasked average preserves (``|Gamma_ss| = 1``) -- the
    diag(v_qssa) == v_qsa test pins that theorem. Blocks mixing live and
    dead modes are skipped.

    Returns ``(e_qsi, v_qssa, v_qsa)``.
    """
    e_qsi = np.array(e_qsi, copy=True)
    v_qssa = np.array(v_qssa, copy=True)
    v_qsa = np.array(v_qsa, copy=True)
    # Global scale + absolute floor: a per-q relative guard lets eigh
    # rotate pure noise on all-TRIM grids where the operator is ~1e-16.
    scale = max(float(np.abs(v_qssa).max()), 1e-12)
    tol = max(1e-9 * scale, 1e-14)
    for qi, i0, i1 in degenerate_sets(np.sqrt(np.abs(w2))):
        B_avg = np.asarray(v_qssa[qi, i0:i1, i0:i1, :], dtype=np.complex128)
        mb = i1 - i0
        live = np.abs(v_qssa[qi, i0:i1, i0:i1]).sum(axis=(1, 2)) > 0
        spreads = [float(np.abs(B_avg[:, :, a] - np.trace(B_avg[:, :, a])
                                / mb * np.eye(mb)).max()) for a in range(3)]
        if live.any() and not live.all():
            continue
        if max(spreads) <= tol:
            # scalar block: no preferred basis, and the averaged diagonal is
            # the basis-independent block mean (= TDEP's trace-mean closure;
            # the raw eigh-basis diagonals spread around it arbitrarily)
            v_qsa[qi, i0:i1, :] = np.einsum(
                "ssa->sa", B_avg).real
            continue
        Q = np.eye(mb, dtype=np.complex128)
        groups = [np.arange(mb)]
        # decreasing-structure order: an eigh of a noise-level component
        # would lock in a garbage basis before the real structure is seen
        for a in np.argsort(spreads)[::-1]:
            Ba = Q.conj().T @ B_avg[:, :, a] @ Q
            newgroups = []
            for g in groups:
                sub = 0.5 * (Ba[np.ix_(g, g)] + Ba[np.ix_(g, g)].conj().T)
                if len(g) == 1 or float(np.abs(
                        sub - np.trace(sub) / len(g) * np.eye(len(g))).max()) \
                        <= tol:
                    newgroups.append(g)
                    continue
                ev, V = np.linalg.eigh(sub)
                Q[:, g] = Q[:, g] @ V
                split, start = [], 0
                for k in range(1, len(g) + 1):
                    if k == len(g) or abs(ev[k] - ev[start]) > tol:
                        split.append(g[start:k])
                        start = k
                newgroups.extend(split)
            groups = newgroups
        # co-rotate operator rows/columns and eigenvector rows; block
        # velocities are branch expectations of the symmetrized operator --
        # inside an exact degeneracy no independent raw readout exists.
        v_qssa[qi, i0:i1, :, :] = np.einsum(
            "ts,tja->sja", Q.conj(), v_qssa[qi, i0:i1, :, :])
        v_qssa[qi, :, i0:i1, :] = np.einsum(
            "jta,ts->jsa", v_qssa[qi, :, i0:i1, :], Q)
        e_qsi[qi, i0:i1, :] = np.einsum("ts,ti->si", Q, e_qsi[qi, i0:i1, :])
        v_qsa[qi, i0:i1, :] = np.einsum(
            "qqa->qa", v_qssa[qi, i0:i1, i0:i1, :]).real
    return e_qsi, v_qssa, v_qsa


def eigenvector_transformation(n, q_frac, atoms, rotations_frac, translations_frac,
                                perm):
    """``Gamma(R, q)``, the representation of R on the eigenvector space.

        e_s(Rq)             = Gamma(R, q) e_s(q)
        Gamma[a1, a2]       = R_cart exp(+2 pi i q . v0),   a1 = R(a2)
        v0                  = R^-1 (r_a1 - t) - r_a1
        sum_b R_ab v^b(q)   = Gamma^dag v^a(q) Gamma

    Unitary; commutes with D(q) for R in L(q), so frequency-block-diagonal.
    The phase uses only r_a1 (phonopy's irreps.py form): a lattice-vector
    wrap shifts it by exp(2 pi i G . N) = 1 for every little-group member
    including umklapp. The r_a2 form broke commutation O(1) on out-of-cell
    geometries.

    Args:
        n: index into ``rotations_frac`` / ``translations_frac`` (spglib,
            fractional).
        q_frac: q-point, fractional, 2pi-free.
        atoms: primitive cell (lattice + positions ``r_a``).
        perm: ``perm[a2] = R(a2)``.

    Returns the ``(3 N_p, 3 N_p)`` unitary in the untransformed Bloch gauge;
    ``symmetrize_v_qssa`` conjugates it into the operator's gauge.
    """
    A = np.asarray(atoms.cell)
    Ainv = np.linalg.inv(A)
    R = A.T @ rotations_frac[n] @ np.linalg.inv(A.T)
    t = translations_frac[n] @ A
    pos = np.asarray(atoms.positions)
    na = len(atoms)

    # fp64 pin: Gamma is exact symmetry data, not a precision-switched
    # buffer; consumers downcast on accumulation (symmetrize_v_qssa's acc).
    G = np.zeros((3 * na, 3 * na), dtype=np.complex128)
    Rinv = np.linalg.inv(R)
    for a2 in range(na):
        a1 = perm[a2]
        v0 = Rinv @ (pos[a1] - t) - pos[a1]
        ph = np.exp(2j * np.pi * np.dot(v0 @ Ainv, q_frac))
        G[3 * a1:3 * a1 + 3, 3 * a2:3 * a2 + 3] = R * ph
    return G


def symmetrize_v_qssa(v_qssa, e_qsi, q_points_frac, atoms, symprec=1e-5,
                      w_qs=None):
    """Average the velocity operator over the little group of each q.

        v^a  <-  < G_R^dag v^a G_R >_{R in L(q)},    G_R = U^dag Gamma(R, q) U

    The projector onto the L(q)-invariant part of v: Schur annihilates every
    inter-irrep block, so the surviving off-diagonal couples multiplets that
    share an irrep (exactly zero at ``|L(q)| >= 12`` on the cubic fixtures).
    Gamma mixes only degenerate modes, so the ``1/(2 sqrt(w_s w_s'))`` folded
    into ``v_qssa`` commutes through; inside a multiplet the diagonal is
    mixed too (a different projector from ``_symmetrize_v_site``).

    Args:
        v_qssa: ``(Nq, Ns, Ns, 3)``, MUST be in the TDEP gauge
            (``_to_tdep_bloch`` output); Gamma is conjugated into it.
        e_qsi: ``(Nq, Ns, Ns)`` row eigenvectors (build ``G_R``).
        q_points_frac: ``(Nq, 3)`` fractional, 2pi-free.
        atoms: primitive cell; its space group supplies L(q).
        symprec: spglib tolerance.
        w_qs: optional ``(Nq, Ns)``; pass unthresholded ``sqrt(|w2|)`` (the
            ``_align_degenerate_branches`` partition). When given, G is
            projected frequency-block-diagonal before averaging: exact Gamma
            couples only equal eigenvalues, and cross-block elements are
            ``~ |U^dag [Gamma_T, D_T] U| / |w2_s - w2_s'|`` leakage.

    Returns ``(Nq, Ns, Ns, 3)``.
    """
    A = np.asarray(atoms.cell)
    frac = atoms.get_scaled_positions()
    ds = spglib.get_symmetry((A, frac, atoms.get_atomic_numbers()), symprec=symprec)
    W, w = ds["rotations"], ds["translations"]

    na = len(atoms)
    perms = np.zeros((len(W), na), dtype=int)
    for n in range(len(W)):
        dd = (frac @ W[n].T + w[n])[:, None, :] - frac[None, :, :]
        dd -= np.rint(dd)
        perms[n] = np.argmin(np.linalg.norm(dd @ A, axis=-1), axis=1)
        if len(set(perms[n].tolist())) != na:
            raise ValueError(f"symmetry operation {n} does not permute the atoms")

    out = np.array(v_qssa, copy=True)
    Ns = out.shape[1]
    masks_of_q = None
    if w_qs is not None:
        masks_of_q = np.broadcast_to(np.eye(Ns, dtype=bool),
                                     (out.shape[0], Ns, Ns)).copy()
        for qi, start, stop in degenerate_sets(w_qs):
            masks_of_q[qi, start:stop, start:stop] = True
    for iq, q in enumerate(np.asarray(q_points_frac)):
        ops = little_group(q, W, A) or [None]
        U = np.swapaxes(np.asarray(e_qsi[iq]), -1, -2)      # columns = modes
        # Conjugate Gamma into the TDEP gauge with the same wrapped-position
        # representative as `_to_tdep_bloch`: mixed representatives break
        # commutation O(1) on non-symmorphic groups.
        P = np.repeat(np.exp(-2j * np.pi * (frac @ q)), 3)
        blockmask = None if masks_of_q is None else masks_of_q[iq]
        acc = np.zeros_like(out[iq])
        for n in ops:
            if n is None:
                acc += out[iq]
                continue
            g0 = eigenvector_transformation(n, q, atoms, W, w, perms[n])
            g_T = P[:, None] * np.conj(g0) * np.conj(P)[None, :]
            G = U.conj().T @ g_T @ U
            if blockmask is not None:
                G = np.where(blockmask, G, 0.0)
            acc += np.einsum("ji,jka,kl->ila", G.conj(), out[iq], G, optimize=True)
        out[iq] = acc / len(ops)
    return out

def _to_tdep_bloch(w2, e, M, q_frac, atoms):
    """Map row eigenvectors and mode-basis ``dD/dq`` into TDEP's Bloch convention.

    With ``P = diag(exp(-2 pi i q . r_a))`` and ``D_T = P D* P^dag`` the columns
    follow as ``U_T = P conj(U)``, hence in row form ``e_T = P conj(e)``. For the
    gradient, writing ``R^a = U^T r_a conj(U)``,

        M_T^a = conj(M^a) - 2 pi i R^a_{ss'} (w2_s' - w2_s)

    The commutator term is the whole difference between the conventions. It
    vanishes when ``w2_s == w2_s'``, so the diagonal group velocity is common to
    both and only the off-diagonal moves.
    """
    frac = np.asarray(atoms.get_scaled_positions())
    # r MUST be the Cartesian image of the same wrapped representative as
    # the phases (dP/dq = -2 pi i r P); raw positions here make this the
    # gradient of no gauge -- O(1) wrong on out-of-cell geometries.
    r = np.repeat(frac @ np.asarray(atoms.cell), 3, axis=0)
    q = np.asarray(q_frac, dtype=float)
    # The fp64 phases and the 2j*pi scalar promote fp32 inputs; cast at the
    # boundary so e/M keep the precision-switch dtype they arrived with.
    phase = np.exp(-2j * np.pi * (frac @ q.T)).T.astype(e.dtype)   # (Nq, N_p)
    P = np.repeat(phase, 3, axis=1)                       # (Nq, 3 N_p)

    e_T = P[:, None, :] * np.conj(e)
    Rq = np.einsum("qsi,ia,qti->qast", e, r.astype(e.real.dtype), np.conj(e),
                   optimize=True)
    dw2 = w2[:, None, :] - w2[:, :, None]                 # w2_s' - w2_s
    M_T = np.conj(M) - (2j * np.pi * Rq * dw2[:, None, :, :]).astype(M.dtype)
    return e_T, M_T


def _rotate_degenerate_subspaces(w2, e, M, probe=_PERTURB_PROBE):
    """Rotate eigh's arbitrary degenerate basis to phonopy's _perturb_D convention."""
    e = np.array(e, copy=True)
    M = np.array(M, copy=True)

    for qi, start, stop in degenerate_sets(np.sqrt(np.abs(w2))):
        G = slice(start, stop)
        M_probe = (probe[0] * M[qi, 0, G, G]
                   + probe[1] * M[qi, 1, G, G]
                   + probe[2] * M[qi, 2, G, G])
        M_probe = 0.5 * (M_probe + M_probe.conj().T)
        _, U = np.linalg.eigh(M_probe)
        e[qi, G, :] = U.T @ e[qi, G, :]
        for a in range(3):
            M[qi, a, G, :] = U.conj().T @ M[qi, a, G, :]
            M[qi, a, :, G] = M[qi, a, :, G] @ U
    return e, M


class Phonon:
    """Batched phonon solver for a fixed FC + primitive/supercell pair.

    Standalone with respect to the rest of gkmx: depends only on numpy,
    ASE, and (optionally) jax. No xarray / h5py / phonopy at runtime.

    The eigensolution at each q is built under the smallest-vectors
    Bloch convention (multi-image-averaged phases at boundary atoms;
    matches phonopy at machine precision on cubic + non-cubic fixtures).
    """

    def __init__(self, force_constants, primitive, supercell,
                 backend="numpy", precision="fp64", p2s_map=None, factor=1.0,
                 enforce_translational_invariance=True,
                 enforce_space_group=True, convention="TDEP"):
        """Build the solver.

        Args:
            force_constants: ``(N_p, N_sc, 3, 3)``, not mass-weighted
                (``gkmx.io.parse_force_constants`` loads the file formats).
            primitive, supercell: ASE Atoms.
            backend: ``"numpy"`` (default) or ``"jax"`` (lazily imported).
            precision: ``"fp64"`` (default, machine parity with phonopy)
                or ``"fp32"`` (~1e-6 on basis-invariant quantities).
            p2s_map: optional ``(N_p,)`` supercell indices of the primitive
                basis; pins ``primitive.positions`` to
                ``supercell.positions[p2s_map]`` for phonopy-style builds
                that offset them by a lattice translation.
            factor: FC unit rescale to eV/A^2. Default 1.0; QE Ry/bohr^2:
                ``(108.97 / 15.633)**2 ~ 48.59``.
            enforce_translational_invariance: apply
                ``translational_invariance`` first. Default ``True``.
            enforce_space_group: apply ``space_group_invariance`` before
                solving. Default ``True``: the convention closures assume
                site symmetry, and fitted FCs that break it get corrupted
                averages instead (KI_B2_MLIP diag identity: 2.1e-1 raw,
                1.6e-15 projected). Off = solve the FCs exactly as given.
            convention: ``"TDEP"`` (default), ``"PHONO3PY"``, or ``"RAW"``;
                decides the degenerate-multiplet basis, ``v_qsa`` there,
                and the off-diagonal ``v_qssa``. See ``CONVENTIONS``.
        """
        from .precision import Precision
        if backend not in ("numpy", "jax"):
            raise ValueError(f"Unknown backend: {backend!r}. Valid: 'numpy', 'jax'.")
        p = Precision.from_str(precision)
        dtype_real, dtype_complex = p.real, p.complex

        # phonopy-style builds can offset primitive positions by a lattice
        # translation; pin to the exact supercell images.
        if p2s_map is not None:
            p2s_map = np.asarray(p2s_map, dtype=np.int64)
            if p2s_map.shape != (len(primitive),):
                raise ValueError(
                    f"p2s_map must have shape ({len(primitive)},), got {p2s_map.shape}"
                )
            primitive = primitive.copy()
            primitive.positions = np.asarray(supercell.positions)[p2s_map]

        convention = str(convention).upper()
        if convention not in CONVENTIONS:
            raise ValueError(
                f"convention must be one of {CONVENTIONS}, got {convention!r}")
        self.primitive = primitive
        self.supercell = supercell
        self.backend = backend
        self.precision = precision
        self.convention = convention
        self._dtype_real = dtype_real
        self._dtype_complex = dtype_complex

        # Reciprocal-space rotations act as R^{-T} of spglib's real-space
        # rotations; -R augmentation captures time reversal (matches
        # phonopy.Symmetry.reciprocal_operations, non-magnetic default).
        spg = spglib.get_symmetry(
            (np.asarray(primitive.cell), primitive.get_scaled_positions(),
             primitive.numbers), symprec=1e-5)
        rots = np.asarray(spg["rotations"])
        recip_rots = np.array([np.linalg.inv(r).T for r in rots])
        recip_ops = np.concatenate([recip_rots, -recip_rots], axis=0)
        # Centrosymmetric groups duplicate (-r ≡ r' for some r').
        _, uniq = np.unique(np.round(recip_ops, 6).reshape(len(recip_ops), -1),
                             axis=0, return_index=True)
        self._symm_rots_frac = recip_ops[np.sort(uniq)]

        svec_frac, multi = get_smallest_vectors(primitive, supercell)
        j_of_k = get_s2p_map(primitive, supercell)

        if enforce_translational_invariance:
            force_constants, self.asr_residual = translational_invariance(
                force_constants, primitive, supercell)
        else:
            self.asr_residual = None
        if enforce_space_group:
            force_constants, self.space_group_residual = space_group_invariance(
                force_constants, primitive, supercell)
        else:
            self.space_group_residual = None

        self.force_constants = np.array(force_constants, copy=True)

        masses = np.asarray(_masses.of(primitive), dtype=self._dtype_real)
        fc_np = np.asarray(force_constants, dtype=self._dtype_real) * self._dtype_real(factor)
        mj = masses[j_of_k]
        mm_inv = 1.0 / np.sqrt(masses[:, None] * mj[None, :])
        fc_mw = fc_np * mm_inv[:, :, None, None]

        self._fc_mw = fc_mw
        self._svec_frac = svec_frac.astype(self._dtype_real)
        self._multi = multi
        self._j_of_k = j_of_k
        self._N_p = len(primitive)
        self._N_sc = len(supercell)
        self._Ns = 3 * self._N_p
        self._V_max = int(svec_frac.shape[2])
        # ASE's Cell.__array__ rejects non-float64 dtypes.
        self._pcell_cart = np.asarray(primitive.cell).astype(self._dtype_real)

        self._multi_mask = (
            np.arange(self._V_max)[None, None, :] < multi[:, :, None]
        ).astype(self._dtype_real)

    def solve(self, q_points_frac, with_velocities=True,
              with_group_velocity_matrices=False):
        """Solve the phonon eigenproblem at the given q-points.

        Args:
            q_points_frac: ``(Nq, 3)`` array of q-points in primitive
                reciprocal-fractional coordinates.
            with_velocities: include ``v_qsa_cartesian`` in the result.
                Set False to skip the dD/dq construction when only
                frequencies / eigenvectors are needed.
            with_group_velocity_matrices: include the QHGK off-diagonal
                group-velocity matrix ``v_qssa_cartesian``.

        Returns:
            A ``Solution`` (or ``SolutionWithGVM`` when
            ``with_group_velocity_matrices=True``). See those types for
            field shapes and units.
        """
        q_frac = np.asarray(q_points_frac, dtype=self._dtype_real)
        solve = {"numpy": _numpy_solve, "jax": _jax_solve}[self.backend]
        w2, e, M, D_q = solve(
            self._fc_mw, self._j_of_k, self._svec_frac,
            self._multi_mask, q_frac, self._pcell_cart, self._N_p,
            dtype_real=self._dtype_real,
            dtype_complex=self._dtype_complex,
        )
        return self._build_solution(
            w2, e, M, D_q, q_frac,
            with_velocities=with_velocities,
            with_group_velocity_matrices=with_group_velocity_matrices,
        )

    def _build_solution(self, w2, e, M, D_q, q_frac, *,
                        with_velocities, with_group_velocity_matrices):
        # Fix the degenerate subspaces before extracting per-mode
        # velocities; never flatten the operator itself before the average
        # (spurious +-c diagonals on antiunitary-paired irreps).
        if self.convention == "PHONO3PY":
            e, M = _rotate_degenerate_subspaces(w2, e, M)

        # After the degenerate handling, not before: the validated route fixes the
        # multiplet basis in gkmx's convention and then maps it over.
        if self.convention == "TDEP":
            e, M = _to_tdep_bloch(w2, e, M, q_frac, self.primitive)
            ph = np.repeat(
                np.exp(-2j * np.pi * (np.asarray(self.primitive.get_scaled_positions())
                                      @ np.asarray(q_frac, dtype=float).T)).T,
                3, axis=1).astype(D_q.dtype)
            D_q = ph[:, :, None] * np.conj(D_q) * np.conj(ph)[:, None, :]

        w_qs = np.sign(w2) * np.sqrt(np.abs(w2))

        # Scale near-zero cutoff off the 4th-smallest |w| so the three
        # Gamma acoustic modes don't set the threshold for everything else.
        flat = np.sort(np.abs(w_qs.flatten()))
        # Floor at sqrt(eps) of the working dtype: fp32 Gamma-acoustic
        # residuals sit above the order-statistics gate and would smear
        # into live modes through the TDEP average.
        thresh = max(flat[min(3, len(flat) - 1)] * 1e-5,
                     float(flat[-1]) * float(np.sqrt(np.finfo(self._dtype_real).eps)))
        w_qs = np.where(np.abs(w_qs) < thresh, 0.0, w_qs)
        w2_qs = np.sign(w_qs) * w_qs ** 2

        # w_inv needs 1/w safely bounded (not just zeroed); 1e20 stands in
        # for infinity at the looser cutoff (~4th-smallest |w| itself).
        w_inv = w_qs.copy()
        thresh_inv = flat[min(3, len(flat) - 1)] * 0.9
        w_inv[np.abs(w_inv) < thresh_inv] = 1e20
        w_inv_qs = 1.0 / w_inv
        w_inv_qs[np.abs(w_inv_qs) < 1e-9] = 0.0

        # Terminal dtype guards: these two silently promoted under fp32
        # when a gauge map used fp64 phases.
        e_qsi = e.astype(self._dtype_complex)
        D_qij = D_q.astype(self._dtype_complex)

        if not with_velocities:
            return Solution(
                w_qs=w_qs, w_inv_qs=w_inv_qs, w2_qs=w2_qs,
                v_qsa_cartesian=None, e_qsi=e_qsi, D_qij=D_qij,
            )

        # v[q, j, a] = Re<e_j|dD/dq_a|e_j> / (2 sqrt(|ev_j|)).
        sqrt_abs_ev = np.sqrt(np.abs(w2))
        sqrt_safe = np.where(sqrt_abs_ev > 1e-20, sqrt_abs_ev, 1.0)

        v_diag = np.einsum("qajj->qja", M)
        v_qsa = (v_diag.real / (2.0 * sqrt_safe[:, :, None])) * C.gv_to_AA_fs

        if self.convention == "PHONO3PY":
            recip_lat = np.linalg.inv(np.asarray(self.primitive.cell))
            v_qsa = _symmetrize_v_site(v_qsa, q_frac, self._symm_rots_frac, recip_lat)

        # Cutoff negative frequencies
        v_ok = w_qs > thresh
        v_qsa = np.where(v_ok[:, :, None], v_qsa, 0.0).astype(self._dtype_real)

        # TDEP defines v_qsa through the averaged block operator, so the
        # operator pipeline runs even without with_group_velocity_matrices:
        # both solve paths must return identical v_qsa.
        if not with_group_velocity_matrices and self.convention != "TDEP":
            return Solution(
                w_qs=w_qs, w_inv_qs=w_inv_qs, w2_qs=w2_qs,
                v_qsa_cartesian=v_qsa, e_qsi=e_qsi, D_qij=D_qij,
            )

        # QHGK off-diagonals use 2 sqrt(w_s w_s') in the denominator —
        # 2 w_s w_s' is a common mistake that agrees only on the diagonal.
        denom = 2.0 * np.sqrt(sqrt_safe[:, :, None] * sqrt_safe[:, None, :])
        M_moved = np.moveaxis(M, 1, -1)
        v_qssa = (M_moved / denom[:, :, :, None]) * C.gv_to_AA_fs

        # sum_b R_ab v^b = Gamma^dag v^a Gamma: the Cartesian average is
        # the diagonal case; off it TDEP averages the mode indices while
        # PHONO3PY applies the same site average to every (s, s') pair.
        band_mask = v_ok[:, :, None] & v_ok[:, None, :]
        if self.convention == "TDEP":
            # Mask first: unmasked acoustic rows carry 1/(2 sqrt(w w')) amplified
            # noise into the average. Re-masked below since the average repopulates.
            v_qssa = np.where(band_mask[:, :, :, None], v_qssa, 0.0)
            v_qssa = symmetrize_v_qssa(v_qssa, e_qsi, q_frac, self.primitive,
                                       w_qs=sqrt_abs_ev)
            # The average fixes the block but not the basis inside it. Align so that
            # one eigenvector is one physical branch on TRS-stuck pairs.
            e_qsi, v_qssa, v_qsa = _align_degenerate_branches(
                w2, e_qsi, v_qssa, v_qsa)
        elif self.convention == "PHONO3PY":
            nq, ns = v_qssa.shape[0], v_qssa.shape[1]
            v_qssa = _symmetrize_v_site(
                v_qssa.reshape(nq, ns * ns, 3), q_frac,
                self._symm_rots_frac, recip_lat).reshape(nq, ns, ns, 3)

        if not with_group_velocity_matrices:
            return Solution(
                w_qs=w_qs, w_inv_qs=w_inv_qs, w2_qs=w2_qs,
                v_qsa_cartesian=v_qsa, e_qsi=e_qsi, D_qij=D_qij,
            )

        v_qssa = np.where(band_mask[:, :, :, None], v_qssa, 0.0)
        v_qssa = v_qssa.astype(self._dtype_complex)

        return SolutionWithGVM(
            w_qs=w_qs, w_inv_qs=w_inv_qs, w2_qs=w2_qs,
            v_qsa_cartesian=v_qsa, v_qssa_cartesian=v_qssa,
            e_qsi=e_qsi, D_qij=D_qij,
        )
