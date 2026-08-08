"""Batched phonon solver with numpy / jax backends."""

import collections

import numpy as np
import spglib

from . import _constants as C
from . import masses as _masses
from ._log import warn
from .brillouin import little_group
from .lattice_points import get_p2s_map, get_s2p_map, get_smallest_vectors

ASR_TOL = 1e-4    # warn above this relative sum-rule residual


def translational_invariance(fc, primitive, supercell,
                             asr_tol=ASR_TOL, tol=1e-5):
    """Impose the acoustic sum rule (ASR) on ``fc``; returns ``(fc, residual)``.

    Translational invariance requires

        sum_B Phi[i][B] = 0        for every primitive atom i, each 3x3 block

    Enforced by removing the residual from the origin block:

        Phi[i][I(i, R=0)] -= sum_B Phi[i][B]
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
    """Dynamical matrix on a q-grid, its q-derivative, and both in the mode basis.

    D(q) uses the smallest-vectors Bloch convention: the phase of a pair is
    averaged over its ``N_iK`` equidistant periodic images rather than taken
    from one lattice point (`lattice_points.get_smallest_vectors`), which is
    what matches phonopy on non-primitive supercells --

        D_{ia,jb}(q) = sum_{K in j}  Phi_{iK}^{ab} / sqrt(m_i m_K)
                       * (1/N_iK) sum_v exp(2 pi i q . s_iKv)

    diagonalized as ``D e_s = w_s^2 e_s``. Differentiating the same phase gives
    the velocity operator; ``s`` enters Cartesian there, so the 2 pi of the
    2pi-free reciprocal lattice survives and later cancels the 1/2pi in
    ``gv_to_AA_fs``:

        dD/dq_a = sum_K Phi~ (1/N_iK) sum_v  2 pi i s_v^a exp(2 pi i q . s_v)
        M_a     = e^dag (dD/dq_a) e

    Args:
        xp: array namespace, ``numpy`` or ``jax.numpy``.
        eigh_fn: Hermitian eigensolver from that namespace.
        fc_mw: mass-weighted force constants ``Phi_iK / sqrt(m_i m_K)``,
            ``(N_p, N_sc, 3, 3)``, eV/(A^2 amu).
        j_of_k: primitive-cell index of each supercell atom, ``(N_sc,)``.
        svec_frac: smallest vectors from primitive atom i to supercell atom K,
            fractional in the primitive cell, ``(N_sc, N_p, max_multi, 3)``.
        multi_mask: which of the ``max_multi`` slots hold a real image,
            ``(N_sc, N_p, max_multi)``. 30-58 % of pairs in cubic cells have
            more than one; using only the interior lattice point is wrong.
        q_frac: q-points, fractional reciprocal coordinates, 2pi-free, ``(Nq, 3)``.
        pcell_cart: primitive lattice vectors as rows [A]; converts ``svec_frac``
            to Cartesian for the derivative.
        N_p: atoms in the primitive cell, so ``Ns = 3 N_p`` modes per q.
        dtype_complex: complex dtype carried from the precision switch.

    Returns:
        w2: ``(Nq, Ns)`` eigenvalues ``w^2`` in ASE units; negative marks a soft
            mode, so the caller takes ``sign(w2) sqrt(|w2|)``.
        e: ``(Nq, Ns, Ns)`` row eigenvectors -- ``e[q, s]`` is the polarization
            of mode s in the mass-weighted primitive basis.
        M: ``(Nq, 3, Ns, Ns)`` velocity operator in the mode basis. Its diagonal
            gives the group velocity, ``v_s^a = Re M[q,a,s,s] / (2 w_s)``; the
            off-diagonal is the QHGK coherence velocity between modes s and s'.
        D_q: ``(Nq, Ns, Ns)`` the dynamical matrix itself, Hermitized.
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
    # Matches phonopy's derivative_dynmat.c; see memory/project_dDdq_per_element_drift.md.
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
    real_dt = v_qsa.real.dtype
    recip_lattice = recip_lattice.astype(real_dt)
    recip_inv = np.linalg.inv(recip_lattice)
    r_cart = np.einsum("ij,ojk,kl->oil",
                       recip_lattice, rots_frac.astype(real_dt), recip_inv)

    q_in_BZ = q_frac - np.rint(q_frac)
    rq = np.einsum("oij,qj->oqi", rots_frac, q_in_BZ)
    mask = np.all(np.abs(rq - q_in_BZ[None, :, :]) < tol, axis=-1)
    count = mask.sum(axis=0)
    count_safe = np.where(count > 0, count, 1).astype(real_dt)

    out = np.einsum("oq,oba,qsa->qsb",
                    mask.astype(v_qsa.dtype), r_cart.astype(v_qsa.dtype), v_qsa,
                    optimize=True)
    out = out / count_safe[:, None, None]
    if (count == 0).any():
        empty = count == 0
        out[empty] = v_qsa[empty]
    return out


# Degenerate modes have no preferred basis: any mixture of equal-frequency
# eigenvectors is also an eigenvector, so eigh's choice among them is arbitrary.
# Stepping off q along `probe` splits them, and degenerate perturbation theory
# says the physical basis is the one diagonalizing dD/dq . probe over the block --
# the modes that stay eigenvectors as q moves. The direction has to be generic:
# along a symmetry axis the block can stay degenerate and the basis stays free.
# [1,2,3]/sqrt(14) is phonopy's own (group_velocity.py:157, "Give an random
# direction to break symmetry"), matched so v_qssa is comparable to theirs.
_PERTURB_PROBE = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)


DEGENERACY_TOL = 1e-4 / C.omega_to_THz   # matches phonopy degenerate_sets cutoff

# Which symmetry convention the solver follows. The two codes symmetrize
# exactly the index the other leaves alone, so each is coherent on its own terms
# and neither is a corrected version of the other:
#
#                       "PHONOPY"                    "TDEP"
#   degenerate subspace rotate so dD/dq . probe      every member takes the
#                       is diagonal, per-mode v      subspace mean trace/mb,
#                       kept distinct                eigenvectors unrotated
#   diagonal v_qsa      averaged over L(q) in the    left alone
#                       Cartesian index
#   off-diagonal        left alone                   averaged over L(q) in the
#   v_qssa                                           mode indices via Gamma(R,q)
#
# The pair is not interchangeable: each average is the identity only on the
# index it acts on, so applying one where the other belongs over-projects.
SYMMETRY_METHODS = ("PHONOPY", "TDEP")


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


def _average_degenerate_subspaces(w2, e, M):
    """TDEP's treatment: give every mode of a multiplet the subspace-mean velocity.

    `dD/dq_a` restricted to the multiplet is diagonalised and all members take
    the mean of its eigenvalues -- which is `trace / mb`, so the diagonalisation
    TDEP performs is not needed. The eigenvectors are left alone: TDEP never
    writes a rotated basis back (`type_forceconstant_secondorder_dynamicalmatrix
    .f90:353`, where the zheev output is deallocated unread, and could not be
    written back anyway since each Cartesian direction gives a different one).

    Basis-independent by construction, since a trace is. The cost is that modes
    with genuinely different velocities inside one multiplet are flattened to
    their mean, which the "phonopy" convention keeps distinct.
    """
    M = np.array(M, copy=True)
    for qi, start, stop in degenerate_sets(np.sqrt(np.abs(w2))):
        idx = np.arange(start, stop)
        G = slice(start, stop)
        for a in range(3):
            M[qi, a, idx, idx] = np.trace(M[qi, a, G, G]) / (stop - start)
    return e, M


def eigenvector_transformation(n, q_frac, atoms, rotations_frac, translations_frac,
                                perm):
    """``Gamma(R, q)``, the representation of R on the eigenvector space.

        e_s(Rq) = Gamma(R, q) e_s(q)

        Gamma(R, q)[a1, a2] = R_cart exp(-2 pi i q . v0),   a1 = R(a2)
        v0 = R^-1 (r_a1 - t) - r_a2

    R permutes the atoms and rotates each atom's 3x3 Cartesian block; the phase
    carries the sublattice translation the permutation leaves over. It is
    unitary, and for R in L(q) it commutes with D(q), so it is block-diagonal in
    frequency -- it mixes only degenerate modes.

    Args:
        n: index into ``rotations_frac`` / ``translations_frac``.
        q_frac: q-point, fractional reciprocal coordinates, 2pi-free.
        atoms: primitive cell, supplying the lattice and the positions ``r_a``.
        rotations_frac, translations_frac: space-group operations in fractional
            coordinates, as spglib returns them.
        perm: ``perm[a2] = R(a2)``, the atom permutation induced by ``R``.

    Returns:
        ``(3 N_p, 3 N_p)`` unitary in the atom-Cartesian basis.

    The velocity operator transforms as

        sum_b R_ab v^b(q) = Gamma(R, q)^dag v^a(q) Gamma(R, q)

    so an average over L(q) that omits it imposes the strictly stronger, false
    condition ``Gamma(R, q) = 1``. That is harmless on the diagonal, where
    ``|Gamma_ss| = 1`` for a non-degenerate mode, and wrong off it.
    """
    A = np.asarray(atoms.cell)
    Ainv = np.linalg.inv(A)
    R = A.T @ rotations_frac[n] @ np.linalg.inv(A.T)
    t = translations_frac[n] @ A
    pos = np.asarray(atoms.positions)
    na = len(atoms)

    G = np.zeros((3 * na, 3 * na), dtype=np.complex128)
    Rinv = np.linalg.inv(R)
    for a2 in range(na):
        a1 = perm[a2]
        v0 = Rinv @ (pos[a1] - t) - pos[a2]
        ph = np.exp(-2j * np.pi * np.dot(v0 @ Ainv, q_frac))
        G[3 * a1:3 * a1 + 3, 3 * a2:3 * a2 + 3] = R * ph
    return G


def symmetrize_v_qssa(v_qssa, e_qsi, q_points_frac, atoms, symprec=1e-5):
    """Average the velocity operator over the little group of each q.

        v^a  <-  < G_R^dag v^a G_R >_{R in L(q)},    G_R = U^dag Gamma(R, q) U

    This is the projector onto the L(q)-invariant part of v. By Schur
    orthogonality it annihilates every inter-irrep block, so what survives is
    coherence between distinct multiplets carrying the *same* irreducible
    representation. The off-diagonal is therefore large at general q and can
    vanish outright where the little group is big enough that no two multiplets
    share an irrep -- exactly zero at ``|L(q)| >= 12`` on the cubic fixtures.
    The size is not monotonic in ``|L(q)|``.

    ``Gamma(R, q)`` mixes only degenerate modes, so the ``1/(2 sqrt(w_s w_s'))``
    already folded into ``v_qssa`` commutes through and the average can act on
    it directly.

    The diagonal is not left alone: ``Gamma_ss`` is a pure phase only for a
    non-degenerate mode, so inside a multiplet the diagonal is mixed too. This
    is a different projector from ``_symmetrize_v_site``, which averages the
    Cartesian index instead, and the two do not agree on the diagonal.

    Args:
        v_qssa: ``(Nq, Ns, Ns, 3)`` velocity operator.
        e_qsi: ``(Nq, Ns, Ns)`` row eigenvectors, used to carry Gamma(R, q) into
            the mode basis.
        q_points_frac: ``(Nq, 3)`` q-points, fractional, 2pi-free.
        atoms: primitive cell; its space group supplies L(q).
        symprec: spglib symmetry tolerance.

    Returns:
        ``(Nq, Ns, Ns, 3)`` averaged velocity operator.
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
    for iq, q in enumerate(np.asarray(q_points_frac)):
        ops = little_group(q, W, A) or [None]
        U = np.swapaxes(np.asarray(e_qsi[iq]), -1, -2)      # columns = modes
        acc = np.zeros_like(out[iq])
        for n in ops:
            if n is None:
                acc += out[iq]
                continue
            G = U.conj().T @ eigenvector_transformation(n, q, atoms, W, w, perms[n]) @ U
            acc += np.einsum("ji,jka,kl->ila", G.conj(), out[iq], G, optimize=True)
        out[iq] = acc / len(ops)
    return out

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
                 enforce_translational_invariance=True, symmetry_method="PHONOPY"):
        """Build the solver.

        Args:
            force_constants: phonopy-shape ``(N_p, N_sc, 3, 3)`` array,
                not mass-weighted. Use ``gkmx.io.parse_force_constants``
                to load from FORCE_CONSTANTS / fc2.hdf5 / flat .dat.
            primitive: ASE Atoms — the primitive cell (``N_p`` atoms).
            supercell: ASE Atoms — the supercell (``N_sc`` atoms).
            backend: ``"numpy"`` (default, CPU) or ``"jax"`` (CPU or
                GPU; lazily imported).
            precision: ``"fp64"`` (default, machine-precision parity
                with phonopy) or ``"fp32"`` (~1e-6 relative on
                basis-invariant quantities, GPU speedup).
            p2s_map: optional ``(N_p,)`` int array of supercell indices
                identifying which supercell atoms are the primitive
                basis. Needed for phonopy ``primitive_matrix`` /
                ``supercell_matrix`` workflows where ``primitive.positions``
                may sit on a primitive-lattice translate of
                ``supercell.positions[p2s_map]``.
            factor: scalar rescaling for non-eV/Å² FCs. Default 1.0
                (FHI-aims / phonopy convention). For QE Ry/bohr² FCs use
                ``(108.97 / 15.633)**2 ≈ 48.59``.
            symmetry_method: ``"PHONOPY"`` (default) or ``"TDEP"`` -- the whole
                symmetry convention, not one knob. PHONOPY rotates a degenerate
                subspace so ``dD/dq . probe`` is diagonal and averages the
                Cartesian index of ``v_qsa``; TDEP gives every member of a
                multiplet the subspace mean and averages the mode indices of
                ``v_qssa``. Each leaves alone the index the other symmetrizes.
        """
        from .precision import Precision
        if backend not in ("numpy", "jax"):
            raise ValueError(f"Unknown backend: {backend!r}. Valid: 'numpy', 'jax'.")
        p = Precision.from_str(precision)
        dtype_real, dtype_complex = p.real, p.complex

        # Phonopy primitive_matrix + supercell_matrix builds can offset
        # `primitive.positions` from `supercell.positions[p2s_map]` by a
        # primitive-lattice translation; pin to the exact supercell images
        # when the caller passes p2s_map.
        if p2s_map is not None:
            p2s_map = np.asarray(p2s_map, dtype=np.int64)
            if p2s_map.shape != (len(primitive),):
                raise ValueError(
                    f"p2s_map must have shape ({len(primitive)},), got {p2s_map.shape}"
                )
            primitive = primitive.copy()
            primitive.positions = np.asarray(supercell.positions)[p2s_map]

        symmetry_method = str(symmetry_method).upper()
        if symmetry_method not in SYMMETRY_METHODS:
            raise ValueError(
                f"symmetry_method must be one of {SYMMETRY_METHODS}, "
                f"got {symmetry_method!r}")
        self.primitive = primitive
        self.supercell = supercell
        self.backend = backend
        self.precision = precision
        self.symmetry_method = symmetry_method
        self._dtype_real = dtype_real
        self._dtype_complex = dtype_complex

        # Reciprocal-space rotations act as R^{-T} of spglib's real-space
        # rotations; -R augmentation captures time reversal (matches
        # phonopy.Symmetry.reciprocal_operations, non-magnetic default).
        try:
            import spglib
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
        except Exception:
            self._symm_rots_frac = np.eye(3, dtype=self._dtype_real)[None, :, :]

        svec_frac, multi = get_smallest_vectors(primitive, supercell)
        j_of_k = get_s2p_map(primitive, supercell)

        # `factor` rescales calculator-native FC to gkmx-internal eV/A^2;
        # for QE Ry/bohr^2 use (108.97 / 15.633)**2 ~ 48.59.
        if enforce_translational_invariance:
            force_constants, self.asr_residual = translational_invariance(
                force_constants, primitive, supercell)
        else:
            self.asr_residual = None

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
        # Fix the degenerate subspaces before extracting per-mode velocities;
        # see memory/project_dDdq_boundary_bug.md.
        if self.symmetry_method == "TDEP":
            e, M = _average_degenerate_subspaces(w2, e, M)
        else:
            e, M = _rotate_degenerate_subspaces(w2, e, M)

        w_qs = np.sign(w2) * np.sqrt(np.abs(w2))

        # Scale near-zero cutoff off the 4th-smallest |w| so the three
        # Gamma acoustic modes don't set the threshold for everything else.
        flat = np.sort(np.abs(w_qs.flatten()))
        thresh = flat[min(3, len(flat) - 1)] * 1e-5
        w_qs = np.where(np.abs(w_qs) < thresh, 0.0, w_qs)
        w2_qs = np.sign(w_qs) * w_qs ** 2

        # w_inv needs 1/w safely bounded (not just zeroed); 1e20 stands in
        # for infinity at the looser cutoff (~4th-smallest |w| itself).
        w_inv = w_qs.copy()
        thresh_inv = flat[min(3, len(flat) - 1)] * 0.9
        w_inv[np.abs(w_inv) < thresh_inv] = 1e20
        w_inv_qs = 1.0 / w_inv
        w_inv_qs[np.abs(w_inv_qs) < 1e-9] = 0.0

        e_qsi = e
        D_qij = D_q

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

        if self.symmetry_method == "PHONOPY":
            recip_lat = np.linalg.inv(np.asarray(self.primitive.cell))
            v_qsa = _symmetrize_v_site(v_qsa, q_frac, self._symm_rots_frac, recip_lat)

        # Signed (not |.|) threshold: zeros both Gamma acoustics and soft
        # modes, matching phonopy's cutoff_frequency.
        v_ok = w_qs > thresh
        v_qsa = np.where(v_ok[:, :, None], v_qsa, 0.0).astype(self._dtype_real)

        if not with_group_velocity_matrices:
            return Solution(
                w_qs=w_qs, w_inv_qs=w_inv_qs, w2_qs=w2_qs,
                v_qsa_cartesian=v_qsa, e_qsi=e_qsi, D_qij=D_qij,
            )

        # QHGK off-diagonals use 2 sqrt(w_s w_s') in the denominator —
        # 2 w_s w_s' is a common mistake that agrees only on the diagonal.
        denom = 2.0 * np.sqrt(sqrt_safe[:, :, None] * sqrt_safe[:, None, :])
        M_moved = np.moveaxis(M, 1, -1)
        v_qssa = (M_moved / denom[:, :, :, None]) * C.gv_to_AA_fs

        # Under R in L(q) the velocity operator obeys
        #     sum_b R_ab v^b = Gamma(R, q)^dag v^a Gamma(R, q)
        # so the Cartesian average applied to v_qsa above is the diagonal case,
        # where Gamma(R, q) is a phase and cancels. Off the diagonal it does
        # not, and the mode-index average is the one that belongs here.
        if self.symmetry_method == "TDEP":
            v_qssa = symmetrize_v_qssa(v_qssa, e_qsi, q_frac, self.primitive)

        band_mask = v_ok[:, :, None] & v_ok[:, None, :]
        v_qssa = np.where(band_mask[:, :, :, None], v_qssa, 0.0)
        v_qssa = v_qssa.astype(self._dtype_complex)

        return SolutionWithGVM(
            w_qs=w_qs, w_inv_qs=w_inv_qs, w2_qs=w2_qs,
            v_qsa_cartesian=v_qsa, v_qssa_cartesian=v_qssa,
            e_qsi=e_qsi, D_qij=D_qij,
        )
