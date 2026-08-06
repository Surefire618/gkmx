"""Batched phonon solver with numpy / jax backends."""

import collections

import numpy as np

from . import _constants as C
from ._log import warn
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
    """Build D(q) under smallest-vectors Bloch and diagonalize; returns (w2, e, M, D_q)."""
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
    )
    dD_dq = dD_dq.reshape(-1, 3, Ns, Ns)
    # Matches phonopy's derivative_dynmat.c; see memory/project_dDdq_per_element_drift.md.
    dD_dq = 0.5 * (dD_dq + xp.conj(xp.swapaxes(dD_dq, -1, -2)))

    M = xp.einsum("qjn,qanm,qkm->qajk", xp.conj(e), dD_dq, e)

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
                    mask.astype(v_qsa.dtype), r_cart.astype(v_qsa.dtype), v_qsa)
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

# How a degenerate subspace fixes its group velocities.
#   "phonopy" -- rotate the basis so dD/dq . probe is diagonal, keep per-mode v
#                distinct (phonopy group_velocity.py::_rot_eigsets).
#   "tdep"    -- no rotation; every mode of the multiplet takes the subspace
#                mean sum(eigenvalues)/mb of dD/dq_a, invariant by construction
#                (type_forceconstant_secondorder_dynamicalmatrix.f90:353).
DEGENERACY_CONVENTIONS = ("phonopy", "tdep")


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
                 enforce_translational_invariance=True, degeneracy="phonopy"):
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

        if degeneracy not in DEGENERACY_CONVENTIONS:
            raise ValueError(
                f"degeneracy must be one of {DEGENERACY_CONVENTIONS}, got {degeneracy!r}")
        if degeneracy == "tdep":
            raise NotImplementedError(
                "degeneracy='tdep' is reserved but not built: it needs the subspace-mean "
                "velocity (sum(eigenvalues)/mb of dD/dq_a per Cartesian direction, "
                "eigenvectors left unrotated) in place of the probe rotation. "
                "Use degeneracy='phonopy'.")

        self.primitive = primitive
        self.supercell = supercell
        self.backend = backend
        self.precision = precision
        self.degeneracy = degeneracy
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

        masses = np.asarray(primitive.get_masses(), dtype=self._dtype_real)
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
        # Pin a canonical basis inside each degenerate subspace before
        # extracting per-mode velocities; see memory/project_dDdq_boundary_bug.md.
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

        # Average v over the little group of q (phonopy's second half).
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

        # Site-symmetrize v_qssa too; on sum-rule-violating FC this shrinks
        # |v_qssa| where the little group is non-trivial (~10-25% lower
        # QHGK κ vs vibes). See memory/project_qhgk_v_qssa_site_symm.md.
        Nq_q, Ns_q, _, _ = v_qssa.shape
        v_qssa = _symmetrize_v_site(
            v_qssa.reshape(Nq_q, Ns_q * Ns_q, 3),
            q_frac, self._symm_rots_frac, recip_lat,
        ).reshape(Nq_q, Ns_q, Ns_q, 3)

        band_mask = v_ok[:, :, None] & v_ok[:, None, :]
        v_qssa = np.where(band_mask[:, :, :, None], v_qssa, 0.0)
        v_qssa = v_qssa.astype(self._dtype_complex)

        return SolutionWithGVM(
            w_qs=w_qs, w_inv_qs=w_inv_qs, w2_qs=w2_qs,
            v_qsa_cartesian=v_qsa, v_qssa_cartesian=v_qssa,
            e_qsi=e_qsi, D_qij=D_qij,
        )
