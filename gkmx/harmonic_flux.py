"""Measured harmonic heat fluxes: real-space ``J_hm-R`` and its mode-space
counterparts ``J_hm-q`` / ``J_quasi-hm``.

``J_hm-R`` is the heat flux of the harmonic model Hamiltonian evaluated on
the MD trajectory in real space -- no eigenvectors, no q-points -- and is the
reference every mode-resolved reconstruction is judged against. On the
commensurate grid it decomposes exactly as::

    J_hm-R = J_quasi-hm + J_disp

``J_quasi-hm`` is the pair (``r0``) term in mode variables -- resonant and
antiresonant bilinears; ``J_disp`` is the ``(u_I - u_J)``-weighted term,
third order in the dynamical variables.
``J_hm-q`` is the diagonal (particle-like) restriction, so that
``J_quasi-hm - J_hm-q`` isolates the coherent inter-branch channel.
The time integral of the ``J_quasi-hm`` autocorrelation is what
``kappa.get_kappa_QHGK`` models via Wick factorization and the fitted
Lorentzian linewidths.

The Green-Kubo pipeline consumes these through ``greenkubo._get_gk_dataset``
(the ``--harmonic-flux`` flag).
"""

import numpy as np
from ase import units

from . import keys
from . import masses as _masses
from ._log import talk
from .lattice_points import get_pair_vectors
from .precision import Precision

_prefix = "gkmx"


def _talk(msg):
    talk(msg, prefix=_prefix)


def _flux_R_common(dataset, dmx, dtype):
    if dtype is None:
        dtype = getattr(dmx, "_dtype_real", None) or Precision.default().real
    N = len(dmx.supercell)
    Phi = np.asarray(dmx.remapped, dtype=dtype).reshape(3 * N, 3 * N)
    time_dim = dataset.displacements.dims[0]
    Nt = dataset.displacements.shape[0]
    volume = dtype(float(np.nanmean(np.asarray(dataset[keys.volume]))))
    return dtype, N, Phi, time_dim, Nt, volume


def _stream_uv(dataset, time_dim, Nt, t_chunk, dtype):
    """Yield ``(t0, t1, u, v)`` trajectory chunks; ``* units.fs`` puts the
    ASE-native velocities on the fs base of the eV/AA^2/fs flux convention."""
    for t0 in range(0, Nt, t_chunk):
        t1 = min(Nt, t0 + t_chunk)
        u = np.asarray(dataset.displacements.isel(
            {time_dim: slice(t0, t1)}).data, dtype=dtype)
        v = np.asarray(dataset.velocities.isel(
            {time_dim: slice(t0, t1)}).data, dtype=dtype) * dtype(units.fs)
        yield t0, t1, u, v


def _disp_rows(u, v, Phi, N):
    """``sum_I u_Ia (s_I - g_I)`` with ``s_I = sum_J W_IJ``,
    ``g_J = sum_I W_IJ``, ``W_IJ = sum_ab v_Ia Phi_IJab u_Jb`` -- the
    ``(u_I - u_J)``-weighted part of the flux, before the ``1/2V``."""
    nt = u.shape[0]
    U, V = u.reshape(nt, -1), v.reshape(nt, -1)
    s = (V * (U @ Phi.T)).reshape(nt, N, 3).sum(axis=2)
    g = (U * (V @ Phi)).reshape(nt, N, 3).sum(axis=2)
    return np.einsum("tIa,tI->ta", u, s - g, optimize=True)


def compute_harmonic_heat_flux_R(dataset, dmx, max_bytes=200 << 20, dtype=None,
                                 verbose=True):
    r"""Real-space harmonic heat flux ``J_hm-R`` from the force constants.

    The flux of the harmonic model energy with instantaneous pair vectors
    ``r0_IJ + u_I - u_J``, in the trajectory's flux units (eV/AA^2/fs)::

        PU_IJa = sum_b  Phi_IJab u_Jb
        s_IJab = (r0_IJa + u_Ia - u_Ja) PU_IJb / V
        J_a    = 1/2 sum_I sum_b (sum_J s_IJab) v_Ib

    ``r0_IJa`` is the multiplicity-averaged smallest pair vector, so pairs
    straddling the supercell boundary enter through every tied image -- the
    same averaging ``D(q)`` uses. Evaluated without forming a pair tensor:
    the displacement terms are row and column sums of
    ``W_IJ = sum_ab v_Ia Phi_IJab u_Jb``, the ``r0`` term is the bilinear
    ``v R^a u`` with the time-independent operator
    ``R^a[3I+p, 3J+q] = r0_IJa Phi_IJpq``. numpy only, no ``backend=``;
    ``max_bytes`` bounds the streamed time chunks.
    """
    dtype, N, Phi, time_dim, Nt, volume = _flux_R_common(dataset, dmx, dtype)
    N3 = 3 * N

    # The solver already searched the smallest vectors, as (N, N_p); the pair
    # table follows from those by translation, so no image search runs here.
    r0 = get_pair_vectors(dmx.primitive, dmx.supercell,
                          *dmx._ensure_phonon().smallest_vectors).astype(dtype)

    Phi4 = Phi.reshape(N, 3, N, 3)
    R_a = [(Phi4 * r0[:, None, :, None, a]).reshape(N3, N3) for a in range(3)]
    del r0

    # Six real (t_chunk, 3N) buffers are live at the peak.
    row = 6 * N3 * np.dtype(dtype).itemsize
    tc = max(1, int(max_bytes // row))
    if verbose:
        ops_gb = 4 * N3 * N3 * np.dtype(dtype).itemsize / 1e9
        _talk(f"harmonic heat flux (real space): N={N}, Nt={Nt}, "
              f"t_chunk={tc}, chunk={tc * row / 1e9:.2f} GB, "
              f"operators={ops_gb:.2f} GB")

    J = np.zeros((Nt, 3), dtype=dtype)
    for t0, t1, u, v in _stream_uv(dataset, time_dim, Nt, tc, dtype):
        nt = u.shape[0]
        U, V = u.reshape(nt, N3), v.reshape(nt, N3)
        J[t0:t1] = _disp_rows(u, v, Phi, N)
        for a in range(3):
            J[t0:t1, a] += (V * (U @ R_a[a].T)).sum(axis=1)
    J *= dtype(0.5) / volume
    return J


def compute_harmonic_heat_flux_disp(dataset, dmx, max_bytes=200 << 20,
                                    dtype=None, verbose=True):
    r"""Displacement term ``J_disp`` of the real-space harmonic flux.

    The ``(u_I - u_J)``-weighted part of ``J_hm-R``::

        J_disp_a = 1/(2V) sum_I u_Ia [ v_I . (Phi u)_I - u_I . (Phi v)_I ]

    (written for the symmetrized ``remapped`` ``Phi``; the kernel contracts
    the exact pairwise form and does not rely on that symmetry). Third order
    in the dynamical variables, so it has no counterpart in any quadratic
    mode theory, cannot be interpolated in q, and enters no kappa channel.
    Benchmark-only: it closes the decomposition
    ``J_hm-R = J_quasi-hm + J_disp``; the Green-Kubo pipeline never computes
    or stores it.
    """
    dtype, N, Phi, time_dim, Nt, volume = _flux_R_common(dataset, dmx, dtype)

    row = 6 * 3 * N * np.dtype(dtype).itemsize
    tc = max(1, int(max_bytes // row))
    if verbose:
        _talk(f"harmonic heat flux (displacement term): N={N}, Nt={Nt}, "
              f"t_chunk={tc}")

    J = np.zeros((Nt, 3), dtype=dtype)
    for t0, t1, u, v in _stream_uv(dataset, time_dim, Nt, tc, dtype):
        J[t0:t1] = _disp_rows(u, v, Phi, N)
    J *= dtype(0.5) / volume
    return J


def compute_harmonic_heat_flux_q(dataset, dmx, v_qssa=None, t_chunk=None,
                                 max_bytes=512 << 20, dtype_u=None,
                                 verbose=True):
    r"""Time-resolved harmonic heat fluxes rebuilt from the mode amplitudes.

    Unlike ``hf_acf_BTE`` / ``hf_acf_QHGK``, which are analytical kernels built
    from ``(w, v, tau, cv)``, these are *measured*: the trajectory is projected
    onto modes and the flux is reassembled per time step::

        J_hm-q_a(t)     = 1/V  sum_qs   E_qs(t) v_qsa,   E_qs = 2 |a_qs|^2 w2_qs
        J_quasi-hm_a(t) = 1/V  Re[ i sum_q sum_ss'  sqrt(w_qs w_qs') v_qss'a
                                   p_qs(t) conj(u_qs'(t)) ]

    with ``u_qs = sum_I e_qsI sqrt(m_I) u_I`` and ``p_qs`` likewise from the
    ASE-native velocities. ``J_hm-q`` keeps only ``s == s'`` (the
    particle-like channel); ``J_quasi-hm`` sums the whole block.

    ``J_quasi-hm`` is the pair (``r0``) term of ``J_hm-R`` in mode variables:
    in the amplitudes ``a, b = (u_qs -+ i w^-1 p_qs)/2`` it carries the
    resonant ``a a*`` and the antiresonant ``a b*`` bilinears, with the
    conjugation consistent with the ``e``-projection above (``u_qs`` is the
    conjugate of the true expansion coefficient, so the operator pairs as
    ``p v u*``). Exactness is convention-conditional: with the native
    operator (RAW, PHONO3PY) ``J_hm-R - J_quasi-hm`` is the displacement
    term alone on the commensurate grid; under TDEP the little-group-averaged
    operator makes the reconstruction the TDEP model's own pair flux. The
    ``J_quasi-hm`` autocorrelation is the measured counterpart of the
    Wick-factorized kernel behind ``kappa.get_kappa_QHGK``.

    ``t_chunk`` is sized so the live buffers stay under ``max_bytes`` (an
    explicit ``t_chunk`` overrides).

    Args:
        v_qssa: velocity operator to reconstruct with. Defaults to the DMX's
            own; a DMX loaded from a cache written without
            ``include_group_velocity_matrices`` has none, so the pipeline
            passes the array its kappa was built from.

    Returns:
        ``(J_hm_q, J_quasi_hm)``, each ``(Nt, 3)`` in the trajectory's flux
        units.
    """
    if dtype_u is None:
        dtype_u = getattr(dmx, "_dtype_real", None) or Precision.default().real
    dtype_c = np.result_type(dtype_u, np.complex64)

    time_dim = dataset.displacements.dims[0]
    Nt = dataset.displacements.shape[0]

    e_qsI = np.asarray(dmx.e_qsI)
    Nq, Ns, I = e_qsI.shape
    nmodes = Nq * Ns
    e_re = np.ascontiguousarray(e_qsI.reshape(nmodes, I).real.astype(dtype_u))
    e_im = np.ascontiguousarray(e_qsI.reshape(nmodes, I).imag.astype(dtype_u))
    w_inv_m = np.asarray(dmx.w_inv_qs).reshape(nmodes).astype(dtype_u)
    w2_m = np.asarray(dmx.w2_qs).reshape(nmodes).astype(dtype_u)
    w_qs = np.asarray(dmx.solution.w_qs).astype(dtype_u)
    v_ma = np.asarray(dmx.solution.v_qsa_cartesian).reshape(nmodes, 3).astype(dtype_u)
    if v_qssa is None:
        v_qssa = dmx.solution.v_qssa_cartesian
    v_qssa = np.asarray(v_qssa, dtype=dtype_c)
    # The bilinear p v u* is eigh-gauge-invariant only when the operator
    # carries the projector's conjugation flavor. TDEP's e_qsI is built from
    # conj(e_qsi) (`_build_e_qsI`), so its operator enters conjugated;
    # RAW / PHONO3PY are native, which is also the exact-identity flavor.
    if dmx._convention == "TDEP":
        v_qssa = np.conj(v_qssa)

    # Acoustic gate: a soft/near-zero branch gets no sqrt(w) weight rather
    # than the sqrt of a negative or denormal number. Independent of
    # phonon.py's w_inv order-statistics cutoff.
    w_sqrt = np.where(w_qs < 1e-4, dtype_u(0.0), w_qs) ** dtype_u(0.5)

    # Same mass source as `compute_cv_tau`: the structure the force constants
    # were weighted with, never the trajectory's own copy.
    m_sqrt = np.sqrt(np.asarray(_masses.of(dmx.supercell))).astype(dtype_u)
    volume = float(np.nanmean(np.asarray(dataset[keys.volume])))

    J_hm_q = np.zeros((Nt, 3), dtype=dtype_u)
    J_quasi_hm = np.zeros((Nt, 3), dtype=dtype_u)

    # Live at the peak, in units of one real (3N, t_chunk) plane: U + V (2),
    # u_c + p_c (2), the four projections (4), P + Uc complex (4) = 12.
    br = np.dtype(dtype_u).itemsize
    row = 12 * 3 * (I // 3) * br
    if t_chunk is None:
        t_chunk = max(1, int(max_bytes // row))
    if verbose:
        peak_gb = row * min(t_chunk, Nt) / 1e9
        _talk(f"harmonic heat flux (mode space): Nq={Nq}, Ns={Ns}, Nt={Nt}, "
              f"t_chunk={t_chunk}, peak={peak_gb:.2f} GB")

    for t0 in range(0, Nt, t_chunk):
        t1 = min(Nt, t0 + t_chunk)
        U = np.asarray(dataset.displacements.isel(
            {time_dim: slice(t0, t1)}).data, dtype=dtype_u)
        V = np.asarray(dataset.velocities.isel(
            {time_dim: slice(t0, t1)}).data, dtype=dtype_u)
        tc = U.shape[0]
        u_c = (U * m_sqrt[None, :, None]).reshape(tc, I).T
        p_c = (V * m_sqrt[None, :, None]).reshape(tc, I).T
        del U, V

        u_re, u_im = e_re @ u_c, e_im @ u_c
        p_re, p_im = e_re @ p_c, e_im @ p_c
        del u_c, p_c
        a_re = dtype_u(0.5) * (u_re - w_inv_m[:, None] * p_im)
        a_im = dtype_u(0.5) * (u_im + w_inv_m[:, None] * p_re)

        E = dtype_u(2.0) * (a_re * a_re + a_im * a_im) * w2_m[:, None]
        J_hm_q[t0:t1] = (E.T @ v_ma) / dtype_u(volume)
        del E, a_re, a_im

        # sqrt(w_s w_s') split one factor per side. Built in place:
        # `re + 1j * im` would promote to complex128 and blow the fp32 bound.
        Uc = np.empty((Nq, Ns, tc), dtype=dtype_c)
        Uc.real = u_re.reshape(Nq, Ns, tc)
        Uc.imag = -u_im.reshape(Nq, Ns, tc)                    # conj(u_qs)
        Uc *= w_sqrt[..., None]
        del u_re, u_im
        P = np.empty((Nq, Ns, tc), dtype=dtype_c)
        P.real = p_re.reshape(Nq, Ns, tc)
        P.imag = p_im.reshape(Nq, Ns, tc)
        P *= w_sqrt[..., None]
        del p_re, p_im
        # One Cartesian component at a time, so the (Nq,Ns,Ns,tc) outer
        # product never exists. Re[i z] = -Im[z].
        for ia in range(3):
            tmp = np.einsum("qsS,qSt->qst", v_qssa[..., ia], Uc, optimize=True)
            J_quasi_hm[t0:t1, ia] = -np.imag(
                np.einsum("qst,qst->t", P, tmp, optimize=True)) / dtype_u(volume)
            del tmp
        del P, Uc

    return J_hm_q, J_quasi_hm
