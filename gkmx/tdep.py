"""TDEP's Bloch convention, for comparing velocity operators across the codes.

gkmx and TDEP diagonalise the *same* dynamical matrix but express it in different
Bloch conventions, so their eigenvectors differ by a per-atom phase and a complex
conjugation. With ``P(q) = diag(exp(-2 pi i q . r_a))`` and ``D*(q) = D(-q)``
(real force constants),

    U_T  = P conj(U)
    D_T  = P D* P^dag
    dD_T = P [ (dD/dq_a)* - 2 pi i [r_a, D*] ] P^dag

The commutator vanishes on the diagonal and does not off it. That is precisely
why the two codes agree on group velocities and disagree on generalized ones:
**the off-diagonal velocity is a property of the representation, not of the
crystal alone**, so any cross-code comparison of ``v_ss'`` -- or of a kappa_C
built from it -- has to fix the convention first.

The ``2 pi`` is load-bearing and easy to drop. It rides on the same 2pi-free
reciprocal lattice as the ``2 pi i s`` in ``dD/dq`` (`phonon._solve_kernel`).
Scanning the coefficient against TDEP's own ``velocity_offdiagonal``:

    coefficient c in dD_T = P[dD* + c i [r_a, D*]]P^dag
    c        KI_bcc     Rb2O      KPTe2
    0        3.8e-01    9.2e-01   9.9e-01
    -1       3.2e-01    7.8e-01   8.4e-01
    -2 pi    9.3e-10    2.2e-09   7.1e-09
    +2 pi    7.6e-01    1.8e+00   2.0e+00
"""
from __future__ import annotations

import numpy as np


def bloch_phase(q_frac, scaled_positions):
    """``P(q) = diag(exp(-2 pi i q . r_a))``, repeated over the three Cartesian slots."""
    phase = np.exp(-2j * np.pi * (np.asarray(scaled_positions) @ np.asarray(q_frac)))
    return np.repeat(phase, 3)


def to_tdep_convention(D_q, dD_dq, q_frac, scaled_positions, cart_positions):
    """Map gkmx's ``(D, dD/dq)`` at one q into TDEP's Bloch convention.

    Args:
        D_q: ``(Ns, Ns)`` mass-weighted dynamical matrix.
        dD_dq: ``(3, Ns, Ns)`` its q-gradient, Cartesian.
        q_frac: q-point in fractional reciprocal coordinates, 2pi-free.
        scaled_positions: primitive-cell atom positions, fractional.
        cart_positions: the same positions in Cartesian [A]; the commutator
            ``[r_a, D*]`` is Cartesian because ``dD/dq`` is.

    Returns:
        ``(D_T, dD_T)`` in the same shapes, in TDEP's convention.
    """
    p = bloch_phase(q_frac, scaled_positions)
    D_star = np.conj(D_q)
    r = np.repeat(np.asarray(cart_positions), 3, axis=0)

    D_T = p[:, None] * D_star * np.conj(p)[None, :]
    dD_T = np.empty_like(dD_dq)
    for a in range(3):
        commutator = r[:, a][:, None] * D_star - D_star * r[:, a][None, :]
        dD_T[a] = p[:, None] * (np.conj(dD_dq[a]) - 2j * np.pi * commutator) \
            * np.conj(p)[None, :]
    return D_T, dD_T
