"""BTE and QHGK thermal conductivity from mode-resolved inputs (q-first)."""

import numpy as np
import xarray as xr

from . import _constants as C
from . import keys


def get_kappa_BTE(v_qsa, tau_qs, cv_qs=None, weights=None, scalar=False):
    """BTE thermal conductivity ``kappa_ab = sum_qs cv * tau * v_a * v_b``.

    The particle-like channel: every mode carries heat on its own, so only the
    diagonal group velocity enters and coherence between modes is absent.

    Args:
        v_qsa: ``(Nq, Ns, 3)`` diagonal group velocity [A/fs].
        tau_qs: ``(Nq, Ns)`` mode lifetimes [fs]. NaN marks a failed fit and
            drops out of the sum.
        cv_qs: ``(Nq, Ns)`` mode heat capacities [eV/(K A^3)]; ``None`` leaves
            kappa unweighted, i.e. the bare ``v^2 tau`` sum.
        weights: per-q integration weights; ``None`` weights every q equally.
        scalar: return the cubic average ``Tr(kappa)/3`` instead of the tensor.

    Returns:
        ``(3, 3)`` tensor in W/(m K), or a scalar when ``scalar=True``.
    """
    if weights is None:
        weights = 1.0
    if cv_qs is None:
        cv_qs = 1.0

    cv_qs = np.asarray(cv_qs)
    v_qsa = np.asarray(v_qsa)
    tau_qs = np.nan_to_num(tau_qs)

    v2_qsab = v_qsa[..., :, None] * v_qsa[..., None, :]
    kappa_qsab = C.to_W_mK * np.asarray(weights * cv_qs * tau_qs)[..., None, None] * v2_qsab
    kappa_ab = kappa_qsab.sum(axis=(0, 1))

    result = xr.DataArray(kappa_ab, dims=keys.tensor, name=keys.kappa_ha)

    if scalar:
        return float(np.diagonal(kappa_ab).mean())
    return result


def qhgk_tau_eff(w_qs, w_inv_qs, tau_qs, tol=1e-4):
    """Off-diagonal QHGK effective lifetime ``tau_eff[q, s, s']``, shape ``(Nq, Ns, Ns)``.

    Quasi-harmonic Green-Kubo coherence kernel of Isaeva, Barbalinardo, Donadio and
    Baroni, Nat. Commun. 10, 3853 (2019), retaining the antiresonant ``(w_s + w_sp)``
    term.

    Args:
        w_qs: ``(Nq, Ns)`` frequencies in gkmx units; converted to rad/fs here
            so the Lorentzian mixes frequencies and linewidths in one unit.
        w_inv_qs: ``(Nq, Ns)`` ``1/w``, converted alongside.
        tau_qs: ``(Nq, Ns)`` mode lifetimes [fs]; NaN is treated as zero.
        tol: frequencies below this (rad/fs) are zeroed, masking the Gamma
            acoustics out of the pair sum.

    Returns:
        ``(Nq, Ns, Ns)`` effective lifetime [fs]. On ``s == s'`` the pair weight
        is 1 and the beat frequency 0, so it reduces to ``tau_s``.
    """
    w = np.asarray(w_qs).copy() * C.omega_to_rad_fs
    wi = np.asarray(w_inv_qs).copy() / C.omega_to_rad_fs
    tau = np.nan_to_num(np.asarray(tau_qs))
    w = np.where(w < tol, 0, w)
    wi = np.where(w == 0, 0, wi)

    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = 0.5 / tau
        w_s = w[:, :, None]; w_sp = w[:, None, :]
        wi_s = wi[:, :, None]; wi_sp = wi[:, None, :]
        w_plus = (w_s + w_sp) ** 2 * (wi_s * wi_sp) / 4
        w_minus = (w_s - w_sp) ** 2 * (wi_s * wi_sp) / 4
        gamma_ss = gamma[:, :, None] + gamma[:, None, :]
        gamma_plus = gamma_ss / (gamma_ss ** 2 + (w_s + w_sp) ** 2)
        gamma_minus = gamma_ss / (gamma_ss ** 2 + (w_s - w_sp) ** 2)
    return np.nan_to_num(w_plus * gamma_minus + w_minus * gamma_plus)


def get_kappa_QHGK(v_qssa, tau_qs, w_qs, w_inv_qs, cv_qs=None, weights=None,
                    scalar=False, tol=1e-4):
    """QHGK thermal conductivity: the coherence channel.

    Two modes ``s != s'`` carry heat together when their linewidths overlap
    their frequency splitting, which the Lorentzian in ``qhgk_tau_eff``
    measures. This is what the BTE leaves out, and it matters where modes are
    dense and short-lived.

    Args:
        v_qssa: ``(Nq, Ns, Ns, 3)`` velocity operator [A/fs]. Its diagonal is
            the group velocity, so the ``s == s'`` terms reproduce the BTE.
        tau_qs: ``(Nq, Ns)`` mode lifetimes [fs].
        w_qs: ``(Nq, Ns)`` frequencies in gkmx units (no 2pi).
        w_inv_qs: ``(Nq, Ns)`` ``1/w`` with the near-zero modes already zeroed.
            Take it from the solver rather than recomputing: the cutoff is set
            from order statistics over the whole q batch, so a locally chosen
            threshold silently disagrees whenever the batch differs.
        cv_qs: ``(Nq, Ns)`` mode heat capacities [eV/(K A^3)].
        weights: per-q integration weights; ``None`` weights every q equally.
        scalar: return ``Tr(kappa)/3`` instead of the tensor.
        tol: frequencies below this count as zero, masking the Gamma acoustics
            out of the pair sum.

    Returns:
        ``(3, 3)`` tensor in W/(m K), or a scalar when ``scalar=True``.
    """
    if weights is None:
        weights = 1.0
    if cv_qs is None:
        cv_qs = 1.0

    cv_qs = np.asarray(cv_qs)
    tau_eff = qhgk_tau_eff(w_qs, w_inv_qs, tau_qs, tol=tol)

    v2 = v_qssa[..., :, None] * v_qssa[..., None, :].conj()

    # cv is indexed by the primed (second) mode, matching the old (s, q) convention.
    if cv_qs.ndim == 2:
        cv_bcast = cv_qs[:, None, :]
    else:
        cv_bcast = cv_qs

    kappa = C.to_W_mK * np.asarray(weights * cv_bcast * tau_eff)[..., None, None] * v2
    kappa_ab = kappa.sum(axis=(0, 1, 2)).real

    result = xr.DataArray(kappa_ab, dims=keys.tensor, name=keys.kappa_QHGK)
    if scalar:
        return float(np.diagonal(kappa_ab).mean())
    return result




def symmetrize_kappa(kappa_ab, primitive, symprec=1e-5):
    """Project a conductivity tensor onto the crystal point group.

        kappa <- (1/|G|) sum_R  R kappa R^T

    A conductivity tensor satisfies ``R kappa R^T = kappa``, but what
    ``get_kappa_QHGK`` returns need not: in the TDEP Bloch convention
    ``D_T = P D* P^dag`` the phase runs over lattice vectors alone, so under a
    symmetry operation it retains
    ``exp(-2 pi i Rq . (L_b - L_a))`` from the lattice vectors the atoms are
    displaced by. That phase depends on q, so it survives differentiation and
    leaves ``v_ss'`` non-covariant.

    This function averages over all rotation operators, enforcing the crystal
    symmetry on the kappa tensor. The projection preserves the trace exactly
    (each ``R kappa R^T`` does), so kappa_avg is untouched and only the
    anisotropy changes.

    TDEP has the same defect and applies exactly this average to every
    channel before writing output (`kappa.f90::symmetrize_kappa`).

    Args:
        kappa_ab: ``(3, 3)`` conductivity tensor, array or DataArray.
        primitive: ASE Atoms whose space group supplies the rotations.
        symprec: spglib symmetry tolerance.

    Returns:
        The symmetrized tensor, same type as ``kappa_ab``.
    """
    import spglib

    A = np.asarray(primitive.cell)
    dataset = spglib.get_symmetry(
        (A, primitive.get_scaled_positions(), primitive.get_atomic_numbers()),
        symprec=symprec)
    rotations = dataset["rotations"]

    K = np.asarray(kappa_ab.data if hasattr(kappa_ab, "data") else kappa_ab)
    A_inv_T = np.linalg.inv(A.T)
    out = np.zeros((3, 3), dtype=np.float64)
    for R in rotations:
        R_cart = A.T @ R @ A_inv_T
        out += R_cart @ np.asarray(K, dtype=np.float64) @ R_cart.T
    out = (out / len(rotations)).astype(K.dtype)

    if isinstance(kappa_ab, xr.DataArray):
        return xr.DataArray(out, dims=kappa_ab.dims, name=kappa_ab.name)
    return out
