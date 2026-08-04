"""BTE and QHGK thermal conductivity from mode-resolved inputs (q-first)."""

import numpy as np
import xarray as xr

from . import _constants as C
from . import keys


def get_kappa_BTE(v_qsa, tau_qs, cv_qs=None, weights=None, scalar=False):
    """BTE thermal conductivity ``kappa_ab = sum_qs cv * tau * v_a * v_b``."""
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
    """QHGK thermal conductivity
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


