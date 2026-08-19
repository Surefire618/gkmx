"""Dense-q-grid interpolation + linear extrapolation of kappa to infinite supercell."""


import numpy as np
import xarray as xr
from scipy.interpolate import LinearNDInterpolator, griddata
from scipy.optimize import curve_fit
from scipy.spatial import QhullError

from . import keys
from ._log import talk
from .kappa import get_kappa_BTE, get_kappa_QHGK
from .lattice_points import get_unit_grid_extended

_prefix = "gkmx.interpolation"


def _talk(msg):
    talk(msg, prefix=_prefix)


def interpolate_to_gamma(q_points, array_qs, extend_minus=True, tol=1e-9):
    """Linear interpolation of ``array_qs`` at Gamma; mirrors via ``-q`` when ``extend_minus=True``."""
    assert np.linalg.norm(q_points[0]) < tol, "First q-point must be Gamma"

    train_qs = q_points[1:]
    train_arr = array_qs[1:]

    if extend_minus:
        train_qs = np.concatenate([train_qs, -train_qs])
        train_arr = np.concatenate([train_arr, train_arr], axis=0)

    q_gamma = np.zeros((1, 3))
    Ns = array_qs.shape[1]
    result = np.zeros(Ns)
    for ns in range(Ns):
        interp = LinearNDInterpolator(train_qs, train_arr[:, ns])
        result[ns] = float(np.asarray(interp(q_gamma)).ravel()[0])

    return result


def interpolate_to_grid(q_points, train_array_qs, train_points, tol=1e-9):
    """SciPy `griddata` linear interpolation from `train_points` to `q_points`."""
    new_pts = (q_points + tol) % 1 - tol
    Ns = train_array_qs.shape[1]
    out = np.empty((len(new_pts), Ns))
    for ns in range(Ns):
        out[:, ns] = griddata(train_points, train_array_qs[:, ns], new_pts)
    return out


def fit_kappa_ladder(nqs, Ks):
    """Per-component weighted linear fit ``kappa(1/nq) = m/nq + y0``.

    ``sigma = 1/nq`` so the denser grids dominate the intercept. Returns
    ``(slope_ab, intercept_ab, slope_stderr_ab, intercept_stderr_ab)``, each
    ``(3, 3)``.
    """
    x = np.asarray(nqs, dtype=float) ** -1.0
    Ks = np.asarray(Ks)
    out = np.zeros((4, 3, 3))
    for a, b in np.ndindex(3, 3):
        popt, pcov = curve_fit(lambda _x, m, y0: m * _x + y0, x, Ks[:, a, b],
                               p0=(-1, 10), sigma=x)
        out[0, a, b], out[1, a, b] = popt
        out[2, a, b], out[3, a, b] = np.sqrt(np.diag(pcov))
    return tuple(out)


def get_interpolation_data(dmx, lifetimes, cv, nq_max=20, quasi_harmonic_greenkubo=False):
    """Dense-grid interpolation of kappa + linear extrapolation ``kappa(1/nq) -> kappa(inf)``.

    Scales ``tau -> l = w**2 * tau`` (grid-independent), interpolates on
    the extended unit grid, sweeps ``nq = 4..nq_max``. Pass
    ``quasi_harmonic_greenkubo=True`` for the QHGK variant.

    ``cv`` is passed through to the kappa kernels unchanged unless it is callable,
    in which case it is called as ``cv(solution)`` on each mesh.
    """
    def heat_capacity(solution):
        """A callable cv follows the mesh; a scalar or array is used as given."""
        return cv(solution) if callable(cv) else cv

    l_qs = dmx.w2_qs * np.nan_to_num(lifetimes)

    try:
        l_qs[0, :] = interpolate_to_gamma(dmx.q_points, l_qs, extend_minus=True)
    except QhullError:
        _talk("** QhullError: q-point sampling insufficient for interpolation")
        return {}

    train_grid = get_unit_grid_extended(dmx.q_points)
    train_l_qs = l_qs[train_grid.map2extended]
    kw_train = {"train_array_qs": train_l_qs, "train_points": train_grid.points_extended}

    # Idempotency check: interpolating back to the training grid must
    # return the same values, otherwise the mesh is misaligned.
    l_qs_check = interpolate_to_grid(dmx.q_points, **kw_train)
    assert np.allclose(l_qs, l_qs_check, atol=1e-6), "Interpolation not idempotent"

    if quasi_harmonic_greenkubo:
        kappa_ha = get_kappa_QHGK(
            v_qssa=dmx.solution.v_qssa_cartesian, tau_qs=lifetimes,
            w_qs=dmx.w_qs, w_inv_qs=dmx.w_inv_qs, cv_qs=heat_capacity(dmx.solution),
        )
    else:
        kappa_ha = get_kappa_BTE(dmx.v_qsa_cartesian, tau_qs=lifetimes,
                                 cv_qs=heat_capacity(dmx.solution))

    nqs = np.arange(4, nq_max + 1, 2)
    Nq_init = len(dmx.q_points)
    Ks = np.zeros((len(nqs), 3, 3))
    Ks_QHGK = np.zeros((len(nqs), 3, 3)) if quasi_harmonic_greenkubo else None

    for ii, nq in enumerate(nqs):
        mesh = (nq, nq, nq)
        grid, solution = dmx.get_mesh_and_solution(
            mesh, reduced=False, monkhorst=False,
            with_group_velocity_matrices=quasi_harmonic_greenkubo,
        )

        # Interpolate on the irreducible grid, then expand by symmetry
        # — `l` is a scalar per mode so this is exact.
        ir_l = interpolate_to_grid(q_points=grid.ir.points, **kw_train)
        tau_int = ir_l[grid.ir.map2full] * solution.w_inv_qs ** 2

        Nq_eff = len(grid.points) / Nq_init
        cv_mesh = heat_capacity(solution)
        KK = get_kappa_BTE(solution.v_qsa_cartesian, tau_int, cv_mesh) / Nq_eff
        Ks[ii] = np.asarray(KK)
        _talk(f"nq={nq:3d}, Nq_eff={Nq_eff:6.2f}, kappa={np.diagonal(KK).mean():.3f} W/mK")

        if quasi_harmonic_greenkubo:
            KK_Q = get_kappa_QHGK(
                v_qssa=solution.v_qssa_cartesian, tau_qs=tau_int,
                w_qs=solution.w_qs, w_inv_qs=solution.w_inv_qs, cv_qs=cv_mesh,
            ) / Nq_eff
            Ks_QHGK[ii] = np.asarray(KK_Q)
            _talk(f"nq={nq:3d}, kappa_QHGK={np.diagonal(KK_Q).mean():.3f} W/mK")

    Ks_da = xr.DataArray(Ks, dims=("nq", *keys.tensor), coords={"nq": nqs})
    if quasi_harmonic_greenkubo:
        Ks_QHGK_da = xr.DataArray(Ks_QHGK, dims=("nq", *keys.tensor), coords={"nq": nqs})

    slope_ab, intercept_ab, correction_ab_stderr, _ = fit_kappa_ladder(
        nqs, Ks_QHGK if quasi_harmonic_greenkubo else Ks)
    correction_ab = intercept_ab - np.asarray(kappa_ha)
    # The scalar fit_* entries report the zz component.
    m_last, y0_last, stderr_last = (slope_ab[2, 2], intercept_ab[2, 2],
                                    correction_ab_stderr[2, 2])

    k_ha = float(np.diagonal(kappa_ha).mean())
    correction = float(np.diagonal(correction_ab).mean())
    nq = len(dmx.q_points) ** (1 / 3)
    correction_factor = 1 + correction / k_ha if k_ha != 0 else 1.0
    correction_factor_err = float(np.diagonal(correction_ab_stderr).mean()) / nq / k_ha if k_ha != 0 else 0.0

    _talk(f"Initial harmonic kappa:   {k_ha:.3f} W/mK")
    _talk(f"Correction:               {correction:.3f} +/- {stderr_last/nq:.3f} W/mK")
    _talk(f"Correction factor:        {correction_factor:.3f}")

    dims_qs = (keys.q_int, keys.s)
    dims_qa = (keys.q_int, keys.a)
    dims_qsa = (keys.q_int, keys.s, keys.a)

    # Internal compute (curve_fit, Qhull, scipy) runs at fp64 for stability;
    # final dataset values follow dmx._dtype_real.
    real_dt = dmx._dtype_real
    _r = lambda x: np.asarray(x, dtype=real_dt)
    results = {
        keys.interpolation_fit_slope: _r(m_last),
        keys.interpolation_fit_intercept: _r(y0_last),
        keys.interpolation_fit_stderr: _r(stderr_last),
        keys.interpolation_correction: _r(correction),
        keys.interpolation_correction_ab: (keys.tensor, _r(correction_ab)),
        keys.interpolation_correction_ab_stderr: (keys.tensor, _r(correction_ab_stderr)),
        keys.interpolation_correction_factor: _r(correction_factor),
        keys.interpolation_correction_factor_err: _r(correction_factor_err),
        keys.interpolation_kappa_array: Ks_da.astype(real_dt),
        keys.interpolation_q_points: (dims_qa, _r(grid.points)),
        keys.interpolation_w_qs: (dims_qs, _r(solution.w_qs)),
        keys.interpolation_tau_qs: (dims_qs, _r(tau_int)),
    }

    if quasi_harmonic_greenkubo:
        results[keys.kappa_ha_QHGK] = kappa_ha.astype(real_dt)
        results[keys.interpolation_kappa_array_QHGK] = Ks_QHGK_da.astype(real_dt)
        results[keys.interpolation_v_qsa] = (dims_qsa, _r(solution.v_qsa_cartesian))

    return results
