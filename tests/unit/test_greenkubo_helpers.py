"""Tests for the private helpers in ``gkmx.greenkubo``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from ase import io as ase_io, units

from gkmx import _constants as C
from gkmx import keys
from gkmx.dynamical_matrix import DynamicalMatrix
from gkmx.greenkubo import (
    _autocorrelation_Nd,
    _correlate,
    _cumtrapz,
    _fit_tau,
    _gk_prefactor,
    compute_cv_tau,
)
from gkmx.io import parse_force_constants

from .._tolerances import TOL_FP64

DATA_DIR = Path(__file__).parent.parent / "datasets" / "KI_B2_MLIP"


def test_correlate_zero_lag_norm_and_hann_taper():
    """``_correlate`` returns an (Nt,) array; at zero lag it equals
    ⟨x²⟩ for normalize=2; the Hann taper zeros the last element."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(128)
    corr = _correlate(x, x, normalize=2, hann=False)
    assert corr.shape == (128,)
    assert corr[0] == pytest.approx(float((x * x).sum()) / 128, abs=TOL_FP64)
    # Hann taper: last lag → 0 for a constant input.
    c_no = _correlate(np.ones(64), np.ones(64), normalize=2, hann=False)
    c_h = _correlate(np.ones(64), np.ones(64), normalize=2, hann=True)
    assert c_no[-1] == pytest.approx(1.0)
    assert c_h[-1] == pytest.approx(0.0, abs=TOL_FP64)


def test_autocorrelation_Nd_trace_identity():
    """Diagonal ACF summed over the d-axis must equal the trace of the
    full off-diagonal ACF — the rotation invariant that ties the two
    code paths together."""
    rng = np.random.default_rng(0)
    x = xr.DataArray(rng.standard_normal((32, 3)), dims=(keys.time, "d"))
    diag = _autocorrelation_Nd(x, off_diagonal=False).sum(dim="d")
    full = _autocorrelation_Nd(x, off_diagonal=True)
    trace = sum(full.values[:, i, i] for i in range(3))
    np.testing.assert_allclose(diag.values, trace, atol=TOL_FP64)
    assert diag.dims == (keys.time,)


def test_cumtrapz_ndarray_and_xarray():
    """Cumulative trapezoid of constant 1 with unit spacing → 0, 1, 2, ...
    Must work for both ndarray (returned as ndarray) and xarray
    (returned as DataArray)."""
    expected = np.arange(10, dtype=float)
    np.testing.assert_allclose(_cumtrapz(np.ones(10)), expected, atol=TOL_FP64)
    da = xr.DataArray(np.ones(10), dims=(keys.time,),
                      coords={keys.time: np.arange(10, dtype=float)})
    out = _cumtrapz(da)
    assert isinstance(out, xr.DataArray)
    np.testing.assert_allclose(out.values, expected, atol=TOL_FP64)


def test_fit_tau_recovers_decay_and_scales_with_dt():
    """For ``g(t) = exp(-t/τ₀)``, ``_fit_tau`` recovers ``τ₀·dt`` to a
    fraction of a percent; doubling dt doubles the reported lifetime."""
    tau0 = 50.0
    g = np.exp(-np.arange(300, dtype=float) / tau0)
    tau1 = _fit_tau(g, dt=1.0, thresh=0.1)
    tau2 = _fit_tau(g, dt=2.0, thresh=0.1)
    assert tau1 == pytest.approx(tau0, rel=0.02)
    assert tau2 == pytest.approx(2.0 * tau1, rel=1e-6)


def test_fit_tau_returns_nan_on_silent_and_non_decaying():
    """No-signal and never-crosses-threshold inputs must return NaN
    (not a spurious extrapolated number)."""
    assert np.isnan(_fit_tau(np.zeros(100), dt=1.0))
    assert np.isnan(_fit_tau(np.ones(100), dt=1.0, thresh=0.5))


def test_gk_prefactor_closed_form_and_scaling():
    """Prefactor matches ``V / (k_B T²) · to_W_mK`` and scales correctly
    with V (linear) and T (1/T²)."""
    V, T = 1000.0, 300.0
    pf = _gk_prefactor(V, T, verbose=False)
    assert pf == pytest.approx(V / units.kB / T**2 * C.to_W_mK, rel=TOL_FP64)
    assert _gk_prefactor(V, 2 * T, verbose=False) / pf == pytest.approx(0.25, rel=TOL_FP64)
    assert _gk_prefactor(2 * V, T, verbose=False) / pf == pytest.approx(2.0, rel=TOL_FP64)


# ---------------------------------------------------------------------------
# factorization=wick vs vertex
# ---------------------------------------------------------------------------

def test_factorization_invalid_raises():
    """Bad ``factorization`` value must reject at the entry point."""
    class _Dummy:
        pass
    with pytest.raises(ValueError, match="factorization"):
        compute_cv_tau(dataset=xr.Dataset(), dmx=_Dummy(),
                       factorization="bubble")


def test_factorization_paths_share_cv_and_mask_acoustic():
    """End-to-end on the KI fixture: both ``"wick"`` and ``"vertex"``
    produce the same ``cv_qs`` (the fork only affects τ, not the Var-of-|a|²
    estimator), both NaN-mask the acoustic Γ modes, and ``"vertex"`` fits
    τ on a non-trivial fraction of modes."""
    if not (DATA_DIR / "nve" / "000000.nc").exists():
        pytest.skip(f"fixture missing: {DATA_DIR}")
    prim = ase_io.read(str(DATA_DIR / "geometry.in.primitive"), format="aims")
    sc = ase_io.read(str(DATA_DIR / "geometry.in.supercell"), format="aims")
    fc = parse_force_constants(str(DATA_DIR / "FORCE_CONSTANTS_tdep"),
                                two_dim=False)
    dmx = DynamicalMatrix(force_constants=np.asarray(fc),
                          primitive=prim, supercell=sc, backend="numpy")
    ds = xr.open_dataset(str(DATA_DIR / "nve" / "000000.nc")).load()
    ds["displacements"] = ((keys.time, "I", "a"),
                            ds["positions"].data - sc.positions)
    ds.attrs["masses"] = sc.get_masses()

    cv_w, tau_w = compute_cv_tau(ds, dmx, factorization="wick")
    cv_v, tau_v = compute_cv_tau(ds.copy(deep=True), dmx,
                                  factorization="vertex")

    np.testing.assert_allclose(cv_w, cv_v, rtol=1e-10, atol=0)
    gamma = np.asarray(dmx.w_qs) < 1e-6
    assert np.all(np.isnan(np.asarray(tau_w)[gamma]))
    assert np.all(np.isnan(np.asarray(tau_v)[gamma]))
    assert np.mean(np.isfinite(tau_v) & (np.asarray(tau_v) > 0)) > 0.1


def test_factorization_paths_agree_on_synthetic_gaussian():
    """Wick's theorem: for a clean stationary Gaussian mode amplitude,
    the un-factorized ``⟨n(t) n(0)⟩`` and Wick-reduced ``|g(t)|²`` paths
    must recover the same τ to within sampling noise. Catches a
    factor-of-2 convention bug in either fork."""
    from scipy.fft import fft, ifft, next_fast_len

    rng = np.random.default_rng(42)
    Nt, tau0 = 20000, 80.0
    decay = np.exp(-1.0 / tau0)
    sigma = np.sqrt(1.0 - decay ** 2)
    a = np.empty(Nt, dtype=np.complex128)
    a[0] = rng.standard_normal() + 1j * rng.standard_normal()
    for t in range(1, Nt):
        a[t] = decay * a[t - 1] + sigma * (rng.standard_normal()
                                            + 1j * rng.standard_normal())
    a -= a.mean()
    nfft = next_fast_len(2 * Nt)
    norm = np.arange(Nt, 0, -1)

    F = fft(a, n=nfft)
    Cg = ifft(F * np.conj(F))[:Nt] / norm
    g2 = (Cg.real ** 2 + Cg.imag ** 2)
    g2 /= g2[0]
    tau_w = _fit_tau(g2, dt=1.0, thresh=0.5)

    n = np.abs(a) ** 2
    Fn = fft(n - n.mean(), n=nfft)
    Cnn = ifft(Fn * np.conj(Fn))[:Nt].real / norm
    Cnn /= Cnn[0]
    tau_v = _fit_tau(Cnn, dt=1.0, thresh=0.5)

    assert np.isfinite(tau_w) and np.isfinite(tau_v)
    rel = abs(tau_w - tau_v) / tau_w
    assert rel < 0.15, (
        f"τ_wick={tau_w:.3f}, τ_vertex={tau_v:.3f}, rel={rel:.3f}; "
        "the two factorization paths disagree on a clean-SMA synthetic."
    )
