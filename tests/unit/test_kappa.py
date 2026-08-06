"""Tests for ``gkmx.kappa.qhgk_tau_eff`` — the QHGK coherence kernel.

The tests that matter here are the off-diagonal ones. The ``s == s'`` diagonal has
``w_s - w_sp = 0``, which makes ``w_minus`` vanish and ``gamma_minus`` collapse to
``1/gamma_ss``, so a diagonal-only test is blind to the frequency unit, to ``tol``,
and to the antiresonant term — i.e. blind to everything the function does beyond
``gamma = 1/(2 tau)``.
"""

from __future__ import annotations

import numpy as np
import pytest

from gkmx.kappa import qhgk_tau_eff

from .._tolerances import TOL_FP64, TOL_FP64_DERIVED

# Written out rather than imported so this file pins the conversion's VALUE, not
# merely its self-consistency. gkmx `w` is rad per ASE time unit; this is rad/fs.
W_TO_RAD_FS = 0.09822694788464063


def _wi(w):
    """Exact ``1/w`` where ``|w|`` is non-trivial, else 0 (acoustic guard)."""
    return np.where(np.abs(w) > 1e-6, 1.0 / w, 0.0)


def test_qhgk_tau_eff_off_diagonal_matches_closed_form():
    """Off-diagonal value against an independently written scalar formula.

    This is the test with teeth: it fails if the rad/fs conversion is dropped,
    rounded to 0.1, or applied to only one of ``w`` / ``w_inv``.
    """
    w0, w1 = 0.12, 0.31                      # gkmx units
    t0, t1 = 900.0, 2500.0                   # fs
    w = np.array([[w0, w1]])
    te = qhgk_tau_eff(w, _wi(w), np.array([[t0, t1]]))

    # independent scalar reconstruction, in rad/fs throughout
    o0, o1 = w0 * W_TO_RAD_FS, w1 * W_TO_RAD_FS
    G = 0.5 / t0 + 0.5 / t1
    w_plus = (o0 + o1) ** 2 / (4 * o0 * o1)
    w_minus = (o0 - o1) ** 2 / (4 * o0 * o1)
    expected = (w_plus * G / (G ** 2 + (o0 - o1) ** 2)
                + w_minus * G / (G ** 2 + (o0 + o1) ** 2))

    assert te[0, 0, 1] == pytest.approx(expected, rel=TOL_FP64_DERIVED)
    assert te[0, 1, 0] == pytest.approx(expected, rel=TOL_FP64_DERIVED)


def test_qhgk_tau_eff_diagonal_equals_lifetime():
    """The ``s == s'`` diagonal collapses to the mode's own lifetime: ``w_plus`` is
    exactly 1, ``w_minus`` exactly 0, and ``gamma_minus`` is ``1/gamma_ss``. This
    pins ``gamma = 1/(2 tau)`` and nothing else — it is invariant under any
    frequency scaling, which is why the off-diagonal tests above exist."""
    rng = np.random.default_rng(1)
    w = rng.uniform(2.0, 40.0, size=(3, 4))
    tau = rng.uniform(50.0, 5000.0, size=(3, 4))
    diag = np.diagonal(qhgk_tau_eff(w, _wi(w), tau), axis1=1, axis2=2)
    assert diag == pytest.approx(tau, rel=TOL_FP64_DERIVED)


def test_qhgk_tau_eff_nan_lifetimes_give_zero():
    """``tau = NaN`` -> ``gamma = inf`` -> Lorentzians ``inf/inf`` -> 0 via nan_to_num.

    The LIFETIME path, not the frequency mask: a NaN lifetime zeroes the mode
    whatever ``tol`` does, so the mask needs its own test with a finite ``tau``.
    """
    w = np.array([[1e-9, 8.0, 20.0]])
    tau = np.array([[np.nan, 400.0, 900.0]])
    te = qhgk_tau_eff(w, _wi(w), tau)
    assert np.all(np.isfinite(te))
    assert te[0, 0, :] == pytest.approx(0.0, abs=0.0)
    assert te[0, 1, 1] > 0.0                 # unmasked modes survive


def test_qhgk_tau_eff_sub_tol_modes_are_masked():
    """A sub-``tol`` mode must be zeroed by the frequency mask alone.

    ``tau`` is finite and ``w_inv`` is passed in finite (not via ``_wi``) so no other
    mechanism can produce the zero — this fails if ``tol`` is disabled.
    """
    w = np.array([[5e-4, 8.0, 20.0]])        # 5e-4 * omega_to_rad_fs = 4.9e-5 < tol
    tau = np.array([[600.0, 400.0, 900.0]])  # finite: gamma cannot do the zeroing
    te = qhgk_tau_eff(w, 1.0 / w, tau)
    assert te[0, 0, :] == pytest.approx(0.0, abs=0.0)
    assert te[0, :, 0] == pytest.approx(0.0, abs=0.0)
    assert te[0, 1, 1] > 0.0


def test_qhgk_tau_eff_soft_modes_do_not_leak_negative():
    """A soft mode has ``w < 0`` and ``w_inv < 0``. If only ``w`` were masked, its
    off-diagonal would survive as ``2*w_plus*gamma_minus`` with a negative sign and
    subtract from kappa."""
    w = np.array([[-0.5, 8.0]])
    te = qhgk_tau_eff(w, 1.0 / w, np.array([[500.0, 400.0]]))
    assert np.all(te >= 0.0)
    assert te[0, 0, 1] == pytest.approx(0.0, abs=0.0)


def test_qhgk_tau_eff_is_symmetric_in_the_mode_pair():
    """``tau_eff[q, s, s'] == tau_eff[q, s', s]``; the kappa sum contracts it against
    a Hermitian ``v_ss'`` and relies on this."""
    rng = np.random.default_rng(2)
    w = rng.uniform(2.0, 30.0, size=(2, 5))
    tau = rng.uniform(50.0, 3000.0, size=(2, 5))
    te = qhgk_tau_eff(w, _wi(w), tau)
    assert te.shape == (2, 5, 5)
    assert te == pytest.approx(np.swapaxes(te, 1, 2), rel=TOL_FP64)


@pytest.mark.parametrize("dt", [np.float32, np.float64])
def test_qhgk_tau_eff_preserves_dtype(dt):
    """Precision-switch contract: both call sites feed pre-cast arrays and rely on
    the dtype flowing through. ``fp32_array * fp64_scalar`` silently promotes, so
    the conversion constant must not widen the result."""
    w = np.array([[3.0, 12.0, 25.0]], dtype=dt)
    te = qhgk_tau_eff(w, _wi(w).astype(dt), np.array([[500.0, 800.0, 1200.0]], dtype=dt))
    assert te.dtype == dt
