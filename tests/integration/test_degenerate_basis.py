"""Bound how far the kappa *estimate* moves when the degenerate basis is re-chosen.

Physically kappa cannot depend on it: inside a multiplet any unitary mixture of
the eigenvectors is equally valid, so the choice carries no information. What
follows the choice is our estimator -- tau and cv are fitted per mode on the
projected amplitudes, which mix under the rotation, and on a short trajectory
those fits are noisy enough for the mixing to show.

So this is a convergence artifact, not a property of kappa, and it shrinks with
trajectory length: the KI Gamma triple spreads 2.98x over 4 ps and 1.17x over
100 ps, while the fit to the invariant trace of g holds at ~420 fs throughout.
The bound below is therefore a property of this 4 ps fixture. `eigh` does pick
differently across precisions and backends (see test_precision_switch.py,
test_backends_agree.py), which is what makes the artifact visible there too.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from gkmx import phonon
from gkmx.greenkubo import get_kappa
from gkmx.phonon import degenerate_sets

# Averaging gamma/cv over the multiplet suppresses the sampling noise the basis
# rotation exposes, but does not remove it: tau comes from a nonlinear threshold
# fit and cv is a 4th-order moment, so neither has an invariant multiplet mean.
# Exact invariance needs the fit taken on the invariant trace of g. Measured on
# this fixture: 2.5e-4 before the averaging, 5.9e-5 after -- so require at least
# a 3x reduction against the pre-averaging value rather than pretending the
# residual is zero.
SWING_BEFORE_AVERAGING = 2.51e-4
MAX_SWING = SWING_BEFORE_AVERAGING / 3


def _kappa_with_rotated_degenerate_basis(traj, fc, seed):
    """Run the pipeline with an extra random unitary applied to each multiplet."""
    original = phonon._rotate_degenerate_subspaces

    def patched(w2, e, M, probe=phonon._PERTURB_PROBE):
        e, M = original(w2, e, M, probe)
        if seed is None:
            return e, M
        rng = np.random.default_rng(seed)
        e, M = np.array(e, copy=True), np.array(M, copy=True)
        for qi, start, stop in degenerate_sets(np.sqrt(np.abs(w2))):
            n = stop - start
            G = slice(start, stop)
            U, _ = np.linalg.qr(rng.standard_normal((n, n))
                                + 1j * rng.standard_normal((n, n)))
            e[qi, G, :] = U.T @ e[qi, G, :]
            for a in range(3):
                M[qi, a, G, :] = U.conj().T @ M[qi, a, G, :]
                M[qi, a, :, G] = M[qi, a, :, G] @ U
        return e, M

    phonon._rotate_degenerate_subspaces = patched
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ds = get_kappa(traj.copy(deep=True), fc_file=fc, interpolate=True,
                           nq_max=8, backend="numpy")
        return float(np.mean(np.diag(ds["thermal_conductivity_corrected"].data)))
    finally:
        phonon._rotate_degenerate_subspaces = original


@pytest.mark.integration
def test_kappa_is_stable_under_degenerate_basis_rotation(tiny_trajectory, tiny_fc_file):
    """The estimate follows the basis choice on this short fixture; hold it bounded."""
    values = np.array([
        _kappa_with_rotated_degenerate_basis(tiny_trajectory, tiny_fc_file, s)
        for s in (None, 0, 1, 2)
    ])
    swing = (values.max() - values.min()) / abs(values.mean())
    assert swing < MAX_SWING, (
        f"kappa_corrected swings {swing:.2e} across degenerate-basis rotations "
        f"(limit {MAX_SWING:.2e}); multiplet averaging in compute_cv_tau may have "
        f"regressed. values={values}"
    )


@pytest.mark.integration
def test_degenerate_multiplets_share_tau_and_cv(tiny_trajectory, tiny_fc_file):
    """The averaging must actually fire: symmetry-equivalent modes carry one
    lifetime and one heat capacity. Guards against `degenerate_sets` and the
    averaging drifting apart on the tolerance that defines a multiplet."""
    from gkmx import keys
    with contextlib.redirect_stdout(io.StringIO()):
        ds = get_kappa(tiny_trajectory.copy(deep=True), fc_file=tiny_fc_file,
                       interpolate=False, backend="numpy")
    tau = np.asarray(ds[keys.mode_lifetime].data)
    cv = np.asarray(ds[keys.mode_heat_capacity].data)
    w = np.abs(np.asarray(ds["w_qs"].data))

    blocks = list(degenerate_sets(w))
    assert blocks, "fixture has no degenerate multiplets; this test is vacuous"
    for qi, start, stop in blocks:
        for name, arr in (("tau", tau), ("cv", cv)):
            blk = arr[qi, start:stop]
            finite = blk[np.isfinite(blk)]
            if finite.size > 1:
                assert np.allclose(finite, finite[0], rtol=1e-6), (
                    f"{name} differs across the multiplet at q={qi} "
                    f"modes {start}..{stop - 1}: {blk}")
