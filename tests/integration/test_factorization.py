"""Smoke + plumbing consistency for the ``factorization`` kwarg.

Both ``"wick"`` and ``"vertex"`` paths must run end-to-end and share the
``factorization``-invariant outputs (``thermal_conductivity`` HFACF κ
and ``mode_heat_capacity``); only τ should differ. Pinned on both an
MLIP simple crystal (KI) and an ab-initio rare-event slice (CuI), since
the schema and dtype mix differ between the two trajectory styles.

Physical κ-ratio claims are NOT made — the 1000-step in-repo slices are
too short. Those live in ``dev_scripts/bench_factorization_converged.py``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from gkmx.greenkubo import get_kappa


def _run_both(trajectory, fc_file):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            f: get_kappa(trajectory.copy(deep=True), fc_file=fc_file,
                         interpolate=False, backend="numpy",
                         precision="fp64", factorization=f)
            for f in ("wick", "vertex")
        }


@pytest.mark.parametrize("traj_fixture,fc_fixture", [
    ("tiny_trajectory", "tiny_fc_file"),  # MLIP simple crystal (KI)
    ("cui_trajectory",  "cui_fc_file"),   # ab-initio rare-event slice (CuI)
], ids=["KI", "CuI"])
def test_wick_and_vertex_share_invariants(request, traj_fixture, fc_fixture):
    """On both KI (MLIP) and CuI (ab-initio):
        - both factorizations fit τ on > 10% of modes
        - ``thermal_conductivity`` (HFACF κ) is bit-identical (HFACF
          path doesn't read ``factorization``)
        - ``mode_heat_capacity`` is bit-identical (same Var-of-|a|²)
    Only τ differs between branches."""
    traj = request.getfixturevalue(traj_fixture)
    fc = request.getfixturevalue(fc_fixture)
    out = _run_both(traj, fc)

    for f in ("wick", "vertex"):
        tau = np.asarray(out[f]["mode_lifetime"])
        fit = np.isfinite(tau) & (tau > 0)
        assert fit.sum() > 0.1 * tau.size, (
            f"{f}: fit only {fit.sum()}/{tau.size} modes"
        )

    np.testing.assert_array_equal(
        np.asarray(out["wick"]["thermal_conductivity"]),
        np.asarray(out["vertex"]["thermal_conductivity"]),
        err_msg="HFACF κ must be factorization-independent",
    )
    np.testing.assert_array_equal(
        np.asarray(out["wick"]["mode_heat_capacity"]),
        np.asarray(out["vertex"]["mode_heat_capacity"]),
        err_msg="mode_heat_capacity must be factorization-independent",
    )
