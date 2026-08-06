"""numpy and jax backends must agree on the full pipeline.

``TestBatchedBackends`` in the unit suite covers ``DynamicalMatrix``
construction. This file additionally covers ``compute_cv_tau`` (mode
projection + FFT ACF) and ``_harmonic_force_residuals`` (anharmonicity
matmul), whose jax branches have no unit-level coverage.
"""

from __future__ import annotations

import numpy as np
import pytest

from gkmx.greenkubo import get_kappa

from .._tolerances import TOL_DEGENERATE_BASIS, TOL_NUMPY_VS_JAX
from ._helpers import rel_max


@pytest.fixture(scope="module")
def ds_per_backend(tiny_trajectory, tiny_fc_file):
    """Run the full pipeline once per backend; share across assertions."""
    pytest.importorskip("jax")
    return {
        b: get_kappa(
            tiny_trajectory.copy(deep=True),
            fc_file=tiny_fc_file, interpolate=True, nq_max=8, backend=b,
        )
        for b in ("numpy", "jax")
    }


def test_numpy_and_jax_agree_on_kappa_sigma_and_full_tensor(ds_per_backend):
    """Every user-facing scalar (κ_raw, κ_corrected, σ) and the full κ
    tensor must agree between numpy and jax. A divergence here is a
    regression in the jax branch of ``compute_cv_tau`` or
    ``_harmonic_force_residuals``.

    κ_raw and σ are held to the rounding floor: they do not read the
    degenerate eigenvectors, and numpy/jax agree on κ_raw exactly. κ_corrected
    rides the QHGK channel and inherits whichever basis `eigh` picked inside a
    degenerate multiplet, which is a physical arbitrariness rather than a
    numerical one -- hence the far wider band. See tests/_tolerances.py.
    """
    n, j = ds_per_backend["numpy"], ds_per_backend["jax"]

    for name, getter, tol in [
        ("kappa_raw",
         lambda d: np.diag(d["thermal_conductivity"].data).mean(),
         TOL_NUMPY_VS_JAX),
        ("sigma",
         lambda d: float(d.attrs["sigma"]),
         TOL_NUMPY_VS_JAX),
        ("kappa_corrected",
         lambda d: np.diag(d["thermal_conductivity_corrected"].data).mean(),
         TOL_DEGENERATE_BASIS),
    ]:
        a, b = getter(n), getter(j)
        rel = rel_max(a, b)
        assert rel < tol, (
            f"{name}: numpy={a:.6f} jax={b:.6f} rel={rel:.2e} (tol {tol:.0e})")

    # Full tensor agreement catches axis-mixing bugs the diagonal scalars
    # would average out. Same channel as κ_corrected, so the same band.
    a = np.asarray(n["thermal_conductivity_corrected"].data)
    b = np.asarray(j["thermal_conductivity_corrected"].data)
    assert np.allclose(a, b, rtol=TOL_DEGENERATE_BASIS,
                       atol=1e-5 * np.abs(a).max())
