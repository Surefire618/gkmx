"""fp32 and fp64 precision modes must produce the same κ on the in-repo fixture.

A regression that introduces a silent fp64 island in fp32 mode (or
loses precision in fp32) breaks the agreement and fails here.
"""

from __future__ import annotations

import numpy as np
import pytest

from gkmx import precision
from gkmx.greenkubo import get_kappa

from .._tolerances import TOL_DEGENERATE_BASIS, TOL_FP32_VS_FP64
from ._helpers import rel_max


@pytest.fixture(scope="module", autouse=True)
def _restore_default_precision():
    original = precision.get_default()
    yield
    precision.set_default(original)


@pytest.fixture(scope="module")
def ds_at_each_precision(tiny_trajectory, tiny_fc_file):
    """Run the full pipeline once per precision; share across assertions."""
    out = {}
    for p in ("fp32", "fp64"):
        precision.set_default(p)
        out[p] = get_kappa(
            tiny_trajectory.copy(deep=True),
            fc_file=tiny_fc_file, interpolate=True, nq_max=8,
            backend="numpy",
        )
    return out


@pytest.mark.parametrize("name,getter,tol", [
    ("kappa_corrected",
     lambda ds: np.diag(ds["thermal_conductivity_corrected"].data).mean(),
     TOL_DEGENERATE_BASIS),
    ("kappa_raw",
     lambda ds: np.diag(ds["thermal_conductivity"].data).mean(),
     TOL_FP32_VS_FP64),
    ("sigma",
     lambda ds: float(ds.attrs["sigma"]),
     TOL_FP32_VS_FP64),
])
def test_fp32_vs_fp64_agree(ds_at_each_precision, name, getter, tol):
    """fp32 and fp64 must agree to `tol`.

    κ_raw and σ are held to the rounding floor. κ_corrected rides the QHGK
    channel, where a narrower mantissa makes `eigh` pick a different — equally
    valid — basis inside a degenerate multiplet, so its band is set by that
    arbitrariness rather than by precision. See tests/_tolerances.py."""
    a = getter(ds_at_each_precision["fp32"])
    b = getter(ds_at_each_precision["fp64"])
    rel = rel_max(a, b)
    assert rel < tol, f"{name}: fp32={a:.6f} fp64={b:.6f} rel={rel:.2e} (tol {tol:.0e})"
