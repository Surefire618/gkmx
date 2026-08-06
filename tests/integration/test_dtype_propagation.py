"""End-to-end dtype propagation in ``get_kappa``.

The 2026-04-19 dtype audit found 11 separate fp64 leaks (HFACF temps,
ts/ks results, q_points, volumes, force_constants, …) that the
per-kernel unit tests ignored because they inspect kernel outputs in
isolation. This file walks the assembled output dataset and asserts
dtype matches the requested precision throughout, plus pins that an
explicit ``precision=`` kwarg overrides the process-level default.
"""
from __future__ import annotations

import numpy as np
import pytest

from gkmx import precision
from gkmx.greenkubo import get_kappa
from gkmx.precision import Precision

_GKMX_PRODUCED_ATTRS = {"sigma", "gk_window_fs", "gk_prefactor",
                        "filter_prominence"}


def _dtype_leaks(ds, p):
    """Return ``[(name, dtype, expected), ...]`` for floats/complexes
    in the dataset that don't match ``p``. Trajectory-inherited attrs
    are NOT audited (the precision switch only owns gkmx-produced ones)."""
    leaks = []
    for name in list(ds.data_vars) + list(ds.coords):
        dt = ds[name].dtype
        if dt.kind == "f" and dt != p.real:
            leaks.append((name, str(dt), str(np.dtype(p.real))))
        elif dt.kind == "c" and dt != p.complex:
            leaks.append((name, str(dt), str(np.dtype(p.complex))))
    for name in _GKMX_PRODUCED_ATTRS:
        if name in ds.attrs and isinstance(ds.attrs[name],
                                            (np.floating, np.complexfloating)):
            dt = np.dtype(type(ds.attrs[name]))
            if dt.kind == "f" and dt != p.real:
                leaks.append((f"attrs[{name!r}]", str(dt), str(np.dtype(p.real))))
    return leaks


@pytest.fixture(scope="module", autouse=True)
def _restore_default_precision():
    original = precision.get_default()
    yield
    precision.set_default(original)


@pytest.mark.parametrize("requested", ["fp32", "fp64"])
def test_full_pipeline_dtype_uniform(tiny_trajectory, tiny_fc_file, requested):
    """Run ``get_kappa(..., precision=requested)`` end-to-end; every
    float/complex in the assembled dataset must match the requested
    precision. A leak from any kernel — HFACF temp arrays, q_points,
    interpolation results — fails this test."""
    ds = get_kappa(
        tiny_trajectory.copy(deep=True),
        fc_file=tiny_fc_file, interpolate=True, nq_max=8,
        backend="numpy", precision=requested,
    )
    leaks = _dtype_leaks(ds, Precision.from_str(requested))
    assert not leaks, (
        f"{requested} pipeline leaked: "
        + "; ".join(f"{n}={d} (expected {e})" for n, d, e in leaks)
    )


@pytest.mark.parametrize("global_,kwarg_,expected", [
    ("fp32", "fp64", np.float64),
    ("fp64", "fp32", np.float32),
])
def test_precision_kwarg_overrides_global(
        tiny_trajectory, tiny_fc_file, global_, kwarg_, expected):
    """An explicit ``precision=`` kwarg wins over ``set_default``, in
    both directions. This is the library-safety contract a caller
    relies on when threading precision through a wrapping API."""
    precision.set_default(global_)
    ds = get_kappa(
        tiny_trajectory.copy(deep=True),
        fc_file=tiny_fc_file, interpolate=False,
        backend="numpy", precision=kwarg_,
    )
    assert ds["thermal_conductivity"].dtype == expected
