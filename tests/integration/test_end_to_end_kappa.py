"""End-to-end κ regression test on the shipped KI_B2_MLIP trajectory.

Mirrors the vibes_anisotropy test_sigma pattern: run the full pipeline
on a checked-in fixture and assert the scientific answer lands in a
reasonable band. Catches regressions that silently change κ (a wrong
unit conversion, a wrong GK window, a broken mode-decomposition) that
the lower-level invariant tests in ``test_dynamical_matrix.py`` would
miss.

``KI_B2_MLIP`` is a trimmed version of KI_B2_n128 (same primitive, same
FC, same reference DMX) with the trajectory cut to the first **200 MD
steps** so the whole fixture fits in the repo (~3 MB). The 200 steps
exercise the full Green-Kubo pipeline code path end-to-end but are
**nowhere near enough to produce a converged κ** — the phonon lifetimes
in KI are ~1–10 ps and 200 fs of trajectory only samples a fraction of
one relaxation. The pins in `tests/_pins.py` therefore catch numerical
regressions, not physics convergence; the converged-κ regression bank
lives in `gkmx/benchmarks/bench_pipeline.py` (driven against the
out-of-tree `gkmx_project/datasets/KI_B2_n128/` and beyond, never
imported by the release-time test suite).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from gkmx import keys
from gkmx.greenkubo import get_kappa

from ._pins import KAPPA_CORRECTED_REF, KAPPA_CORRECTED_TOL, KAPPA_RAW_REF, KAPPA_RAW_TOL

DATA_DIR = Path(__file__).parent.parent / "datasets" / "KI_B2_MLIP"


@pytest.fixture(scope="module")
def trajectory():
    path = DATA_DIR / "nve" / "000000.nc"
    if not path.exists():
        pytest.skip(f"trajectory fixture not found: {path}")
    return xr.open_dataset(path).load()


@pytest.fixture(scope="module")
def fc_file():
    path = DATA_DIR / "FORCE_CONSTANTS_tdep"
    if not path.exists():
        pytest.skip(f"FC fixture not found: {path}")
    return str(path)


class TestEndToEndKappa:
    """Pin the KI_B2_MLIP κ output to the values in `tests/_pins.py`.

    The shipped 200-step trajectory is not enough to converge κ. These
    tests catch numerical regressions in the pipeline (broken units,
    wrong GK window, mode-decomposition drift, dD/dq Hermitization
    changes), not physics convergence. The same pins are consumed by
    `tests/integration/test_kappa_convergence.py` — single source of
    truth in `tests/_pins.py` so a pin update lands once.
    """

    @pytest.mark.parametrize("backend", ["numpy", "jax"])
    def test_kappa_no_interp(self, trajectory, fc_file, backend):
        """Raw κ pinned within the band in `_pins.KAPPA_RAW_TOL`. fp32
        and fp64 agree bit-identically post the 2026-04-19 dtype-
        propagation pass, so a single ref serves both backends."""
        ds = get_kappa(
            trajectory.copy(deep=True),
            fc_file=fc_file,
            interpolate=False,
            backend=backend,
        )
        assert "thermal_conductivity" in ds
        k = np.asarray(ds["thermal_conductivity"].data)
        assert k.shape == (3, 3)
        assert np.all(np.isfinite(k))
        k_scalar = float(np.mean(np.diag(k)))
        assert abs(k_scalar - KAPPA_RAW_REF) < KAPPA_RAW_TOL, (
            f"{backend}: kappa diag-mean = {k_scalar:.6f} W/mK; "
            f"ref = {KAPPA_RAW_REF} W/mK (tol {KAPPA_RAW_TOL})"
        )

    @pytest.mark.parametrize("backend", ["numpy", "jax"])
    def test_kappa_corrected(self, trajectory, fc_file, backend):
        """Extrapolated κ pinned within `_pins.KAPPA_CORRECTED_TOL`. The
        size-extrapolation linear fit amplifies any upstream lifetime
        drift, so this is the most sensitive single regression scalar."""
        ds = get_kappa(
            trajectory.copy(deep=True),
            fc_file=fc_file,
            interpolate=True,
            nq_max=8,
            backend=backend,
        )
        assert "thermal_conductivity_corrected" in ds
        k_corr = np.asarray(ds["thermal_conductivity_corrected"].data)
        assert k_corr.shape == (3, 3)
        assert np.all(np.isfinite(k_corr))
        k_scalar = float(np.mean(np.diag(k_corr)))
        assert abs(k_scalar - KAPPA_CORRECTED_REF) < KAPPA_CORRECTED_TOL, (
            f"{backend}: kappa_corrected diag-mean = {k_scalar:.6f} W/mK; "
            f"ref = {KAPPA_CORRECTED_REF} W/mK (tol {KAPPA_CORRECTED_TOL})"
        )


@pytest.mark.parametrize("traj_fixture,fc_fixture", [
    ("trajectory", "fc_file"),          # KI: every mode fits a lifetime
    ("cui_trajectory", "cui_fc_file"),  # CuI: 3 of 648 fits fail
], ids=["KI", "CuI"])
def test_analytical_emits_bte_and_qhgk_hfacfs(request, traj_fixture, fc_fixture):
    """``analytical=True`` lights up the BTE / QHGK time-resolved HFACFs
    branch in ``_analytical_hfacfs``. Pins shape + finiteness on all four
    added arrays.

    CuI is in the parametrization because it is the only in-repo fixture
    carrying failed lifetime fits, which reach the QHGK kernel as γ = inf
    and NaN the whole einsum — KI has none, so it cannot see that path.
    The count is asserted below so the coverage cannot quietly vanish.
    """
    traj = request.getfixturevalue(traj_fixture)
    fc = request.getfixturevalue(fc_fixture)
    ds = get_kappa(
        traj.copy(deep=True),
        fc_file=fc,
        interpolate=False,
        backend="numpy",
        analytical=True,
    )
    Nt = ds["heat_flux_acf"].sizes[keys.time]
    for name in ("heat_flux_BTE_acf",
                 "heat_flux_BTE_acf_integral",
                 "heat_flux_QHGK_acf",
                 "heat_flux_QHGK_acf_integral"):
        arr = np.asarray(ds[name].data)
        assert arr.shape == (Nt, 3, 3), f"{name} has shape {arr.shape}"
        assert np.all(np.isfinite(arr)), f"{name} contains NaN / inf"

    n_failed = int(np.isnan(np.asarray(ds[keys.mode_lifetime].data)).sum())
    if traj_fixture.startswith("cui"):
        assert n_failed > 0, (
            "CuI no longer carries failed lifetime fits; the finiteness "
            "assertion above no longer covers the γ = inf path")
