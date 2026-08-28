"""Align eigenvector remapping across Bloch conventions.

Each convention reports `e_qsi` in the Bloch representative its `v_ss'` is
defined in. `_build_e_qsI` remaps them onto the supercell against the phase
origin that representative pairs with, so every convention's projector selects the same
physical branch. The mode amplitude is then convention-free, and so is anything
fitted from it: outside a degenerate multiplet tau is identical in all of them,
which is what this test measures.

Resolution scales with the per-atom phase `exp(-2 pi i q . r_a)`, so the sharper
of the two fixtures is `Ga2O3_alpha_aiGK` (10-atom R-3c, general Wyckoff
positions) rather than the 2-atom cubic `KI_B2_MLIP`.
`tests/unit/test_bloch_projector.py` checks the projector itself across 13
geometries, no trajectory needed.

Inside a multiplet the basis is free, so those modes are excluded
(test_degenerate_basis.py bounds what the choice does to the estimator there).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from gkmx import keys
from gkmx.greenkubo import get_kappa
from gkmx.phonon import CONVENTIONS, DEGENERACY_TOL

from .._tolerances import TOL_FP64_DERIVED

DATA_ROOT = Path(__file__).parent.parent / "datasets"

# fixture directory -> FC file, or None when the trajectory carries its own.
FIXTURES = {
    "Ga2O3_alpha_aiGK": None,
    "KI_B2_MLIP": "FORCE_CONSTANTS_tdep",
}


@pytest.fixture(scope="module", params=sorted(FIXTURES))
def kappa_per_convention(request):
    name = request.param
    traj = DATA_ROOT / name / "nve" / "000000.nc"
    if not traj.exists():
        pytest.skip(f"trajectory fixture not found: {traj}")
    fc_name = FIXTURES[name]
    fc_file = str(DATA_ROOT / name / fc_name) if fc_name else None
    ds = xr.open_dataset(traj, engine="h5netcdf").load()
    out = {}
    for c in CONVENTIONS:
        out[c] = get_kappa(ds, fc_file=fc_file, interpolate=False,
                           precision="fp64", convention=c)
    return name, out


def _non_degenerate(w_qs, tol=DEGENERACY_TOL):
    """Mask of modes whose frequency is isolated within its q-point."""
    gap = np.abs(w_qs[:, :, None] - w_qs[:, None, :])
    np.einsum("qss->qs", gap)[...] = np.inf
    return gap.min(axis=-1) > tol


def test_lifetimes_are_convention_independent(kappa_per_convention):
    name, per_conv = kappa_per_convention
    ref = per_conv["RAW"]
    w = np.asarray(ref[keys.w_qs])
    tau_ref = np.asarray(ref[keys.mode_lifetime])
    live = _non_degenerate(w) & np.isfinite(tau_ref) & (tau_ref > 0)
    assert live.sum() > 0.2 * tau_ref.size, "too few isolated modes to test"

    for conv, ds in per_conv.items():
        tau = np.asarray(ds[keys.mode_lifetime])
        sel = live & np.isfinite(tau)
        rel = np.abs(tau[sel] - tau_ref[sel]) / tau_ref[sel]
        assert rel.max() < TOL_FP64_DERIVED, (
            f"{name} [{conv}]: tau on isolated modes differs from RAW by "
            f"{rel.max():.2e} -- the convention is reaching the mode projection")


def test_raw_kappa_is_convention_independent(kappa_per_convention):
    """Control: raw GK never touches modes, so it must be bit-identical."""
    name, per_conv = kappa_per_convention
    ref = float(np.trace(np.asarray(per_conv["RAW"][keys.kappa])))
    for conv, ds in per_conv.items():
        assert float(np.trace(np.asarray(ds[keys.kappa]))) == ref, f"{name} [{conv}]"
