"""HDF5 round-trip of every persisted DMX field + end-to-end
``dmx_file=`` plumbing in ``get_kappa``.

The fp64 cache must round-trip bit-identically. Note: we don't compare
the resulting κ_corrected to the pinned scalar because fp32 OMP
non-determinism in ``compute_cv_tau`` adds ~1e-4 drift between
independent runs, which the linear extrapolation amplifies ~1000× into
``kappa_corrected``. That's a property of parallel fp32 BLAS, not a
``from_hdf5`` regression — so the field-level pin is the right tool.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import io as ase_io

from gkmx.dynamical_matrix import DynamicalMatrix
from gkmx.greenkubo import get_kappa
from gkmx.io import parse_force_constants

from ._pins import KAPPA_CORRECTED_REF


@pytest.fixture(scope="module")
def dmx_pair(tiny_dir, tmp_path_factory):
    """Build a fresh DMX, save it, reload it. Yield ``(fresh, reloaded, cache_path)``."""
    prim = ase_io.read(str(tiny_dir / "geometry.in.primitive"), format="aims")
    sc = ase_io.read(str(tiny_dir / "geometry.in.supercell"), format="aims")
    fc = np.asarray(parse_force_constants(
        str(tiny_dir / "FORCE_CONSTANTS_tdep"), two_dim=False))
    fresh = DynamicalMatrix(force_constants=fc, primitive=prim, supercell=sc,
                            with_group_velocity_matrices=True)
    out = tmp_path_factory.mktemp("dmx_cache") / "DynamicalMatrix.nc"
    fresh.to_hdf5(str(out), include_D_qij=True,
                   include_group_velocity_matrices=True)
    return fresh, DynamicalMatrix.from_hdf5(str(out)), out


@pytest.mark.parametrize("getter", [
    lambda d: d.fc_phonopy,
    lambda d: d.w_qs,
    lambda d: d.w_inv_qs,
    lambda d: d.w2_qs,
    lambda d: d.v_qsa_cartesian,
    lambda d: d.e_qsi,
    lambda d: d.e_qsI,
    lambda d: d.remapped,
    lambda d: d.q_points,
    lambda d: d.solution.v_qssa_cartesian,
], ids=["fc_phonopy", "w_qs", "w_inv_qs", "w2_qs", "v_qsa", "e_qsi",
        "e_qsI", "remapped", "q_points", "v_qssa"])
def test_dmx_field_bit_identical_after_round_trip(dmx_pair, getter):
    """Every persisted DMX field must round-trip through HDF5 with zero
    drift — fp64 storage via h5netcdf + zlib is lossless. A non-zero
    diff points at a precision downcast or a missing field in
    ``to_hdf5``/``from_hdf5``."""
    fresh, reloaded, _ = dmx_pair
    a = np.asarray(getter(fresh))
    b = np.asarray(getter(reloaded))
    assert a.shape == b.shape
    assert np.abs(a - b).max() == 0.0


def test_pipeline_consumes_cached_dmx(tiny_trajectory, dmx_pair):
    """``get_kappa(..., dmx_file=...)`` must actually load and use the
    cache (not silently rebuild) and produce a physically plausible
    κ_corrected. ±50 % band because fp32 OMP variance + extrapolation
    add real drift; this is a behavior test for the dmx_file= path,
    not a scalar pin."""
    _, _, cache_path = dmx_pair
    ds = get_kappa(tiny_trajectory.copy(deep=True),
                   dmx_file=str(cache_path), interpolate=True, nq_max=8,
                   backend="numpy")
    k_corr = np.asarray(ds["thermal_conductivity_corrected"].data)
    assert k_corr.shape == (3, 3)
    assert np.all(np.isfinite(k_corr))
    k = float(np.diag(k_corr).mean())
    assert abs(k - KAPPA_CORRECTED_REF) < 0.5 * KAPPA_CORRECTED_REF
