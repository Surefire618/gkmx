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
from ase import units

from gkmx import DynamicalMatrix, Precision, keys
from gkmx.greenkubo import get_kappa
from gkmx.harmonic_flux import (
    compute_harmonic_heat_flux_disp,
    compute_harmonic_heat_flux_q,
    compute_harmonic_heat_flux_R,
)
from gkmx.io import json2atoms
from gkmx.lattice_points import get_pair_vectors
from gkmx.phonon import CONVENTIONS

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
        """Raw κ pinned within the band in `_pins.KAPPA_RAW_TOL`, at
        fp64 (the precision the pins are baked at; the default fp32
        path lands 7e-9 away on this scalar)."""
        ds = get_kappa(
            trajectory.copy(deep=True),
            fc_file=fc_file,
            interpolate=False,
            backend=backend,
            precision="fp64",
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
            precision="fp64",
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


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
def test_harmonic_flux_emits_three_measured_fluxes(trajectory, fc_file,
                                                   precision):
    """``harmonic_flux=True`` lights up ``compute_harmonic_heat_flux_R/_q``.

    Pins shape, dims, dtype and finiteness on the three fluxes and their six
    ACF / integral arrays, and pins that all nine share the per-MD-step
    ``time_md`` axis with ``sigma_per_sample`` rather than introducing a
    second axis of the same length.

    ``precision`` is passed to ``get_kappa`` rather than set process-wide on
    purpose: ``Precision.resolve`` does not touch the global default, so a
    kernel that falls back to ``Precision.default()`` instead of taking the
    resolved dtype is only visible when the two disagree.
    """
    traj = trajectory.copy(deep=True)
    Nt = traj.sizes[keys.time]
    ds = get_kappa(traj, fc_file=fc_file, interpolate=False,
                   backend="numpy", harmonic_flux=True, precision=precision)

    real = Precision.from_str(precision).real
    assert ds.sizes[keys.time_md] == Nt
    for name in (keys.heat_flux_harmonic, keys.heat_flux_harmonic_q,
                 keys.heat_flux_QHGK_ta):
        arr = ds[name]
        assert arr.dims == keys.time_md_vec, f"{name} has dims {arr.dims}"
        assert arr.shape == (Nt, 3)
        assert arr.dtype == real, f"{name} is {arr.dtype}, not {real}"
        assert np.all(np.isfinite(np.asarray(arr.data))), f"{name} has NaN / inf"
    for name in (keys.hf_acf_ha, keys.hf_acf_ha_integral,
                 keys.hf_acf_ha_q, keys.hf_acf_ha_q_integral,
                 keys.hf_acf_qhgk_ta, keys.hf_acf_qhgk_ta_integral):
        arr = ds[name]
        assert arr.dims == keys.time_md_tensor, f"{name} has dims {arr.dims}"
        assert arr.shape == (Nt, 3, 3)
        assert np.all(np.isfinite(np.asarray(arr.data))), f"{name} has NaN / inf"


def test_harmonic_flux_prefers_the_stored_J_hm_R_unless_it_is_padded(
        trajectory, fc_file):
    """No in-repo fixture carries ``heat_flux_harmonic``, so the read path is
    unreachable in CI unless the field is injected. Inject a marker array and
    assert it is passed through verbatim; then NaN one row — the stride vibes
    writes flux on — and assert gkmx falls back to the FC rebuild instead of
    shortening the axis under ``_get_hf_data``'s dropna.
    """
    Nt = trajectory.sizes[keys.time]
    marker = np.tile(np.arange(Nt, dtype=np.float64)[:, None] * 1e-6, (1, 3))

    traj = trajectory.copy(deep=True)
    traj[keys.heat_flux_harmonic] = (keys.time_vec, marker)
    ds = get_kappa(traj, fc_file=fc_file, interpolate=False,
                   backend="numpy", harmonic_flux=True)
    stored = np.asarray(ds[keys.heat_flux_harmonic].data)
    assert np.allclose(stored, marker, rtol=1e-6, atol=0), (
        "stored heat_flux_harmonic was not passed through verbatim")

    padded = marker.copy()
    padded[1::2] = np.nan
    traj = trajectory.copy(deep=True)
    traj[keys.heat_flux_harmonic] = (keys.time_vec, padded)
    ds = get_kappa(traj, fc_file=fc_file, interpolate=False,
                   backend="numpy", harmonic_flux=True)
    rebuilt = np.asarray(ds[keys.heat_flux_harmonic].data)
    assert rebuilt.shape == (Nt, 3)
    assert np.all(np.isfinite(rebuilt)), "NaN-padded flux was not rebuilt"
    assert not np.allclose(rebuilt[::2], marker[::2], rtol=1e-6, atol=0)


@pytest.fixture(scope="module")
def solved(trajectory, fc_file):
    """Trajectory carrying the FC and reference atoms `get_kappa` attaches,
    plus a matching DMX — the pair the flux kernels take directly.

    ``precision="fp64"`` is load-bearing, not a numerics preference: under
    fp32 `get_kappa` recasts the time coord, which rebinds its local
    ``dataset`` and leaves the caller's copy without the FC.
    """
    traj = trajectory.copy(deep=True)
    get_kappa(traj, fc_file=fc_file, interpolate=False, backend="numpy",
              precision="fp64")
    return traj, DynamicalMatrix.from_dataset(
        traj, with_group_velocity_matrices=True)


def test_harmonic_flux_diagonal_channel_reproduces_J_hm_q(solved):
    """Restricting ``v_qssa`` to its diagonal must collapse ``J_quasi-hm``
    onto ``J_hm-q``: ``w_pair(s, s) = w_s`` and ``diag(v_qssa) == v_qsa``, so
    the ``s == s'`` term is exactly ``2/V sum_qs |a|^2 w^2 v_qsa``.

    A transposed ``v_op`` index, a ``w2``-vs-``w**2`` slip, or a wrong factor
    on either channel breaks this; the shape and finiteness pins above would
    not notice any of them.
    """
    traj, dmx = solved
    v = np.asarray(dmx.solution.v_qssa_cartesian)
    v_diag = np.zeros_like(v)
    idx = np.arange(v.shape[1])
    v_diag[:, idx, idx, :] = v[:, idx, idx, :]

    J_hm_q, J_quasi_hm = compute_harmonic_heat_flux_q(
        traj, dmx, v_qssa=v_diag, dtype_u=np.float64, verbose=False)
    err = np.abs(J_quasi_hm - J_hm_q).max() / np.abs(J_hm_q).max()
    assert err < 1e-12, f"diagonal J_quasi-hm departs from J_hm-q by {err:.2e}"


def test_harmonic_flux_R_and_q_share_a_magnitude(solved):
    """``J_hm-R`` carries `* units.fs` on the velocities (vibes' FCCalculator
    does the same) while the mode channels take velocities as-is through the
    eigenvectors. Dropping that factor leaves the two perfectly correlated but
    ``1/units.fs = 10.1805`` apart, which only a scale check can see.
    """
    traj, dmx = solved
    J_hm_R = compute_harmonic_heat_flux_R(traj, dmx, dtype=np.float64,
                                          verbose=False)
    _, J_quasi_hm = compute_harmonic_heat_flux_q(
        traj, dmx, dtype_u=np.float64, verbose=False)
    ratio = float(J_hm_R.std() / J_quasi_hm.std())
    assert 0.2 < ratio < 5.0, (
        f"std(J_hm-R) / std(J_quasi-hm) = {ratio:.3f}; a missing or doubled "
        f"units.fs conversion moves this by 10.1805")


def test_harmonic_flux_survives_a_dmx_cache_without_gvm(solved, trajectory,
                                                        tmp_path):
    """``to_hdf5`` defaults to ``include_group_velocity_matrices=False``, so a
    DMX read back from such a cache carries a plain ``Solution`` with no
    ``v_qssa_cartesian``. The flux block must take the operator the pipeline
    already built rather than reaching into that ``Solution``.
    """
    _, dmx = solved
    cache = tmp_path / "dmx_no_gvm.nc"
    dmx.to_hdf5(str(cache), include_group_velocity_matrices=False)

    ds = get_kappa(trajectory.copy(deep=True), dmx_file=str(cache),
                   interpolate=False, backend="numpy", harmonic_flux=True)
    for name in (keys.heat_flux_harmonic, keys.heat_flux_harmonic_q,
                 keys.heat_flux_QHGK_ta):
        assert np.all(np.isfinite(np.asarray(ds[name].data))), name


def test_harmonic_flux_R_matches_the_literal_pair_tensor(solved):
    """Transcribe vibes' ``FCCalculator`` pair sum literally and compare.

    The kernel evaluates this as five GEMMs against time-independent operators;
    this reference shares no code path with it, so it pins the contraction
    sides and the ``r0`` orientation. Both matter and neither is otherwise
    covered: transposing ``R^a`` yields a flux 190 % wrong that still
    correlates -0.97 with the truth, moves ``std(J_hm-R)/std(J_quasi-hm)`` by
    only 1.159 -> 1.182, and leaves every other assertion in this file green.

    Five steps on the 128-atom fixture — the ``(I, J, a, b)`` tensor this
    builds is what the kernel exists to avoid, so it stays deliberately tiny.
    """
    traj, dmx = solved
    nsteps = 5
    sc = dmx.supercell
    N = len(sc)

    # r0 itself is pinned against the direct image search in
    # tests/unit/test_lattice_points.py; what this reference exists to pin is
    # the contraction built on top of it.
    r0 = get_pair_vectors(dmx.primitive, sc)
    Phi = np.asarray(dmx.remapped, dtype=np.float64).reshape(
        N, 3, N, 3).swapaxes(1, 2)                          # Phi_IJab

    u = np.asarray(traj[keys.displacements][:nsteps].data, dtype=np.float64)
    v = np.asarray(traj[keys.velocities][:nsteps].data,
                   dtype=np.float64) * units.fs
    volume = float(np.nanmean(np.asarray(traj[keys.volume])))

    ref = np.empty((nsteps, 3))
    for t in range(nsteps):
        PU = np.einsum("IJab,Jb->IJa", Phi, u[t], optimize=True)
        d = r0 + u[t][:, None, :] - u[t][None, :, :]
        s = np.einsum("IJa,IJb->Iab", d, PU, optimize=True)
        ref[t] = 0.5 * np.einsum("Iab,Ib->a", s, v[t], optimize=True) / volume

    J = compute_harmonic_heat_flux_R(traj, dmx, dtype=np.float64,
                                     verbose=False)[:nsteps]
    err = np.abs(J - ref).max() / np.abs(ref).max()
    assert err < 1e-12, f"GEMM form departs from the literal sum by {err:.2e}"


def test_J_quasi_hm_is_the_complete_pair_flux(solved):
    """``J_quasi-hm`` carries the resonant AND antiresonant bilinears with the
    conjugation consistent with the ``e``-projection, so under a convention
    whose operator is the exact Bloch image of ``r0 Phi`` (RAW/PHONO3PY) it
    must equal ``J_hm-R`` minus the displacement term exactly::

        J_quasi-hm = J_hm-R - (1/2V) sum_I u_Ia [v_I.(Phi u)_I - u_I.(Phi v)_I]

    The resonant-only formula fails this at O(0.4) rel-rms; the wrong
    conjugation at O(1). ``J_disp`` comes from
    ``compute_harmonic_heat_flux_disp`` fed the TDEP fixture's DMX on
    purpose -- ``remapped`` is convention-independent, and this leans on
    that -- so the assertion is also the closure of the three-function
    decomposition ``J_hm-R = J_quasi-hm + J_disp``. Derivation:
    ``dev_scripts/harmonic_flux_decomposition.md``.
    """
    traj, dmx = solved

    J_hm_R = compute_harmonic_heat_flux_R(traj, dmx, dtype=np.float64,
                                          verbose=False)
    J_disp = compute_harmonic_heat_flux_disp(traj, dmx, dtype=np.float64,
                                             verbose=False)

    # `solved` is TDEP (the default); its little-group-averaged operator is
    # deliberately not the Bloch image of r0 Phi, so rebuild in RAW.
    dmx_raw = DynamicalMatrix(
        force_constants=np.asarray(traj[keys.fc]),
        primitive=json2atoms(traj.attrs[keys.reference_primitive]),
        supercell=json2atoms(traj.attrs[keys.reference_supercell]),
        with_group_velocity_matrices=True, precision="fp64", convention="RAW")
    _, J_quasi_raw = compute_harmonic_heat_flux_q(
        traj, dmx_raw, dtype_u=np.float64, verbose=False)
    err = np.abs(J_quasi_raw - (J_hm_R - J_disp)).max() / np.abs(J_hm_R).max()
    assert err < 1e-12, f"J_quasi-hm departs from the pair term by {err:.2e}"

    # Resolving power: the conjugate operator must break the identity badly.
    _, J_wrong = compute_harmonic_heat_flux_q(
        traj, dmx_raw, v_qssa=np.conj(dmx_raw.solution.v_qssa_cartesian),
        dtype_u=np.float64, verbose=False)
    err_wrong = np.abs(J_wrong - (J_hm_R - J_disp)).max() / np.abs(J_hm_R).max()
    assert err_wrong > 0.05, (
        f"conjugated operator still satisfies the identity ({err_wrong:.2e}); "
        f"the assertion above has no resolving power")


def test_J_quasi_hm_is_gauge_stable(trajectory, fc_file):
    """fp32 and fp64 eigh pick O(1)-different bases inside degenerate
    multiplets (and arbitrary per-mode phases everywhere), so agreement of the
    flux across the two solves is a direct gauge-invariance measurement on
    real data. The pairing ``p v u*`` is invariant only when the operator
    carries the projector's conjugation flavor -- TDEP's projector is built
    from ``conj(e_qsi)``, so its operator enters conjugated; without that the
    TDEP flux moves by 49 % rel-rms here.
    """
    traj = trajectory.copy(deep=True)
    get_kappa(traj, fc_file=fc_file, interpolate=False, backend="numpy",
              precision="fp64")
    kw = {"force_constants": np.asarray(traj[keys.fc], dtype=np.float64),
          "primitive": json2atoms(traj.attrs[keys.reference_primitive]),
          "supercell": json2atoms(traj.attrs[keys.reference_supercell]),
          "with_group_velocity_matrices": True}
    for convention in CONVENTIONS:
        d64 = DynamicalMatrix(precision="fp64", convention=convention, **kw)
        d32 = DynamicalMatrix(precision="fp32", convention=convention, **kw)
        _, J64 = compute_harmonic_heat_flux_q(traj, d64, dtype_u=np.float64,
                                              verbose=False)
        _, J32 = compute_harmonic_heat_flux_q(traj, d32, dtype_u=np.float64,
                                              verbose=False)
        err = np.sqrt(((J32 - J64) ** 2).mean()) / J64.std()
        assert err < 1e-5, f"{convention}: gauge noise {err:.2e}"
        if convention == "TDEP":
            # Resolving power: feeding the conjugate operator (undoing the
            # internal conj) must expose the gauge sensitivity.
            _, J_bad = compute_harmonic_heat_flux_q(
                traj, d32, v_qssa=np.conj(d32.solution.v_qssa_cartesian),
                dtype_u=np.float64, verbose=False)
            err_bad = np.sqrt(((J_bad - J64) ** 2).mean()) / J64.std()
            assert err_bad > 0.1, (
                f"unconjugated TDEP operator is gauge-stable ({err_bad:.2e}); "
                f"the assertion above has no resolving power")


def test_harmonic_flux_dmx_cache_rebuild_keeps_the_convention(
        trajectory, fc_file, tmp_path):
    """A gvm-less DMX cache forces `_get_gk_interpolate` to rebuild the DMX;
    the rebuild must inherit the cache's convention. A default rebuild hands
    the flux kernel a TDEP operator against a RAW projector -- measured 1.5x
    wrong on ``heat_flux_QHGK_ta`` while ``attrs['convention']`` still says
    RAW. Pin the RAW identity ``J_quasi-hm == J_hm-R - J_disp`` through the
    full ``dmx_file`` path.
    """
    traj = trajectory.copy(deep=True)
    get_kappa(traj, fc_file=fc_file, interpolate=False, backend="numpy",
              precision="fp64", convention="RAW")
    dmx_raw = DynamicalMatrix(
        force_constants=np.asarray(traj[keys.fc], dtype=np.float64),
        primitive=json2atoms(traj.attrs[keys.reference_primitive]),
        supercell=json2atoms(traj.attrs[keys.reference_supercell]),
        with_group_velocity_matrices=True, precision="fp64", convention="RAW")
    cache = tmp_path / "dmx_raw_no_gvm.nc"
    dmx_raw.to_hdf5(str(cache), include_group_velocity_matrices=False)

    ds = get_kappa(trajectory.copy(deep=True), dmx_file=str(cache),
                   interpolate=False, backend="numpy", precision="fp64",
                   harmonic_flux=True)
    assert ds.attrs["convention"] == "RAW"
    J_quasi = np.asarray(ds[keys.heat_flux_QHGK_ta].data, dtype=np.float64)
    J_hm_R = np.asarray(ds[keys.heat_flux_harmonic].data, dtype=np.float64)
    J_disp = compute_harmonic_heat_flux_disp(traj, dmx_raw, dtype=np.float64,
                                             verbose=False)
    err = np.abs(J_quasi - (J_hm_R - J_disp)).max() / np.abs(J_hm_R).max()
    assert err < 1e-10, (
        f"RAW identity broken through the cache-rebuild path by {err:.2e} "
        f"(a convention-dropping rebuild gives ~1.5)")
