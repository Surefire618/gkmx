"""Green-Kubo pipeline: mode decomposition, ACF lifetimes, extrapolation.

Top-level entry: `get_kappa`. The heavy kernel is
`compute_cv_tau` (mode heat capacities + lifetimes from an MD trajectory).
"""

import collections
import gc
import json
import os
from pathlib import Path

import numpy as np
import scipy.optimize as so
import scipy.signal as sl
import xarray as xr
from ase import Atoms, units
from scipy.integrate import cumulative_trapezoid

from . import _constants as C
from . import keys
from . import masses as _masses
from ._backend import get_backend
from ._log import Timer, talk, warn
from ._resources import mode_block_peak_gb
from .brillouin import get_symmetrized_array
from .dynamical_matrix import DynamicalMatrix
from .harmonic_flux import compute_harmonic_heat_flux_q, compute_harmonic_heat_flux_R
from .interpolation import get_interpolation_data
from .io import parse_force_constants
from .kappa import get_kappa_BTE, qhgk_tau_eff
from .kappa import symmetrize_kappa as _symmetrize_kappa
from .mic import fold as mic_fold
from .mic import is_orthogonal
from .phonon import DEGENERACY_TOL, degenerate_sets
from .precision import Precision
from .trajectory import gk_prefactor

_prefix = "gkmx"


def _talk(msg):
    talk(msg, prefix=_prefix)


def _gk_prefactor(volume, temperature, verbose=True):
    """GK prefactor ``V / (kB * T**2)`` converted to W/mK."""
    pf = gk_prefactor(volume, temperature)
    if verbose:
        _talk(f"GK prefactor: V={float(volume):.1f} AA^3, "
              f"T={float(temperature):.1f} K -> {pf:.3e}")
    return pf


def _gk_prefactor_from_dataset(dataset, verbose=True):
    volume = dataset.attrs[keys.volume]
    temperature = float(dataset[keys.temperature].mean())
    return _gk_prefactor(volume, temperature, verbose=verbose)


def _disp_block(positions_block, sc_positions, cell, search_images):
    """Per-block displacement with MIC fold (used by lazy dask path)."""
    dtype = positions_block.dtype
    sc_positions = np.asarray(sc_positions, dtype=dtype)
    cell = np.asarray(cell, dtype=dtype)
    raw = positions_block - sc_positions
    return mic_fold(raw, cell, search=search_images)


def check_disp_magnitudes(disp, cell, *, fraction_of_safe=0.5):
    """Warn if any frame's max |disp| exceeds ``fraction_of_safe * L_min/2``; returns the stats dict."""
    from .mic import safe_radius
    disp = np.asarray(disp)
    if disp.shape[-1] != 3:
        raise ValueError(f"disp last axis must be 3; got {disp.shape}")
    half_L_min = safe_radius(cell)
    threshold = fraction_of_safe * half_L_min
    norms = np.linalg.norm(disp, axis=-1)  # (..., Na)
    max_per_frame = norms.reshape(norms.shape[0], -1).max(axis=-1)
    Nt = len(max_per_frame)
    stats = {
        "Nt": Nt, "half_L_min": half_L_min,
        "mean": float(np.mean(max_per_frame)),
        "median": float(np.median(max_per_frame)),
        "p99": float(np.percentile(max_per_frame, 99)),
        "max": float(np.max(max_per_frame)),
        "ratio": float(np.max(max_per_frame) / max(half_L_min, 1e-30)),
    }
    _talk(f"disp diagnostic: Nt={Nt}, max|disp|={stats['max']:.3f} Å, "
          f"L_min/2={half_L_min:.3f} Å, ratio={stats['ratio']:.3g} "
          f"(mean={stats['mean']:.3f}, median={stats['median']:.3f}, "
          f"p99={stats['p99']:.3f})")
    flagged = np.flatnonzero(max_per_frame > threshold)
    if flagged.size:
        first10 = [(int(i), float(max_per_frame[i])) for i in flagged[:10]]
        warn(f"check_disp_magnitudes: {flagged.size}/{Nt} frames exceed "
             f"{fraction_of_safe}·L_min/2 = {threshold:.3f} Å (defect "
             f"hopping or PBC wrap?); first 10 (frame, |disp|max Å): "
             f"{first10}")
    return stats


def _correlate(f1, f2, normalize=2, hann=True):
    """Cross-correlate ``f1`` and ``f2`` along time, with optional Hann taper."""
    a1, a2 = np.asarray(f1), np.asarray(f2)
    Nt = min(len(a1), len(a2))
    corr = sl.correlate(a1[:Nt], a2[:Nt])[Nt - 1:]
    if normalize == 2:
        corr /= np.arange(Nt, 0, -1)
    elif normalize == 1 or normalize is True:
        corr /= Nt
    if hann:
        corr *= sl.windows.hann(2 * Nt)[Nt:]
    return corr


def _autocorrelation_Nd(array, off_diagonal=False):
    """ACF of a multi-dim ``xr.DataArray`` with time on the leading axis."""
    Nt, *shape = np.shape(array)
    data = np.moveaxis(np.asarray(array), 0, -1)

    if not off_diagonal:
        corr = np.zeros((*shape, Nt), dtype=data.dtype)
        for idx in np.ndindex(*shape):
            corr[idx] = _correlate(data[idx], data[idx])
        corr = np.moveaxis(corr, -1, 0)
        da = array.copy()
        da.data = corr
        return da

    from itertools import product as iprod
    flat = data.reshape(-1, Nt)
    N = flat.shape[0]
    corr = np.zeros((N * N, Nt), dtype=data.dtype)
    for ii, (d1, d2) in enumerate(iprod(flat, flat)):
        corr[ii] = _correlate(d1, d2)
    corr = np.moveaxis(corr.reshape((*shape, *shape, Nt)), -1, 0)

    dims = list(array.dims)
    new_dims = [dims[0]]
    for d in dims[1:]:
        new_dims.extend([d + "1", d + "2"])

    return xr.DataArray(corr, dims=new_dims[:len(corr.shape)], coords=array.coords)


def _cumtrapz(array):
    """Cumulative trapezoid along the time axis."""
    if isinstance(array, xr.DataArray):
        times = np.asarray(array[keys.time]) if hasattr(array, keys.time) else None
        result = cumulative_trapezoid(np.asarray(array), x=times, axis=0, initial=0)
        da = array.copy()
        da.data = result
        da.name = (array.name or "") + "_" + keys.integral
        return da
    return cumulative_trapezoid(np.asarray(array), axis=0, initial=0)


def _get_hf_data(flux, prefactor=1.0, total=False):
    """HFACF and cumulative kappa from a heat-flux array."""
    flux = flux.dropna(keys.time)
    flux_avg = 0 if total else flux.mean(axis=0)
    hfacf = _autocorrelation_Nd(flux - flux_avg, off_diagonal=True) * prefactor
    kappa = _cumtrapz(hfacf)
    HFData = collections.namedtuple("hf_data", (keys.hf_acf, keys.kappa_cumulative))
    return HFData(hfacf, kappa)


def _get_lowest_vib_freq(velocities, prominence=0.2, threshold_freq=0.1,
                          backend="numpy", max_mem_gb=4.0):
    """Lowest VDOS peak above ``threshold_freq``, in THz."""
    v = velocities.dropna(keys.time)
    # FFT scratch follows the precision switch (no fp64 promotion).
    dtype = np.dtype(Precision.default().real)
    Nt = v.shape[0]
    data = np.moveaxis(np.asarray(v, dtype=dtype), 0, -1).reshape(-1, Nt)
    nfft = 2 * Nt
    bk = get_backend(backend)
    norm = bk.to_device(np.arange(Nt, 0, -1, dtype=dtype))
    hann = bk.to_device(sl.windows.hann(2 * Nt)[Nt:].astype(dtype))
    v_acf = bk.rfft_power_sum(
        data, nfft=nfft, max_mem_gb=max_mem_gb, norm=norm, hann=hann)
    npad = 10000
    data_padded = np.pad(v_acf, (0, npad))
    times = np.asarray(v[keys.time])
    dt = float(times[1] - times[0])

    fft_result = np.fft.fft(data_padded)[:len(data_padded) // 2]
    max_freq = 1.0 / (2.0 * dt) * 1000  # THz
    freqs = np.linspace(0, max_freq, len(fft_result))
    vdos = np.abs(fft_result).real
    vdos -= vdos.min()
    mask = freqs > threshold_freq
    if mask.any():
        vdos /= vdos[mask].max()

    peaks, _ = sl.find_peaks(vdos, prominence=prominence)
    if len(peaks) == 0:
        warn("No peaks found in VDOS, using default frequency")
        return 1.0

    peak_freqs = freqs[peaks]
    freq = peak_freqs[0]
    if freq < threshold_freq and len(peak_freqs) > 1:
        freq = peak_freqs[1]
    return float(freq)


def _savgol_filter(array, window_fs=None, window=None, antisymmetric=False, polyorder=1):
    """Savitzky-Golay filter along the time axis."""
    if window_fs is not None:
        times = array[keys.time]
        window = len(times[times < window_fs])
    if window is None:
        raise ValueError("Either window_fs or window must be specified")
    window = window // 2 * 2 + 1  # savgol needs odd length

    if antisymmetric:
        data = np.concatenate((-np.asarray(array)[::-1], np.asarray(array)))
    else:
        data = np.asarray(array)

    filtered = np.zeros_like(data)
    data = np.moveaxis(data, 0, -1)
    filtered = np.moveaxis(filtered, 0, -1)
    for idx in np.ndindex(data.shape[:-1]):
        filtered[idx] = sl.savgol_filter(data[idx], window_length=window, polyorder=polyorder)
    filtered = np.moveaxis(filtered, -1, 0)

    result = array.copy()
    if antisymmetric:
        result.data = filtered[len(array):]
    else:
        result.data = filtered
    return result


def compute_cv_tau(dataset, dmx, stride=1, t_chunk=5000, mode_block=None,
                   max_mem_gb=4.0, backend="numpy",
                   dtype_u=None, dtype_a=None,
                   hann=True, normalize=2,
                   correct_finite_time=True,
                   factorization="wick", lifetime_fit_cutoff=0.5):
    """Mode heat capacities and lifetimes from an MD trajectory.

    Always uses the full-trajectory FFT ACF (Wiener-Khinchin); time-axis
    chunking of the autocorrelation is biased at every lag inside the chunk
    (Bartlett triangle: <g_est(tau)> = g(tau) (1 - |tau|/L)), so the
    threshold-crossing tau depends on L, not the physics.

    Args:
        dataset: trajectory ``xr.Dataset`` carrying ``displacements`` and
            ``velocities``. Masses come from ``dmx.supercell``, not from the
            trajectory's own copy, so the projection cannot end up on a
            different mass table than the force constants.
        dmx: solved ``DynamicalMatrix`` supplying ``e_qsI``, ``w_qs`` and
            ``w_inv_qs`` on the commensurate grid.
        stride: keep every ``stride``-th frame. Coarsens the time axis, so it
            raises the shortest resolvable lifetime.
        t_chunk: frames per streaming block, for the projection only.
            Projection, mass-weighting and the moments are linear in time and
            chunk freely; the ACF never does -- it is one full-length FFT, and
            chunking it is biased at every lag.
        backend: ``"numpy"`` (default) or ``"jax"``.
        dtype_u: dtype of the displacement / momentum buffers.
        dtype_a: dtype of the complex mode amplitudes. Both default to the
            precision switch.
        hann: taper the ACF with a Hann window sized to the full trajectory.
        normalize: ACF normalization; ``2`` is the unbiased ``1/(N - lag)``.
        correct_finite_time: apply ``1/tau -> 1/tau - 1/T_max``. Affects
            long-lifetime modes only.
        lifetime_fit_cutoff: the fraction of ``g(0)`` the fit crosses to define
            tau. Lower samples further into the tail: better for long-tau
            modes, noisier for short-tau ones.
        factorization: ``"wick"`` (default, ``|g|^2`` fit, SMA /
            dressed-bubble per dissertation Sec. 5.4.2 +
            Fiorentino-Baroni PRB 107 054311 (2023)) or ``"vertex"``
            (un-factorized ``<n(t) n(0)>``; vibes_tom parity, NOT a
            full Mori-Zwanzig treatment of the vertex).
        mode_block: auto-sized against ``max_mem_gb`` when ``None``;
            raises ``MemoryError`` if even ``mode_block=1`` overflows.
        max_mem_gb: pass ``"auto"`` to detect device free memory.

    Returns:
        ``(cv_qs, tau_qs)`` as ``xr.DataArray`` of shape ``(Nq, Ns)``;
        acoustic modes near Gamma are NaN in ``tau_qs``.
    """
    if factorization not in ("wick", "vertex"):
        raise ValueError(
            f"factorization must be 'wick' or 'vertex', got "
            f"{factorization!r}"
        )

    bk = get_backend(backend)

    # Inherit DMX precision so DynamicalMatrix(precision="fp32") flows
    # end-to-end without the module default quietly upcasting.
    p_default = Precision.default()
    if dtype_u is None:
        dtype_u = getattr(dmx, "_dtype_real", None) or p_default.real
    if dtype_a is None:
        dtype_a = getattr(dmx, "_dtype_complex", None) or p_default.complex
    dtype_u, dtype_a = bk.resolve_dtypes(dtype_u, dtype_a)
    # Real companion of `dtype_a` — keeps `C_arr / norm_d` from silently
    # upcasting complex64 to complex128 on split-precision calls.
    dtype_a_real = np.zeros((), dtype=dtype_a).real.dtype
    bytes_per_complex = np.dtype(dtype_a).itemsize
    bytes_per_real = np.dtype(dtype_u).itemsize

    Nt = dataset.displacements.isel(
        {dataset.displacements.dims[0]: slice(None, None, stride)}).shape[0]
    dt = float(dataset.displacements.time[1] - dataset.displacements.time[0])

    e_qsI = np.asarray(dmx.e_qsI)
    Nq, Ns, I = e_qsI.shape
    nmodes = Nq * Ns

    # Split complex e_qsI into real/imag parts: two real matmuls beat
    # one complex matmul. Reshape puts q outside s: `mode = q*Ns + s`.
    e_re_np = np.ascontiguousarray(e_qsI.reshape(nmodes, I).real.astype(dtype_u))
    e_im_np = np.ascontiguousarray(e_qsI.reshape(nmodes, I).imag.astype(dtype_u))
    w_inv_m = np.asarray(dmx.w_inv_qs).reshape(nmodes).astype(dtype_u)
    w2_m = np.asarray(dmx.w2_qs).reshape(nmodes).astype(dtype_u)

    # One mass source for the whole pipeline: taken from the structure the
    # force constants were weighted with, not the trajectory's own copy, so
    # the projection cannot end up on a different table than the eigenvectors.
    m = np.asarray(_masses.of(dmx.supercell)).astype(dtype_u)
    m_sqrt = np.sqrt(m).astype(dtype_u)

    # Full-trajectory ACF only — see the Bartlett-bias note in the docstring.
    tau_max_eff = Nt
    nfft = bk.next_fast_len(2 * Nt)

    # Choose how many modes to autocorrelate at once. The ACF keeps three
    # (mode_block, Nt) complex buffers alive simultaneously, so memory grows
    # linearly with the block and the largest affordable block is just the
    # budget divided by the cost of one mode.
    #
    # peak_factor=1.0 asks for the raw allocation with no headroom of its own,
    # because max_mem_gb already carries the margin: it is the user's number
    # verbatim when given as a float, and an already-reduced value when it came
    # from "auto".
    if mode_block is None:
        single_mode_gb = mode_block_peak_gb(1, Nt, bytes_per_complex, peak_factor=1.0)
        # One mode is the floor. If even that exceeds the budget there is no
        # smaller block to retreat to, so stop here instead of dying later in
        # the FFT. The only other lever would be shortening the ACF, and that
        # changes the physics rather than the memory use — hence the refusal.
        if single_mode_gb > max_mem_gb:
            raise MemoryError(
                f"compute_cv_tau: even mode_block=1 needs "
                f"{single_mode_gb:.2f} GB > max_mem_gb={max_mem_gb:.2f} GB "
                f"(Nt={Nt}, bytes_per_complex={bytes_per_complex}). "
                f"Raise --max-mem-gb to ≥ {single_mode_gb:.1f}, use a "
                f"larger-memory device, or shrink Nt/stride. Time-axis "
                f"chunking of the ACF is not an option — it biases the "
                f"estimator at every lag (Bartlett triangle).")
        mode_block = min(nmodes, max(1, int(max_mem_gb / single_mode_gb)))

    real_peak_gb = mode_block_peak_gb(mode_block, Nt, bytes_per_complex, peak_factor=1.0)
    _talk(f"backend={backend} [{bk.device_description()}], "
          f"dtype={np.dtype(dtype_u).name}/{np.dtype(dtype_a).name}, "
          f"mode_block={mode_block}, t_chunk={t_chunk}, "
          f"peak={real_peak_gb:.2f} GB, "
          f"traj chunk={2*t_chunk*I*bytes_per_real/1e9:.2f} GB")

    if normalize == 2:
        norm_np = np.arange(Nt, 0, -1, dtype=dtype_a_real)
    else:
        norm_np = dtype_a_real.type(Nt)
    # Hann sized to the full trajectory length so the taper at any lag
    # inside the `_fit_tau` window (≤ Nt/2) is ≈ 1.
    hann_vec_np = (sl.windows.hann(2 * Nt)[Nt:Nt + tau_max_eff]
                   .astype(dtype_a_real) if hann else None)
    Tmax = (Nt - 1) * dt
    time_dim = dataset.displacements.dims[0]

    norm_d = bk.to_device(norm_np) if normalize == 2 else norm_np
    hann_d = bk.to_device(hann_vec_np) if hann_vec_np is not None else None
    e_re_d = bk.to_device(e_re_np)
    e_im_d = bk.to_device(e_im_np)
    w_inv_d = bk.to_device(w_inv_m)

    cv_mode = np.empty(nmodes, dtype=dtype_u)
    tau_mode = np.full(nmodes, np.nan, dtype=dtype_u)

    def _project_chunk_host(t0, t1):
        src0, src1 = t0 * stride, t1 * stride
        U = np.asarray(dataset.displacements.isel(
            {time_dim: slice(src0, src1, stride)}).data, dtype=dtype_u)
        V = np.asarray(dataset.velocities.isel(
            {time_dim: slice(src0, src1, stride)}).data, dtype=dtype_u)
        tc = U.shape[0]
        u_c = (U * m_sqrt[None, :, None]).reshape(tc, I).T
        p_c = (V * m_sqrt[None, :, None]).reshape(tc, I).T
        up_np = np.empty((I, 2 * tc), dtype=dtype_u)
        up_np[:, :tc] = u_c
        up_np[:, tc:] = p_c
        return bk.to_device(up_np), tc

    def _project_chunk(e_re_blk, e_im_blk, w_inv_blk, t0, t1):
        up, tc = _project_chunk_host(t0, t1)
        up_re = e_re_blk @ up
        up_im = e_im_blk @ up
        a_re_c = 0.5 * (up_re[:, :tc] - w_inv_blk[:, None] * up_im[:, tc:])
        a_im_c = 0.5 * (up_im[:, :tc] + w_inv_blk[:, None] * up_re[:, tc:])
        return a_re_c, a_im_c, tc

    def _full_block(e_re_blk, e_im_blk, w_inv_blk, B):
        """Full-Nt FFT autocorrelation; memory bounding is on the mode axis."""
        a_re = bk.xp.empty((B, Nt), dtype=dtype_u)
        a_im = bk.xp.empty((B, Nt), dtype=dtype_u)
        is_jax = backend == "jax"
        for t0 in range(0, Nt, t_chunk):
            t1 = min(Nt, t0 + t_chunk)
            a_re_c, a_im_c, _ = _project_chunk(
                e_re_blk, e_im_blk, w_inv_blk, t0, t1)
            if is_jax:
                a_re = a_re.at[:, t0:t1].set(a_re_c)
                a_im = a_im.at[:, t0:t1].set(a_im_c)
            else:
                a_re[:, t0:t1] = a_re_c
                a_im[:, t0:t1] = a_im_c
            del a_re_c, a_im_c

        sum_a_re = a_re.sum(axis=-1)
        sum_a_im = a_im.sum(axis=-1)
        abs2 = a_re * a_re + a_im * a_im
        sum_abs2 = abs2.sum(axis=-1)
        sum_abs4 = (abs2 * abs2).sum(axis=-1)
        del abs2

        if factorization == "wick":
            # Subtract the mode mean before the ACF (the vertex path already
            # centers n): a static offset <a> != 0 -- metastable / defect
            # trajectories -- adds a |<a>|^2 plateau to |g|^2 that biases
            # the threshold fit low, and the symmetrized sum below would
            # add the plateaus of equivalent modes coherently.
            a_re = a_re - a_re.mean(axis=-1, keepdims=True)
            a_im = a_im - a_im.mean(axis=-1, keepdims=True)
        a_blk = bk.complex(a_re, a_im, dtype=dtype_a)
        del a_re, a_im
        if factorization == "wick":
            # Wick-factorize the 4-point correlator → products of
            # 2-point; fit `|g(t)|²` (thesis Eq. 5.43).
            signal = a_blk
        else:
            # Un-factorized `<n(t)n(0)>` with `n = |a|²` — keeps the
            # connected 4-point cumulant. Fiorentino-Baroni PRB 107
            # 054311 (2023).
            n_blk = a_blk.real * a_blk.real + a_blk.imag * a_blk.imag
            signal = n_blk - n_blk.mean(axis=-1, keepdims=True)
            del n_blk
        del a_blk
        C_arr = bk.fft_autocorrelate(
            signal, nfft=nfft, tau_max=tau_max_eff,
            norm=norm_d, hann=hann_d)
        return (sum_a_re, sum_a_im, sum_abs2, sum_abs4), C_arr

    def _cv_and_inv_var(sums, w2_blk):
        """Mode heat capacities and per-mode ``1/var(a)^2`` from accumulated moments."""
        sum_a_re, sum_a_im, sum_abs2, sum_abs4 = sums
        mean_abs2 = sum_abs2 / Nt
        mean_abs4 = sum_abs4 / Nt
        mean_a_re = sum_a_re / Nt
        mean_a_im = sum_a_im / Nt
        var_abs2 = mean_abs4 - mean_abs2 * mean_abs2
        cv_vals = (2 * w2_blk) ** 2 * bk.to_host(var_abs2)
        var_a = bk.xp.maximum(mean_abs2 - (mean_a_re * mean_a_re
                                            + mean_a_im * mean_a_im),
                               bk.xp.asarray(0, dtype=dtype_u))
        eps2 = bk.xp.asarray(1e-16, dtype=dtype_u)
        one = bk.xp.asarray(1, dtype=dtype_u)
        var_a_safe = bk.xp.where(var_a > eps2, var_a, one)
        inv_var2 = bk.to_host(one / (var_a_safe * var_a_safe))
        var_h = bk.to_host(var_a)
        mean2 = bk.to_host(mean_a_re * mean_a_re + mean_a_im * mean_a_im)
        dc = mean2 / np.maximum(var_h, np.finfo(dtype_u).tiny)
        return cv_vals, inv_var2, var_h, dc

    def _fit_taus(C_arr, inv_var2, var_a, segs):
        """Exponential fits; returns tau in fs for the chunk's modes.

        Wick path: one fit per group of symmetry-equivalent modes on
        ``|sum_m g_m|^2 / (sum_m var_m)^2``. Within a multiplet the sum is
        the trace of the block correlation matrix, so the fitted tau cannot
        see which intra-multiplet basis eigh picked; across equivalent
        q-points it adds their statistics to one smooth curve instead of M
        noisy ones (a nonlinear fit of noisy curves is biased; averaging
        the fits keeps the bias, averaging the curves shrinks it).
        Singleton groups reduce to the per-mode curve exactly. The vertex
        path stays per-mode: its diagonal ``<nn>`` sum is quartic in `a`
        and has no invariant assembled from diagonals alone.
        """
        B = segs[-1][1]
        taus = np.full(B, np.nan, dtype=dtype_u)
        if factorization == "wick":
            g = bk.to_host(C_arr)
            for i0, i1 in segs:
                if i1 - i0 == 1:
                    g2 = g[i0].real ** 2 + g[i0].imag ** 2
                    curve = g2 * inv_var2[i0]
                else:
                    tr = g[i0:i1].sum(axis=0)
                    var_sum = var_a[i0:i1].sum()
                    if not var_sum > 0:
                        continue
                    curve = (tr.real ** 2 + tr.imag ** 2) / (var_sum
                                                             * var_sum)
                taus[i0:i1] = _fit_tau(curve, dt=dt,
                                        thresh=lifetime_fit_cutoff)
            return taus
        nn = bk.to_host(C_arr.real)
        for i in range(B):
            c0 = nn[i, 0]
            if c0 <= 0 or not np.isfinite(c0):
                continue
            taus[i] = _fit_tau(nn[i] / c0, dt=dt,
                                thresh=lifetime_fit_cutoff)
        return taus

    # Groups of symmetry-equivalent modes: one per irreducible q and
    # degenerate multiplet, spanning the equivalent q-points.
    w_abs = np.abs(np.asarray(dmx.w_qs))
    map2ir = np.asarray(dmx.q_grid.map2ir)
    map2full = np.asarray(dmx.q_grid.ir.map2full)

    def _mode_groups(qrep, qset):
        out, covered = [], np.zeros(Ns, dtype=bool)
        for _, b0, b1 in degenerate_sets(w_abs[[qrep]]):
            out.append((np.asarray(qset)[:, None] * Ns
                        + np.arange(b0, b1)).ravel())
            covered[b0:b1] = True
        out += [np.asarray(qset) * Ns + si for si in np.flatnonzero(~covered)]
        return out

    groups, n_misaligned = [], 0
    for ii, qrep in enumerate(map2ir):
        members = np.flatnonzero(map2full == ii)
        if np.abs(w_abs[members] - w_abs[qrep]).max() <= DEGENERACY_TOL:
            groups += _mode_groups(qrep, members)
        else:
            n_misaligned += 1
            for m in members:
                groups += _mode_groups(m, [m])
    if n_misaligned:
        warn(f"compute_cv_tau: {n_misaligned} groups of symmetry-equivalent "
             f"q-points have spectra deviating beyond the degeneracy "
             f"tolerance; their tau fits fall back to per-q groups.")

    dc_mode = np.zeros(nmodes, dtype=dtype_u)
    k0 = 0
    while k0 < len(groups):
        k1, total = k0, 0
        while k1 < len(groups) and (total + len(groups[k1]) <= mode_block
                                    or k1 == k0):
            total += len(groups[k1])
            k1 += 1
        idx = np.concatenate(groups[k0:k1])
        segs, pos = [], 0
        for g in groups[k0:k1]:
            segs.append((pos, pos + len(g)))
            pos += len(g)
        sums, C_arr = _full_block(
            e_re_d[idx], e_im_d[idx], w_inv_d[idx], len(idx))
        cv_mode[idx], inv_var2, var_a, dc = _cv_and_inv_var(sums, w2_m[idx])
        tau_mode[idx] = _fit_taus(C_arr, inv_var2, var_a, segs)
        dc_mode[idx] = dc
        del sums, C_arr, inv_var2, var_a, dc
        k0 = k1

    volume = float(np.asarray(dataset.volume).mean())
    temperature = float(np.asarray(dataset.temperature).mean())
    cv_mode /= np.dtype(dtype_u).type(units.kB * temperature ** 2 * volume)

    if correct_finite_time:
        good = np.isfinite(tau_mode) & (tau_mode > 0)
        corrected = np.full_like(tau_mode, np.nan)
        corrected[good] = 1.0 / (1.0 / tau_mode[good] - 1.0 / Tmax)
        tau_mode = corrected

    cv_qs = cv_mode.reshape(Nq, Ns)
    tau_qs = tau_mode.reshape(Nq, Ns)
    # In-place mask (not np.where) to avoid fp64 upcast of tau_qs.
    tau_qs[np.asarray(dmx.w_qs) < 1e-6] = np.nan

    # Non-stationarity gate on the fitted modes only (the tau mask above is
    # the liveness criterion; numerically dead acoustics have var ~ 0 and a
    # meaningless ratio).
    dc_live = dc_mode.reshape(Nq, Ns)[np.isfinite(tau_qs)]
    dc_count = int((dc_live > 0.1).sum())
    if dc_count:
        warn(f"compute_cv_tau: {dc_count} of {dc_live.size} fitted modes "
             f"are non-stationary (|<a>|^2/var > 0.1, max "
             f"{float(dc_live.max()):.2f}) -- the trajectory may straddle "
             f"a structural transition. The ACF uses mean-centered "
             f"amplitudes, but a two-basin slow component can still "
             f"contaminate the fit tail.")

    _average_over_multiplets(cv_qs, tau_qs, np.asarray(dmx.w_qs))

    return (
        xr.DataArray(cv_qs, dims=keys.q_s, name=keys.mode_heat_capacity),
        xr.DataArray(tau_qs, dims=keys.q_s, name=keys.mode_lifetime),
    )


def _average_over_multiplets(cv_qs, tau_qs, w_qs):
    """Average cv and the linewidth 1/tau over degenerate modes, in place.

    Follows phonopy's `degenerate_sets` and phono3py's
    `imag_self_energy.py::average_by_degeneracy`, which averages the imaginary
    self-energy: gamma is what enters linearly and has the invariant subspace sum.
    """
    for qi, start, stop in degenerate_sets(np.abs(np.asarray(w_qs))):
        G = slice(start, stop)
        cv_blk, tau_blk = cv_qs[qi, G], tau_qs[qi, G]

        ok = np.isfinite(cv_blk)
        if ok.any():
            cv_qs[qi, G] = np.where(ok, cv_blk[ok].mean(), cv_blk)

        # A failed fit carries no linewidth to contribute; averaging over the
        # survivors keeps one bad mode from poisoning its partners.
        ok = np.isfinite(tau_blk) & (tau_blk > 0)
        if ok.any():
            tau_qs[qi, G] = np.where(ok, 1.0 / np.mean(1.0 / tau_blk[ok]), tau_blk)


def _harmonic_force_residuals(disp, forces, fc_remapped, backend="numpy",
                                t_chunk=5000, dtype=None):
    """Stream ``std(forces - f_ha) / std(forces)`` per-sample and global; never materializes f_ha."""
    bk = get_backend(backend)
    if dtype is None:
        dtype = Precision.default().real
    dtype, _ = bk.resolve_dtypes(dtype, None)
    fc_remapped = np.ascontiguousarray(fc_remapped, dtype=dtype)

    Nt, Na, _ = disp.shape

    sigma_ps = np.empty(Nt, dtype=dtype)
    n_total = Nt * Na * 3
    sum_res = sum_res_sq = sum_for = sum_for_sq = dtype(0.0)

    fc_d = bk.to_device(fc_remapped)

    def _chunk_stats(d_chunk, f_chunk, fc_d):
        xp = bk.xp
        f_ha = -(d_chunk.reshape(d_chunk.shape[0], -1) @ fc_d
                 ).reshape(f_chunk.shape)
        res = f_chunk - f_ha
        sigma_per = xp.std(res, axis=(1, 2)) / xp.std(f_chunk, axis=(1, 2))
        return (sigma_per,
                xp.sum(res), xp.sum(res * res),
                xp.sum(f_chunk), xp.sum(f_chunk * f_chunk))
    _chunk_stats = bk.jit(_chunk_stats)

    for t0 in range(0, Nt, t_chunk):
        t1 = min(Nt, t0 + t_chunk)
        d_chunk = np.ascontiguousarray(disp[t0:t1], dtype=dtype)
        f_chunk = np.ascontiguousarray(forces[t0:t1], dtype=dtype)
        sigma_per, sr, srs, sf, sfs = _chunk_stats(
            bk.to_device(d_chunk), bk.to_device(f_chunk), fc_d,
        )
        sigma_ps[t0:t1] = np.asarray(bk.to_host(sigma_per))
        sum_res    += dtype(bk.to_host(sr))
        sum_res_sq += dtype(bk.to_host(srs))
        sum_for    += dtype(bk.to_host(sf))
        sum_for_sq += dtype(bk.to_host(sfs))
        del d_chunk, f_chunk

    var_res = sum_res_sq / n_total - (sum_res / n_total) ** 2
    var_for = sum_for_sq / n_total - (sum_for / n_total) ** 2
    sigma = float(np.sqrt(max(var_res, dtype(0.0)))
                  / np.sqrt(max(var_for, dtype(1e-30))))
    return sigma_ps, sigma


def _fit_tau(g2, dt, thresh=0.1, maxfev=2000):
    """Exponential fit to ``g2(t)``; returns lifetime in fs, NaN on failure."""
    y = np.asarray(g2, dtype=np.float64)
    if not np.isfinite(y[0]) or y[0] < 1e-12:
        return np.nan
    idx = np.where(y < thresh)[0]
    if idx.size == 0 or idx.min() < 2:
        return np.nan
    n = int(idx.min())
    x = np.arange(n, dtype=np.float64)
    yy = y[:n]
    slope = np.polyfit(x, np.log(np.clip(yy, 1e-300, None)), 1)[0]
    tau0 = (-1.0 / slope) if slope < 0 else max(2.0, n / 5.0)
    try:
        (tau, _), _ = so.curve_fit(
            lambda x, tau, y0: y0 * np.exp(-x / tau),
            x, yy, p0=(tau0, float(yy[0])),
            bounds=([1e-12, 1e-300], [np.inf, np.inf]), maxfev=maxfev,
        )
        return float(tau) * float(dt)
    except Exception:
        return np.nan


def _qhgk_coherence_kernel(t_fs, w, wi, gamma):
    """Time-domain QHGK coherence kernel; its integral over t is `qhgk_tau_eff`.

        K(t) = e^{-G t} [ w_plus cos((w_s - w_s') t) + w_minus cos((w_s + w_s') t) ]

    `w`, `wi` in rad/fs (scaled by `C.omega_to_rad_fs`), `gamma` in 1/fs, `t` in fs.
    """
    w_s = w[:, :, None]; w_sp = w[:, None, :]
    wi_s = wi[:, :, None]; wi_sp = wi[:, None, :]
    w_plus = (w_s + w_sp) ** 2 * (wi_s * wi_sp) / 4
    w_minus = (w_s - w_sp) ** 2 * (wi_s * wi_sp) / 4
    Gamma = gamma[:, :, None] + gamma[:, None, :]

    tt = np.asarray(t_fs)[:, None, None, None]
    return np.exp(-Gamma[None] * tt) * (
        w_plus * np.cos((w_s - w_sp)[None] * tt)
        + w_minus * np.cos((w_s + w_sp)[None] * tt))


def _analytical_hfacfs(time_fs, tau_qs, cv_qs, v_qsa, v_qssa, w_qs, w_inv_qs,
                        dtype_real, tol=1e-4):
    """Analytical BTE and QHGK time-resolved HFACFs (thesis Eq. 5.52, Fiorentino-Baroni form)."""
    t = np.asarray(time_fs, dtype=dtype_real)
    Nt = t.size
    tau = np.nan_to_num(np.asarray(tau_qs, dtype=dtype_real))
    cv = np.asarray(cv_qs, dtype=dtype_real)
    v = np.asarray(v_qsa, dtype=dtype_real)
    v2_qsab = v[..., :, None] * v[..., None, :]

    with np.errstate(divide="ignore", invalid="ignore"):
        decay = np.exp(-t[:, None, None]
                        / np.where(tau > 0, tau[None, :, :], np.inf))
    C_BTE = C.to_W_mK * np.einsum(
        "tqs,qs,qsab->tab", decay, cv, v2_qsab, optimize=True,
    ).astype(dtype_real)
    kappa_BTE = np.asarray(
        _cumtrapz(xr.DataArray(C_BTE, dims=(keys.time,) + keys.tensor,
                                coords={keys.time: t})),
        dtype=dtype_real)

    # Same rad/fs conversion and acoustic mask as `kappa.qhgk_tau_eff`: Gamma is
    # in 1/fs, so w must be too or the coherences beat 1/omega_to_rad_fs too fast.
    scale = np.dtype(dtype_real).type(C.omega_to_rad_fs)
    w = np.asarray(w_qs, dtype=dtype_real) * scale
    wi = np.asarray(w_inv_qs, dtype=dtype_real) / scale
    w = np.where(w < tol, 0, w)
    wi = np.where(w == 0, 0, wi)

    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = 0.5 / tau
    # tau = 0 marks a mode whose lifetime fit failed. It has no linewidth, so it
    # carries no coherence and is dropped from the pair sum below; keeping
    # gamma = inf would give exp(-inf * 0) = nan at t = 0.
    no_linewidth = ~np.isfinite(gamma)
    gamma = np.where(no_linewidth, 0.0, gamma)

    v_ssa = np.asarray(v_qssa)
    vv = (v_ssa[..., :, None] * np.swapaxes(v_ssa, 1, 2)[..., None, :]).real
    vv = vv.astype(dtype_real)
    pair_ok = ~(no_linewidth[:, :, None] | no_linewidth[:, None, :])
    prefactor = cv[:, None, :] * pair_ok

    C_QHGK = np.zeros((Nt, 3, 3), dtype=dtype_real)
    t_chunk = max(1, min(Nt, 256))
    for t0 in range(0, Nt, t_chunk):
        t1 = min(Nt, t0 + t_chunk)
        kernel = _qhgk_coherence_kernel(t[t0:t1], w, wi, gamma)
        C_QHGK[t0:t1] = C.to_W_mK * np.einsum(
            "tqsp,qsp,qspab->tab", kernel, prefactor, vv, optimize=True,
        ).astype(dtype_real)
    kappa_QHGK = np.asarray(
        _cumtrapz(xr.DataArray(C_QHGK, dims=(keys.time,) + keys.tensor,
                                coords={keys.time: t})),
        dtype=dtype_real)

    return C_BTE, kappa_BTE, C_QHGK, kappa_QHGK


def get_kappa(dataset, fc_file=None, dmx_file=None,
              interpolate=False, nq_max=20, backend="numpy",
              precision=None, max_mem_gb=4.0, freq=None,
              factorization="wick", analytical=False, harmonic_flux=False,
              lifetime_fit_cutoff=0.5,
              correct_finite_time=True,
              enforce_translational_invariance=True,
              enforce_space_group=True,
              convention="PHONO3PY"):
    """Run the full Green-Kubo thermal-conductivity pipeline on an MD trajectory.

    The pipeline:
        1. (optional) build / load a ``DynamicalMatrix`` from FC.
        2. MIC-fold displacements; check residuals against the
           harmonic prediction.
        3. compute the heat-flux autocorrelation and its integral;
           apply a Savitzky-Golay filter; pick the cutoff time per
           tensor component.
        4. (with DMX) project trajectory onto modes; fit per-mode
           lifetimes ``tau_qs`` and heat capacities ``cv_qs``.
        5. (optional) dense-q-grid interpolation + linear
           extrapolation to the infinite-supercell limit.

    Args:
        dataset: trajectory ``xr.Dataset`` (loaded by ``open_dataset``).
            Must carry ``positions``, ``velocities``, ``heat_flux`` data
            vars and the ``reference_primitive`` / ``reference_supercell``
            JSON attrs.
        fc_file: optional path to a ``FORCE_CONSTANTS`` / ``fc2.hdf5`` /
            flat ``.dat`` file. Used when no DMX cache exists yet. With
            both ``fc_file`` and ``dmx_file`` ``None``, the
            mode-decomposed branch is skipped and only the bare HFACF
            kappa is returned.
        dmx_file: optional path to a cached ``DynamicalMatrix.nc``.
            Read if the file exists; otherwise built from ``fc_file``
            (or the dataset's embedded FC) and saved here.
        interpolate: enable dense-q-grid linear extrapolation of
            ``kappa_corrected``. No-op when no DMX is available.
        nq_max: maximum mesh density for the dense-grid sweep
            (``nq = 4, 6, ..., nq_max``).
        backend: ``"numpy"`` (default, CPU) or ``"jax"`` (CPU/GPU,
            lazily imported). Forwarded to all backend-aware kernels.
        precision: ``"fp64"`` / ``"fp32"`` / ``None``. ``None`` falls
            back to ``GKMX_PRECISION`` (default ``"fp32"``). When
            ``dmx_file`` is provided AND the file pins a ``precision``
            attr, that pinned value wins.
        max_mem_gb: per-block memory budget for ``compute_cv_tau``.
            Float (e.g. ``4.0``) or the string ``"auto"`` to detect
            device free memory and apply the safety model. Honors the
            ``GKMX_MAX_MEM_GB`` env override.
        freq: explicit filter-window frequency in THz. ``None``
            auto-detects from the velocities VDOS (lowest peak above
            ``threshold_freq=0.1 THz``).
        factorization: ``"wick"`` (default; dressed-bubble / SMA fit
            on ``|g(t)|²``) or ``"vertex"`` (un-factorized
            ``<n(t) n(0)>`` fit, vibes_tom-parity).
        analytical: emit analytical BTE / QHGK time-resolved HFACFs
            alongside the simulated one (adds 4 × ``(Nt, 3, 3)`` arrays
            to the output).
        harmonic_flux: emit the measured harmonic heat fluxes — real-space
            ``J_hm-R`` (read from the trajectory, or rebuilt from the force
            constants when it is absent) and its mode-space counterparts
            ``J_hm-q`` and ``J_quasi-hm`` — each with its ACF and running
            integral. Written as ``heat_flux_harmonic`` / ``heat_flux_harmonic_q`` /
            ``heat_flux_QHGK_ta`` (plus ``_acf`` and ``_acf_integral``) on the
            per-MD-step ``time_md`` axis.
        lifetime_fit_cutoff: ACF threshold for the per-mode exponential
            fit. Lower → fit further into the tail (more robust for
            long-tau modes, noisier for short-tau).
        correct_finite_time: apply the ``1/τ → 1/τ − 1/T_max``
            finite-time correction. Affects long-lifetime modes only.
        enforce_translational_invariance: impose the acoustic sum rule on the
            force constants before solving; see ``Phonon``. Default ``True``.
        enforce_space_group: project the force constants onto the
            space-group-invariant subspace before solving; see ``Phonon``.
            Default ``True``.
        convention: ``"PHONO3PY"`` (default), ``"TDEP"``, or ``"RAW"``; see
            ``gkmx.phonon.CONVENTIONS``. The eigenvectors enter the mode
            projection, so tau and cv move with it and kappa_BTE moves too, not
            only kappa_QHGK. The raw HFACF kappa never sees it.

    Returns:
        ``xr.Dataset`` with kappa tensor, HFACF, mode-resolved
        ``cv_qs`` / ``tau_qs``, group velocities, cutoff time per
        tensor component, and (with ``interpolate``) the
        infinite-supercell-extrapolated correction.
    """
    if max_mem_gb == "auto":
        from ._resources import Resources
        Nt_auto = int(dataset.sizes.get(keys.time, 0))
        # Use the trajectory's atom count directly; reading the DMX cache
        # here would load 100s of MB just to get one dim size.
        Nat_auto = int(np.asarray(dataset.positions).shape[1])
        bpr = np.dtype(Precision.resolve(precision).real).itemsize
        plan = Resources.auto_resolve_for_compute_cv_tau(
            backend, Nt_auto, Nat_auto, bytes_per_real=bpr)
        free_gb = plan.resources.free_gb if plan.resources else float("nan")
        _talk(f"max_mem_gb=auto → {plan.max_mem_gb:.2f} GB "
              f"(backend={backend!r} Nt={Nt_auto} Nat={Nat_auto} "
              f"free={free_gb:.1f} GB)")
        max_mem_gb = plan.max_mem_gb

    p = Precision.resolve(precision)

    # One-shot dtype commitment: every downstream array inherits p.real.
    for _name in (keys.positions, keys.velocities, keys.forces, keys.heat_flux,
                  keys.temperature):
        if _name in dataset and dataset[_name].dtype != p.real:
            dataset[_name] = dataset[_name].astype(p.real)
    if keys.time in dataset.coords and dataset[keys.time].dtype != p.real:
        dataset = dataset.assign_coords({keys.time: dataset[keys.time].astype(p.real)})

    # heat_flux must be eV/(Å²·fs); ps-base vibes/gkx fixtures must be
    # divided by 1000 at fixture creation, NOT here.

    primitive = Atoms(**json.loads(dataset.attrs[keys.reference_primitive]))
    supercell = Atoms(**json.loads(dataset.attrs[keys.reference_supercell]))

    sc_positions = np.asarray(supercell.positions)
    cell = np.asarray(supercell.cell)
    search_images = not is_orthogonal(cell)
    positions_da = dataset[keys.positions]

    # Eager fold avoids re-running mic_fold per chunk in both
    # anharmonicity and compute_cv_tau; lazy fallback above GKMX_EAGER_DISP_GB.
    eager_budget_gb = float(os.environ.get("GKMX_EAGER_DISP_GB", "8.0"))
    disp_bytes = int(positions_da.size * positions_da.dtype.itemsize)
    if disp_bytes <= eager_budget_gb * 1e9:
        from . import mic
        timer = Timer("MIC-fold displacements (eager)", prefix=_prefix)
        raw = np.asarray(positions_da.values, dtype=p.real) - sc_positions
        # Skip the fold when the trajectory is unwrapped relative to the
        # reference (gkx-style writers) — fold would be a very expensive
        # identity. vibes-style writers with boundary-atom image mismatch
        # take the else branch and run the fold normally.
        if not mic.needs_fold(raw, cell):
            disp_arr = raw
            timer(f"MIC-fold displacements skipped ({disp_bytes/1e9:.2f} GB, "
                  f"max raw |d| < L_min/2 = {mic.safe_radius(cell):.2f} Å; "
                  f"trajectory unwrapped)")
        else:
            disp_arr = mic_fold(raw, cell, search=search_images)
            timer(f"MIC-fold displacements ({disp_bytes/1e9:.2f} GB, "
                  f"search_images={search_images}; some rows wrap)")
        dataset[keys.displacements] = (positions_da.dims, disp_arr)
        check_disp_magnitudes(disp_arr, cell)
    else:
        _talk(f"MIC-fold displacements: lazy per-chunk "
              f"({disp_bytes/1e9:.2f} GB > {eager_budget_gb:.1f} GB; "
              f"override with GKMX_EAGER_DISP_GB)")
        disp_da = xr.apply_ufunc(
            _disp_block,
            positions_da,
            kwargs={"sc_positions": sc_positions, "cell": cell,
                    "search_images": search_images},
            dask="parallelized",
            output_dtypes=[positions_da.dtype],
        )
        dataset[keys.displacements] = disp_da

    volumes = np.full(positions_da.shape[0],
                      float(dataset.attrs[keys.volume]), dtype=p.real)
    dataset[keys.volume] = (keys.time, volumes)

    dmx_path = Path(dmx_file) if dmx_file else None
    fc_path = Path(fc_file) if fc_file else None
    dmx = None

    # FC lookup: cached DMX → external FC file → embedded FC → none.
    if dmx_path and dmx_path.exists():
        _talk(f"Loading DynamicalMatrix from {dmx_path}")
        dmx = DynamicalMatrix.from_hdf5(str(dmx_path), backend=backend, precision=p.name)
    elif fc_path and fc_path.exists():
        timer = Timer("Building DynamicalMatrix from FC file", prefix=_prefix)
        fc = parse_force_constants(str(fc_path), two_dim=False)
        dmx = DynamicalMatrix(
            force_constants=np.asarray(fc), primitive=primitive, supercell=supercell,
            with_group_velocity_matrices=True, backend=backend, precision=p.name,
            enforce_translational_invariance=enforce_translational_invariance,
            enforce_space_group=enforce_space_group,
            convention=convention,
        )
        if dmx_path:
            _talk(f"Saving DynamicalMatrix to {dmx_path}")
            dmx.to_hdf5(str(dmx_path), include_D_qij=True, include_group_velocity_matrices=True)
        timer()
    elif keys.fc in dataset.data_vars:
        timer = Timer("Building DynamicalMatrix from trajectory-embedded FC",
                      prefix=_prefix)
        fc = np.asarray(dataset[keys.fc])
        dmx = DynamicalMatrix(
            force_constants=fc, primitive=primitive, supercell=supercell,
            with_group_velocity_matrices=True, backend=backend, precision=p.name,
            enforce_translational_invariance=enforce_translational_invariance,
            enforce_space_group=enforce_space_group,
            convention=convention,
        )
        if dmx_path:
            _talk(f"Saving DynamicalMatrix to {dmx_path}")
            dmx.to_hdf5(str(dmx_path), include_D_qij=True, include_group_velocity_matrices=True)
        timer()

    if dmx is not None:
        dataset.update({keys.fc: (keys.dim_fc, np.asarray(dmx.fc_phonopy, dtype=p.real))})
        dataset.update({keys.fc_remapped: (keys.dim_fc_remapped, np.asarray(dmx.remapped))})
        dataset.attrs[keys.map_supercell_to_primitive] = np.asarray(dmx.I2iL_map[:, 0])

    if dmx is None and harmonic_flux:
        warn("harmonic_flux=True but no FC / DMX provided — the measured "
             "harmonic fluxes need a mode decomposition and are skipped.",
             prefix=_prefix)

    if dmx is None and interpolate:
        warn("interpolate=True but no FC / DMX provided — skipping "
             "mode-decomposition and interpolation, returning bare "
             "HFACF-based kappa.", prefix=_prefix)

    ds_gk = _get_gk_dataset(
        dataset, dmx=dmx, interpolate=interpolate,
        quasi_harmonic_greenkubo=True, nq_max=nq_max, backend=backend,
        max_mem_gb=max_mem_gb, freq=freq, precision=p.name,
        factorization=factorization,
        analytical=analytical,
        harmonic_flux=harmonic_flux,
        lifetime_fit_cutoff=lifetime_fit_cutoff,
        correct_finite_time=correct_finite_time,
    )
    return ds_gk


def _get_gk_dataset(dataset, dmx=None, interpolate=False,
                     quasi_harmonic_greenkubo=False, nq_max=20,
                     window_factor=C.window_factor,
                     filter_prominence=C.default_filter_prominence,
                     freq=None,
                     total=False, cross_offdiag=False, verbose=True,
                     backend="numpy", max_mem_gb=4.0, precision=None,
                     factorization="wick", analytical=False, harmonic_flux=False,
                     lifetime_fit_cutoff=0.5,
                     correct_finite_time=True):
    """HFACF -> integrated kappa -> cutoff times, plus optional mode decomposition."""
    heat_flux = dataset[keys.heat_flux]
    if total and keys.heat_flux_aux in dataset:
        heat_flux = heat_flux + dataset[keys.heat_flux_aux]

    gk_pf = _gk_prefactor_from_dataset(dataset, verbose=verbose)
    hfacf, kappa = _get_hf_data(heat_flux, prefactor=gk_pf, total=total)

    if freq is None:
        freq = _get_lowest_vib_freq(
            dataset[keys.velocities], prominence=filter_prominence,
            backend=backend, max_mem_gb=max_mem_gb)
        if verbose:
            _talk(f"Filter frequency (auto): {freq:.4f} THz "
                  f"(prominence={filter_prominence})")
    else:
        freq = float(freq)
        if verbose:
            _talk(f"Filter frequency (user): {freq:.4f} THz")
    window_fs = window_factor / freq * 1000
    if verbose:
        _talk(f"Filter window: freq={freq:.4f} THz, window={window_fs:.1f} fs")

    k_filtered = _savgol_filter(kappa, window_fs=window_fs, antisymmetric=True)

    # Derivative-of-filtered-kappa is smoother than direct filter of HFACF.
    dt = float(kappa.time[1] - kappa.time[0])
    k_grad = kappa.copy()
    k_grad.data = np.gradient(k_filtered, axis=0) / dt
    j_filtered = _savgol_filter(k_grad, window_fs=window_fs)

    # Diagonal cutoff = first HFACF zero-crossing; off-diagonal reuses
    # diagonal averaged time (unless cross_offdiag asks for its own).
    p = Precision.resolve(precision)
    ts = np.zeros([3, 3], dtype=p.real)
    ks = np.zeros([3, 3], dtype=p.real)
    j_sym = 0.5 * (j_filtered + np.swapaxes(j_filtered, 1, 2))
    k_sym = 0.5 * (k_filtered + np.swapaxes(k_filtered, 1, 2))

    for ii, jj in np.array(list(np.ndindex(3, 3)))[[0, 4, 8, 1, 2, 3, 5, 6, 7]]:
        j = j_sym[:, ii, jj]
        if ii == jj:
            times = j.time[j < 0]
        elif cross_offdiag:
            cross_time = (ts[ii, ii] + ts[jj, jj]) / 2
            times = j.time[j.time > cross_time]
        else:
            times = j.time[j / j[0] < 0]

        ta = float(times.min()) if len(times) > 1 else 0
        ks[ii, jj] = float(k_sym[:, ii, jj].sel(time=ta))
        ts[ii, jj] = ta

    if verbose:
        k_mean = np.mean(np.diag(ks))
        k_err = np.std(np.diag(ks)) / 3 ** 0.5
        _talk(f"Kappa: {k_mean:.3f} +/- {k_err:.3f} W/mK")

    attrs = dataset.attrs.copy()
    attrs.update({keys.gk_window_fs: window_fs, keys.gk_prefactor: gk_pf,
                  keys.filter_prominence: filter_prominence})

    data = {
        keys.heat_flux: heat_flux,
        keys.hf_acf: hfacf,
        keys.hf_acf_filtered: j_filtered,
        keys.kappa_cumulative: kappa,
        keys.kappa_cumulative_filtered: k_filtered,
        keys.kappa: (keys.tensor, ks),
        keys.time_cutoff: (keys.tensor, ts),
    }

    if dmx is not None:
        data[keys.kappa_symmetrized] = (
            keys.tensor, np.asarray(_symmetrize_kappa(ks, dmx.primitive)))

    if dmx is not None:
        timer = Timer("Anharmonicity", prefix=_prefix)
        sigma_ps, sigma = _harmonic_force_residuals(
            disp=dataset.displacements.data,
            forces=dataset.forces.data,
            fc_remapped=np.asarray(dmx.remapped),
            backend=backend, dtype=getattr(dmx, "_dtype_real", None) or p.real,
        )
        # Distinct dim from keys.time: heat_flux.time is dropna'd while
        # sigma_per_sample lives on every MD step.
        data[keys.sigma_per_sample] = (keys.time_md, sigma_ps)
        attrs[keys.sigma] = sigma
        # Provenance: tau, cv and every mode-resolved field below depend on it,
        # and nothing downstream can infer it from the arrays.
        attrs["convention"] = dmx._convention
        timer()

        data_ha, dmx = _get_gk_interpolate(
            dataset, dmx=dmx, interpolate=interpolate,
            nq_max=nq_max, quasi_harmonic_greenkubo=quasi_harmonic_greenkubo,
            backend=backend, max_mem_gb=max_mem_gb,
            factorization=factorization,
            lifetime_fit_cutoff=lifetime_fit_cutoff,
            correct_finite_time=correct_finite_time,
        )
        data.update(data_ha._asdict())
        data[keys.fc] = dataset[keys.fc]

        if analytical:
            timer = Timer("Analytical BTE / QHGK HFACFs", prefix=_prefix)
            C_BTE, k_BTE, C_QHGK, k_QHGK = _analytical_hfacfs(
                time_fs=np.asarray(kappa.time),
                tau_qs=data_ha.mode_lifetime,
                cv_qs=data_ha.mode_heat_capacity,
                v_qsa=data_ha.v_qsa_cartesian,
                v_qssa=data_ha.v_qssa_cartesian,
                w_qs=data_ha.w_qs,
                w_inv_qs=data_ha.w_inv_qs,
                dtype_real=p.real,
            )
            data[keys.hf_acf_BTE] = (keys.time_tensor, C_BTE)
            data[keys.hf_acf_BTE_integral] = (keys.time_tensor, k_BTE)
            data[keys.hf_acf_QHGK] = (keys.time_tensor, C_QHGK)
            data[keys.hf_acf_QHGK_integral] = (keys.time_tensor, k_QHGK)
            timer()

        if harmonic_flux:
            timer = Timer("Harmonic heat fluxes", prefix=_prefix)
            J_hm_q, J_quasi_hm = compute_harmonic_heat_flux_q(
                dataset, dmx, v_qssa=data_ha.v_qssa_cartesian,
                dtype_u=p.real, verbose=verbose)
            J_hm_R = (np.asarray(dataset[keys.heat_flux_harmonic], dtype=p.real)
                      if keys.heat_flux_harmonic in dataset else None)
            if J_hm_R is None or np.isnan(J_hm_R).any():
                J_hm_R = compute_harmonic_heat_flux_R(
                    dataset, dmx, dtype=p.real, verbose=verbose)
            t_coord = np.asarray(
                dataset.displacements[dataset.displacements.dims[0]])
            data[keys.time_md] = (keys.time_md, t_coord)
            for name, J, k_acf, k_int in (
                    (keys.heat_flux_harmonic, J_hm_R,
                     keys.hf_acf_ha, keys.hf_acf_ha_integral),
                    (keys.heat_flux_harmonic_q, J_hm_q,
                     keys.hf_acf_ha_q, keys.hf_acf_ha_q_integral),
                    (keys.heat_flux_QHGK_ta, J_quasi_hm,
                     keys.hf_acf_qhgk_ta, keys.hf_acf_qhgk_ta_integral)):
                da = xr.DataArray(J, dims=keys.time_vec,
                                  coords={keys.time: t_coord})
                acf, cum = _get_hf_data(da, prefactor=gk_pf)
                data[name] = (keys.time_md_vec, J)
                data[k_acf] = (keys.time_md_tensor, np.asarray(acf))
                data[k_int] = (keys.time_md_tensor, np.asarray(cum))
            timer()

        # The interpolation fields are absent when the commensurate grid cannot
        # triangulate (QhullError, e.g. a supercell sampling a 2D q-plane) —
        # degrade to the uncorrected output instead of crashing.
        if interpolate and hasattr(data_ha, keys.interpolation_correction):
            correction = data_ha.interpolation_correction
            kappa_corrected = ks + correction * np.eye(3, dtype=ks.dtype)
            data[keys.kappa_corrected] = (keys.tensor, kappa_corrected)
            k_mean = np.mean(np.diag(ks))
            _talk(f"Corrected kappa: {k_mean + correction:.3f} W/mK")

    data.update({key: dataset[key] for key in (keys.volume, keys.temperature)})
    return xr.Dataset(data, coords=kappa.coords, attrs=attrs)


def _get_gk_interpolate(dataset, dmx=None, interpolate=False,
                         quasi_harmonic_greenkubo=False, nq_max=20,
                         backend="numpy", max_mem_gb=4.0,
                         factorization="wick", lifetime_fit_cutoff=0.5,
                         correct_finite_time=True):
    """Mode decomposition (and optional dense-grid interpolation) on top of a DMX."""
    timer = Timer("DynamicalMatrix setup", prefix=_prefix)
    need_vssq = bool(quasi_harmonic_greenkubo)

    if dmx is not None:
        sol = getattr(dmx, "solution", None)
        have_vssq = sol is not None and getattr(sol, "v_qssa_cartesian", None) is not None
        if need_vssq and not have_vssq:
            dmx = DynamicalMatrix.from_dataset(
                dataset, with_group_velocity_matrices=True,
                convention=dmx._convention)
    else:
        dmx = DynamicalMatrix.from_dataset(dataset, with_group_velocity_matrices=need_vssq)
    timer()

    # Free anharmonic tensors for the cuFFT planner workspace; never
    # call jax.clear_caches() — it evicts compiled binaries, not memory,
    # and slows interpolation 30-40 %.
    gc.collect()

    timer = Timer(f"compute_cv_tau ({backend} [{get_backend(backend).device_description()}])",
                  prefix=_prefix)
    cv_qs, tau_qs = compute_cv_tau(
        dataset=dataset, dmx=dmx, backend=backend, max_mem_gb=max_mem_gb,
        factorization=factorization,
        lifetime_fit_cutoff=lifetime_fit_cutoff,
        correct_finite_time=correct_finite_time,
    )
    timer()

    map2ir, map2full = dmx.q_grid.map2ir, dmx.q_grid.ir.map2full
    tau_sym = get_symmetrized_array(tau_qs, map2ir=map2ir, map2full=map2full)

    v_qsa = dmx.v_qsa_cartesian
    K_ha = get_kappa_BTE(v_qsa=v_qsa, tau_qs=tau_qs, cv_qs=cv_qs)
    K_ha.name = keys.kappa_ha

    # Scalar cv chosen so K_ha_sym matches K_ha's diagonal average;
    # mean(cv_qs) fallback when K_ha is near zero (trivial fixtures).
    real_dt = getattr(dmx, "_dtype_real", None) or Precision.default().real
    k = float(K_ha.data.diagonal().mean())
    if k < 1e-4:
        cv = cv_qs.mean()
    else:
        cv = k / get_kappa_BTE(v_qsa=v_qsa, tau_qs=tau_qs, scalar=True)
    cv = xr.DataArray(np.asarray(cv, dtype=real_dt), name=keys.heat_capacity)

    K_ha_sym = get_kappa_BTE(v_qsa=v_qsa, tau_qs=tau_sym, cv_qs=cv)
    K_ha_sym.name = keys.kappa_ha_symmetrized

    v_qsa_arr = np.asarray(v_qsa, dtype=real_dt)
    tau_arr = np.nan_to_num(np.asarray(tau_qs, dtype=real_dt))
    cv_arr = np.asarray(cv_qs, dtype=real_dt)
    v2_qsab = v_qsa_arr[..., :, None] * v_qsa_arr[..., None, :]
    mode_kappa_bte_arr = (C.to_W_mK
                           * (cv_arr * tau_arr)[..., None, None]
                           * v2_qsab).astype(real_dt)
    mode_kappa_BTE = xr.DataArray(
        mode_kappa_bte_arr, dims=(keys.q, keys.s) + keys.tensor, name=keys.mode_kappa_BTE)

    arrays = [K_ha, K_ha_sym, cv_qs, cv, tau_qs, tau_sym,
              mode_kappa_BTE] + dmx._get_arrays()

    sol = getattr(dmx, "solution", None)
    v_qssa_cart = getattr(sol, "v_qssa_cartesian", None) if sol is not None else None
    if v_qssa_cart is not None:
        complex_dt = getattr(dmx, "_dtype_complex", None) or Precision.default().complex
        v_qssa_arr = np.asarray(v_qssa_cart, dtype=complex_dt)
        arrays.append(xr.DataArray(v_qssa_arr, dims=keys.q_s_s_a,
                                    name=keys.v_qssa_cartesian))

        # Per-pair QHGK kappa (.real exact: imag sums to zero by v·v.conj() Hermiticity).
        tau_eff = qhgk_tau_eff(
            np.asarray(dmx.w_qs, dtype=real_dt),
            np.asarray(dmx.w_inv_qs, dtype=real_dt),
            tau_arr,
        )
        cv_bcast = cv_arr[:, None, :] if cv_arr.ndim == 2 else cv_arr
        v2_qssab = (v_qssa_arr[..., :, None]
                    * v_qssa_arr[..., None, :].conj())
        mode_kappa_qhgk_arr = (C.to_W_mK
                                * (cv_bcast * tau_eff)[..., None, None]
                                * v2_qssab).real.astype(real_dt)
        arrays.append(xr.DataArray(
            mode_kappa_qhgk_arr, dims=keys.q_s_s_a[:-1] + keys.tensor,
            name=keys.mode_kappa_QHGK))

    data = {ar.name: ar for ar in arrays}

    if interpolate:
        results = get_interpolation_data(
            dmx=dmx, lifetimes=tau_sym, cv=cv, nq_max=nq_max,
            quasi_harmonic_greenkubo=quasi_harmonic_greenkubo,
        )
        data.update(results)

    return collections.namedtuple("gk_ha_q_data", data.keys())(**data), dmx
