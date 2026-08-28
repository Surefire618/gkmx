"""Cepstral analysis of the heat flux: kappa from the zero-frequency PSD.

Ercole, Marcolongo & Baroni, Sci. Rep. 7, 15835 (2017) (the SporTran method).
The Green-Kubo integral equals half the flux power spectrum at zero frequency,

    kappa = pf/2 * S(0),      pf = V / (kB T^2)

so instead of an ACF plateau one estimates S(0) from the periodogram:

    S_k     = dt/N |sum_t J_t e^{-2 pi i k t / N}|^2      (chi^2-distributed)
    log S_k = smooth log-PSD + noise of KNOWN mean psi(l) - log(l)
              and variance psi'(l)   (l = averaged components/runs)
    c_n     = inverse FFT of log S_k                       (the cepstrum)

The true log-PSD is smooth, so only the first P* cepstral coefficients carry
signal; P* is chosen by the Akaike information criterion, and

    log S(0) = c_0 + 2 sum_{n=1}^{P*-1} c_n,   var = psi'(l) (4P* - 2) / M

gives kappa with a built-in statistical error bar. Whole-series, no Welch
segmentation -- the smoothing lives in the cepstral truncation, which is
unbiased where chunked ACFs are not (see the repo's ACF-kernel history).

Input: a trajectory .nc (uses ``heat_flux``) or a gk/ensemble .nc (rebuilds
the periodogram from the stored ACF; ``--ell`` should then count the runs x
components averaged into it). Components listed in ``--components`` are
treated as statistically equivalent and averaged -- only symmetry-equivalent
ones belong together (cubic: xx,yy,zz; uniaxial: xx,yy).

    python tools/cepstral_kappa.py traj.nc --components xx,yy,zz -o figs/
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scipy.special import polygamma, psi

from gkmx import gk_prefactor, keys, open_dataset
from gkmx.trajectory import _guess_heat_flux_time_unit, _heat_flux_units_attr

COMP = {"xx": (0, 0), "yy": (1, 1), "zz": (2, 2)}


def periodogram_from_flux(J, dt):
    """Two-sided PSD estimate of a flux component, one-sided grid.

    The sample mean is NOT subtracted: the physical flux mean is zero, and
    subtracting the sample mean zeroes the DC bin exactly (log -> -inf)
    instead of leaving it a legitimate chi^2 sample.
    """
    J = J[np.isfinite(J)]
    N = len(J)
    F = np.fft.rfft(J)
    S = dt / N * np.abs(F) ** 2
    f = np.fft.rfftfreq(N, d=dt)
    return f, S


def periodogram_from_acf(acf, dt):
    """Approximate periodogram from a stored one-sided ACF.

    The even extension of a truncated linear ACF is not exactly a
    periodogram and can go negative at high frequency; those bins are
    floored at a small positive quantile so they read as "no power" rather
    than log(0) = -inf, which would poison the cepstrum. Prefer the raw-flux
    path when the trajectory is available -- there the chi^2 statistics are
    exact.
    """
    acf = acf[np.isfinite(acf)]
    sym = np.concatenate([acf, acf[-2:0:-1]])
    S = dt * np.fft.rfft(sym).real
    f = np.fft.rfftfreq(len(sym), d=dt)
    pos = S[S > 0]
    floor = 1e-6 * np.median(pos) if len(pos) else 1e-300
    return f, np.maximum(S, floor)


def cepstral(S_bar, ell):
    """log S(0) and its variance via the AIC-truncated cepstrum.

    Names follow the paper / SporTran: ``S_bar`` is their averaged
    periodogram S-bar, ``ell`` their sample count l.

    Args:
        S_bar: one-sided periodogram, the mean of ``ell`` statistically
            equivalent samples (symmetry-equivalent components x runs).
        ell: number of independent averages in ``S_bar``; sets the known
            log-chi^2 bias psi(ell) - log(ell) and variance psi'(ell).

    Returns:
        ``(L0, varL0, P_star, aic, S_smooth)``: ``exp(L0)`` estimates
        ``S(0)`` with log-variance ``varL0``; ``S_smooth`` is the spectrum
        rebuilt from the ``P_star`` retained coefficients, for plotting.
    """
    S_bar = np.maximum(np.array(S_bar, dtype=float), 1e-300)
    M = len(S_bar)
    lam = psi(ell) - np.log(ell)          # log-chi^2 bias
    sig2 = polygamma(1, ell)              # log-chi^2 variance
    logS = np.log(S_bar) - lam
    ext = np.concatenate([logS, logS[-2:0:-1]])
    c = np.fft.ifft(ext).real[:M]
    # AIC(P) = M/sig2 * sum_{n>=P} c_n^2 + 2P
    tail = np.cumsum((c ** 2)[::-1])[::-1]
    Pmax = min(M - 1, 2000)
    aic = np.array([M / sig2 * tail[P] + 2 * P for P in range(1, Pmax)])
    P_star = int(np.argmin(aic)) + 1
    L0 = c[0] + 2.0 * c[1:P_star].sum()
    varL0 = sig2 * (4 * P_star - 2) / (2 * M - 2)
    # smoothed spectrum from the retained coefficients, for the plot
    keep = np.zeros(M)
    keep[:P_star] = c[:P_star]
    ext_s = np.concatenate([keep, keep[-2:0:-1]])
    logS_smooth = np.fft.fft(ext_s).real[:M]
    return L0, varL0, P_star, aic, np.exp(logS_smooth)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nc", type=Path)
    ap.add_argument("--field", default=None,
                    help="flux (time, atom-free (t,3)) or ACF field name; "
                         "default: heat_flux, else the stored raw ACF")
    ap.add_argument("--components", default="xx",
                    help="symmetry-EQUIVALENT components to average "
                         "(cubic: xx,yy,zz; uniaxial: xx,yy)")
    ap.add_argument("--ell", type=int, default=None,
                    help="independent averages in the periodogram (default: "
                         "n components; multiply by ensemble_n_runs yourself "
                         "for ensemble-mean ACF inputs)")
    ap.add_argument("-o", "--outdir", default=Path("cepstral"), type=Path)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--heat-flux-time-unit", default=None,
                    choices=["fs", "ps"])
    args = ap.parse_args()

    # Resolve the heat-flux time base exactly as the pipeline does, but
    # NEVER silently: vibes-era writers are ps-base, gkx-era fs-base, and a
    # wrong base moves kappa by 1e6. Declared attr > --heat-flux-time-unit >
    # magnitude guess (warned).
    raw = xr.open_dataset(str(args.nc), engine="h5netcdf")
    declared = (_heat_flux_units_attr(raw)
                or raw.attrs.get("heat_flux_time_unit"))
    if "heat_flux" in raw and declared is None and             args.heat_flux_time_unit is None:
        guess, k_rough = _guess_heat_flux_time_unit(raw)
        print(f"WARNING: {args.nc.name} declares no heat-flux unit; "
              f"magnitude guess = {guess}-base (rough kappa "
              f"{k_rough:.2e} W/mK). Pass --heat-flux-time-unit to pin it.")
    raw.close()
    ds = open_dataset(str(args.nc),
                      heat_flux_time_unit=args.heat_flux_time_unit).load()
    unit_src = (f"declared {declared}" if declared
                else f"forced {args.heat_flux_time_unit}"
                if args.heat_flux_time_unit else "guessed")
    comps = args.components.split(",")
    tag = args.tag or args.nc.stem

    field = args.field or ("heat_flux" if "heat_flux" in ds else "acf")
    if field == "acf":
        field = keys.hf_acf
    arr = np.asarray(ds[field].data, dtype=np.float64)
    tdim = ds[field].dims[0]
    tt = np.asarray(ds[tdim], dtype=float)
    dt = float(tt[1] - tt[0])
    if arr.ndim == 2:
        # (time, 3): a flux time series -- exact chi^2 statistics
        J = arr
        specs = []
        for cname in comps:
            a, _ = COMP[cname]
            f, S = periodogram_from_flux(J[:, a], dt)
            specs.append(S)
        source = f"flux field {field!r}"
        is_flux = True
    elif arr.ndim == 3:
        # (time, 3, 3): a stored ACF tensor (already prefactor-scaled by the
        # pipeline) -- approximate periodogram reconstruction
        specs = []
        for cname in comps:
            a, b = COMP[cname]
            f, S = periodogram_from_acf(arr[:, a, b], dt)
            specs.append(S)
        source = f"ACF field {field!r}"
        is_flux = False
        prefactor_applied = True
    else:
        raise SystemExit(f"field {field!r} has ndim {arr.ndim}; expected a "
                         f"(t, 3) flux or (t, 3, 3) ACF")
    S_bar = np.mean(specs, axis=0)
    ell = args.ell or len(comps)

    prefactor_applied = locals().get("prefactor_applied", False)
    if keys.volume in ds:
        volume = float(np.nanmean(np.asarray(ds[keys.volume])))
    else:
        volume = float(ds.attrs[keys.volume])
    if keys.temperature in ds:
        temperature = float(np.nanmean(np.asarray(ds[keys.temperature])))
    else:
        temperature = float(ds.attrs["temperature"])
    pf = 1.0 if prefactor_applied else gk_prefactor(volume, temperature)

    L0, varL0, P_star, aic, S_smooth = cepstral(S_bar, ell)
    kappa = pf * np.exp(L0) / 2.0
    sigma = kappa * np.sqrt(varL0)
    print(f"{tag}: kappa = {kappa:.4f} +/- {sigma:.4f} W/mK   "
          f"(P* = {P_star}, ell = {ell}, {source}, unit: {unit_src}, "
          f"T = {temperature:.1f} K, V = {volume:.1f} A^3)")

    # Same-series GK cross-check: the running integral computed from the
    # identical loaded series with the identical prefactor. A unit slip moves
    # this and the cepstral number together by the same factor, so the
    # cepstral-vs-plateau comparison is unit-consistent by construction.
    if is_flux:
        Jm = np.nanmean([np.asarray(ds[field].data, dtype=np.float64)[:, COMP[c][0]]
                         for c in comps], axis=0)
        Jm = Jm[np.isfinite(Jm)]
        acf_x = np.correlate(Jm[:4000] - 0.0, Jm[:4000], "full")[len(Jm[:4000]) - 1:]
        acf_x /= len(Jm[:4000])
        cum = np.cumsum(acf_x) * dt * pf
        for i_ps in (1, 5, 20):
            i = int(i_ps * 1000 / dt)
            if i < len(cum):
                print(f"  cross-check GK integral({i_ps} ps) = {cum[i]:+.4f} "
                      f"W/mK  (same series, same prefactor)")

    # ---- side-by-side: time domain (running GK integral) vs frequency
    # domain (kappa(f) = pf/2 S(f)); the cutoff-choice kappa and the
    # cepstral kappa are both marked on both panels ----
    if is_flux:
        acfs = []
        for cname in comps:
            a, _ = COMP[cname]
            Jc = J[:, a][np.isfinite(J[:, a])]
            n = len(Jc)
            Fz = np.fft.rfft(Jc, 2 * n)
            ac = np.fft.irfft(np.abs(Fz) ** 2, 2 * n)[:n] / n
            acfs.append(ac)
        k_t = np.cumsum(np.mean(acfs, axis=0)) * dt * pf
        t_ps = np.arange(len(k_t)) * dt / 1000.0
    else:
        integral = field + "_" + keys.integral
        if integral in ds:
            cum = np.asarray(ds[integral].data, dtype=np.float64)
            k_t = np.mean([cum[:, COMP[c][0], COMP[c][1]] for c in comps],
                          axis=0)
        else:
            am = np.mean([np.asarray(ds[field].data, dtype=np.float64)
                          [:, COMP[c][0], COMP[c][1]] for c in comps], axis=0)
            k_t = np.cumsum(np.nan_to_num(am)) * dt
        t_ps = np.asarray(ds[tdim], dtype=float)[:len(k_t)] / 1000.0

    # the analytical model kernel (fitted cv/tau/v; integral already in
    # W/mK, ties to tau_eff exactly) -- the smooth counterpart of the noisy
    # measured integral, and the channel the size extrapolation rests on
    k_model = None
    if keys.hf_acf_QHGK_integral in ds:
        km = np.asarray(ds[keys.hf_acf_QHGK_integral].data, dtype=np.float64)
        k_model = np.mean([km[:, COMP[c][0], COMP[c][1]] for c in comps],
                          axis=0)
        t_model = np.asarray(
            ds[ds[keys.hf_acf_QHGK_integral].dims[0]], dtype=float) / 1000.0

    # frequency-domain version of the same model kernel: the one-sided
    # cosine transform kappa(f) = int_0^inf K(t) cos(2 pi f t) dt, so its
    # f -> 0 limit is exactly the model plateau on the time panel. K is
    # already in kappa units per fs -- no GK prefactor.
    kf_model = None
    if keys.hf_acf_QHGK in ds:
        Km = np.asarray(ds[keys.hf_acf_QHGK].data, dtype=np.float64)
        Kc = np.nan_to_num(np.mean(
            [Km[:, COMP[c][0], COMP[c][1]] for c in comps], axis=0))
        tm = np.asarray(ds[ds[keys.hf_acf_QHGK].dims[0]], dtype=float)
        dtm = float(tm[1] - tm[0])
        sym_m = np.concatenate([Kc, Kc[-2:0:-1]])
        kf_model = dtm * np.fft.rfft(sym_m).real / 2.0
        f_model = np.fft.rfftfreq(len(sym_m), d=dtm)

    kappa_cutoff = None
    if "thermal_conductivity" in ds:
        kk = np.asarray(ds["thermal_conductivity"].data)
        kappa_cutoff = float(np.mean([kk[COMP[c]] for c in comps]))

    THz = 1e3  # 1/fs -> THz
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.3))
    axs[0].axhline(0, color="k", lw=0.5)
    axs[0].plot(t_ps[1:], k_t[1:], color="tab:blue", lw=1.2,
                label=r"$\kappa(t)$ (GK integral)")
    if k_model is not None:
        axs[0].plot(t_model[1:], k_model[1:], color="tab:green", lw=1.6,
                    label="analytical QHGK (model)")
    axs[0].set_xscale("log")
    axs[0].set_xlabel("t (ps)")
    axs[0].set_ylabel(r"$\kappa$ (W/mK)")
    axs[0].set_title("time domain", fontsize=10)

    axs[1].plot(f[1:] * THz, pf / 2.0 * S_bar[1:], color="0.75", lw=0.5,
                label="periodogram")
    axs[1].plot(f[1:] * THz, pf / 2.0 * S_smooth[1:], "r-", lw=1.4,
                label=f"cepstral (P*={P_star})")
    if kf_model is not None:
        axs[1].plot(f_model[1:] * THz, kf_model[1:], color="tab:green",
                    lw=1.6, label="analytical QHGK (model)")
    axs[1].set_xscale("log")
    axs[1].set_xlabel("f (THz)")
    axs[1].set_ylabel(r"$\kappa(f) = \mathrm{pf}/2\, S(f)$ (W/mK)")
    axs[1].set_title("frequency domain", fontsize=10)

    for ax in axs:
        ax.axhspan(kappa - sigma, kappa + sigma, color="r", alpha=0.15)
        ax.axhline(kappa, color="r", lw=0.9, ls=":",
                   label=rf"cepstral $\kappa$ = {kappa:.2f}")
        if kappa_cutoff is not None:
            ax.axhline(kappa_cutoff, color="k", lw=0.9, ls="--",
                       label=rf"cutoff $\kappa$ = {kappa_cutoff:.2f}")
    lo = min(0.0, float(np.nanmin(k_t)) * 1.05)
    hi = max(float(np.nanquantile(pf / 2.0 * S_smooth, 0.98)),
             float(np.nanmax(k_t)) * 1.05,
             (kappa_cutoff or 0) * 1.2)
    for ax in axs:
        ax.set_ylim(lo, hi)
    axs[0].legend(frameon=False, fontsize="x-small", loc="upper left")
    axs[1].legend(frameon=False, fontsize="x-small", loc="upper right")
    fig.suptitle(f"{tag}: cepstral {kappa:.3f} +/- {sigma:.3f} W/mK"
                 + (f"   |   cutoff {kappa_cutoff:.3f}"
                    if kappa_cutoff is not None else ""),
                 fontsize=10)
    fig.tight_layout()
    args.outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.outdir / f"cepstral_{tag}.{ext}",
                    dpi=450 if ext == "png" else None, bbox_inches="tight")
    (args.outdir / f"cepstral_{tag}.json").write_text(json.dumps({
        "kappa": kappa, "sigma": sigma, "P_star": P_star, "ell": ell,
        "components": comps, "source": source, "temperature": temperature,
        "volume": volume, "file": str(args.nc)}, indent=1))
    print(f"-> {args.outdir}/cepstral_{tag}.png")


if __name__ == "__main__":
    main()
