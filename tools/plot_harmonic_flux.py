"""Plot the measured harmonic heat fluxes from a ``gkmx out gk`` .nc file.

Plots one trajectory's gk output. For each requested Cartesian component:
C_JJ(t) on top, the running kappa(t) below, log time axis. Curves:

    J_v         measured virial heat flux (the raw Green-Kubo input)
    J_hm-R      real-space harmonic flux -- the reference the mode
                decomposition is judged against (carries the displacement term)
    J_quasi-hm  the pair flux, resonant + antiresonant
    J_hm-q      its particle-like diagonal; J_quasi-hm minus J_hm-q is the
                coherent channel

The tool is convention-agnostic: it plots whatever the file contains and
prints the ``convention`` attribute in the title as provenance. Curves whose
fields are absent (runs without ``--harmonic-flux``) are skipped. The mode
fluxes live on the full ``time_md`` axis; the virial ACF on the dropna'd
``time`` axis.

    python tools/plot_harmonic_flux.py gk_1.nc -o figs
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
from matplotlib import colormaps
from matplotlib import pyplot as plt

from gkmx import keys

cmap = colormaps["tab10"]
FIGSIZE = (4, 3)
VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

# (acf key, time dim, label, color, linestyle, alpha)
CURVES = [
    (keys.hf_acf, keys.time, r"$J_v$", "k", "-", 0.3),
    (keys.hf_acf_ha, keys.time_md, r"$J_{\rm hm-R}$", "k", "--", 0.5),
    (keys.hf_acf_qhgk_ta, keys.time_md, r"$J_{\rm quasi-hm}$", cmap(1), "-", 0.7),
    (keys.hf_acf_ha_q, keys.time_md, r"$J_{\rm hm-q}$", cmap(0), "-", 0.7),
]


def plot_file(ncfile, outdir, tag, components, xlim, ylim_acf):
    gk = xr.load_dataset(ncfile)
    conv = gk.attrs.get("convention", "?")
    times = {d: np.asarray(gk[d], dtype=float) / 1000.0 for d in
             (keys.time, keys.time_md) if d in gk.dims or d in gk.coords}
    for t in times.values():
        t[0] = 0.0
    outdir.mkdir(parents=True, exist_ok=True)
    for ab in components:
        a, b = VOIGT[ab]
        fig, axs = plt.subplots(nrows=2, sharex=True, figsize=FIGSIZE)
        axs[0].plot(times[keys.time], np.zeros_like(times[keys.time]),
                    "k-", linewidth=0.5)
        for key, tdim, label, color, ls, alpha in CURVES:
            if key not in gk:
                continue
            t = times[tdim]
            axs[0].plot(t, np.asarray(gk[key])[:, a, b], ls, color=color,
                        label=label, alpha=alpha)
            axs[1].plot(t, np.asarray(gk[key + "_" + keys.integral])[:, a, b],
                        ls, color=color, alpha=alpha)
        axs[0].set_ylabel(r"$C_{JJ}(t)$ (eV/Å/fs)")
        if ylim_acf:
            axs[0].set_ylim(ylim_acf)
        axs[1].set_ylabel(r"$\kappa(t)$ (W/mK)")
        axs[1].set_xlabel(r"Time $t$ (ps)")
        axs[1].set_xscale("log")
        axs[1].set_xlim(xlim)
        fig.legend(loc="upper right", bbox_to_anchor=(0.96, 0.96), ncol=2,
                   frameon=False)
        fig.suptitle(f"{tag}  ({conv})",
                     fontsize=plt.rcParams["font.size"] - 1, y=1.02)
        fig.tight_layout()
        names = "xyz"
        comp = f"{names[a]}{names[b]}"
        for ext in ("png", "pdf"):
            fig.savefig(outdir / f"HFACF_{tag}_{comp}.{ext}",
                        dpi=600 if ext == "png" else None, bbox_inches="tight")
        plt.close(fig)
        print(f"{tag} {comp}: -> {outdir}/HFACF_{tag}_{comp}.png")
    gk.close()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="a gk .nc file")
    ap.add_argument("-o", "--outdir", default="harmonic_flux", type=Path)
    ap.add_argument("--components", default="xx,yy,zz,xz",
                    help="comma-separated from xx,yy,zz,yz,xz,xy")
    ap.add_argument("--xlim", nargs=2, type=float, default=(3e-3, 60.0))
    ap.add_argument("--ylim-acf", nargs=2, type=float, default=(-0.02, 0.1),
                    help="C_JJ panel limits; pass 0 0 to autoscale")
    ap.add_argument("--style", default=None,
                    help="matplotlib style name or .mplstyle path")
    ap.add_argument("--usetex", action="store_true",
                    help="honor the style's TeX text rendering (needs a full "
                         "TeX install); off by default for portability")
    args = ap.parse_args()

    if args.style:
        plt.style.use(args.style)
    if not args.usetex:
        plt.rcParams["text.usetex"] = False
    names = ["xx", "yy", "zz", "yz", "xz", "xy"]
    components = [names.index(c) for c in args.components.split(",")]
    ylim = None if tuple(args.ylim_acf) == (0.0, 0.0) else tuple(args.ylim_acf)

    f = args.input
    # gk outputs stack suffixes (gk_0.gk.interpolate.nc); keep the run id
    plot_file(f, args.outdir, f"{f.parent.name}_{f.stem.split('.')[0]}",
              components, args.xlim, ylim)


if __name__ == "__main__":
    main()
