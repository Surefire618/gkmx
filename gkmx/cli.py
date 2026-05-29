"""Stand-alone gkmx CLI: ``gkmx out gk trajectory.nc --fc ... -o gk.nc``."""
from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr

from ._log import Timer, talk, warn

xr.set_options(file_cache_maxsize=512)

_CLI_PREFIX = "gkmx.cli"


def _talk(msg):
    talk(msg, prefix=_CLI_PREFIX)


def _warn(msg):
    warn(msg, prefix=_CLI_PREFIX)


def _gpu_visible():
    """GPU probe without importing jax (avoids CUDA init noise on CPU hosts)."""
    import os
    import shutil
    import subprocess
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        return bool(cvd.strip()) and cvd.strip() != "-1"
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return False
    try:
        r = subprocess.run([smi, "-L"], capture_output=True,
                           timeout=3, check=False, text=True)
    except Exception:
        return False
    return r.returncode == 0 and "GPU " in r.stdout


def _resolve_backend(choice):
    """Map ``--backend auto`` to ``jax`` when a real GPU is usable, else ``numpy``."""
    if choice != "auto":
        return choice
    if not _gpu_visible():
        _talk("auto backend: no GPU visible → numpy")
        return "numpy"
    try:
        import contextlib
        import io
        # Suppress the "GPU may be present but CUDA jaxlib not installed"
        # stderr note during the probe — we handle that case below.
        with contextlib.redirect_stderr(io.StringIO()):
            import jax
            devs = jax.devices()
    except ImportError:
        _talk("auto backend: GPU visible but jax not importable → numpy")
        return "numpy"
    except Exception as e:
        _talk(f"auto backend: jax probe failed ({type(e).__name__}) → numpy")
        return "numpy"
    if not any(d.platform != "cpu" for d in devs):
        _talk("auto backend: jax landed on cpu (no CUDA jaxlib) → numpy")
        return "numpy"
    _talk(f"auto backend: jax on {devs[0].platform} → jax")
    return "jax"


def _add_gk_parser(subparsers):
    p = subparsers.add_parser(
        "gk",
        help="Green-Kubo postprocessor.",
        description="Run Green-Kubo postprocessing on a trajectory FILE.",
    )
    p.add_argument("file", type=Path, help="Trajectory file.")
    p.add_argument(
        "-fc", "--fc_file", default=None, type=Path,
        help="FORCE_CONSTANTS / fc2.hdf5 file. When omitted and the "
             "trajectory has an embedded `force_constants` data_var, "
             "that FC is used instead.")
    p.add_argument(
        "-dmx", "--dmx_file", default=None, type=Path,
        help="cached DynamicalMatrix.nc. If the file exists it is "
             "loaded; if not and a FC source is available, the built "
             "DMX is cached here.")
    p.add_argument(
        "-o", "--outfile", default=None, type=Path,
        help="output path. Default: sibling of FILE with "
             "suffix `.gk.nc` (plus `.interpolate` if --interpolate).")
    p.add_argument("--maxsteps", default=None, type=int,
                   help="truncate trajectory to the first N MD steps")
    p.add_argument("--offset", default=None, type=int,
                   help="skip the first N MD steps")
    p.add_argument("--spacing", default=None, type=int,
                   help="stride: use every Nth step")
    p.add_argument("--interpolate", action="store_true",
                   help="dense-q-grid interpolation + linear extrapolation")
    p.add_argument("--maxnq", default=20, type=int,
                   help="dense-grid sweep goes up to (maxnq, maxnq, maxnq)")
    p.add_argument("--backend", default="auto",
                   choices=["auto", "numpy", "jax"],
                   help="compute backend for the mode kernel. `auto` "
                        "(default) picks jax if importable, else numpy.")
    p.add_argument("--precision", default=None, choices=["fp32", "fp64"],
                   help="override the GKMX_PRECISION env default")
    p.add_argument(
        "--factorization", default="wick", choices=["wick", "vertex"],
        help="lifetime correlator: wick = SMA / dressed-bubble |g|² "
             "fit (default), vertex = un-factorized <nn>")
    p.add_argument("--freq", default=None, type=float,
                   help="filter-window frequency in THz; bypasses VDOS auto-detect")
    p.add_argument(
        "--heat-flux-time-unit", dest="heat_flux_time_unit",
        default=None, choices=["fs", "ps"],
        help="override heat_flux time base (default: auto-detect)")
    p.add_argument(
        "--analytical", action="store_true",
        help="emit analytical time-resolved BTE and QHGK HFACFs "
             "alongside the virial HFACF (adds 4 × (Nt, 3, 3) arrays)")
    p.add_argument(
        "--lifetime-fit-cutoff", dest="lifetime_fit_cutoff",
        default=0.5, type=float,
        help="ACF threshold for the per-mode exponential fit: the fit "
             "spans lags where the ACF stays above this value. Lower "
             "values (0.1) fit further into the tail — more robust for "
             "long-lifetime modes, noisier for short ones. Default 0.5.")
    p.add_argument(
        "--no-finite-time-correction", dest="correct_finite_time",
        action="store_false", default=True,
        help="Disable the 1/τ → 1/τ − 1/Tmax finite-time correction on "
             "mode lifetimes. Use for parity with pipelines that omit it "
             "(e.g. vibes_anisotropy). Lets τ exceed Tmax when the fit "
             "supports it — affects long-lifetime modes only.")
    p.set_defaults(_command=_cmd_gk)


def _cmd_gk(args):
    from . import open_dataset, get_kappa

    file = args.file
    outfile = args.outfile
    if outfile is None:
        suffix = ".gk.interpolate.nc" if args.interpolate else ".gk.nc"
        outfile = file.with_suffix(suffix)

    if outfile.is_file():
        _warn(f"{outfile} exists, skipping")
        return

    t_load = Timer(f"Loading trajectory {file}", prefix=_CLI_PREFIX)
    dataset = open_dataset(file, heat_flux_time_unit=args.heat_flux_time_unit)
    Nt = len(dataset.time)
    Na = int(dataset.attrs.get("natoms", dataset.positions.shape[1]))
    dt = float(dataset.time[1] - dataset.time[0]) if Nt > 1 else float("nan")
    t_load(f"Loaded {Nt} timesteps × {Na} atoms (dt = {dt:.2f} fs)")

    offset = args.offset
    if offset is not None and offset < 0:
        offset = len(dataset.time) + offset
    if offset is not None:
        _talk(f"starting from timestep {offset}")
        dataset = dataset.isel(time=slice(offset, len(dataset.time)))
    if args.maxsteps is not None:
        if len(dataset.time) < args.maxsteps:
            _warn(f"dataset too short ({len(dataset.time)} < "
                  f"{args.maxsteps}); aborting")
            return
        _talk(f"truncating to {args.maxsteps} timesteps")
        dataset = dataset.isel(time=slice(0, args.maxsteps))
    if args.spacing is not None:
        _talk(f"using spacing {args.spacing}")
        dataset = dataset.isel(time=slice(0, len(dataset.time), args.spacing))

    backend = _resolve_backend(args.backend)
    from ._backend import get_backend
    _talk(f"backend: {backend} [{get_backend(backend).device_description()}]")

    _talk(f"fc_file={args.fc_file}, dmx_file={args.dmx_file}, "
          f"interpolate={args.interpolate}, "
          f"precision={args.precision}, factorization={args.factorization}, "
          f"lifetime_fit_cutoff={args.lifetime_fit_cutoff}, "
          f"correct_finite_time={args.correct_finite_time}")

    ds_gk = get_kappa(
        dataset,
        fc_file=args.fc_file,
        dmx_file=args.dmx_file,
        interpolate=args.interpolate,
        nq_max=args.maxnq,
        backend=backend,
        precision=args.precision,
        factorization=args.factorization,
        freq=args.freq,
        analytical=args.analytical,
        lifetime_fit_cutoff=args.lifetime_fit_cutoff,
        correct_finite_time=args.correct_finite_time,
    )

    t_write = Timer(f"Writing {outfile}", prefix=_CLI_PREFIX)
    # h5netcdf + invalid_netcdf so complex v_qssa_cartesian round-trips
    # (netCDF4 engine rejects complex dtypes without auto_complex=True).
    ds_gk.to_netcdf(outfile, engine="h5netcdf", invalid_netcdf=True)
    t_write()


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="gkmx",
        description="gkmx: mode-decomposed Green-Kubo thermal conductivity.",
    )
    top = parser.add_subparsers(dest="_group", required=True)

    out = top.add_parser("out", help="Output-writing commands (gk, ...).")
    out_sub = out.add_subparsers(dest="_subcmd", required=True)
    _add_gk_parser(out_sub)

    return parser


def gkmx(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    args._command(args)


if __name__ == "__main__":
    gkmx()
