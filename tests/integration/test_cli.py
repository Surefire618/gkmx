"""Smoke tests for the ``gkmx`` argparse CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
import xarray as xr

from gkmx.cli import _resolve_backend, gkmx

DATA_DIR = Path(__file__).parent.parent / "datasets" / "KI_B2_MLIP"


def test_help_lists_every_documented_flag(capsys):
    """``gkmx out gk --help`` must mention every flag the docs advertise;
    a quietly-dropped flag would otherwise pass through the rest of the
    suite undetected."""
    with pytest.raises(SystemExit):
        gkmx(["out", "gk", "--help"])
    out = capsys.readouterr().out
    for flag in ("--fc_file", "--dmx_file", "--interpolate", "--maxnq",
                 "--backend", "--precision", "--factorization",
                 "--analytical", "--lifetime-fit-cutoff",
                 "--no-finite-time-correction"):
        assert flag in out, f"{flag} missing from gk --help"


def test_unknown_subcommand_and_missing_arg_exit():
    """Top-level dispatch must reject unknown subcommands and missing
    required positionals — both via argparse's SystemExit."""
    with pytest.raises(SystemExit):
        gkmx(["nope"])
    with pytest.raises(SystemExit):
        gkmx(["out", "gk"])


def test_resolve_backend_explicit_and_auto(monkeypatch):
    """Explicit ``"numpy"`` / ``"jax"`` pass through; ``"auto"`` falls
    back to ``"numpy"`` when no GPU is visible."""
    import gkmx.cli as cli
    assert _resolve_backend("numpy") == "numpy"
    assert _resolve_backend("jax") == "jax"
    monkeypatch.setattr(cli, "_gpu_visible", lambda: False)
    assert _resolve_backend("auto") == "numpy"


def test_gk_end_to_end_writes_and_skips(tmp_path):
    """End-to-end: ``gkmx out gk`` on the KI_B2_MLIP fixture writes a
    netCDF with ``thermal_conductivity`` of shape (3,3); a second run
    against an existing outfile is a no-op (idempotency for batch sweeps)."""
    nve, fc = DATA_DIR / "nve", DATA_DIR / "FORCE_CONSTANTS_tdep"
    if not nve.exists() or not fc.exists():
        pytest.skip(f"KI_B2_MLIP fixture missing under {DATA_DIR}")

    out = tmp_path / "out.nc"
    gkmx(["out", "gk", str(nve), "--fc_file", str(fc),
          "--outfile", str(out), "--backend", "numpy"])
    assert out.is_file()
    with xr.open_dataset(out, engine="h5netcdf") as ds:
        assert ds["thermal_conductivity"].shape == (3, 3)

    placeholder = tmp_path / "skip.nc"
    placeholder.write_text("placeholder")
    gkmx(["out", "gk", str(nve), "--fc_file", str(fc),
          "--outfile", str(placeholder), "--backend", "numpy"])
    assert placeholder.read_text() == "placeholder"
