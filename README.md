# gkmx

**Mode-decomposed Green-Kubo thermal conductivity from MD trajectories.**

## Overview

`gkmx` postprocesses an equilibrium molecular-dynamics trajectory together
with harmonic force constants and returns the thermal-conductivity tensor
`κ` in W/(m·K). The pipeline projects the heat flux onto phonon modes,
fits per-mode lifetimes and heat capacities from the autocorrelation, and
recombines them into both the Boltzmann (BTE) and quasi-harmonic Green-Kubo
(QHGK) tensors. Finite-supercell effects are removed by a linear
extrapolation in `1/N`.

## Installation

```bash
conda create -n gkmx python=3.11 -c conda-forge
conda activate gkmx
pip install -e .                # core
```

The JAX backend is recommended — `[jax-cuda]` gives a large speed-up on
NVIDIA GPUs. Pick one of:

```bash
pip install -e '.[jax-cuda]'    # JAX with CUDA 12 (NVIDIA GPU, recommended)
pip install -e '.[jax]'         # JAX CPU-only (no GPU available)
```

For a development install, add the `dev` extra: `pip install -e '.[dev]'`.

## Quickstart

```python
from gkmx import open_dataset, get_kappa

ds = open_dataset("trajectory.nc")
result = get_kappa(
    ds,
    fc_file="FORCE_CONSTANTS",   # harmonic FCs (phonopy / TDEP / flat .dat)
    dmx_file="dmx.nc",           # dynamical-matrix cache; built once, reused
    interpolate=True,
    nq_max=8,
)

# All κ tensors are (3, 3), W/(m·K).
print(result.thermal_conductivity_harmonic)       # BTE — particle channel
print(result.thermal_conductivity_harmonic_QHGK)  # QHGK — particle + wave channels
print(result.thermal_conductivity_corrected)      # size-extrapolated to N → ∞
```

The returned `xarray.Dataset` also carries the direct Green-Kubo κ
(`thermal_conductivity`), the heat-flux autocorrelation, the finite-time-corrected
κ integral, and per-mode frequencies, group velocities, lifetimes, and heat
capacities. Save it with `result.to_netcdf(...)`.

## Command-line interface

```bash
gkmx out gk trajectory.nc \
    --fc_file FORCE_CONSTANTS \
    --dmx_file dmx.nc \
    --interpolate --maxnq 8
```

Output is written next to the trajectory as `trajectory.gk.nc`. The backend
defaults to JAX when a CUDA GPU is visible, numpy otherwise; pin with
`--backend numpy|jax`. See `gkmx out gk --help` for all flags.

## Trajectory input

`gkmx` does not run MD — bring an xarray netCDF with `positions`, `velocities`,
and `heat_flux`. Built-in loaders today:

- [`gkx`](https://github.com/Surefire618/gkx) — JAX-based MD driver.
- [FHI-vibes](https://vibes-developers.gitlab.io/vibes/).

Adapters for LAMMPS, i-PI, and ASE are planned.

## Supported force-constant formats

Pass any of the following to `fc_file`; the layout is auto-detected:

- phonopy: `FORCE_CONSTANTS`, `fc2.hdf5`
- TDEP: `infile.forceconstant`
- Flat `.dat` (raw `3·N × 3·N` matrix)

## Development

```bash
pytest tests/unit/        -q
pytest tests/integration/ -q
```

## License

MIT — see `LICENSE`.
