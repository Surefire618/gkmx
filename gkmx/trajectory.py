"""xarray-trajectory helpers (open_dataset, Trajectory, derived properties)."""

from pathlib import Path
import warnings

import numpy as np
import xarray as xr
from ase import units

from . import _constants as C
from . import keys
from .io import json2atoms


# gkmx internals expect heat_flux in eV/(Å²·fs). Vibes / legacy-gkx writers
# emit eV/(Å²·ps) — silent 1e6 error in κ if not converted. See
# memory/feedback_vibes_heat_flux_unit_convention.md.
_UNITS_FS = ("/fs", " fs", "·fs", "*fs", "per fs", "femtosecond")
_UNITS_PS = ("/ps", " ps", "·ps", "*ps", "per ps", "picosecond")

# Physical κ window (W/m/K) for any condensed material < ~1e4 (diamond ~ 2000).
# fs-base κ_rough lands in [1e-3, 1e4]; ps-base is 1e6× larger.
_KAPPA_PHYSICAL_LO = 1e-3
_KAPPA_PHYSICAL_HI = 1e4


def _parse_time_unit(units_attr):
    s = str(units_attr).lower()
    for tok in _UNITS_FS:
        if tok in s:
            return "fs"
    for tok in _UNITS_PS:
        if tok in s:
            return "ps"
    return None


def _heat_flux_units_attr(ds):
    hf = ds.get("heat_flux")
    if hf is not None:
        for k in ("units", "heat_flux_units"):
            v = hf.attrs.get(k)
            if v:
                u = _parse_time_unit(v)
                if u:
                    return u
    for k in ("heat_flux_units", "heat_flux_time_unit"):
        v = ds.attrs.get(k)
        if v:
            u = _parse_time_unit(v)
            if u:
                return u
    return None


def _guess_heat_flux_time_unit(ds):
    """Guess fs- vs ps-base by computing rough κ and comparing to physical window."""
    if "heat_flux" not in ds:
        return "fs", None
    j = ds["heat_flux"].dropna(keys.time) if keys.time in ds["heat_flux"].dims else ds["heat_flux"]
    j_arr = np.asarray(j, dtype=np.float64)
    j_std = float(np.std(j_arr))
    if not np.isfinite(j_std) or j_std == 0:
        return "fs", None

    V = float(np.asarray(ds.get("volume", ds.attrs.get("volume", 1.0))).mean())
    T = float(np.asarray(ds.get("temperature", ds.attrs.get("temperature", 300.0))).mean())
    if V <= 0 or T <= 0:
        return "fs", None

    tau_typ_fs = 1000.0
    pf = V / units.kB / T**2 * C.to_W_mK
    kappa_rough = pf * j_std**2 * tau_typ_fs

    if kappa_rough > 1e5:
        return "ps", kappa_rough
    if _KAPPA_PHYSICAL_LO <= kappa_rough <= _KAPPA_PHYSICAL_HI:
        return "fs", kappa_rough
    return "fs", kappa_rough


def _apply_heat_flux_time_unit(ds, time_unit, *, verbose=True):
    if "heat_flux" not in ds:
        return ds
    if time_unit == "ps":
        ds["heat_flux"] = ds["heat_flux"] / 1000.0
        for aux in ("heat_flux_aux", "heat_flux_potential",
                    "heat_flux_convective"):
            if aux in ds.data_vars:
                ds[aux] = ds[aux] / 1000.0
        if verbose:
            warnings.warn(
                "[gkmx.trajectory] heat_flux appears to be in "
                "eV/(Å²·ps); divided by 1000 to match gkmx's "
                "fs-base convention. Set `heat_flux_units = "
                "'eV/angstrom^2/fs'` in your trajectory writer to "
                "silence this.",
                stacklevel=3,
            )
    ds["heat_flux"].attrs["units"] = "eV / angstrom^2 / fs"
    ds.attrs["heat_flux_time_unit"] = "fs"
    return ds


def open_dataset(folder, chunks=None, heat_flux_time_unit=None):
    """Open a trajectory NetCDF file or directory of chunked NetCDFs.

    Args:
        folder: path to either a single ``.nc`` file or a directory
            containing ``*.nc`` chunks (loaded via ``xr.open_mfdataset``).
        chunks: xarray/dask chunk spec. Default ``{"time": 5000}``
            matches the gkmx kernel's I/O streaming. Pass ``-1`` to
            disable chunking and load each file eagerly.
        heat_flux_time_unit: ``"fs"``, ``"ps"``, or ``None``. ``None``
            auto-detects from the ``heat_flux`` attrs or, failing that,
            from a magnitude heuristic (computes a rough κ and compares
            to the physical window [1e-3, 1e4] W/m/K). On return,
            ``ds.heat_flux`` is always fs-base regardless of input
            convention; ``ds.heat_flux.attrs["units"]`` and
            ``ds.attrs["heat_flux_time_unit"]`` are annotated.

    Returns:
        An ``xr.Dataset`` with the trajectory's data variables and
        attrs. ``heat_flux`` is normalized to eV/(Å²·fs).

    Raises:
        ValueError: ``heat_flux_time_unit`` not one of ``"fs"`` /
            ``"ps"`` / ``None``.
    """
    folder = Path(folder)
    if chunks is None:
        chunks = {keys.time: 5000}
    if folder.is_file():
        ds = xr.open_dataset(folder, chunks=chunks)
    else:
        ds = xr.open_mfdataset(f"{folder}/*.nc", chunks=chunks)

    if heat_flux_time_unit is None:
        unit = _heat_flux_units_attr(ds)
        if unit is None:
            unit, kappa_rough = _guess_heat_flux_time_unit(ds)
            if kappa_rough is not None and not (
                    _KAPPA_PHYSICAL_LO <= kappa_rough <= _KAPPA_PHYSICAL_HI
                    or kappa_rough > 1e5):
                warnings.warn(
                    f"[gkmx.trajectory] heat_flux magnitude yields "
                    f"κ_rough ≈ {kappa_rough:.2e} W/m/K (τ=1ps guess), "
                    f"which is outside both the expected fs-base window "
                    f"[{_KAPPA_PHYSICAL_LO}, {_KAPPA_PHYSICAL_HI}] and "
                    f"the clean ps-base band (>1e5). Defaulting to "
                    f"'fs' — pass `heat_flux_time_unit=` explicitly if "
                    f"this trajectory uses a different convention.",
                    stacklevel=2,
                )
    else:
        unit = heat_flux_time_unit
        if unit not in ("fs", "ps"):
            raise ValueError(
                f"heat_flux_time_unit must be 'fs', 'ps', or None; "
                f"got {unit!r}"
            )

    return _apply_heat_flux_time_unit(ds, unit)


# Derived-property registry ported from stepson (avoids vibes runtime dep).
getters = {}


def getter(name, requires=()):
    """Register a getter that derives ``name`` from a dataset."""

    def decorator(function):
        def wrapper(dataset, ensure_required=True):
            if ensure_required:
                for required in requires:
                    add_property(dataset, required)
            dataarray = function(dataset)
            return dataarray.rename(name)

        getters[name] = wrapper
        return wrapper

    return decorator


def add_property(dataset, key):
    """Ensure ``key`` is in ``dataset``, deriving it (and any ``requires=``) if needed."""
    if key not in dataset:
        get = getters[key]
        dataarray = get(dataset)
        dataset.update({key: dataarray})


def _reference_atoms(dataset):
    return json2atoms(dataset.attrs[keys.reference_atoms])


@getter(keys.cell)
def get_cell(dataset):
    """Lattice vectors ``(time, a, b)`` broadcast from the reference Atoms."""
    return xr.DataArray(
        _reference_atoms(dataset).get_cell().array, dims=keys.tensor,
    ).expand_dims(dim={keys.time: dataset.time})


@getter(keys.volume, requires=[keys.cell])
def get_volume(dataset):
    """Cell volume per time step via ``a · (b × c)``."""
    cell = dataset[keys.cell]
    volume = np.sum(
        cell[:, 0, :] * np.cross(cell[:, 1, :], cell[:, 2, :]),
        axis=1,
    )
    return volume


@getter(keys.momenta)
def get_momenta(dataset):
    """``p = m * v``."""
    masses = xr.DataArray(np.asarray(dataset.attrs["masses"]), dims=keys.I)
    return dataset[keys.velocities] * masses  # order sensitive — keeps dim order


@getter(keys.energy_kinetic, requires=[keys.momenta])
def get_energy_kinetic(dataset):
    """``E_k = 0.5 * sum(p · v)`` over atoms and Cartesian components."""
    momenta = dataset[keys.momenta]
    velocities = dataset[keys.velocities]
    return (0.5 * momenta * velocities).sum(dim=(keys.I, keys.a))


@getter(
    keys.stress_kinetic,
    requires=[keys.volume, keys.momenta, keys.stress_potential],
)
def get_stress_kinetic(dataset):
    """``sigma_kin = -(1/V) sum_i p_i p_i^T / m_i``; NaN-aligned with stress_potential."""
    invmasses = 1.0 / xr.DataArray(
        np.asarray(dataset.attrs["masses"]), dims=keys.I,
    )
    invvolume = 1.0 / dataset[keys.volume]
    momenta = dataset[keys.momenta]

    stress_kinetic = (
        -1.0 * (momenta * momenta.rename(a="b") * invmasses).sum(dim=keys.I)
        * invvolume
    )

    stress_potential = dataset[keys.stress_potential]
    return stress_kinetic.where(~np.isnan(stress_potential))


@getter(keys.stress, requires=[keys.stress_kinetic, keys.stress_potential])
def get_stress(dataset):
    """Total stress = kinetic + potential."""
    return dataset[keys.stress_kinetic] + dataset[keys.stress_potential]


def _xtrace(arr, axis1=-2, axis2=-1):
    if axis1 < 0:
        axis1 = len(arr.shape) + axis1
    if axis2 < 0:
        axis2 = len(arr.shape) + axis2
    data = np.trace(arr, axis1=axis1, axis2=axis2)
    new_dims = list(arr.dims)
    new_dims.pop(max(axis1, axis2))
    new_dims.pop(min(axis1, axis2))
    return xr.DataArray(data, dims=new_dims, coords=arr.coords)


@getter(keys.pressure, requires=[keys.stress])
def get_pressure(dataset):
    return _xtrace(dataset[keys.stress]).rename(keys.pressure) * (-1.0 / 3.0)


@getter(keys.pressure_kinetic, requires=[keys.stress_kinetic])
def get_pressure_kinetic(dataset):
    return _xtrace(dataset[keys.stress_kinetic]).rename(
        keys.pressure_kinetic,
    ) * (-1.0 / 3.0)


@getter(keys.pressure_potential, requires=[keys.stress_potential])
def get_pressure_potential(dataset):
    return _xtrace(dataset[keys.stress_potential]).rename(
        keys.pressure_potential,
    ) * (-1.0 / 3.0)


@getter(keys.temperature, requires=[keys.energy_kinetic, keys.forces])
def get_temperature(dataset):
    """``T = 2 E_k / (3N k_B)``."""
    ekin = dataset[keys.energy_kinetic]
    dof = len(dataset[keys.forces][0]) * 3
    return 2 * ekin / (dof * units.kB)


def gk_prefactor(volume, temperature):
    """Compute the Green-Kubo prefactor ``V / (k_B T²)`` in W/mK per (eV/Å²/fs).

    Multiplies the time-integrated heat-flux autocorrelation to give κ.

    Args:
        volume: cell volume in Å³.
        temperature: temperature in K.

    Returns:
        Prefactor (float), in W/m/K per (eV/Å²/fs)² · fs.
    """
    from ._constants import to_W_mK
    V = float(volume)
    T = float(temperature)
    return V / (units.kB * T * T) * to_W_mK


class Trajectory:
    """Minimal xarray-backed trajectory wrapper.

    Two operations: ``len(traj)`` to count MD steps and
    ``traj.get_atoms(index)`` to reconstruct an ASE ``Atoms`` (positions
    + velocities + optionally cell) for a given step. Intended for MD
    restart workflows where gkmx ships the trajectory as ``.nc``.
    """

    def __init__(self, folder):
        """Open a trajectory from a ``.nc`` file or directory of chunks.

        Args:
            folder: path forwarded to ``open_dataset``.
        """
        self.data = open_dataset(Path(folder))
        self._template = json2atoms(self.data.attrs[keys.reference_atoms])

    def __len__(self):
        """Return the number of MD timesteps in the trajectory."""
        return int(self.data[keys.time].size)

    def get_atoms(self, index):
        """Rebuild an ASE ``Atoms`` for the ``index``-th MD step.

        Args:
            index: integer time index (0-based; negative not supported).

        Returns:
            An ASE ``Atoms`` with positions set, plus velocities and
            cell if present in the dataset.
        """
        atoms = self._template.copy()
        step = self.data.isel({keys.time: index})
        atoms.set_positions(np.asarray(step[keys.positions].compute().data))
        if keys.velocities in step:
            atoms.set_velocities(np.asarray(step[keys.velocities].compute().data))
        if keys.cell in step:
            atoms.set_cell(np.asarray(step[keys.cell].compute().data))
        return atoms
