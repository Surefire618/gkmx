"""Smoke + closed-form coverage of ``gkmx.trajectory``'s derived-property registry.

Builds a synthetic dataset and verifies that every registered getter
resolves end-to-end; pins the deepest-chain quantity (``temperature``,
which depends on ``energy_kinetic`` → ``momenta``) against a closed
form so a regression in any step of the chain surfaces here.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from ase import Atoms, units

from gkmx import keys
from gkmx.io import atoms2json
from gkmx.trajectory import add_property, getters

_REGISTERED = (
    keys.cell, keys.volume, keys.momenta, keys.energy_kinetic,
    keys.stress_kinetic, keys.stress, keys.pressure,
    keys.pressure_kinetic, keys.pressure_potential, keys.temperature,
)


@pytest.fixture
def synthetic_dataset():
    """3-step trajectory of a 2-atom He box at v=1 (so kinetic
    quantities are closed-form), zero potential stress, zero forces."""
    Nt, Na = 3, 2
    atoms = Atoms(
        "He2",
        positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        cell=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        pbc=True,
    )
    return xr.Dataset(
        data_vars={
            keys.velocities: ((keys.time, keys.I, keys.a),
                              np.ones((Nt, Na, 3))),
            keys.stress_potential: ((keys.time,) + keys.tensor,
                                    np.zeros((Nt, 3, 3))),
            keys.forces: ((keys.time, keys.I, keys.a),
                          np.zeros((Nt, Na, 3))),
        },
        coords={keys.time: np.arange(Nt, dtype=np.float64)},
        attrs={
            keys.reference_atoms: atoms2json(atoms),
            "masses": atoms.get_masses().tolist(),
        },
    ), atoms


def test_every_registered_getter_resolves(synthetic_dataset):
    """``add_property`` must produce every documented derived field
    without error; calling it twice is idempotent."""
    ds, _ = synthetic_dataset
    for key in _REGISTERED:
        assert key in getters, f"{key!r} not in registry"
        add_property(ds, key)
        before = np.asarray(ds[key].data).copy()
        add_property(ds, key)
        np.testing.assert_array_equal(ds[key].data, before)


def test_temperature_closed_form(synthetic_dataset):
    """``T = 2 E_k / (3 N k_B)`` with ``E_k = 0.5 Σ p·v = 0.5 · 3 · Σm``
    (since v ≡ 1). Pins the deepest chain (temperature ← E_k ← momenta),
    so a regression in any of those three steps fails this test."""
    ds, atoms = synthetic_dataset
    add_property(ds, keys.temperature)
    ekin = 0.5 * 3 * atoms.get_masses().sum()
    dof = 3 * len(atoms)
    expected = 2 * ekin / (dof * units.kB)
    np.testing.assert_allclose(ds[keys.temperature].data,
                               np.full(3, expected))
