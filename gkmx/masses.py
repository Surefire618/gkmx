"""Which atomic-mass table the whole pipeline runs on.

gkmx runs on ASE masses, because that is what the MD was run with -- the
trajectory, the force constants fitted from it, and the mode projection all have
to agree, and ASE's table is what produced them. Nothing in normal use should
change that.

The exception is developing or testing against TDEP. TDEP computes its masses as
the natural-abundance-weighted mean over stable isotopes, from IUPAC/CIAAW
abundances and AME isotope masses (`naturaldistribution`,
tdep/src/libolle/type_crystalstructure.f90:1522); ASE quotes the published,
*rounded* standard atomic weight. Same quantity, different precision -- but
``omega ~ 1/sqrt(m)`` turns ``dm/m`` into ``dw/w = dm/2m``, which floors any TDEP
reproduction at ~1e-05 for O, Cl, Se, Te, N. Reproducing TDEP therefore needs
TDEP's table, and needs it *everywhere*: using it for the force constants but not
for the mode projection gives eigenvectors and amplitudes on different mass
tables, silently.

So it is one process-wide switch, off by default, set through the environment
exactly like ``GKMX_PRECISION``::

    GKMX_MASSES=TDEP  python -m pytest tests/integration/test_tdep_reference.py

Do not add a CLI flag or a per-call argument: a half-applied mass table is worse
than either table applied consistently.
"""
from __future__ import annotations

import os

import numpy as np

from ._constants import TDEP_ATOMIC_MASSES

_VALID = ("ASE", "TDEP")

_DEFAULT: str = os.environ.get("GKMX_MASSES", "ASE").upper()
if _DEFAULT not in _VALID:
    raise ValueError(f"GKMX_MASSES={_DEFAULT!r} must be one of {_VALID}")


def set_default(table: str) -> None:
    """Set the process-wide mass table (``"ASE"`` or ``"TDEP"``)."""
    global _DEFAULT
    t = table.upper()
    if t not in _VALID:
        raise ValueError(f"mass table must be one of {_VALID}, got {table!r}")
    _DEFAULT = t


def get_default() -> str:
    """Return the current process-wide mass table name."""
    return _DEFAULT


def for_numbers(numbers, table: str | None = None) -> np.ndarray:
    """Masses [amu] for a sequence of atomic numbers, under the active table."""
    t = (table or _DEFAULT).upper()
    if t not in _VALID:
        raise ValueError(f"mass table must be one of {_VALID}, got {table!r}")
    Z = np.asarray(numbers, dtype=int)
    if t == "TDEP":
        return np.asarray([TDEP_ATOMIC_MASSES[z] for z in Z], dtype=float)
    from ase.data import atomic_masses
    return np.asarray(atomic_masses, dtype=float)[Z]


def of(atoms, table: str | None = None) -> np.ndarray:
    """Masses [amu] of an ASE ``Atoms`` under the active table.

    Under ``"ASE"`` this returns ``atoms.get_masses()`` unchanged, so any masses
    a caller set by hand -- isotope substitution, an alloy average -- survive.
    Under ``"TDEP"`` they are replaced from the element, which is the point.
    """
    if (table or _DEFAULT).upper() == "ASE":
        return np.asarray(atoms.get_masses(), dtype=float)
    return for_numbers(atoms.get_atomic_numbers(), table=table)


def of_dataset(dataset) -> np.ndarray:
    """Masses [amu] for a trajectory's supercell, under the active table.

    Under ``"ASE"`` this is the trajectory's own ``attrs["masses"]``, which is
    what the MD actually ran with. Under ``"TDEP"`` it is resolved from the
    elements of the stored supercell, so the derived momenta and kinetic stress
    move with the rest of the pipeline instead of being left behind.
    """
    if _DEFAULT == "ASE":
        return np.asarray(dataset.attrs["masses"], dtype=float)
    import json

    from ase import Atoms

    from . import keys
    sc = Atoms(**json.loads(dataset.attrs[keys.reference_supercell]))
    return for_numbers(sc.get_atomic_numbers())
