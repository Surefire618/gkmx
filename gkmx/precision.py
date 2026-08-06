"""Bundled real/complex dtype pair for the gkmx pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

_VALID = ("fp32", "fp64")
_DTYPES = {
    "fp32": (np.float32, np.complex64),
    "fp64": (np.float64, np.complex128),
}

_DEFAULT: str = os.environ.get("GKMX_PRECISION", "fp32").lower()
if _DEFAULT not in _VALID:
    raise ValueError(f"GKMX_PRECISION={_DEFAULT!r} must be one of {_VALID}")


@dataclass(frozen=True, slots=True)
class Precision:
    """A real / complex numpy dtype pair, identified by name.

    Attributes:
        real: numpy real dtype (``np.float32`` for fp32, ``np.float64`` for fp64).
        complex: numpy complex dtype (``np.complex64`` or ``np.complex128``).
        name: ``"fp32"`` or ``"fp64"``.
    """

    real: type
    complex: type
    name: str

    @classmethod
    def from_str(cls, name: str) -> Precision:
        """Build a Precision from a string.

        Args:
            name: ``"fp32"`` or ``"fp64"`` (case-insensitive).

        Returns:
            The matching Precision instance.

        Raises:
            ValueError: if ``name`` is not one of ``"fp32"`` / ``"fp64"``.
        """
        name = name.lower()
        if name not in _DTYPES:
            raise ValueError(f"Unknown precision: {name!r}. Valid: {_VALID}.")
        real, complex_ = _DTYPES[name]
        return cls(real=real, complex=complex_, name=name)

    @classmethod
    def default(cls) -> Precision:
        """Return the process-wide default Precision.

        Resolution order: ``set_default(...)`` last call wins; otherwise
        the ``GKMX_PRECISION`` env var (read once at import); otherwise
        ``"fp32"``.
        """
        return cls.from_str(_DEFAULT)

    @classmethod
    def resolve(cls, name: str | None = None) -> Precision:
        """Return ``from_str(name)`` if ``name`` is given, else ``default()``."""
        return cls.from_str(name) if name is not None else cls.default()


def set_default(precision: str) -> None:
    """Set the process-wide default precision.

    Args:
        precision: ``"fp32"`` or ``"fp64"`` (case-insensitive).

    Raises:
        ValueError: if ``precision`` is not one of ``"fp32"`` / ``"fp64"``.
    """
    global _DEFAULT
    p = precision.lower()
    if p not in _VALID:
        raise ValueError(f"precision must be one of {_VALID}, got {precision!r}")
    _DEFAULT = p


def get_default() -> str:
    """Return the current process-wide default precision string (``"fp32"`` or ``"fp64"``)."""
    return _DEFAULT
