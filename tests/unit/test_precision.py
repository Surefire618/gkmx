"""Tests for ``gkmx.precision``."""

from __future__ import annotations

import numpy as np
import pytest

from gkmx.precision import Precision, get_default, set_default


@pytest.fixture(autouse=True)
def _restore_default():
    original = get_default()
    yield
    set_default(original)


@pytest.mark.parametrize("name,real,complex_", [
    ("fp32", np.float32, np.complex64),
    ("fp64", np.float64, np.complex128),
])
def test_from_str_dtype_mapping_and_case_insensitive(name, real, complex_):
    """``from_str`` maps fp32/fp64 to the documented numpy dtype pair,
    case-insensitively."""
    p = Precision.from_str(name)
    assert (p.name, p.real, p.complex) == (name, real, complex_)
    assert Precision.from_str(name.upper()) == p


@pytest.mark.parametrize("bad", ["fp16", "double", "", "fp32 "])
def test_from_str_rejects_unknown(bad):
    with pytest.raises(ValueError, match="Unknown precision"):
        Precision.from_str(bad)


def test_default_and_set_default_round_trip():
    """``set_default`` updates the process-wide default; ``default()``
    and ``get_default()`` reflect it. Invalid input raises before any
    state change."""
    for name in ("fp32", "fp64"):
        set_default(name)
        assert get_default() == name
        assert Precision.default().name == name
    with pytest.raises(ValueError, match="precision must be one of"):
        set_default("fp16")


def test_resolve_obeys_default_and_explicit_override():
    """``resolve(None)`` is what every ``get_kappa``-class entry point
    falls back to — must equal ``default()``. An explicit string wins
    over the global default regardless of which side was set last."""
    set_default("fp64")
    assert Precision.resolve(None) == Precision.from_str("fp64")
    assert Precision.resolve("fp32").name == "fp32"
    set_default("fp32")
    assert Precision.resolve(None) == Precision.from_str("fp32")
    assert Precision.resolve("fp64").name == "fp64"


def test_dataclass_is_frozen_and_equal_by_value():
    """``@dataclass(frozen=True, slots=True)`` — assignment raises;
    equality is by field value, so two ``from_str("fp32")`` calls compare
    equal."""
    p = Precision.from_str("fp32")
    with pytest.raises((AttributeError, TypeError)):
        p.real = np.float64
    assert Precision.from_str("fp64") == Precision.from_str("fp64")
    assert Precision.from_str("fp32") != Precision.from_str("fp64")
