"""Tests for ``gkmx.io`` — FC parsers and Atoms <-> JSON round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from ase import Atoms

from gkmx.io import (
    _expand_compressed,
    _NumpyEncoder,
    atoms2json,
    json2atoms,
    parse_force_constants,
)

DATA_DIR = Path(__file__).parent.parent / "datasets" / "KI_B2_MLIP"


def _write_hdf5_fc(path, fc):
    with h5py.File(path, "w") as f:
        f.create_dataset("force_constants", data=fc)


def test_parse_force_constants_three_formats_and_two_dim(tmp_path):
    """All three parser branches (text / fc2.hdf5 / flat .dat) must:
        - return float64
        - produce the same shape & values for the same logical FC
        - respect ``two_dim`` flatten ``(i, a) → 3i+a``"""
    fc4 = np.arange(2 * 4 * 9, dtype=np.float64).reshape(2, 4, 3, 3)
    expected_2d = fc4.transpose(0, 2, 1, 3).reshape(6, 12)

    hdf5 = tmp_path / "fc2.hdf5"
    _write_hdf5_fc(hdf5, fc4)
    np.testing.assert_array_equal(parse_force_constants(hdf5), fc4)
    np.testing.assert_array_equal(
        parse_force_constants(hdf5, two_dim=True), expected_2d)

    flat_path = tmp_path / "fc.dat"
    np.savetxt(flat_path, expected_2d)
    np.testing.assert_array_equal(parse_force_constants(flat_path), expected_2d)

    remapped_path = tmp_path / "fc_remapped.txt"
    np.savetxt(remapped_path, expected_2d)
    np.testing.assert_array_equal(parse_force_constants(remapped_path),
                                  expected_2d)

    text_path = DATA_DIR / "FORCE_CONSTANTS_tdep"
    if text_path.exists():
        fc_text = parse_force_constants(text_path)
        assert fc_text.ndim == 4 and fc_text.shape[2:] == (3, 3)
        assert fc_text.dtype == np.float64


def test_parse_force_constants_unknown_format_raises(tmp_path):
    path = tmp_path / "mystery.xyz"
    path.write_text("not actually FC")
    with pytest.raises(ValueError, match="Unknown force constants file format"):
        parse_force_constants(path)


def test_atoms_json_round_trip_with_masses():
    """``atoms2json`` + ``json2atoms`` preserves numbers, positions,
    cell, pbc, AND custom masses (the only branch on the encoder side)."""
    atoms = Atoms("Fe2", positions=[[0, 0, 0], [1.4, 0, 0]],
                  cell=[3, 3, 3], pbc=True)
    custom = np.array([60.0, 100.0])
    atoms.set_masses(custom)
    decoded = json2atoms(atoms2json(atoms))
    np.testing.assert_array_equal(decoded.numbers, atoms.numbers)
    np.testing.assert_allclose(decoded.positions, atoms.positions)
    np.testing.assert_allclose(decoded.cell.array, atoms.cell.array)
    np.testing.assert_array_equal(decoded.pbc, atoms.pbc)
    np.testing.assert_allclose(decoded.get_masses(), custom)


def test_json2atoms_handles_vibes_extras():
    """vibes-flavored payloads carry ``velocities`` and ``info`` keys
    that ASE's ``Atoms(**d)`` would reject, plus run-length-compressed
    ``symbols`` and ``masses``. ``json2atoms`` must absorb both."""
    d = {
        "symbols": [[3, "C"], [2, "O"]],
        "positions": [[i, 0, 0] for i in range(5)],
        "cell": [5.0, 5.0, 5.0],
        "pbc": [True, True, True],
        "masses": [[3, 12.011], [2, 15.999]],
        "velocities": [[0, 0, 0]] * 5,   # ASE rejects this kwarg
        "info": {"meta": "drop me"},     # idem
    }
    atoms = json2atoms(json.dumps(d))
    assert atoms.get_chemical_symbols() == ["C", "C", "C", "O", "O"]
    np.testing.assert_allclose(
        atoms.get_masses(), [12.011, 12.011, 12.011, 15.999, 15.999])


@pytest.mark.parametrize("value,expected", [
    (np.array([1, 2, 3]), "[1, 2, 3]"),
    (np.int64(7),         "7"),
    (np.float64(2.5),     "2.5"),
    (np.bool_(True),      "true"),
])
def test_numpy_encoder_handles_numpy_types(value, expected):
    assert json.dumps(value, cls=_NumpyEncoder) == expected


def test_numpy_encoder_falls_through_for_unknown():
    with pytest.raises(TypeError):
        json.dumps(object(), cls=_NumpyEncoder)


def test_expand_compressed_mixed():
    """Bare items pass through; ``[count, value]`` pairs expand. Covers
    both branches in one input."""
    assert _expand_compressed(
        ["x", [2, "y"], "z"]
    ) == ["x", "y", "y", "z"]
