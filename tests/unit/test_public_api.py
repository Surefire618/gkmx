"""Pin the public surface of `gkmx`."""

import gkmx


EXPECTED_ALL = {
    "DynamicalMatrix",
    "Phonon",
    "Precision",
    "Trajectory",
    "get_kappa",
    "gk_prefactor",
    "keys",
    "open_dataset",
}


def test_all_matches_expected():
    assert set(gkmx.__all__) == EXPECTED_ALL


def test_each_exported_symbol_resolves():
    for name in gkmx.__all__:
        assert hasattr(gkmx, name), f"gkmx.__all__ lists {name!r} but gkmx.{name} is missing"


def test_submodules_resolve():
    assert gkmx.keys is not None
    assert gkmx.mic is not None
    assert gkmx.mic.fold is not None
    assert gkmx.mic.is_orthogonal is not None
