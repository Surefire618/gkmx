"""Shared fixtures for the gkmx integration tests.

Backed exclusively by the in-repo ``tests/datasets/KI_B2_MLIP/``
fixture — a 200-step slice of the KI_B2_n128 NVE trajectory shipped
in git (~3 MB). Same primitive, supercell, FC, and DMX as the full
n128 dataset; only the trajectory length is trimmed.

Every test in `gkmx/tests/` runs on this in-repo fixture; **no test
in the released gkmx code depends on out-of-tree datasets**. The
out-of-tree `gkmx_project/datasets/` (KI_B2_n128, n2000, CsCl_fcc_n4096,
…) is reserved for benchmarks (`gkmx/benchmarks/`) and the
diagnostic scripts under `gkmx_project/dev_scripts/`.

Every test in this directory is auto-tagged with the ``integration``
marker, so ``pytest tests/unit/`` never picks them up. Run explicitly
with::

    pytest tests/integration/
    pytest -m integration

All heavy setup happens once per session in the fixtures below; each
test gets a fresh ``trajectory.copy(deep=True)`` because
``get_kappa`` mutates its input dataset in place.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import xarray as xr


# ---------------------------------------------------------------------------
# Mark the whole package as integration.
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(config, items):
    """Auto-tag every test in tests/integration/ with the 'integration' mark."""
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


# ---------------------------------------------------------------------------
# In-repo KI_B2_MLIP fixture
# ---------------------------------------------------------------------------

# tests/integration/conftest.py → parents[1] == tests/, parents[1]/datasets
_TINY_DIR = Path(__file__).resolve().parents[1] / "datasets" / "KI_B2_MLIP"


@pytest.fixture(scope="session")
def tiny_dir() -> Path:
    """Directory containing the in-repo KI_B2_MLIP fixture."""
    if not _TINY_DIR.is_dir():
        pytest.skip(f"in-repo fixture KI_B2_MLIP not found at {_TINY_DIR}")
    for required in ("FORCE_CONSTANTS_tdep", "geometry.in.primitive",
                     "geometry.in.supercell", "nve/000000.nc",
                     "DynamicalMatrix.nc"):
        if not (_TINY_DIR / required).exists():
            pytest.skip(f"KI_B2_MLIP missing {required}")
    return _TINY_DIR


@pytest.fixture(scope="session")
def tiny_fc_file(tiny_dir) -> str:
    return str(tiny_dir / "FORCE_CONSTANTS_tdep")


@pytest.fixture(scope="session")
def tiny_dmx_file(tiny_dir) -> str:
    """Pre-built `DynamicalMatrix.nc` shipped with the fixture."""
    return str(tiny_dir / "DynamicalMatrix.nc")


@pytest.fixture(scope="session")
def tiny_trajectory(tiny_dir) -> xr.Dataset:
    """200-step NVE trajectory, loaded once per session.

    Callers should take a ``trajectory.copy(deep=True)`` before passing
    to ``get_kappa`` — the pipeline mutates its input.
    """
    return xr.open_dataset(tiny_dir / "nve" / "000000.nc").load()


# ---------------------------------------------------------------------------
# In-repo CuI_aiGK fixture (ab-initio, SMA-breakdown regime)
# ---------------------------------------------------------------------------

_CUI_DIR = Path(__file__).resolve().parents[1] / "datasets" / "CuI_aiGK"


@pytest.fixture(scope="session")
def cui_dir() -> Path:
    """Directory containing the in-repo CuI_aiGK fixture (200-step NVE
    slice starting at 35 ps — the metastable/second-minimum regime
    where wick and vertex factorization paths diverge)."""
    if not _CUI_DIR.is_dir():
        pytest.skip(f"in-repo fixture CuI_aiGK not found at {_CUI_DIR}")
    for required in ("FORCE_CONSTANTS_tdep", "geometry.in.primitive",
                     "geometry.in.supercell", "nve/000000.nc",
                     "DynamicalMatrix.nc"):
        if not (_CUI_DIR / required).exists():
            pytest.skip(f"CuI_aiGK missing {required}")
    return _CUI_DIR


@pytest.fixture(scope="session")
def cui_fc_file(cui_dir) -> str:
    return str(cui_dir / "FORCE_CONSTANTS_tdep")


@pytest.fixture(scope="session")
def cui_dmx_file(cui_dir) -> str:
    return str(cui_dir / "DynamicalMatrix.nc")


@pytest.fixture(scope="session")
def cui_trajectory(cui_dir) -> xr.Dataset:
    """200-step NVE trajectory from the CuI rare-event regime."""
    return xr.open_dataset(cui_dir / "nve" / "000000.nc").load()
