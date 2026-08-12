"""``gkmx.kappa.symmetrize_kappa`` against the three Ga2O3 polymorphs.

A perturbed tensor through the projector, checked against the kappa tensor form
each crystal class allows:

    alpha  R-3c (167)   trigonal      kxx = kyy, no off-diagonal
    beta   C2/m (12)    monoclinic    kxz allowed, kxy = kyz = 0
    kappa  Pna2_1 (33)  orthorhombic  diagonal
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from gkmx.kappa import symmetrize_kappa

from .._tolerances import TOL_FP64

DATA_DIR = Path(__file__).parent.parent / "datasets" / "Ga2O3"

# phase -> (space group, off-diagonals the group forbids)
PHASES = {
    "alpha": ("R-3c (167)", ("xy", "xz", "yz")),
    "beta": ("C2/m (12)", ("xy", "yz")),
    "kappa": ("Pna2_1 (33)", ("xy", "xz", "yz")),
}

OFF = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}


@pytest.mark.parametrize("name", sorted(PHASES), ids=sorted(PHASES))
def test_kappa_tensor_takes_the_form_the_crystal_class_allows(name):
    spacegroup, forbidden = PHASES[name]
    prim = read(str(DATA_DIR / f"geometry.in.primitive.{name}"), format="aims")

    rng = np.random.default_rng(0)
    K = rng.normal(size=(3, 3))
    K = K + K.T                       # symmetric, but with no symmetry of its own
    S = np.asarray(symmetrize_kappa(K, prim))
    scale = np.abs(S).max()

    for label in forbidden:
        i, j = OFF[label]
        assert abs(S[i, j]) / scale < TOL_FP64, (
            f"{name}: k{label} is forbidden by {spacegroup} but is {S[i, j]:.3e}")

    if name == "alpha":
        kxx, kyy, _ = np.diagonal(S)
        assert abs(kxx - kyy) / scale < TOL_FP64, (
            f"{name}: {spacegroup} requires kxx = kyy, got {kxx:.6f}, {kyy:.6f}")
