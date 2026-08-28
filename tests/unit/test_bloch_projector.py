"""``e_qsI`` projects onto the same physical branch in every convention.

`e_qsI` contracts a supercell displacement field into a mode amplitude, so its
eigenvector gauge and its phase origin are one pair, carrying the intra-cell
offset exactly once: the native conventions hold it in the phase (the full atom
position, matching `D(q)`'s inter-atomic phases), TDEP in the eigenvector
(`e_T = P conj(e)`, `P = diag(exp(-2 pi i q . r_a))`).

Two conventions' projectors may then differ only by a phase per mode, which the
ACF divides out, and by a rotation inside an exactly degenerate multiplet, which
is a free basis choice (`test_degenerate_basis.py` bounds what it does to the
estimator).

Geometry-only, so it runs on every shipped fixture rather than the two with
trajectories -- including the low-symmetry and non-symmorphic ones where the
per-atom phases weigh most.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import io as ase_io

from gkmx.dynamical_matrix import DynamicalMatrix
from gkmx.io import parse_force_constants
from gkmx.phonon import CONVENTIONS, degenerate_sets

from .._tolerances import TOL_FP64_DERIVED

DATA_ROOT = Path(__file__).parent.parent / "datasets"

FIXTURES = [
    "KI_B2_MLIP", "CuI_aiGK", "tdep_KI_bcc", "tdep_KPTe2", "tdep_Rb2O",
    "phonopy_Si", "phonopy_NaCl", "phonopy_SrTiO3", "phonopy_AgErTe2",
    "phonopy_Li2ZnGeO4", "phonopy_LiCdBO3", "phonopy_Mg2YbSb2",
    "phonopy_RbIn3F10",
]


def _load(name):
    d = DATA_ROOT / name
    prim = ase_io.read(str(d / "geometry.in.primitive"), format="aims")
    sc = ase_io.read(str(d / "geometry.in.supercell"), format="aims")
    if (d / "force_constants.npy").exists():
        fc = np.load(d / "force_constants.npy")
    elif (d / "FORCE_CONSTANTS_tdep").exists():
        fc = np.asarray(parse_force_constants(
            str(d / "FORCE_CONSTANTS_tdep"), two_dim=False))
    else:
        pytest.skip(f"{name}: no force constants")
    return prim, sc, fc


@pytest.mark.parametrize("name", FIXTURES)
def test_projector_is_convention_independent(name):
    d = DATA_ROOT / name
    if not d.is_dir():
        pytest.skip(f"fixture not found: {name}")
    prim, sc, fc = _load(name)

    projectors, freqs = {}, None
    for conv in CONVENTIONS:
        dmx = DynamicalMatrix(force_constants=fc, primitive=prim,
                              supercell=sc, precision="fp64",
                              convention=conv)
        projectors[conv] = np.asarray(dmx.e_qsI)
        if freqs is None:
            freqs = np.asarray(dmx.w_qs)

    # Partition modes exactly as the conventions' own basis rotations do, so
    # the test cannot disagree with them about what counts as degenerate.
    # (`degenerate_sets` groups on |w|, which also merges the Gamma acoustics
    # that eigh scatters to +/- a few 1e-3 THz around zero.)
    block = np.tile(np.arange(freqs.shape[1]), (freqs.shape[0], 1))
    for qi, start, stop in degenerate_sets(freqs):
        block[qi, start:stop] = start

    ref = projectors["RAW"]
    for conv, e in projectors.items():
        # Rows are orthonormal per q, so the overlap is unitary; the invariant
        # is that it is block-diagonal over degenerate multiplets.
        overlap = np.abs(np.einsum("qsI,qtI->qst", e, ref.conj()))
        for q in range(overlap.shape[0]):
            same = block[q][:, None] == block[q][None, :]
            leak = np.where(same, 0.0, overlap[q]).max()
            assert leak < TOL_FP64_DERIVED, (
                f"{name} [{conv}] q={q}: projector puts {leak:.2e} of a mode's "
                f"weight on modes of a different frequency -- the convention's "
                f"eigenvector gauge and phase origin are mismatched")
