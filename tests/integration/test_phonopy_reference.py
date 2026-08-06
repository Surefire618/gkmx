"""Cross-validate ``gkmx.Phonon`` against pre-computed phonopy references.

Reference data is checked in under ``tests/datasets/phonopy_*/``; phonopy
itself is **not** imported at test time. To regenerate, run
``tests/datasets/_generate_phonopy_references.py``.

Coverage: eight materials spanning six of the seven crystal systems
(missing triclinic — phonopy ships no triclinic fixture).

Each material is parametrized over six independent contracts that
catch different failure modes:
  1. space group tag — fixture metadata sanity
  2. signed frequencies — eigenvalue solve
  3. V_ab — diagonal group velocities (rotation-invariant)
  4. v_qssa — QHGK off-diagonal group-velocity matrix
  5. degenerate-subspace projectors — gauge-invariant eigenvectors
  6. frequency range — fixture unit-sanity (catches Ry/bohr² mixups)

A module-scoped cache solves each material once; the six tests then
consume different fields.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest
from ase import io as ase_io

from gkmx import _constants as C
from gkmx.phonon import Phonon

from .._tolerances import TOL_FP64, TOL_FP64_DERIVED

DATA_ROOT = Path(__file__).parent.parent / "datasets"

FIXTURES = [
    # cubic ×3
    ("phonopy_Si",         227),  # Fd-3m   diamond
    ("phonopy_NaCl",       225),  # Fm-3m   rock salt
    ("phonopy_SrTiO3",     221),  # Pm-3m   perovskite (soft mode)
    # monoclinic
    ("phonopy_Li2ZnGeO4",    7),  # Pc
    # orthorhombic
    ("phonopy_RbIn3F10",    17),  # P222_1
    # tetragonal non-symmorphic
    ("phonopy_AgErTe2",    113),  # P-42_1m
    # trigonal
    ("phonopy_Mg2YbSb2",   164),  # P-3m1
    # hexagonal
    ("phonopy_LiCdBO3",    174),  # P-6
]


@cache
def _solve(fixture_name):
    """Load fixture and run gkmx once per material; cache for the rest of the session."""
    d = DATA_ROOT / fixture_name
    if not d.is_dir():
        pytest.skip(f"fixture {fixture_name} not found at {d}")

    prim = ase_io.read(str(d / "geometry.in.primitive"), format="aims")
    sc = ase_io.read(str(d / "geometry.in.supercell"), format="aims")
    fc = np.load(d / "force_constants.npy")
    ref = np.load(d / "reference.npz")
    # ASE's aims writer drops masses; restore phonopy's exact values
    # or frequencies drift by ~1e-5.
    prim.set_masses(np.asarray(ref["primitive_masses"]))
    sc.set_masses(np.asarray(ref["supercell_masses"]))

    factor = (float(ref["unit_conversion_factor"]) / C.omega_to_THz) ** 2
    # The references are generated with phonopy's own symmetrize_fc=False, so
    # both sides must consume the raw force constants for this to compare
    # solvers rather than preprocessing.
    sol = Phonon(fc, prim, sc, backend="numpy", factor=factor,
                 enforce_translational_invariance=False).solve(
        np.asarray(ref["q_points"]),
        with_velocities=True, with_group_velocity_matrices=True,
    )
    out = {
        "freqs_sorted_signed":
            np.sort(sol.w_qs * C.omega_to_THz, axis=-1),
        "V_ab": np.einsum(
            "qsa,qsb->qab",
            sol.v_qsa_cartesian * 1e3, sol.v_qsa_cartesian * 1e3),
        # phonopy GVM layout: (Nq, 3, Ns, Ns) in THz·Å
        "v_qssa_THzA": np.moveaxis(
            np.asarray(sol.v_qssa_cartesian), -1, 1) * 1e3,
        # phonopy uses column eigenvectors `e[q, i, s]`
        "e_col": np.moveaxis(np.asarray(sol.e_qsi), -1, -2),
    }
    return ref, out


def _max_rel(a, b):
    return float(np.abs(a - b).max()) / max(float(np.abs(b).max()), 1e-30)


@pytest.mark.parametrize("fixture_name,expected_sg", FIXTURES,
                          ids=[name for name, _ in FIXTURES])
class TestPhonopyReference:
    """One material per parametrize ID; six tests per material, each
    pinning a distinct physical contract so failures self-describe."""

    def test_space_group_tag(self, fixture_name, expected_sg):
        """Fixture metadata sanity — catches a misplaced reference file."""
        ref, _ = _solve(fixture_name)
        assert int(ref["space_group_number"]) == expected_sg

    def test_frequencies_match(self, fixture_name, expected_sg):
        """Sorted **signed** frequencies match phonopy at fp64. Comparing
        signed (not absolute) preserves the soft-mode convention."""
        ref, out = _solve(fixture_name)
        freqs_ref = np.sort(np.asarray(ref["frequencies_THz"]), axis=-1)
        assert _max_rel(out["freqs_sorted_signed"], freqs_ref) < TOL_FP64

    def test_velocity_tensor_V_ab_matches(self, fixture_name, expected_sg):
        """``V_ab(q) = Σ_s v_sa·v_sb`` matches phonopy. Rotation-
        invariant and therefore gauge-safe even with degenerate modes
        — the right per-q comparison for diagonal group velocities."""
        ref, out = _solve(fixture_name)
        V_ref = np.asarray(ref["velocity_tensor_THz2_A2"])
        assert _max_rel(out["V_ab"], V_ref) < TOL_FP64_DERIVED

    def test_group_velocity_matrix_matches(self, fixture_name, expected_sg):
        """QHGK off-diagonal ``v_qssa(q)`` matches phonopy element-wise.
        A regression here is most likely in ``_perturb_D`` /
        ``_symmetrize_group_velocity`` or in the dD/dq Hermitization."""
        ref, out = _solve(fixture_name)
        gvm_ref = np.asarray(ref["group_velocity_matrix_THzA"])
        assert _max_rel(out["v_qssa_THzA"], gvm_ref) < TOL_FP64_DERIVED

    def test_eigenvector_projectors_match(self, fixture_name, expected_sg):
        """Per-degenerate-group projector ``P = Σ_{s∈grp} |e_s⟩⟨e_s|``
        matches phonopy. Gauge-invariant under unitary rotations within
        a degenerate subspace — the right comparison because phonopy
        applies an extra ``_perturb_D`` rotation that gkmx does not."""
        ref, out = _solve(fixture_name)
        e_ref = np.asarray(ref["eigenvectors_raw"])
        freqs = np.asarray(ref["frequencies_THz"])
        e_gx = out["e_col"]
        Nq, Ns, _ = e_gx.shape
        max_dP = 0.0
        for qi in range(Nq):
            fq = freqs[qi]
            start = 0
            for si in range(1, Ns + 1):
                if si == Ns or abs(fq[si] - fq[start]) > 1e-4:
                    P_gx = e_gx[qi, :, start:si] @ e_gx[qi, :, start:si].conj().T
                    P_ref = e_ref[qi, :, start:si] @ e_ref[qi, :, start:si].conj().T
                    max_dP = max(max_dP, float(np.abs(P_gx - P_ref).max()))
                    start = si
        assert max_dP < TOL_FP64_DERIVED, (
            f"{fixture_name}: degenerate-subspace |ΔP| = {max_dP:.3e}"
        )

    def test_frequency_range_plausible(self, fixture_name, expected_sg):
        """Fixture unit-sanity: every physically reasonable solid has
        ``f_max`` in 1–100 THz. Catches Ry/bohr² ↔ eV/Å² unit mixups,
        wrong FC shape, or a corrupted ``.npy``. Not a stability check
        — small soft modes from real instabilities are fine."""
        ref, _ = _solve(fixture_name)
        freqs = np.asarray(ref["frequencies_THz"])
        assert 1.0 < float(freqs.max()) < 100.0
        assert float(freqs.min()) > -5.0
