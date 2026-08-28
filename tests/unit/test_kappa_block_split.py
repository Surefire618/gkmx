"""Population / coherence split of kappa_QHGK, PRX definition.

Simoncelli, Marzari & Mauri, PRX 12, 041011 (2022): diagonal and degenerate
velocity-operator elements belong to the populations conductivity kappa_P;
coherences kappa_C come only from non-degenerate pairs. gkmx implements the
prescription basis-free -- the whole intra-multiplet block goes to kappa_P
(``qhgk_block_mask``), so no rotation is needed and the full 3x3 tensor is
covered at once.

Pinned across every FC-bearing fixture (13 datasets, 21 space groups):
exact closure and invariance across the intra-block-unitary conventions
(RAW, PHONO3PY, WIGNER), plus block kappa_C <= label kappa_C. The
resolving-power controls stay on KI_B2_MLIP, where their floors were
measured: TDEP's little-group average is a contraction there (18.9 % of
operator weight), so it moves what a vacuous invariance test would not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import io as ase_io

from gkmx.io import parse_force_constants
from gkmx.kappa import get_kappa_BTE, get_kappa_QHGK, get_kappa_QHGK_block_split
from gkmx.lattice_points import get_commensurate_q_points
from gkmx.phonon import Phonon

from .._tolerances import TOL_FP64, TOL_FP64_DERIVED

DATASETS = Path(__file__).resolve().parents[1] / "datasets"
# Every fixture that ships force constants. Ga2O3_alpha_aiGK embeds its FC
# in a trajectory and is exercised at integration level instead.
FIXTURES = [
    "KI_B2_MLIP", "CuI_aiGK",
    "tdep_KI_bcc", "tdep_Rb2O", "tdep_KPTe2", "tdep_Ga2O3_kappa",
    "phonopy_Si", "phonopy_NaCl", "phonopy_SrTiO3", "phonopy_AgErTe2",
    "phonopy_Li2ZnGeO4", "phonopy_LiCdBO3", "phonopy_Mg2YbSb2",
    "phonopy_RbIn3F10",
]


def _load_fc(d):
    if (d / "FORCE_CONSTANTS_tdep").exists():
        return np.asarray(parse_force_constants(
            str(d / "FORCE_CONSTANTS_tdep"), two_dim=False))
    if (d / "force_constants.npy").exists():
        return np.load(d / "force_constants.npy")
    return np.load(d / "force_constants.npz")["force_constants"]


@pytest.fixture(scope="module", params=FIXTURES)
def solutions(request):
    d = DATASETS / request.param
    if not d.is_dir():
        pytest.skip(f"{request.param} fixture not found")
    prim = ase_io.read(d / "geometry.in.primitive", format="aims")
    sc = ase_io.read(d / "geometry.in.supercell", format="aims")
    fc = _load_fc(d)
    q = np.asarray(get_commensurate_q_points(prim.cell, sc.cell,
                                             fractional=True))
    out = {"name": request.param}
    for conv in ("PHONO3PY", "RAW", "TDEP", "WIGNER"):
        out[conv] = Phonon(fc, prim, sc, precision="fp64",
                           convention=conv).solve(
            q, with_velocities=True, with_group_velocity_matrices=True)
    return out


def _split(sol, tau=500.0, cv=1e-5):
    shape = np.asarray(sol.w_qs).shape
    tau_qs = np.full(shape, tau)
    cv_qs = np.full(shape, cv)
    kP, kC = get_kappa_QHGK_block_split(
        sol.v_qssa_cartesian, tau_qs, sol.w_qs, sol.w_inv_qs, cv_qs)
    kQ = get_kappa_QHGK(sol.v_qssa_cartesian, tau_qs, sol.w_qs,
                        sol.w_inv_qs, cv_qs)
    return np.asarray(kP), np.asarray(kC), np.asarray(kQ)


def test_block_split_closes_exactly(solutions):
    """Every pair counted once: kappa_P + kappa_C == kappa_QHGK at fp64."""
    kP, kC, kQ = _split(solutions["RAW"])
    assert np.abs(kP + kC - kQ).max() / np.abs(kQ).max() < TOL_FP64


def test_block_split_is_convention_invariant(solutions):
    """Both channels are basis-invariant: the intra-block-unitary
    conventions differ only inside multiplets, which the split assigns
    wholesale to kappa_P. Exact only for exact degeneracies: tol-merged
    near-degenerate blocks leave the kernel varying by the intra-block
    frequency spread, so the invariance floor is that spread (worst
    measured 2.2e-10 on Mg2YbSb2, vs 2e-16 on KI's exact multiplets) --
    hence the DERIVED tier. The KI-only controls below prove the
    assertion could fail (their floors are measured values there)."""
    ref = _split(solutions["RAW"])
    for conv in ("PHONO3PY", "WIGNER"):
        got = _split(solutions[conv])
        for a, b, name in zip(ref[:2], got[:2], ("kappa_P", "kappa_C")):
            rel = np.abs(a - b).max() / np.abs(ref[2]).max()
            assert rel < TOL_FP64_DERIVED, f"{conv} {name}: rel {rel:.2e}"

    if solutions["name"] != "KI_B2_MLIP":
        return
    # Resolving power: the label split must NOT be invariant. TDEP's
    # contraction moves diagonal weight strongly; WIGNER's unitary moves it
    # weakly (2.2e-6) but still three decades above the second floor.
    shape = np.asarray(solutions["RAW"].w_qs).shape
    tau_qs, cv_qs = np.full(shape, 500.0), np.full(shape, 1e-5)
    label = {c: float(np.mean(np.diag(np.asarray(get_kappa_BTE(
        np.real(np.einsum("qssa->qsa", np.asarray(
            solutions[c].v_qssa_cartesian))), tau_qs, cv_qs)))))
        for c in ("RAW", "TDEP", "WIGNER")}
    assert abs(label["RAW"] - label["TDEP"]) / label["RAW"] > 1e-3, (
        "label split agrees across conventions on KI; the invariance "
        "assertion above has no resolving power")
    assert abs(label["RAW"] - label["WIGNER"]) / label["RAW"] > 1e-9


def test_block_split_moves_weight_off_the_label_split(solutions):
    """Intra-multiplet off-diagonal weight is population, not coherence:
    block kappa_C cannot exceed the label (s != s') sum. Measured
    reassignment on KI_B2_MLIP: 3.6 % of label_C (uniform tau)."""
    sol = solutions["RAW"]
    _kP, kC, kQ = _split(sol)
    dg = np.real(np.einsum("qssa->qsa", np.asarray(sol.v_qssa_cartesian)))
    shape = np.asarray(sol.w_qs).shape
    k_diag = np.asarray(get_kappa_BTE(dg, np.full(shape, 500.0),
                                      np.full(shape, 1e-5)))
    label_C = float(np.mean(np.diag(kQ - k_diag)))
    block_C = float(np.mean(np.diag(kC)))
    assert block_C <= label_C, (
        f"block kappa_C ({block_C:.6g}) exceeds label kappa_C "
        f"({label_C:.6g})")
