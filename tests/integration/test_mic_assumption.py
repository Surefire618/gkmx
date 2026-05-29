"""MIC fold correctness and eager-vs-lazy parity in ``get_kappa``.

The pipeline used to assume ``positions - sc_positions`` was its own
MIC fold; that broke on CuI nve.1 (MD writer wraps positions to the
box). The fix is to fold each per-block displacement through
``mic_fold`` regardless of how positions were stored.
"""
from __future__ import annotations

import os

import numpy as np
from ase import io as ase_io

from gkmx.greenkubo import _disp_block, get_kappa
from gkmx.mic import is_orthogonal


def test_mic_is_identity_on_unwrapped_trajectory(tiny_trajectory, tiny_dir):
    """KI_B2_MLIP positions are stored unwrapped; ``mic_fold`` must be
    the identity. Any non-zero difference means either the trajectory
    is wrapped or ``mic_fold`` is misbehaving."""
    sc = ase_io.read(str(tiny_dir / "geometry.in.supercell"), format="aims")
    sc_pos = np.asarray(sc.positions, dtype=np.float64)
    cell = np.asarray(sc.cell, dtype=np.float64)
    pos = np.asarray(tiny_trajectory["positions"][::100].data, dtype=np.float64)
    folded = _disp_block(pos, sc_pos, cell,
                         search_images=not is_orthogonal(cell))
    assert np.abs(pos - sc_pos - folded).max() < 1e-12


def test_pipeline_invariant_under_wrap_and_eager_lazy(
        tiny_trajectory, tiny_fc_file, tiny_dir):
    """``get_kappa`` must produce the same κ on:
        (a) the original trajectory,
        (b) a wrapped trajectory where one atom is shifted by a full
            lattice vector,
        (c) the lazy MIC-per-chunk branch (``GKMX_EAGER_DISP_GB=0``).
    All three must match to fp64 — MIC undoes the wrap exactly, and
    eager / lazy differ only in *when* MIC runs, not in the result."""
    sc = ase_io.read(str(tiny_dir / "geometry.in.supercell"), format="aims")
    cell = np.asarray(sc.cell)

    def _run(traj, env_override=None):
        prev = os.environ.get("GKMX_EAGER_DISP_GB")
        if env_override is not None:
            os.environ["GKMX_EAGER_DISP_GB"] = env_override
        try:
            ds = get_kappa(traj, fc_file=tiny_fc_file, interpolate=False,
                           backend="numpy", precision="fp64")
        finally:
            if env_override is not None:
                if prev is None:
                    del os.environ["GKMX_EAGER_DISP_GB"]
                else:
                    os.environ["GKMX_EAGER_DISP_GB"] = prev
        return float(np.diag(ds.thermal_conductivity).mean())

    k_baseline = _run(tiny_trajectory.copy(deep=True))

    wrapped = tiny_trajectory.copy(deep=True)
    wrapped["positions"].data[:, 0, :] += cell[0]
    k_wrapped = _run(wrapped)

    k_lazy = _run(tiny_trajectory.copy(deep=True), env_override="0")

    scale = max(abs(k_baseline), 1e-30)
    assert abs(k_wrapped - k_baseline) / scale < 1e-6, (
        f"MIC failed to undo +cell wrap: {k_wrapped} vs {k_baseline}"
    )
    assert abs(k_lazy - k_baseline) / scale < 1e-6, (
        f"eager / lazy MIC branches disagree: {k_lazy} vs {k_baseline}"
    )
