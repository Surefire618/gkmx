"""Pin σ and κ-tensor symmetry on the in-repo KI_B2_MLIP trajectory.

The κ_raw / κ_corrected scalar pins live in
``test_end_to_end_kappa.py`` (which exercises numpy + jax). This file
covers the regression bank for two scalars that file does not:

  - σ (anharmonicity score) — regression bank for the remapped FC +
    ``_harmonic_force_residuals`` path.
  - κ tensor symmetry — axis-mixing guard distinct from the diagonal
    scalar pin.

200 steps × 128 atoms is a unit-test fixture, not a converged Green-
Kubo run. The full-physics pin lives in the n4096 benchmark.
"""

from __future__ import annotations

import numpy as np
import pytest

from gkmx.greenkubo import get_kappa

from ._pins import SIGMA_REF, SIGMA_TOL


class TestKappaPipelineTiny:

    @pytest.fixture(scope="class")
    def gk_dataset(self, tiny_trajectory, tiny_fc_file):
        """Run the full pipeline once for the whole class."""
        return get_kappa(
            tiny_trajectory.copy(deep=True),
            fc_file=tiny_fc_file,
            interpolate=True,
            nq_max=8,
            backend="numpy",
        )

    def test_sigma_anharmonicity(self, gk_dataset):
        """Anharmonicity score σ pinned. Regression bank for the
        remapped FC + `_harmonic_force_residuals` path."""
        sigma = float(gk_dataset.attrs["sigma"])
        assert abs(sigma - SIGMA_REF) < SIGMA_TOL, (
            f"sigma = {sigma:.6f} differs from pin {SIGMA_REF} by more "
            f"than {SIGMA_TOL}. Likely a regression in the remapped FC "
            "or in the harmonic-force matmul."
        )

    def test_kappa_tensor_symmetric(self, gk_dataset):
        """κ tensor should be symmetric (axis-mixing guard). On 200
        steps the off-diagonals are dominated by HFACF noise so we
        only enforce symmetry, not isotropy."""
        k = np.asarray(gk_dataset["thermal_conductivity_corrected"].data)
        assert np.allclose(k, k.T, atol=1e-3 * np.abs(k).max()), (
            f"κ tensor not symmetric:\n{k}"
        )
