"""Pinned scalar references on KI_B2_MLIP.

This is not a converged calculation, only a light-weight fixture.
These pins test fp32/fp64 and numpy/jax agreement.
"""

# Cubic average of `thermal_conductivity` (raw HFACF), W/mK.
KAPPA_RAW_REF = 0.11722172
KAPPA_RAW_TOL = 1e-6

# Cubic average of `thermal_conductivity_corrected` (size-extrapolated), W/mK.
KAPPA_CORRECTED_REF = 0.44318744
KAPPA_CORRECTED_TOL = 5e-5

# Anharmonicity score: std(f_DFT - f_harmonic) / std(f_DFT).
SIGMA_REF = 0.40990110
SIGMA_TOL = 1e-6
