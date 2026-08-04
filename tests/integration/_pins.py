"""Single source of truth for pinned scalar references on KI_B2_MLIP.

Used by both unit and integration tests so a pin update lands in one
place. Captured 2026-04-20 from a clean fp32/fp64 run on the in-repo
KI_B2_MLIP fixture (1000 MD steps × 128 atoms, dt = 4 fs ⇒ 4 ps,
nq_max=8). Both fp32 and fp64 produce these values bit-identically —
that's the whole point of the dtype-propagation pass.

Physics note on the 2026-04-20 bump:

The fixture was extended from 200 → 1000 MD steps so the short-τ
end of the ACF has enough samples to fit more modes reliably
(200 steps × 4 fs = 0.8 ps was barely long enough for even the
shortest-lived modes; 1000 steps = 4 ps covers the bulk of the
fittable distribution). This legitimately moves κ_raw and κ_corr
upward because the longer integration window resolves more of
the HFACF decay before the cutoff. σ is unchanged to four figures
because it's a per-sample anharmonicity score that doesn't depend
on the ACF length.

Old 200-step pins (kept here as an archaeology note, NOT active):

    KAPPA_RAW_REF        = 0.034386
    KAPPA_CORRECTED_REF  = 0.083394
    SIGMA_REF            = 0.410520

Bumping policy: only when the physics legitimately moves (an
anharmonic correction, a fixed off-by-2π, a better FC fitter, a
trajectory extension like the one above). Note the bump in the
commit message; do not widen the bands silently.
"""

# Cubic average of `thermal_conductivity` (raw HFACF), W/mK.
KAPPA_RAW_REF = 0.117222
KAPPA_RAW_TOL = 1e-3

# Cubic average of `thermal_conductivity_corrected` (size-extrapolated), W/mK.
#
# 0.367793 -> 0.443956 (2026-08-04): sum rule now enforced by default;
# KI_B2_MLIP violates it by 6.9 % of max|Phi|, moving the Gamma acoustics off
# zero. Bisected: enforce_translational_invariance=False reproduces
# 0.367793173. fp64 gives 0.444028414, fp32 0.443955690.
#
# 0.367602 -> 0.367793 (2026-08-03): qhgk_tau_eff converts w to rad/fs via
# _constants.omega_to_rad_fs, not the rounded literal 0.1 (1.81 % high).
# Only the off-diagonal moves. Bisected: restoring 0.1 reproduces 0.367602. NB
# the shift is inside the tolerance, so this pin would not have failed alone.
KAPPA_CORRECTED_REF = 0.443956
KAPPA_CORRECTED_TOL = 5e-3

# Anharmonicity score: std(f_DFT − f_harmonic) / std(f_DFT).
SIGMA_REF = 0.409901
SIGMA_TOL = 5e-3
