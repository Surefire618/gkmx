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
# Rides on the QHGK channel, and is not reproducible below ~1e-4 across
# precisions or backends: rotating a degenerate multiplet mixes the mode
# amplitudes tau and cv are fitted on, so their values follow whichever basis
# eigh returned. Averaging over multiplets suppresses that but cannot remove it
# -- tau is a nonlinear fit and cv a 4th-order moment, so neither has an
# invariant multiplet mean. REF is the midpoint of the fp32/fp64 x numpy/jax
# spread (0.443910 to 0.443953) so no configuration sits at the edge of the
# band. Tightening TOL needs the fit taken on the invariant trace of g.
# See test_degenerate_basis.py and test_precision_switch.py.
KAPPA_CORRECTED_REF = 0.443931
KAPPA_CORRECTED_TOL = 1e-4

# Anharmonicity score: std(f_DFT − f_harmonic) / std(f_DFT).
SIGMA_REF = 0.409901
SIGMA_TOL = 5e-3
