"""Three-tier tolerance constants for the gkmx test suite.

Single source of truth used by both unit and integration tests
(`from .._tolerances import ...`).

Use these named constants instead of bare literals so a tolerance bump
or a precision-class change lands in one place. Picked from the actual
empirical bands seen by the cross-validated reference tests:

  - ``TOL_FP64`` (1e-12)         — direct fp64 linalg + matmul. Two
    matmul / round chains in fp64 produce ~1e-12 worst-case rounding
    on the n128-scale problems we test. Use for: frequencies vs ref,
    eigenvalue identities, scaled-positions ↔ cartesian round-trips.
  - ``TOL_FP64_DERIVED`` (1e-6)  — sum-of-products derived quantities
    (V_ab kernel, eigenvector spectral resolution, group-velocity
    cross-products). Each multiplication adds ~one decimal of
    rounding; 1e-6 is a comfortable band on the cross-validated
    reference tensors at fp64.
  - ``TOL_KAPPA_PIN`` (1e-4)     — relative band for `κ_corrected`
    pinned-scalar regression on the tiny fixture. Wider because the
    interpolation linear-fit amplifies any upstream lifetime noise.

If a new test wants a fourth tier, add it here with a one-paragraph
justification rather than picking a bare literal in the test body.
"""

TOL_FP64 = 1e-12
TOL_FP64_DERIVED = 1e-6
TOL_KAPPA_PIN = 1e-4

# Cross-configuration agreement. Split by cause, because the two have different
# floors and a single band cannot serve both:
#
#   - ``TOL_FP32_VS_FP64`` / ``TOL_NUMPY_VS_JAX`` (1e-6) — *numerics*. Quantities
#     limited only by rounding: a narrower mantissa, or XLA reducing in a
#     different order than OpenBLAS. Measured on KI_B2_MLIP: kappa_raw 1.4e-07
#     across precisions and exactly 0 across backends, sigma <1e-07 for both.
#     1e-6 is ~7x the observed floor.
#
#   - ``TOL_DEGENERATE_BASIS`` (1e-3) — *sampling*, neither rounding nor physics.
#     kappa cannot depend on the eigenbasis `eigh` picks inside a degenerate
#     multiplet: any unitary mixture of equal-frequency modes is equally valid.
#     Our estimate can, because tau and cv are fitted per mode on the projected
#     amplitudes, which mix under that choice — and a 1e-7 change of input
#     rotates the basis by tens of degrees, so different precisions and backends
#     land on different fits. That is why the band is 4 orders wider than the
#     rounding one on identical arithmetic. It is a convergence artifact and
#     shrinks with trajectory length (KI Gamma triple: 2.98x spread over 4 ps,
#     1.17x over 100 ps), so this band is sized for the 4 ps fixture, not for
#     production runs. Measured 5.2e-05 (backends), 9.8e-05 (precisions), 1.3e-04
#     on CI with a different jax build. Multiplet averaging in `compute_cv_tau`
#     suppresses it but cannot remove it — tau is a nonlinear fit and cv a
#     4th-order moment, so neither has an invariant multiplet mean. Do not
#     tighten without first taking the fit on the invariant trace of g.
TOL_FP32_VS_FP64 = 1e-6
TOL_NUMPY_VS_JAX = 1e-6
TOL_DEGENERATE_BASIS = 1e-3
