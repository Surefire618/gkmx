"""Named tolerance tiers; use these instead of bare literals in tests."""

# fp64 rounding of direct linalg/matmul. For identities and reference
# comparisons of frequencies, eigenvalues, position round-trips.
TOL_FP64 = 1e-12

# fp64 rounding accumulated by sum-of-products. For derived tensors:
# V_ab kernel, spectral resolution, group-velocity cross-products.
TOL_FP64_DERIVED = 1e-6

# Rounding across precisions / backends. For basis-invariant scalars
# (kappa_raw, sigma) in test_precision_switch / test_backends_agree;
# measured floor 1.4e-7 on KI_B2_MLIP.
TOL_FP32_VS_FP64 = 1e-6
TOL_NUMPY_VS_JAX = 1e-6

# Finite-sampling spread of the basis-sensitive cv estimator across
# precisions / backends. For kappa_corrected in the same two tests;
# worst measured pair 1.2e-3 on the 4 ps KI_B2_MLIP fixture.
TOL_DEGENERATE_BASIS = 5e-3

# fp32-vs-fp64 agreement of the velocity operator, compared per
# (q, degenerate block) row power. For the parity test in test_phonon;
# measured 9e-7 on KI_B2_MLIP over all three conventions.
TOL_FP32_BLOCK_POWER = 1e-5
