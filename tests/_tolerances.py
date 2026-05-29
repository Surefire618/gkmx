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
