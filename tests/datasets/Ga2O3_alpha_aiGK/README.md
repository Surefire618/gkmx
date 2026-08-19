# Ga2O3_alpha_aiGK — low-symmetry fixture for Bloch-convention regressions

1000-MD-step (5 ps, dt = 5 fs) head slice of the α-Ga₂O₃ 300 K
ab-initio NVE trajectory from the 2024 FHI-vibes aiGK study
(`240129_Ga2O3_all/alpha_300_100ps/gk_0`, 100 ps original).

## Why this fixture exists

The cubic fixtures cannot police the Bloch-representative pairing in
`DynamicalMatrix._build_e_qsI`. That pairing breaks when a
convention's eigenvector gauge is contracted against another
convention's phase table, and the damage scales with how non-trivial
the per-atom phase `exp(-2πi q·r_a)` is — i.e. with the number of
atoms in the primitive and how general their Wyckoff positions are.

α-Ga₂O₃ is R-3c (167) with a **10-atom primitive** (Ga at 12c, O at
18e), against 2-atom primitives at special positions for
`KI_B2_MLIP` (Pm-3m) and `CuI_aiGK` (Fm-3m). Measured with the
projector deliberately mispaired:

| fixture | median τ, TDEP vs RAW | worst isolated mode |
|---|---|---|
| `Ga2O3_alpha_aiGK` | 58.0 fs vs 478.4 fs (8.2×) | 98 % |
| `KI_B2_MLIP` | 428 fs vs 689 fs (1.6×) | 79 % |

The 8.2 × matches what the full production trajectories show, so this
fixture reproduces the regression at its true magnitude rather than a
symmetry-suppressed remnant. See
`tests/integration/test_convention_projection.py`;
`tests/unit/test_bloch_projector.py` covers structural breadth on
geometry alone.

## Layout

    nve/000000.nc    5 ps trajectory slice, force constants embedded
    README.md        this file

`data_vars` trimmed to what gkmx consumes — `positions`,
`velocities`, `forces`, `heat_flux`, `temperature`, `volume`,
`force_constants` — and stored fp32, which holds the fixture to 3 MB.
`momenta` is dropped (derived as velocities × masses). The embedded
`force_constants` means no separate FC file: `get_kappa(ds)` with no
`fc_file` exercises the trajectory-embedded path.

## Known quirks

- **`heat_flux_time_unit = "fs"` is stored as an attribute**, so the
  loader never falls back to its magnitude heuristic. The vibes-format
  source is fs-base already; no ÷1000 was applied.
- **Heat flux is written every other MD step**, so half its rows are
  NaN by construction (the loader `dropna`s internally).
- **5 ps is far too short to converge κ.** The p90 lifetime is
  comparable to the slice length. This fixture pins convention
  invariance and plumbing, not physics; the converged reference is the
  out-of-tree 100 ps run.
- **Commensurate grid is 8 all-TRIM q-points** (2×2×2 of the 80-atom
  supercell), so `kappa_harmonic` vanishes under TDEP — inversion kills
  the group velocities there. That is correct, not a defect.
