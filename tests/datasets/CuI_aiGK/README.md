# CuI_aiGK — lightweight ab-initio GK fixture (SMA-breakdown regime)

1000-MD-step (5 ps, dt = 5 fs) slice of the CuI ab-initio NVE
trajectory from the Knoop group (F. Knoop *et al.*, Phys Rev Mater
**4**, 083809 (2020); Phys Rev Lett **130**, 236301 (2023)). Sliced
from frames **7100–8100** of the full 60 ps run — i.e. **straddling
the transition** into the second local energy minimum at ~38 ps.
The slice covers roughly 500 MD steps in the equilibrium ground
state (35.5–38.0 ps) plus 500 MD steps in the higher-energy
metastable basin (38.0–40.5 ps), so the ACFs average over both
regimes. This "half-eq + half-metastable" design maximizes the
contrast between the Wick / dressed-bubble lifetime extraction and
the un-factorized (vertex) extraction: the metastable half injects
non-Gaussian `<nn>` content that the Wick factorization discards
but the equilibrium half keeps the fits on cleanly exponential ACF
windows so both methods still converge.

## Scientific context

On the full
60 ps CuI trajectory, the median per-mode ratio τ_wick/τ_vertex is
1.01 (most modes SMA-valid). On this 5 ps half-eq + half-metastable
slice gkmx reports τ-median ratio ≈ 1.13 (modest — most modes
individually converge once the fit window has enough data) but the
**transport-relevant κ_harmonic ratio is ≈ 3.3** (stronger than KI's
2.70 on a comparable 4-ps MLIP-clean slice). The BTE weighting by
cv·v² amplifies the breakdown in the modes that actually carry
heat. See the Notion page "SMA Breakdown for Long-Lifetime Acoustic
Modes — CuI Diagnosis & Cures" for the horizon / future-work
framing.

## Layout (mirrors `KI_B2_MLIP/`)

  geometry.in.primitive       FHI-aims, CuI rocksalt B1 (a ≈ 6.01 Å)
  geometry.in.supercell       3×3×3 of conventional cell, 216 atoms
  FORCE_CONSTANTS_tdep        phonopy/TDEP-format text FC
  DynamicalMatrix.nc          gkmx HDF5 DMX cache
  nve/000000.nc               5-ps trajectory slice at 35.5–40.5 ps
                              (half eq ground state + half metastable)
  README.md                   this file

Trimmed data_vars (kept only what gkmx consumes): `positions`,
`velocities`, `forces`, `heat_flux`, `energy_potential`,
`energy_kinetic`, `temperature`, `volume`, `cell`,
`positions_reference`, `lattice_reference`. Static FC arrays,
`momenta` (= velocities × masses, derivable), per-atom stress
tensors, and aux quantities dropped to keep the fixture to ~16 MB
despite 5× more frames than KI_B2_MLIP.

## Known quirks

- **Trajectory is wrapped.** `positions − positions_reference` overshoots
  by ~1 fractional lattice vector on several atoms; gkmx's `_disp_block`
  re-applies vesin MIC per block so the pipeline handles this correctly.
- **Cell is non-orthogonal at the primitive level** (rocksalt fcc);
  `is_orthogonal(primitive.cell) == False`, so MIC takes the 27-image
  search path.
- **Heat flux is in eV/(Å²·fs)** — already converted from the
  vibes/gkx eV/(Å²·ps) convention at fixture-build time (divided by
  1000). Matches the `to_W_mK = e * 1e25` conversion in
  `gkmx._constants`. The gkmx pipeline no longer divides by 1000;
  any fixture derived from a vibes-format trajectory must apply the
  conversion at fixture time.
- **5 ps is too short to converge κ.** Long-lifetime acoustic modes
  (τ > ~1 ps) still don't fit cleanly here — this fixture exists to
  exercise the factorization code paths and regress the breakdown
  direction, not to produce physically converged κ. The full
  out-of-tree 60 ps trajectory at
  `<gkmx-parent>/datasets/CuI_FK_n216/` is the convergence reference.
