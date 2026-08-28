# KI(B2) MLIP MD fixture

KI in the B2 / CsCl structure, Pm-3m, a = 4.261797 Å. Primary end-to-end fixture;
drives `test_end_to_end_kappa.py` and the `KAPPA_RAW_REF`, `KAPPA_CORRECTED_REF`
and `SIGMA_REF` pins in `tests/integration/_pins.py`.

| file | contents |
|---|---|
| `geometry.in.primitive` | 2-atom primitive cell (K, I), FHI-aims format |
| `geometry.in.supercell` | 128-atom 4×4×4 supercell |
| `FORCE_CONSTANTS_tdep` | fc2, phonopy text format, `(2, 128, 3, 3)`, eV/Å²; extracted with TDEP from the trajectory below, projected onto the ASR + space-group-invariant subspace |
| `nve/000000.nc` | 1000-step MLIP NVE trajectory at 303 K: positions, velocities, forces, heat flux, stress |
| `DynamicalMatrix.nc` | cached solution on the commensurate 4×4×4 q-grid (64 q, 6 modes): `w_qs`, `v_qsa`, `v_qssa`, `e_qsi`, the FC as solved; fp64, WIGNER convention. Reference for the solver tests |

**This is not a converged calculation.** It is sized to keep the test suite fast.
Every κ below is a reproducible number for regression purposes, not a physical
prediction for KI.

# Reproduce

```bash
cd tests/datasets/KI_B2_MLIP
gkmx out gk nve/000000.nc -fc FORCE_CONSTANTS_tdep \
    --interpolate --maxnq 8 --backend numpy -o kappa.gk.nc
```

Tail of the log (library default precision; the `_pins.py` values are baked at
`--precision fp64`):

```
[gkmx.interpolation]  Initial harmonic kappa:   0.156 W/mK
[gkmx.interpolation]  Correction:               0.215 +/- 0.007 W/mK
[gkmx.interpolation]  Correction factor:        2.376
[gkmx]  Corrected kappa: 0.332 W/mK
```

Read the pinned scalar with:

```python
import numpy as np, xarray as xr
ds = xr.open_dataset("kappa.gk.nc", engine="h5netcdf")
np.mean(np.diag(ds["thermal_conductivity_corrected"].data))
```

# Force-constant symmetrization

The raw TDEP fit violates the acoustic sum rule (6.9e-2 of max|Φ|) and the site
symmetry (2.0e-2); unprojected, the acoustic modes sit at 0.44–0.55 THz instead
of zero. The stored file is the projection — identical to the FC every
default-configured solve uses, since the projection is idempotent, so all
pinned results are unaffected. The raw fit is in git history.
