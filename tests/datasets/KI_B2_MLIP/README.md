# KI(B2) MLIP MD fixture

KI in the B2 / CsCl structure, Pm-3m, a = 4.261797 Å. Primary end-to-end fixture;
drives `test_end_to_end_kappa.py` and the `KAPPA_RAW_REF`, `KAPPA_CORRECTED_REF`
and `SIGMA_REF` pins in `tests/integration/_pins.py`.

| file | contents |
|---|---|
| `geometry.in.primitive` | 2-atom primitive cell (K, I), FHI-aims format |
| `geometry.in.supercell` | 128-atom 4×4×4 supercell |
| `FORCE_CONSTANTS_tdep` | fc2, phonopy text format, `(2, 128, 3, 3)`, eV/Å²; extracted with TDEP from the trajectory below |
| `nve/000000.nc` | 1000-step MLIP NVE trajectory at 303 K: positions, velocities, forces, heat flux, stress |
| `DynamicalMatrix.nc` | cached solution on the commensurate 4×4×4 q-grid (64 q, 6 modes): `w_qs`, `v_qsa`, `v_qssa`, `e_qsi`. Reference for the solver tests; predates ASR enforcement, so those tests pass `enforce_translational_invariance=False` |

**This is not a converged calculation.** It is sized to keep the test suite fast.
Every κ below is a reproducible number for regression purposes, not a physical
prediction for KI.

# Reproduce

```bash
cd tests/datasets/KI_B2_MLIP
gkmx out gk nve/000000.nc -fc FORCE_CONSTANTS_tdep \
    --interpolate --maxnq 8 --backend numpy -o kappa.gk.nc
```

Tail of the log:

```
[gkmx.interpolation]  Initial harmonic kappa:   0.187 W/mK
[gkmx.interpolation]  Correction:               0.327 +/- 0.006 W/mK
[gkmx.interpolation]  Correction factor:        2.749
[gkmx]  Corrected kappa: 0.444 W/mK
```

`kappa.gk.nc` carries the κ tensors, the HFACF and its integral, and the per-mode
frequencies, group velocities, lifetimes and heat capacities. Read the pinned
scalars with:

```python
import numpy as np, xarray as xr
ds = xr.open_dataset("kappa.gk.nc", engine="h5netcdf")
np.mean(np.diag(ds["thermal_conductivity_corrected"].data))   # 0.443956
```

Add `--no-enforce-translational-invariance` to use the force constants exactly as
stored — see below for what that changes.

# Enforce translational invariance

The force constants do not satisfy the acoustic sum rule:

```
max|sum_B Phi_aB|  = 1.070e-01 eV/A^2      (6.934e-02 of max|Phi|)
per-atom diagonal  = K +0.0581, I +0.1070
```

Every other force-constant set in the project is clean to 1e-14 or better, so this
is the only fixture exercising the imperfect-FC path — which is why it is kept.
The residual is isotropic and positive and the on-site block is too large by
exactly that amount, so the sum rule was never imposed; it is not a cutoff
artifact. The geometry is fine — spglib gives Pm-3m (221) with 48 operations.

Γ frequencies, THz:

```
default                                 +0.0000  +0.0000  +0.0000   +2.5733  +2.5733  +2.5733
--no-enforce-translational-invariance   +0.4385  +0.4853  +0.5456   +2.6065  +2.6388  +2.6627
```

The three acoustic modes must be exactly zero. Left alone they sit at 0.44-0.55
THz, which the residual predicts: a rigid translation carrying it has
`w^2 = (1/M_tot) sum_{i,B} Phi[i][B]`, giving 0.4931 THz. They then acquire a large
`1/w` that leaks into every `1/w`-weighted mode sum.

κ, W/mK, from the two CLI runs above:

| | raw HFACF | harmonic | QHGK | corrected |
|---|---|---|---|---|
| default | 0.117222 | 0.178491 | 0.186775 | **0.443956** |
| `--no-enforce-…` | 0.117222 | 0.173231 | 0.181404 | 0.367793 |

`raw HFACF` is bit-identical — it comes straight from the trajectory and never
touches the force constants. The harmonic channels move ~3 %, and the
size-extrapolation amplifies that to **+20.7 %** on `corrected` (correction factor
2.381 → 2.749).
