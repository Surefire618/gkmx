# phonopy reference — Li2ZnGeO4

Phonopy reference fixture consumed by
`tests/unit/test_phonopy_reference.py` so gkmx's `Phonon` solver can be
cross-checked against phonopy's parallel C kernel without importing
phonopy at test time.

## Metadata

| Field | Value |
|---|---|
| Material | Li2ZnGeO4 |
| Space group | 7 |
| Primitive symbols | `['Li', 'Li', 'Li', 'Li', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'Zn', 'Zn', 'Ge', 'Ge']` |
| `N_p` / `N_sc` | 16 / 128 |
| Supercell matrix | `diag([2, 2, 2])` |
| Primitive matrix | `[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]` |
| q-grid | 32 uniform-random in `(-0.5, 0.5)^3`, seed `42` |
| phonopy version | 2.28.0 |
| `unit_conversion_factor` | `15.633302300230191` |

## Source

Loaded via `phonopy.load(unitcell=POSCAR, force_sets_filename=FORCE_SETS, ...)` from phonopy's upstream test fixtures:

- [`test/phonon/POSCAR_Pc`](https://github.com/phonopy/phonopy/blob/develop/test/phonon/POSCAR_Pc)
- [`test/phonon/FORCE_SETS_Pc`](https://github.com/phonopy/phonopy/blob/develop/test/phonon/FORCE_SETS_Pc)

## Files

| File | Contents |
|---|---|
| `geometry.in.primitive` | FHI-aims geometry, primitive cell |
| `geometry.in.supercell` | FHI-aims geometry, supercell |
| `force_constants.npy` | `(N_p, N_sc, 3, 3)` float64, compact phonopy FC |
| `reference.npz` | q-points + phonopy solver outputs — frequencies, group velocities, QHGK group-velocity matrix, raw eigenvectors, phonopy's exact masses, `unit_conversion_factor`, `space_group_number` |

## Regeneration

```
cd gkmx
micromamba run -n gkmx python tests/datasets/_generate_phonopy_references.py
```

Do not edit these files by hand — re-run the generator instead.
