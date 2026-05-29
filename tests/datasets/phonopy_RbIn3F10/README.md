# phonopy reference — RbIn3F10

Phonopy reference fixture consumed by
`tests/unit/test_phonopy_reference.py` so gkmx's `Phonon` solver can be
cross-checked against phonopy's parallel C kernel without importing
phonopy at test time.

## Metadata

| Field | Value |
|---|---|
| Material | RbIn3F10 |
| Space group | 17 |
| Primitive symbols | `['Rb', 'Rb', 'In', 'In', 'In', 'In', 'In', 'In', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F']` |
| `N_p` / `N_sc` | 28 / 112 |
| Supercell matrix | `diag([2, 2, 1])` |
| Primitive matrix | `[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]` |
| q-grid | 32 uniform-random in `(-0.5, 0.5)^3`, seed `42` |
| phonopy version | 2.28.0 |
| `unit_conversion_factor` | `15.633302300230191` |

## Source

Loaded via `phonopy.load(unitcell=POSCAR, force_sets_filename=FORCE_SETS, ...)` from phonopy's upstream test fixtures:

- [`test/phonon/POSCAR_P222_1`](https://github.com/phonopy/phonopy/blob/develop/test/phonon/POSCAR_P222_1)
- [`test/phonon/FORCE_SETS_P222_1`](https://github.com/phonopy/phonopy/blob/develop/test/phonon/FORCE_SETS_P222_1)

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
