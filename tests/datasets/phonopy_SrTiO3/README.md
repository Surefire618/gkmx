# phonopy reference — SrTiO3

Phonopy reference fixture consumed by
`tests/unit/test_phonopy_reference.py` so gkmx's `Phonon` solver can be
cross-checked against phonopy's parallel C kernel without importing
phonopy at test time.

## Metadata

| Field | Value |
|---|---|
| Material | SrTiO3 |
| Space group | 221 |
| Primitive symbols | `['O', 'O', 'O', 'Ti', 'Sr']` |
| `N_p` / `N_sc` | 5 / 135 |
| q-grid | 32 uniform-random in `(-0.5, 0.5)^3`, seed `42` |
| phonopy version | 2.28.0 |
| `unit_conversion_factor` | `15.633302300230191` |

## Source

Loaded via `phonopy.load(...)` from phonopy's upstream test fixture:

- [`test/phonopy_SrTiO3.yaml.xz`](https://github.com/phonopy/phonopy/blob/develop/test/phonopy_SrTiO3.yaml.xz)

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
