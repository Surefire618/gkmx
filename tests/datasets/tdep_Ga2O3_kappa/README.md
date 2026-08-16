# tdep_Ga2O3_kappa — kappa-Ga2O3, Pna2_1 (33), 40-atom primitive

TDEP second-order force constants and 300 K conductivity output from
`aiGK_Ga2O3_tdep_results/kappa_lowtemp_300/tdep_7_2.8_10_300_dumpgrid_333`:
cutoff 5.14 A / 53 neighbours, fitted at 315 K on 1001 configurations of a
320-atom cell. 3^3 grid, 120 bands.

    force_constants.npz                      (40, 320, 3, 3), eV/A^2
    outfile.thermal_conductivity             SMA kappa, W/m/K
    outfile.grid_thermal_conductivity.hdf5   mode-resolved

Older TDEP format: the text file is the bare 10-column layout
`T kxx kyy kzz kxz kyz kxy kzx kzy kyx` -- kxz *before* kxy -- and holds the
iterative kappa, while the grid file's `lifetimes` are the RTA ones. The grid
file lacks `qpoints_irreducible`, `velocity_offdiagonal(_im)` and
`n_invariant_operations`, so `_generate_tdep_references.py` cannot build a
`reference.npz` from it.

Eigenvectors are not stored: fixed only up to a per-mode phase and, in a
degenerate multiplet, up to a unitary.
