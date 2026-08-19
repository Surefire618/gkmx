"""Public dataset field-name and dim-label constants emitted by the gkmx pipeline."""


def _join(*parts):
    return "_".join(p for p in parts if p)


# Dimension labels and tuples (xarray axis names).
i, j, I, J, a, b = "i", "j", "I", "J", "a", "b"
ia = "ia"  # composite (atom, cartesian) flat axis used by _get_arrays
s, q, q_ir, q_int = "s", "q", "q_ir", "q_int"
q_s = (q, s)
q_a = (q, a)
q_s_a = (q, s, a)
q_s_i = (q, s, i)
q_s_s_a = (q, s, s + "p", a)
q_i_j = (q, i, j)

tensor = (a, b)

time = "time"
time_md = "time_md"  # full per-MD-step axis (sigma_per_sample, etc.)
time_vec = (time, a)
time_atom_vec = (time, I, a)
time_tensor = (time, a, b)

dim_fc = (i, J, a, b)
dim_fc_remapped = ("Ia", "Jb")
# Cache-specific FC dim tuple — uses ("p", "I") labels so the cached
# Dataset can coexist with e_qsi/D_qij sharing the "i" label.
dim_fc_phonopy = ("p", "I", a, b)


# Internal join-tokens (composed into field names below; not read externally).
_aux = "aux"
_remapped = "remapped"
_reference = "reference"
_filtered = "filtered"
_harmonic = "harmonic"
_corrected = "corrected"
_QHGK = "QHGK"
_acf = "acf"
_interpolation = "interpolation"

# Public tags read by callers.
symmetrized = "symmetrized"
integral = "integral"

fc = "force_constants"
fc_remapped = _join(fc, _remapped)

reference_atoms = _join("atoms", _reference)
reference_primitive = "atoms_primitive"
reference_supercell = "atoms_supercell"
map_supercell_to_primitive = "map_supercell_to_primitive"

volume = "volume"
positions = "positions"
displacements = "displacements"
velocities = "velocities"
momenta = "momenta"
forces = "forces"
cell = "cell"
energy_potential = "energy_potential"
energy_kinetic = "energy_kinetic"
stress_potential = "stress_potential"
stress_kinetic = "stress_kinetic"
stress = "stress"
pressure = "pressure"
pressure_kinetic = "pressure_kinetic"
pressure_potential = "pressure_potential"
temperature = "temperature"

heat_flux = "heat_flux"
heat_flux_aux = _join(heat_flux, _aux)

gk_prefactor = "gk_prefactor"
gk_window_fs = "gk_window_fs"
filter_prominence = "filter_prominence"

heat_capacity = "heat_capacity"
mode_heat_capacity = _join("mode", heat_capacity)
mode_lifetime = "mode_lifetime"
mode_lifetime_symmetrized = _join(mode_lifetime, symmetrized)

kappa = "thermal_conductivity"
kappa_symmetrized = _join(kappa, symmetrized)
kappa_ha = _join(kappa, _harmonic)
kappa_ha_symmetrized = _join(kappa_ha, symmetrized)
kappa_QHGK = _join(kappa, _QHGK)
kappa_corrected = _join(kappa, _corrected)
_mode_kappa = _join("mode", kappa)
mode_kappa_BTE = _join(_mode_kappa, "BTE")
mode_kappa_QHGK = _join(_mode_kappa, _QHGK)

v_qsa_cartesian = "v_qsa_cartesian"
v_qssa_cartesian = "v_qssa_cartesian"

# Dataset variable names (shared by the DMX HDF5 cache and the GK output dataset).
q_points = "q_points"
fc_phonopy = "force_constants_phonopy"
w_qs = "w_qs"
w_inv_qs = "w_inv_qs"
w2_qs = "w2_qs"
v_qsa = "v_qsa"
v_qssa = "v_qssa"  # base; complex stored as v_qssa_re / v_qssa_im
e_qsi = "e_qsi"    # base; complex stored as e_qsi_re   / e_qsi_im
D_qij = "D_qij"    # base; complex stored as D_qij_re   / D_qij_im
q_map2ir = "q_map2ir"
q_map_ir2full = "q_map_ir2full"

time_cutoff = "cutoff_time"

sigma = "sigma"
sigma_per_sample = _join(sigma, "per_sample")

hf_acf = _join(heat_flux, _acf)
hf_acf_filtered = _join(hf_acf, _filtered)
kappa_cumulative = _join(hf_acf, integral)
kappa_cumulative_filtered = _join(kappa_cumulative, _filtered)

hf_acf_BTE = _join(heat_flux, "BTE", _acf)
hf_acf_BTE_integral = _join(hf_acf_BTE, integral)
hf_acf_QHGK = _join(heat_flux, _QHGK, _acf)
hf_acf_QHGK_integral = _join(hf_acf_QHGK, integral)

_interpolation_fit = _join(_interpolation, "fit")
interpolation_fit_slope = _join(_interpolation_fit, "slope")
interpolation_fit_intercept = _join(_interpolation_fit, "intercept")
interpolation_fit_stderr = _join(_interpolation_fit, "stderr")
interpolation_correction = _join(_interpolation, "correction")
interpolation_correction_ab = _join(interpolation_correction, "ab")
interpolation_correction_ab_stderr = _join(interpolation_correction_ab, "stderr")
interpolation_correction_factor = _join(interpolation_correction, "factor")
interpolation_correction_factor_err = _join(interpolation_correction_factor, "err")
interpolation_kappa_array = _join(_interpolation, "kappa_array")
interpolation_q_points = _join(_interpolation, "q_points")
interpolation_w_qs = _join(_interpolation, "w_qs")
interpolation_tau_qs = _join(_interpolation, "tau_qs")
interpolation_v_qsa = _join(_interpolation, "v_qsa")

kappa_ha_QHGK = _join(kappa_ha, _QHGK)
interpolation_kappa_array_QHGK = _join(interpolation_kappa_array, _QHGK)
