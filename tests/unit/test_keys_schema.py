"""Pin every ``gkmx.keys`` string value and assert all values are unique."""

import pytest

from gkmx import keys

# Pinned field-name strings. Renaming any of these is a schema change for
# cached DMX files (``DynamicalMatrix.nc``), GK output datasets
# (``greenkubo.nc``), and external consumers (gkx, analysis scripts).
EXPECTED_VALUES = {
    # Dim labels
    "i": "i", "j": "j", "I": "I", "J": "J", "a": "a", "b": "b",
    "ia": "ia", "s": "s", "q": "q", "q_ir": "q_ir", "q_int": "q_int",
    "time": "time", "time_md": "time_md",
    # FC schema
    "fc": "force_constants",
    "fc_remapped": "force_constants_remapped",
    "fc_phonopy": "force_constants_phonopy",
    # Reference structures
    "reference_atoms": "atoms_reference",
    "reference_primitive": "atoms_primitive",
    "reference_supercell": "atoms_supercell",
    "map_supercell_to_primitive": "map_supercell_to_primitive",
    # Trajectory fields
    "volume": "volume",
    "positions": "positions",
    "displacements": "displacements",
    "velocities": "velocities",
    "momenta": "momenta",
    "forces": "forces",
    "cell": "cell",
    "energy_potential": "energy_potential",
    "energy_kinetic": "energy_kinetic",
    "stress_potential": "stress_potential",
    "stress_kinetic": "stress_kinetic",
    "stress": "stress",
    "pressure": "pressure",
    "pressure_kinetic": "pressure_kinetic",
    "pressure_potential": "pressure_potential",
    "temperature": "temperature",
    # Heat flux
    "heat_flux": "heat_flux",
    "heat_flux_aux": "heat_flux_aux",
    # GK analysis
    "gk_prefactor": "gk_prefactor",
    "gk_window_fs": "gk_window_fs",
    "filter_prominence": "filter_prominence",
    "time_cutoff": "cutoff_time",
    "sigma": "sigma",
    "sigma_per_sample": "sigma_per_sample",
    # Mode-resolved
    "heat_capacity": "heat_capacity",
    "mode_heat_capacity": "mode_heat_capacity",
    "mode_lifetime": "mode_lifetime",
    # Velocities
    "v_qsa_cartesian": "v_qsa_cartesian",
    "v_qssa_cartesian": "v_qssa_cartesian",
    # Kappa
    "kappa": "thermal_conductivity",
    "kappa_symmetrized": "thermal_conductivity_symmetrized",
    "kappa_ha": "thermal_conductivity_harmonic",
    "kappa_ha_symmetrized": "thermal_conductivity_harmonic_symmetrized",
    "kappa_QHGK": "thermal_conductivity_QHGK",
    "kappa_corrected": "thermal_conductivity_corrected",
    "kappa_ha_QHGK": "thermal_conductivity_harmonic_QHGK",
    "mode_kappa_BTE": "mode_thermal_conductivity_BTE",
    "mode_kappa_QHGK": "mode_thermal_conductivity_QHGK",
    # HFACF
    "hf_acf": "heat_flux_acf",
    "hf_acf_filtered": "heat_flux_acf_filtered",
    "hf_acf_BTE": "heat_flux_BTE_acf",
    "hf_acf_BTE_integral": "heat_flux_BTE_acf_integral",
    "hf_acf_QHGK": "heat_flux_QHGK_acf",
    "hf_acf_QHGK_integral": "heat_flux_QHGK_acf_integral",
    # Measured harmonic fluxes (--harmonic-flux). These names are shared with
    # FHI-vibes; changing one silently breaks a cross-code comparison.
    "heat_flux_harmonic": "heat_flux_harmonic",
    "heat_flux_harmonic_q": "heat_flux_harmonic_q",
    "heat_flux_QHGK_ta": "heat_flux_QHGK_ta",
    "hf_acf_ha": "heat_flux_harmonic_acf",
    "hf_acf_ha_integral": "heat_flux_harmonic_acf_integral",
    "hf_acf_ha_q": "heat_flux_harmonic_q_acf",
    "hf_acf_ha_q_integral": "heat_flux_harmonic_q_acf_integral",
    "hf_acf_qhgk_ta": "heat_flux_QHGK_ta_acf",
    "hf_acf_qhgk_ta_integral": "heat_flux_QHGK_ta_acf_integral",
    "kappa_cumulative": "heat_flux_acf_integral",
    "kappa_cumulative_filtered": "heat_flux_acf_integral_filtered",
    # Standalone tags read by callers (brillouin.py, greenkubo.py).
    "symmetrized": "symmetrized",
    "integral": "integral",
    # DMX cache schema
    "q_points": "q_points",
    "w_qs": "w_qs",
    "w_inv_qs": "w_inv_qs",
    "w2_qs": "w2_qs",
    "v_qsa": "v_qsa",
    "v_qssa": "v_qssa",
    "e_qsi": "e_qsi",
    "D_qij": "D_qij",
    "q_map2ir": "q_map2ir",
    "q_map_ir2full": "q_map_ir2full",
    # Interpolation outputs
    "interpolation_fit_slope": "interpolation_fit_slope",
    "interpolation_fit_intercept": "interpolation_fit_intercept",
    "interpolation_fit_stderr": "interpolation_fit_stderr",
    "interpolation_correction": "interpolation_correction",
    "interpolation_correction_ab": "interpolation_correction_ab",
    "interpolation_correction_ab_stderr": "interpolation_correction_ab_stderr",
    "interpolation_correction_factor": "interpolation_correction_factor",
    "interpolation_correction_factor_err": "interpolation_correction_factor_err",
    "interpolation_kappa_array": "interpolation_kappa_array",
    "interpolation_q_points": "interpolation_q_points",
    "interpolation_w_qs": "interpolation_w_qs",
    "interpolation_tau_qs": "interpolation_tau_qs",
    "interpolation_v_qsa": "interpolation_v_qsa",
    "interpolation_kappa_array_QHGK": "interpolation_kappa_array_QHGK",
}


@pytest.mark.parametrize("name,expected", sorted(EXPECTED_VALUES.items()))
def test_key_value_pinned(name, expected):
    assert getattr(keys, name) == expected


def test_string_keys_are_unique():
    """No two public string keys may share a value (catches MF1-style aliasing)."""
    seen = {}
    for name in dir(keys):
        if name.startswith("_"):
            continue
        val = getattr(keys, name)
        if not isinstance(val, str):
            continue
        if val in seen:
            pytest.fail(
                f"keys.{name} = {val!r} duplicates keys.{seen[val]} — "
                f"two distinct constants must not share a string value"
            )
        seen[val] = name
