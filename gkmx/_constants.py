"""Pipeline defaults and unit conversions."""

from ase import units
from numpy import pi


window_factor = 1.0
default_filter_prominence = 0.05


AMU = units._amu
EV = units._e
AA = 1e-10  # m
PICO = 1e-12  # s
FEMTO = 1e-3 * PICO
THZ = 1 / PICO
BOLTZMANN = units._k

omega_to_THz = (EV / AA**2 / AMU) ** 0.5 / THZ / 2 / pi  # ~15.633  (gkmx w -> THz)
THz_to_cm = THZ / units._c / 100                          # ~33.356  (THz -> 1/cm)
gv_to_AA_fs = omega_to_THz / 1000                         # group velocity to A/fs
to_W_mK = EV * 1e25                                       # eV/fs/A -> W/mK
