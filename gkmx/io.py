"""Force-constant file parsing and ASE Atoms <-> JSON."""

import json
from pathlib import Path

import numpy as np
from ase import Atoms


def _parse_force_constants_text(path):
    """Parse phonopy ``FORCE_CONSTANTS`` text (header ``N_p N_sc``, then ``i j`` + 3x3 blocks)."""
    with open(path, "r") as f:
        tokens = f.read().split()

    idx = 0
    N_p = int(tokens[idx]); idx += 1
    N_sc = int(tokens[idx]); idx += 1

    fc = np.zeros((N_p, N_sc, 3, 3), dtype=np.float64)

    for block in range(N_p * N_sc):
        i = int(tokens[idx]) - 1; idx += 1
        j = int(tokens[idx]) - 1; idx += 1
        block_vals = np.asarray(tokens[idx:idx + 9], dtype=np.float64).reshape(3, 3)
        idx += 9
        fc[i, j] = block_vals

    return fc


def _parse_force_constants_hdf5(path):
    import h5py
    with h5py.File(str(path), "r") as f:
        fc = np.asarray(f["force_constants"][...], dtype=np.float64)
    return fc


def parse_force_constants(fc_file, two_dim=False):
    """Load FC from phonopy text / fc2.hdf5 / flat .dat; returns ``(N_p, N_sc, 3, 3)`` (or 2D when ``two_dim=True``)."""
    path = Path(fc_file)
    name = path.name.lower()

    if ".dat" in name or "remapped" in name:
        fc = np.loadtxt(path)
        return fc

    if "hdf5" in name or name.endswith(".hdf5") or name.endswith(".h5"):
        fc = _parse_force_constants_hdf5(path)
    elif "force_constants" in name or name == "force_constants":
        fc = _parse_force_constants_text(path)
    else:
        raise ValueError(f"Unknown force constants file format: {path}")

    if two_dim:
        Np, Ns = fc.shape[:2]
        return fc.transpose(0, 2, 1, 3).reshape(3 * Np, 3 * Ns)

    return fc


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def atoms2json(atoms, reduce=False):
    """Serialize ASE Atoms to a JSON string that round-trips via ``json2atoms``."""
    d = {
        "numbers": atoms.numbers.tolist(),
        "positions": atoms.positions.tolist(),
        "cell": atoms.cell.tolist(),
        "pbc": atoms.pbc.tolist(),
    }
    if atoms.has("masses"):
        d["masses"] = atoms.get_masses().tolist()
    return json.dumps(d, cls=_NumpyEncoder)


def _expand_compressed(compressed):
    result = []
    for item in compressed:
        if isinstance(item, list) and len(item) == 2:
            count, val = item
            result.extend([val] * int(count))
        else:
            result.append(item)
    return result


def json2atoms(json_str):
    """Deserialize ASE Atoms; accepts both gkmx-flat and vibes run-length-compressed input."""
    d = json.loads(json_str)

    if "symbols" in d and isinstance(d["symbols"], list):
        syms = d["symbols"]
        if syms and isinstance(syms[0], list):
            d["symbols"] = _expand_compressed(syms)

    if "masses" in d and isinstance(d["masses"], list):
        masses = d["masses"]
        if masses and isinstance(masses[0], list):
            d["masses"] = _expand_compressed(masses)

    d.pop("velocities", None)
    d.pop("info", None)

    return Atoms(**d)
