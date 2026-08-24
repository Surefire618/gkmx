"""Freeze pair-vector digests from the direct N x N image search.

``lattice_points.get_pair_vectors`` reaches the same table by translation,
using the ``(N, N_p)`` smallest vectors the solver already holds. Proving that
needs the direct route once; running it in the suite would repeat an
``O(N^2 V)`` search on every invocation, which is the cost the helper exists to
remove (0.77 GB and 34 s at N = 2000).

So the direct route runs here, once, and the tests compare against the frozen
digests. A digest rather than the array itself: the full tables are ~12 MB
across these geometries, against ~13 MB for every other in-repo fixture
combined.

Run this only when the smallest-vector convention deliberately changes.

Output: tests/datasets/pair_vectors_reference.npz
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import io as ase_io

from gkmx.lattice_points import _average_images, get_smallest_vectors

HERE = Path(__file__).resolve().parent
GEOMETRIES = [
    "CuI_aiGK", "GaN", "Ga2O3", "KI_B2_MLIP",
    "phonopy_AgErTe2", "phonopy_Li2ZnGeO4", "phonopy_LiCdBO3",
    "phonopy_Mg2YbSb2", "phonopy_NaCl", "phonopy_RbIn3F10",
    "phonopy_Si", "phonopy_SrTiO3",
    "tdep_Ga2O3_kappa", "tdep_KI_bcc", "tdep_KPTe2", "tdep_Rb2O",
]


def digest(r0):
    """Three numbers that a mispaired gather cannot preserve.

    The weights are deliberately asymmetric in (I, J) — a symmetric weight
    would be blind to exactly the transposition this is guarding against.
    """
    n = r0.shape[0]
    i = np.arange(n, dtype=np.float64)
    w = np.cos(1.0 + 0.7 * i[:, None] + 0.13 * i[None, :])
    return np.einsum("IJ,IJa->a", w, r0, optimize=True)


def main():
    out = {}
    for name in GEOMETRIES:
        d = HERE / name
        sc = ase_io.read(str(d / "geometry.in.supercell"), format="aims")
        svec_frac, multi = get_smallest_vectors(sc, sc)
        r0 = _average_images(svec_frac, multi, sc.cell)
        out[f"{name}__digest"] = digest(r0)
        out[f"{name}__absmax"] = np.array(np.abs(r0).max())
        out[f"{name}__norm"] = np.array(np.linalg.norm(r0))
        print(f"{name:22s} N={len(sc):5d} absmax={np.abs(r0).max():.6f}")
    np.savez(HERE / "pair_vectors_reference.npz", **out)
    print(f"wrote {HERE / 'pair_vectors_reference.npz'}")


if __name__ == "__main__":
    main()
