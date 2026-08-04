"""Lattice point enumeration, supercell-primitive mapping, commensurate q-points."""

import collections

import numpy as np
import scipy.linalg as la
from ase.cell import Cell


def get_lattice_points(cell, supercell, extended=True, tolerance=1e-5, sort=True):
    """Lattice points of ``cell`` inside ``supercell``; with ``extended=True`` also returns boundary images and multiplicities."""
    scell = Cell(supercell)
    lattice = np.asarray(cell)
    sc_arr = np.asarray(scell)

    supercell_matrix = np.round(sc_arr @ la.inv(lattice)).astype(int)
    n_expected = int(round(abs(la.det(supercell_matrix))))

    M_abs = np.abs(supercell_matrix)
    bound = int(np.ceil(0.5 * M_abs.sum(axis=0).max())) + 1

    tol = tolerance

    rng = np.arange(-bound, bound + 1)
    ijk = np.stack(np.meshgrid(rng, rng, rng, indexing="ij"), axis=-1).reshape(-1, 3)
    all_pts = ijk @ lattice
    frac = scell.scaled_positions(all_pts)

    inside_mask = np.all(frac > -0.5 - tol, axis=1) & np.all(frac < 0.5 - tol, axis=1)
    lattice_points = all_pts[inside_mask]

    if extended:
        ext_mask = (
            np.all(frac > -0.5 - tol, axis=1)
            & np.all(frac < 0.5 + tol, axis=1)
            & ~inside_mask
        )
        lattice_points_ext = all_pts[ext_mask]

    assert len(np.unique(lattice_points, axis=0)) == n_expected, (
        f"Expected {n_expected} lattice points, found {len(lattice_points)}"
    )

    if sort:
        order = np.argsort(np.linalg.norm(lattice_points + [0, 2e-5, 4e-5], axis=1))
        lattice_points = lattice_points[order]

    lattice_points = np.round(lattice_points, decimals=16)

    if not extended:
        return lattice_points

    ext_with_mult = []
    if len(lattice_points_ext) > 0:
        frac_lps = scell.scaled_positions(lattice_points)
        frac_ext = scell.scaled_positions(lattice_points_ext)
        diffs = (frac_ext[None, :, :] - frac_lps[:, None, :] + tol) % 1 % 1 - tol
        matches = np.linalg.norm(diffs, axis=-1) < tol
        for i, lp in enumerate(lattice_points):
            group = [lp]
            for j in np.where(matches[i])[0]:
                group.append(lattice_points_ext[j])
            ext_with_mult.append(np.round(group, decimals=16))
    else:
        ext_with_mult = [np.round([lp], decimals=16) for lp in lattice_points]

    return lattice_points, ext_with_mult


def map_I_to_iL(primitive, supercell, lattice_points=None, tol=1e-5):
    """Map supercell atom index ``I`` to ``(primitive index i, lattice-point index L)``."""
    prim = primitive.copy()
    sc = supercell.copy()
    sc.wrap()
    # NB: do NOT call prim.wrap() here. For non-orthogonal primitive cells
    # (phonopy primitive_matrix + supercell_matrix workflow with p2s_map),
    # wrapping into the primitive parallelepiped moves basis atoms by a
    # primitive-lattice translation. The orbit still covers the same
    # supercell atoms modulo supercell periodicity, but the wrap-to-[0,1)
    # below would bucket some orbit points on different sides of the
    # boundary, breaking the sorted pairing.

    if lattice_points is None:
        lattice_points = get_lattice_points(prim.cell, sc.cell, extended=False)

    Np = len(prim)
    Nsc = len(sc)
    Nlp = len(lattice_points)

    prim_sym = np.array(prim.get_chemical_symbols())
    sc_sym = np.array(sc.get_chemical_symbols())

    target_cart = (prim.positions[:, None, :] + lattice_points[None, :, :]).reshape(-1, 3)
    target_sym = np.repeat(prim_sym, Nlp)
    target_iL = np.column_stack([
        np.repeat(np.arange(Np), Nlp),
        np.tile(np.arange(Nlp), Np),
    ])

    # ``(x + wrap_tol) % 1.0`` mirrors vibes_anisotropy: absorb fp noise at
    # the cell boundary so frac = -1e-16 wraps to ~0, not ~1 - 1e-16.
    inv_cell = np.linalg.inv(sc.cell[:].T)
    decimals = int(-np.log10(tol)) - 1
    wrap_tol = 1e-8  # fractional; well below any physical position spacing
    sc_frac = (sc.positions @ inv_cell.T + wrap_tol) % 1.0
    target_frac = (target_cart @ inv_cell.T + wrap_tol) % 1.0

    I2iL = np.empty((Nsc, 2), dtype=int)

    for s in np.unique(sc_sym):
        sc_mask = sc_sym == s
        tg_mask = target_sym == s
        sc_ids = np.where(sc_mask)[0]
        tg_ids = np.where(tg_mask)[0]

        sc_pos = sc_frac[sc_ids]
        tg_pos = target_frac[tg_ids]

        sc_key = np.round(sc_pos, decimals=decimals)
        tg_key = np.round(tg_pos, decimals=decimals)

        sc_rec = np.rec.fromarrays(sc_key.T, names="x,y,z", formats="f8,f8,f8")
        tg_rec = np.rec.fromarrays(tg_key.T, names="x,y,z", formats="f8,f8,f8")

        o_sc = np.argsort(sc_rec, order=("x", "y", "z"))
        o_tg = np.argsort(tg_rec, order=("x", "y", "z"))

        I2iL[sc_ids[o_sc]] = target_iL[tg_ids[o_tg]]

    iL2I = np.zeros((Np, Nlp), dtype=int)
    iL2I[I2iL[:, 0], I2iL[:, 1]] = np.arange(Nsc)
    return I2iL, iL2I.squeeze()


def get_commensurate_q_points(cell, supercell, fractional=False, tolerance=1e-5):
    """q-points commensurate with ``supercell``; pass ``fractional=True`` for primitive-reciprocal coords."""
    inv_lattice = la.inv(np.asarray(cell)).T
    inv_superlattice = la.inv(np.asarray(supercell)).T

    q_points = get_lattice_points(
        inv_superlattice, inv_lattice, extended=False, tolerance=tolerance
    )

    if fractional:
        q_frac = la.solve(inv_lattice.T, q_points.T).T
        return np.round(q_frac, decimals=14)

    return q_points


def get_smallest_vectors(primitive, supercell, tol=1e-5):
    """Shortest primitive->supercell vectors with phonopy's multi-image averaging; returns (svec_frac, multi)."""
    N_p = len(primitive)
    N_sc = len(supercell)

    A_prim = np.asarray(primitive.cell, dtype=np.float64)
    A_sc = np.asarray(supercell.cell, dtype=np.float64)
    r_prim = np.asarray(primitive.positions, dtype=np.float64)
    r_sc = np.asarray(supercell.positions, dtype=np.float64)

    r_base = r_sc[:, None, :] - r_prim[None, :, :]

    ns = np.array(
        [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
        dtype=np.float64,
    )
    lattice_cart = ns @ A_sc

    candidates = r_base[:, :, None, :] + lattice_cart[None, None, :, :]
    dist = np.linalg.norm(candidates, axis=-1)

    min_dist = dist.min(axis=-1, keepdims=True)
    mask = dist < (min_dist + tol)
    multi = mask.sum(axis=-1).astype(np.int32)
    V_max = int(multi.max())

    svec_cart = np.zeros((N_sc, N_p, V_max, 3), dtype=np.float64)
    for k in range(N_sc):
        for i in range(N_p):
            valid = np.where(mask[k, i])[0]
            svec_cart[k, i, : len(valid)] = candidates[k, i, valid]

    A_prim_inv = np.linalg.inv(A_prim)
    svec_frac = svec_cart @ A_prim_inv

    return svec_frac, multi


def get_s2p_map(primitive, supercell, lattice_points=None, tol=1e-5):
    """1D map from supercell index to primitive atom index."""
    I2iL, _ = map_I_to_iL(primitive, supercell, lattice_points=lattice_points, tol=tol)
    return I2iL[:, 0].astype(np.int64)


def get_p2s_map(primitive, supercell, lattice_points=None, tol=1e-5):
    """1D map from primitive atom ``i`` to its supercell index ``I`` at the origin.

    The inverse of ``get_s2p_map`` restricted to the origin lattice point
    (``R = 0``) — the atom itself, not one of its ``N_lattice`` periodic images.
    """
    if lattice_points is None:
        lattice_points = get_lattice_points(primitive.cell, supercell.cell,
                                            extended=False)
    I2iL, _ = map_I_to_iL(primitive, supercell, lattice_points=lattice_points, tol=tol)

    L_origin = int(np.argmin(np.linalg.norm(np.asarray(lattice_points), axis=1)))
    untranslated = I2iL[:, 1] == L_origin

    p2s = []
    for i in range(len(primitive)):
        I = np.flatnonzero(untranslated & (I2iL[:, 0] == i))
        if len(I) != 1:
            raise ValueError(
                f"get_p2s_map: primitive atom {i} has {len(I)} supercell atoms at "
                f"the origin lattice point, expected exactly 1.")
        p2s.append(int(I[0]))
    p2s = np.array(p2s, dtype=np.int64)

    # The origin image must sit ON the primitive atom. Counting is not enough:
    # map_I_to_iL returns an assignment for any geometry, so a displaced primitive
    # still yields exactly one match per atom.
    dr = np.asarray(supercell.positions)[p2s] - np.asarray(primitive.positions)
    dr_frac = dr @ np.linalg.inv(np.asarray(supercell.cell))
    mic_distance = np.linalg.norm(
        (dr_frac - np.rint(dr_frac)) @ np.asarray(supercell.cell), axis=1)
    if mic_distance.max() > tol:
        raise ValueError(
            f"get_p2s_map: primitive atom {int(mic_distance.argmax())} is "
            f"{mic_distance.max():.3e} A from its origin supercell atom (tol={tol}); "
            f"primitive and supercell are inconsistent.")
    return p2s


def get_unit_grid_extended(q_points_frac, tol=1e-9):
    """Wrap q-points into ``[0, 1]`` and replicate boundary points across the cube."""
    q_unit = (q_points_frac + tol) % 1 - tol

    map_ext = []
    ext_points = []
    for iq, q in enumerate(q_unit):
        if not (abs(q) < tol).any():
            continue
        mask = abs(q) < tol
        rge = np.where(mask, 2, 1)
        for ii, jj, kk in np.ndindex(*rge):
            if ii == jj == kk == 0:
                continue
            ext_points.append(q + np.array([ii, jj, kk]))
            map_ext.append(iq)

    q_extended = np.concatenate((q_unit, ext_points)) if ext_points else q_unit.copy()
    map_full = np.concatenate((np.arange(len(q_points_frac)), map_ext))

    Result = collections.namedtuple("unit_grid_extended", ("points", "points_extended", "map2extended"))
    return Result(q_unit, q_extended, map_full.astype(int))
