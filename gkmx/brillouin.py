"""Brillouin zone operations: q-grid symmetry reduction, symmetrization, BZ mesh."""

import collections

import numpy as np
import spglib
import xarray as xr

from . import keys


def _to_spglib_cell(atoms):
    return (atoms.cell[:], atoms.get_scaled_positions(), atoms.numbers)


def _get_symmetry_dataset(atoms, index_maps=False, symprec=1e-5):
    cell = _to_spglib_cell(atoms)
    spg_data = spglib.get_symmetry_dataset(cell, symprec=symprec)

    # Newer spglib returns an attribute-style dataset; normalize to dict.
    if hasattr(spg_data, "__dict__"):
        data = {k: v for k, v in spg_data.__dict__.items() if not k.startswith("_")}
    else:
        data = dict(spg_data)

    lat = np.asarray(atoms.cell)
    lat_recip = np.linalg.pinv(lat).T
    rots = data["rotations"]
    rots_cart = np.array([lat.T @ (r @ lat_recip) for r in rots]).round(decimals=15)
    data["rotations_cartesian"] = rots_cart

    trls = data["translations"]
    trls_cart = np.array([t @ lat for t in trls]).round(decimals=15)
    data["translations_cartesian"] = trls_cart

    if index_maps:
        data["index_maps"] = _compute_index_maps(atoms, rots_cart, trls_cart)

    return collections.namedtuple("spg_dataset", data.keys())(**data)


def _compute_index_maps(atoms, rotations_cart, translations_cart, tol=1e-5):
    # ``(x + tol) % 1.0`` shift keeps float noise at x=0 from aliasing to
    # x=1-eps in a different sort bucket (same trick as map_I_to_iL).
    pos = atoms.positions.copy()
    cell = atoms.cell[:]
    inv_cell = np.linalg.inv(cell.T)
    symbols = np.array(atoms.get_chemical_symbols())
    n_ops = len(rotations_cart)
    n_atoms = len(atoms)

    decimals = int(-np.log10(tol)) - 1
    ref_frac = (pos @ inv_cell.T + tol) % 1.0
    ref_keys = np.round(ref_frac, decimals=decimals)
    pos_to_idx = {tuple(k): i for i, k in enumerate(ref_keys)}

    index_maps = np.zeros([n_ops, n_atoms], dtype=int)
    for ii, (rc, tc) in enumerate(zip(rotations_cart, translations_cart)):
        transformed = (rc @ pos.T).T + tc
        frac = (transformed @ inv_cell.T + tol) % 1.0
        pos_keys = np.round(frac, decimals=decimals)
        for a, key in enumerate(pos_keys):
            J = pos_to_idx.get(tuple(key))
            if J is not None and symbols[J] == symbols[a]:
                index_maps[ii, J] = a

    return index_maps


def get_q_grid(q_points, primitive, symprec=1e-5, eps=1e-9):
    """Reduce ``q_points`` by primitive-cell symmetry; returns a q_grid namedtuple with ``.ir``, ``.map2ir``, ``.symop2ir``."""
    spg_data = _get_symmetry_dataset(primitive, index_maps=True, symprec=symprec)
    rotations = spg_data.rotations.swapaxes(1, 2)
    n_rot = len(rotations)
    rot_indices = np.arange(n_rot, dtype=int)

    ir_indices = []
    map2ir_indices = np.arange(len(q_points), dtype=int)
    symop2ir = np.zeros(len(q_points), dtype=int)

    q_norms = np.linalg.norm(q_points, axis=-1)

    for iq, q in enumerate(q_points):
        found = False
        for ir in ir_indices:
            if abs(q_norms[iq] - q_norms[ir]) > eps:
                continue
            rotated = rotations @ q
            diffs = np.square(rotated - q_points[ir]).sum(axis=-1)
            matches = rot_indices[diffs < eps]
            if len(matches) > 0:
                symop2ir[iq] = matches[0]
                map2ir_indices[iq] = ir
                found = True
                break
        if not found:
            ir_indices.append(iq)

    ir_arr, map2ir_points, ir_weights = np.unique(
        map2ir_indices, return_inverse=True, return_counts=True
    )

    q_cart = primitive.cell.reciprocal().cartesian_positions(q_points)

    IrGrid = collections.namedtuple("ir_grid", ["points", "points_cartesian", "weights", "map2full"])
    ir_grid = IrGrid(
        points=q_points[ir_arr],
        points_cartesian=q_cart[ir_arr],
        weights=ir_weights,
        map2full=map2ir_points,
    )

    QGrid = collections.namedtuple(
        "q_grid",
        ["points", "points_cartesian", "map2ir", "map2ir_points",
         "map2ir_indices", "spg_data", "symop2ir", "ir"],
    )
    return QGrid(
        points=q_points.copy(),
        points_cartesian=q_cart,
        map2ir=ir_arr,
        map2ir_points=map2ir_points,
        map2ir_indices=map2ir_indices,
        spg_data=spg_data,
        symop2ir=symop2ir,
        ir=ir_grid,
    )


def get_bz_mesh(mesh, atoms, monkhorst=True, reduced=False, q_points_scale=1.0, symprec=1e-5, eps=1e-9):
    """Regular ``(n1, n2, n3)`` BZ mesh, optionally symmetry-reduced."""
    mesh = np.asarray(mesh)
    cell = _to_spglib_cell(atoms)
    is_shift = -(mesh % 2) + 1 if monkhorst else np.zeros(3)

    mapping, grid = spglib.get_ir_reciprocal_mesh(mesh, cell=cell, is_shift=is_shift)
    points = grid.astype(float) / mesh + 0.5 * is_shift / mesh
    points = ((points + 0.5 + eps) % 1 - 0.5 - eps).round(decimals=14)
    points *= q_points_scale

    if not reduced:
        return get_q_grid(points, atoms)

    map2ir, weights = np.unique(mapping, return_counts=True)
    ir_points = points[map2ir]
    ir_points = ((ir_points + 0.5 + eps) % 1 - 0.5 - eps).round(decimals=14)
    ir_cart = atoms.cell.reciprocal().cartesian_positions(ir_points)

    IrMesh = collections.namedtuple("bz_ir_grid", ["points", "points_cartesian", "weights"])
    return IrMesh(points=ir_points, points_cartesian=ir_cart, weights=weights)


def get_symmetrized_array(array, map2ir, map2full, xarray=True):
    """Average ``array`` over symmetry-equivalent q-points."""
    dims = array.dims
    new_dims = list(dims)
    qi = new_dims.index(keys.q)
    new_dims.insert(0, new_dims.pop(qi))

    arr = array.transpose(*new_dims)
    arr_ir = np.zeros_like(np.asarray(arr)[map2ir])

    for ii, _ in enumerate(map2ir):
        mask = map2full == ii
        arr_ir[ii] = np.asarray(arr)[mask].mean(axis=0)

    name = (array.name or "") + f"_{keys.symmetrized}"
    result = xr.DataArray(arr_ir[map2full], dims=new_dims, name=name)
    result = result.transpose(*dims)

    if xarray:
        return result
    return result.data
