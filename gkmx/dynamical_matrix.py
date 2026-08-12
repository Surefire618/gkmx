"""MD-workflow adapter around `Phonon` with commensurate-grid setup and HDF5 cache."""

import numpy as np
import xarray as xr
from ase.geometry import get_distances

from . import keys
from .brillouin import get_bz_mesh, get_q_grid
from .io import atoms2json, json2atoms
from .lattice_points import (
    get_commensurate_q_points,
    get_lattice_points,
    get_s2p_map,
    map_I_to_iL,
)
from .phonon import Phonon, Solution, SolutionWithGVM
from .precision import Precision


def _remap_force_constants(fc, primitive, supercell, new_supercell=None,
                           symmetrize=True, tol=1e-5, eps=1e-13):
    """Remap FC from ``(N_p, N_sc, 3, 3)`` to ``(N_sc, N_sc, 3, 3)``."""
    if new_supercell is None:
        new_supercell = supercell.copy()

    prim = primitive.copy()
    prim_cell = prim.cell.copy()
    prim.cell = supercell.cell
    prim.wrap(eps=tol)
    sc = supercell.copy()
    sc.wrap(eps=tol)

    n_new = len(new_supercell)

    sc_r = np.zeros((fc.shape[0], fc.shape[1], 3))
    for aa, a1 in enumerate(prim):
        diff = sc.positions - a1.position
        p2s = np.where(np.linalg.norm(diff, axis=1) < tol)[0][0]
        sc_r[aa], _ = get_distances([sc.positions[p2s]], sc.positions,
                                     cell=sc.cell, pbc=True)

    prim.cell = prim_cell
    sc2pc = np.asarray(get_s2p_map(prim, new_supercell, tol=tol))

    inv_cell = np.linalg.inv(new_supercell.get_cell(complete=True).T)
    positions = new_supercell.positions
    decimals = int(-np.log10(tol)) - 1

    scale = 10 ** decimals
    ref_frac = (positions @ inv_cell.T + eps) % 1.0
    ref_keys = np.round(ref_frac * scale).astype(np.int64)

    offsets = sc_r[sc2pc]
    targets = positions[:, None, :] + offsets
    targ_frac = (targets @ inv_cell.T + eps) % 1.0
    targ_keys = np.round(targ_frac * scale).astype(np.int64)
    flat_keys = targ_keys.reshape(-1, 3)

    # Pack (x, y, z) as one int64 so sort+searchsorted handles the whole
    # N^2 batch (with tol=1e-5 each coord lives in ~20 bits — fits 64).
    K = int(scale) + 1
    ref_enc = ref_keys[:, 0] * K * K + ref_keys[:, 1] * K + ref_keys[:, 2]
    flat_enc = flat_keys[:, 0] * K * K + flat_keys[:, 1] * K + flat_keys[:, 2]

    order = np.argsort(ref_enc)
    ref_sorted = ref_enc[order]
    pos = np.searchsorted(ref_sorted, flat_enc)
    pos_clipped = np.clip(pos, 0, len(ref_sorted) - 1)
    matched = ref_sorted[pos_clipped] == flat_enc
    a2_idx = np.where(matched, order[pos_clipped], -1).reshape(n_new, n_new)

    fc_out = np.zeros((n_new, n_new, 3, 3), dtype=fc.dtype)
    valid = a2_idx >= 0
    a1_arr = np.repeat(np.arange(n_new), n_new).reshape(n_new, n_new)
    sc_a2_arr = np.tile(np.arange(n_new), n_new).reshape(n_new, n_new)
    uc_idx_arr = sc2pc[a1_arr]

    a1_valid = a1_arr[valid]
    a2_valid = a2_idx[valid]
    uc_valid = uc_idx_arr[valid]
    sc_a2_valid = sc_a2_arr[valid]

    np.add.at(fc_out, (a1_valid, a2_valid), fc[uc_valid, sc_a2_valid])

    fc_2d = fc_out.swapaxes(1, 2).reshape(3 * n_new, 3 * n_new)
    if symmetrize:
        fc_2d = 0.5 * (fc_2d + fc_2d.T)

    return fc_2d.reshape(n_new, 3, n_new, 3).swapaxes(1, 2)


def get_full_solution_from_ir(q_grid, ir_solution, with_group_velocity_matrices=False):
    """Expand an irreducible-grid ``Solution`` to the full grid by symmetry."""
    Nq = len(q_grid.points)
    Ns = ir_solution.w_qs.shape[1]
    Np = Ns // 3
    real_dtype = ir_solution.w_qs.dtype
    complex_dtype = ir_solution.e_qsi.dtype

    ip = q_grid.map2ir_points
    ir = q_grid.symop2ir
    rot_per_q = q_grid.spg_data.rotations_cartesian[ir].astype(real_dtype)
    index_maps_per_q = q_grid.spg_data.index_maps[ir]

    perm_per_q = np.eye(Np, dtype=real_dtype)[index_maps_per_q]

    G_per_q = (perm_per_q[:, :, None, :, None]
               * rot_per_q[:, None, :, None, :]
               ).reshape(Nq, Ns, Ns).astype(complex_dtype)

    w_qs     = ir_solution.w_qs[ip]
    w_inv_qs = ir_solution.w_inv_qs[ip]
    v_qsa_ir = ir_solution.v_qsa_cartesian[ip]
    e_qsi_ir = ir_solution.e_qsi[ip]
    D_qij_ir = ir_solution.D_qij[ip]

    v_qsa = np.einsum("qsa,qba->qsb", v_qsa_ir, rot_per_q)
    e_qsi = np.einsum("qsi,qji->qsj", e_qsi_ir, G_per_q)
    D_qij = (G_per_q @ D_qij_ir) @ np.swapaxes(G_per_q.conj(), -1, -2)

    if with_group_velocity_matrices:
        v_qssa_ir = ir_solution.v_qssa_cartesian[ip]
        v_qssa = np.einsum("qsta,qba->qstb", v_qssa_ir, rot_per_q)
    else:
        v_qssa = None

    w2_qs = np.sign(w_qs) * w_qs ** 2

    if with_group_velocity_matrices:
        return SolutionWithGVM(w_qs, w_inv_qs, w2_qs, v_qsa, v_qssa, e_qsi, D_qij)
    return Solution(w_qs, w_inv_qs, w2_qs, v_qsa, e_qsi, D_qij)


class DynamicalMatrix:
    """MD-workflow adapter around ``Phonon``.

    At construction, solves on the commensurate q-grid implied by the
    primitive/supercell pair and caches:
        - the ``Solution`` (frequencies, eigenvectors, group velocities)
        - the supercell-basis projection matrix ``e_qsI`` (built lazily)
        - the remapped FC ``(3*N_sc, 3*N_sc)`` for harmonic-force
          predictions (built lazily)

    Persisted to / loaded from HDF5 via ``to_hdf5`` / ``from_hdf5``.

    Use this when running the Green-Kubo pipeline. Use the plain
    ``Phonon`` class instead if you just want batched eigensolves without
    the MD machinery (xarray / h5py / spglib).
    """

    def __init__(self, force_constants, primitive, supercell,
                 with_group_velocity_matrices=False, backend="numpy",
                 precision="fp64", enforce_translational_invariance=True,
                 convention="PHONOPY"):
        """Build the adapter and solve on the commensurate q-grid.

        Args:
            force_constants: phonopy-shape ``(N_p, N_sc, 3, 3)`` array,
                not mass-weighted. Raises ``ValueError`` on any other
                shape.
            primitive: ASE Atoms — the primitive cell.
            supercell: ASE Atoms — the supercell (must be commensurate
                with the primitive lattice).
            with_group_velocity_matrices: solve with the QHGK
                off-diagonal ``v_qssa_cartesian`` included. Required for
                QHGK kappa; small extra cost.
            backend: ``"numpy"`` (default) or ``"jax"``. Forwarded to
                the internal ``Phonon``.
            precision: ``"fp64"`` (default) or ``"fp32"``. Drives the
                dtype of the cached solution and every lazily-built
                derived array.
            enforce_translational_invariance: impose the acoustic sum rule
                ``sum_B Phi[i][B] = 0`` by removing the residual from the
                origin block. Default ``True``; warns when the residual is a
                large fraction of ``max|Phi|``. Turning it off keeps the force
                constants exactly as supplied, at the cost of Gamma acoustics
                that do not sit at zero and a ``1/w`` that leaks into every
                ``1/w``-weighted mode sum.
            convention: ``"PHONOPY"`` (default) or ``"TDEP"``. Same frequencies
                and diagonal group velocity either way; eigenvectors and
                ``v_qssa`` off the diagonal differ. See
                ``gkmx.phonon.CONVENTIONS``.
        """
        self.primitive = primitive.copy()
        self.supercell = supercell.copy()

        Np, Na = len(primitive), len(supercell)
        fc = np.asarray(force_constants)
        if fc.shape != (Np, Na, 3, 3):
            raise ValueError(
                f"force_constants must be in phonopy shape (N_p={Np}, N_sc={Na}, 3, 3); "
                f"got {fc.shape}. Use gkmx.io.parse_force_constants(..., two_dim=False) "
                "or reshape yourself before calling DynamicalMatrix."
            )

        self._fc_phonopy = fc.copy()
        self._backend = backend
        self._precision = precision
        p = Precision.from_str(precision)
        self._dtype_real, self._dtype_complex = p.real, p.complex

        self._convention = convention
        self._setup_lattice_and_grid(primitive, supercell)

        self._phonon = Phonon(
            force_constants=self._fc_phonopy,
            primitive=self.primitive,
            supercell=self.supercell,
            backend=backend,
            precision=precision,
            enforce_translational_invariance=enforce_translational_invariance,
            convention=convention,
        )
        self._solution = self._phonon.solve(
            q_points_frac=self._q_grid.points,
            with_velocities=True,
            with_group_velocity_matrices=with_group_velocity_matrices,
        )

        self._e_qsI = None
        self._remapped_3N_cache = None

    def _setup_lattice_and_grid(self, primitive, supercell, q_points=None):
        lps = get_lattice_points(primitive.cell, supercell.cell, extended=False)
        self._lattice_points = lps
        self._I2iL, self._iL2I = map_I_to_iL(primitive, supercell, lattice_points=lps)
        if q_points is None:
            q_points = get_commensurate_q_points(
                primitive.cell, supercell.cell, fractional=True)
        self._q_grid = get_q_grid(q_points, primitive=primitive)

    def _build_supercell_positions(self):
        positions = np.zeros((len(self.supercell), 3), dtype=self._dtype_real)
        prim_pos = self.primitive.positions
        for I in range(len(self.supercell)):
            i, L = self._I2iL[I]
            positions[I] = prim_pos[i] + self._lattice_points[L]
        return positions

    def _build_e_qsI(self):
        """``e_qsI = (1/sqrt(Nq)) * exp(2*pi*i q·R_k) * e_qsi[..., 3*i(k)+a]``."""
        Rs = self._build_supercell_positions()
        q_cart = np.asarray(self.q_points_cartesian,
                            dtype=self._dtype_real)
        Nq = len(q_cart)
        p_indices = self._I2iL[:, 0]
        i_of_I = np.concatenate([np.arange(3) + 3 * ii for ii in p_indices])
        e_qsI = self._solution.e_qsi[:, :, i_of_I]
        # `2j * np.pi` upcasts to complex128 unless cast to _dtype_complex.
        two_pi_j = self._dtype_complex(2j * np.pi)
        q_dot_R = (q_cart[:, None, :] * Rs[None, :, :]).sum(axis=-1)
        phases_qk = np.exp(two_pi_j * q_dot_R.astype(self._dtype_complex))
        phases_qI = phases_qk.repeat(3, axis=1)
        inv_sqrt_Nq = self._dtype_real(Nq ** -0.5)
        e_qsI = inv_sqrt_Nq * phases_qI[:, None, :] * e_qsI
        self._e_qsI = np.ascontiguousarray(e_qsI)

    # Array-valued @property below return `.copy()` so callers cannot mutate
    # the cached solution; hot loops should bind to a local once.

    @property
    def fc_phonopy(self):
        """``(N_p, N_sc, 3, 3)`` input force constants (phonopy shape, copy)."""
        return self._fc_phonopy.copy()

    @property
    def remapped(self):
        """``(3*N_sc, 3*N_sc)`` non-mass-weighted FC, ``(I, a) -> 3*I + a``.

        Built lazily from ``fc_phonopy`` via supercell remapping; cached
        on first access. Used by the harmonic-force residual diagnostic
        (``f_ha = -disp @ remapped`` in the Green-Kubo pipeline).
        """
        if self._remapped_3N_cache is not None:
            return self._remapped_3N_cache.copy()
        fc_phonopy = np.asarray(self._fc_phonopy, dtype=self._dtype_real)
        fc_remap = _remap_force_constants(fc_phonopy, self.primitive, self.supercell)
        N = len(self.supercell)
        self._remapped_3N_cache = fc_remap.swapaxes(1, 2).reshape(3 * N, 3 * N)
        return self._remapped_3N_cache.copy()

    @property
    def I2iL_map(self):
        """``(N_sc, 2)`` int array mapping supercell atom index ``I``
        to ``(primitive atom index i, lattice-point index L)``."""
        return self._I2iL.copy()

    @property
    def q_grid(self):
        """The commensurate ``q_grid`` namedtuple from ``gkmx.brillouin.get_q_grid``."""
        return self._q_grid

    @property
    def q_points(self):
        """``(Nq, 3)`` commensurate q-points in primitive reciprocal-fractional coords."""
        return self._q_grid.points

    @property
    def q_points_cartesian(self):
        """``(Nq, 3)`` commensurate q-points in Cartesian (1/Å) coords."""
        return self._q_grid.points_cartesian

    @property
    def solution(self):
        """The cached ``Solution`` / ``SolutionWithGVM`` from the construction-time solve."""
        return self._solution

    @property
    def w_qs(self):
        return self._solution.w_qs.copy()

    @property
    def w2_qs(self):
        return self._solution.w2_qs.copy()

    @property
    def w_inv_qs(self):
        return self._solution.w_inv_qs.copy()

    @property
    def e_qsi(self):
        return self._solution.e_qsi.copy()

    @property
    def e_qsI(self):
        """``(Nq, Ns, 3*N_sc)`` mode-projection matrix; built lazily."""
        if self._e_qsI is None:
            self._build_e_qsI()
        return self._e_qsI.copy()

    @property
    def v_qsa_cartesian(self):
        return self._solution.v_qsa_cartesian

    def __repr__(self):
        return f"DynamicalMatrix(fc_shape={self._fc_phonopy.shape}, n_q={len(self.q_points)})"

    def _ensure_phonon(self):
        if self._phonon is None:
            self._phonon = Phonon(
                force_constants=self._fc_phonopy,
                primitive=self.primitive,
                supercell=self.supercell,
                backend=self._backend,
                precision=self._precision,
                convention=self._convention,
            )
        return self._phonon

    def get_solution(self, q_points, with_group_velocity_matrices=False):
        """Solve at arbitrary q-points via the cached ``Phonon`` solver.

        Args:
            q_points: ``(Nq, 3)`` array of q-points in primitive
                reciprocal-fractional coordinates. Need not be
                commensurate with the supercell.
            with_group_velocity_matrices: include the QHGK off-diagonal
                ``v_qssa_cartesian`` in the returned solution.

        Returns:
            A ``Solution`` (or ``SolutionWithGVM``). See ``Phonon.solve``.
        """
        return self._ensure_phonon().solve(
            q_points_frac=np.asarray(q_points, dtype=self._dtype_real),
            with_velocities=True,
            with_group_velocity_matrices=with_group_velocity_matrices,
        )

    def get_mesh_and_solution(self, mesh, reduced=True, monkhorst=True,
                              q_points_scale=1.0, **kwargs):
        """Build a regular BZ mesh and solve on it.

        Args:
            mesh: ``(n1, n2, n3)`` mesh density.
            reduced: solve on the irreducible wedge and expand by
                symmetry (default). False = solve on the full mesh.
            monkhorst: shift by half a mesh step on even axes (default).
            q_points_scale: rescaling for the mesh-fractional coords;
                pass < 1 to sample a sub-box of the BZ.
            **kwargs: forwarded to ``get_solution`` (e.g.
                ``with_group_velocity_matrices=True``).

        Returns:
            ``(q_grid, Solution)`` — the grid namedtuple from
            ``gkmx.brillouin.get_bz_mesh`` and the corresponding
            ``Solution`` / ``SolutionWithGVM``.
        """
        kw_mesh = {"atoms": self.primitive, "monkhorst": monkhorst,
                   "reduced": reduced, "q_points_scale": q_points_scale}
        q_grid = get_bz_mesh(mesh=mesh, **kw_mesh)

        if reduced:
            sol = self.get_solution(q_points=q_grid.points, **kwargs)
        else:
            ir_sol = self.get_solution(q_points=q_grid.ir.points, **kwargs)
            sol = get_full_solution_from_ir(q_grid, ir_sol, **kwargs)

        return q_grid, sol

    def _get_arrays(self):
        qsi = (keys.q, keys.s, keys.ia)
        q_points = np.asarray(self.q_grid.points, dtype=self._dtype_real)
        arrays = [
            xr.DataArray(self.v_qsa_cartesian, dims=keys.q_s_a, name=keys.v_qsa_cartesian),
            xr.DataArray(self.w_qs, dims=keys.q_s, name=keys.w_qs),
            xr.DataArray(self.w_inv_qs, dims=keys.q_s, name=keys.w_inv_qs),
            xr.DataArray(self.e_qsi.real, dims=qsi, name=f"{keys.e_qsi}_re"),
            xr.DataArray(self.e_qsi.imag, dims=qsi, name=f"{keys.e_qsi}_im"),
            xr.DataArray(q_points, dims=keys.q_a, name=keys.q_points),
            xr.DataArray(self.q_grid.map2ir, dims=(keys.q_ir,), name=keys.q_map2ir),
            xr.DataArray(self.q_grid.ir.map2full, dims=(keys.q,), name=keys.q_map_ir2full),
        ]
        return arrays

    def to_hdf5(self, filename, include_D_qij=False,
                include_group_velocity_matrices=False, complevel=3):
        """Persist the solution to HDF5 (via the h5netcdf engine).

        Args:
            filename: output path. Conventional extension ``.nc``.
            include_D_qij: also store the mass-weighted dynamical matrix
                ``D_qij`` (large; useful for debugging, not for the GK
                pipeline).
            include_group_velocity_matrices: also store the QHGK
                off-diagonal ``v_qssa_cartesian``. Required if the cache
                will drive a QHGK kappa computation.
            complevel: zlib compression level (0–9, default 3).

        Returns:
            The output ``filename`` (for chaining).
        """
        ds = xr.Dataset()
        ds.attrs["reference_primitive_json"] = atoms2json(self.primitive)
        ds.attrs["reference_supercell_json"] = atoms2json(self.supercell)
        ds.attrs["precision"] = self._precision

        ds[keys.q_points] = xr.DataArray(self.q_grid.points, dims=keys.q_a)
        ds[keys.fc_phonopy] = xr.DataArray(self._fc_phonopy, dims=keys.dim_fc_phonopy)

        sol = self._solution
        ds[keys.w_qs] = xr.DataArray(sol.w_qs, dims=keys.q_s)
        ds[keys.w_inv_qs] = xr.DataArray(sol.w_inv_qs, dims=keys.q_s)
        ds[keys.w2_qs] = xr.DataArray(sol.w2_qs, dims=keys.q_s)
        ds[keys.v_qsa] = xr.DataArray(sol.v_qsa_cartesian, dims=keys.q_s_a)

        if include_group_velocity_matrices and isinstance(sol, SolutionWithGVM):
            ds[f"{keys.v_qssa}_re"] = xr.DataArray(sol.v_qssa_cartesian.real, dims=keys.q_s_s_a)
            ds[f"{keys.v_qssa}_im"] = xr.DataArray(sol.v_qssa_cartesian.imag, dims=keys.q_s_s_a)

        if sol.e_qsi is not None:
            ds[f"{keys.e_qsi}_re"] = xr.DataArray(sol.e_qsi.real, dims=keys.q_s_i)
            ds[f"{keys.e_qsi}_im"] = xr.DataArray(sol.e_qsi.imag, dims=keys.q_s_i)

        if include_D_qij and sol.D_qij is not None:
            ds[f"{keys.D_qij}_re"] = xr.DataArray(sol.D_qij.real, dims=keys.q_i_j)
            ds[f"{keys.D_qij}_im"] = xr.DataArray(sol.D_qij.imag, dims=keys.q_i_j)

        enc = {k: {"zlib": True, "complevel": complevel} for k in ds.data_vars}
        ds.to_netcdf(filename, engine="h5netcdf", encoding=enc)
        return filename

    @classmethod
    def from_hdf5(cls, filename, backend="numpy", precision=None):
        """Load a cached solution without re-running the eigensolve.

        Args:
            filename: path written by ``to_hdf5``.
            backend: backend for any subsequent ``get_solution`` calls
                (``"numpy"`` or ``"jax"``). Does not affect the cached
                solution itself.
            precision: ``"fp64"`` / ``"fp32"`` / ``None``. Resolution
                order: explicit kwarg → file ``precision`` attr →
                ``Precision.default()``.

        Returns:
            A ``DynamicalMatrix`` whose ``solution``, ``q_grid``, and
            FC are populated from the file. ``e_qsI`` and ``remapped``
            are rebuilt lazily on first access.
        """
        ds = xr.open_dataset(filename, engine="h5netcdf").load()

        prim = json2atoms(ds.attrs["reference_primitive_json"])
        sc = json2atoms(ds.attrs["reference_supercell_json"])

        precision = precision or ds.attrs.get("precision")
        p = Precision.resolve(precision)
        precision = p.name
        dtype_real, dtype_complex = p.real, p.complex

        obj = cls.__new__(cls)
        obj.primitive = prim
        obj.supercell = sc
        obj._fc_phonopy = np.asarray(ds[keys.fc_phonopy].data)
        obj._backend = backend
        obj._precision = precision
        obj._convention = "PHONOPY"
        obj._dtype_real = dtype_real
        obj._dtype_complex = dtype_complex

        obj._setup_lattice_and_grid(
            prim, sc, q_points=np.asarray(ds[keys.q_points].data),
        )

        obj._phonon = None

        def _as_real(name):
            return np.asarray(ds[name].data, dtype=dtype_real)

        def _as_complex(base):
            # `1j` is Python complex (complex128); `re + 1j * im` silently
            # upcasts regardless of re/im dtype.
            re = np.asarray(ds[f"{base}_re"].data, dtype=dtype_real)
            im = np.asarray(ds[f"{base}_im"].data, dtype=dtype_real)
            out = np.empty(re.shape, dtype=dtype_complex)
            out.real = re
            out.imag = im
            return out

        w_qs = _as_real(keys.w_qs)
        w_inv_qs = _as_real(keys.w_inv_qs)
        w2_qs = _as_real(keys.w2_qs)
        v_qsa = _as_real(keys.v_qsa)

        e_qsi = _as_complex(keys.e_qsi) if f"{keys.e_qsi}_re" in ds else None
        D_qij = _as_complex(keys.D_qij) if f"{keys.D_qij}_re" in ds else None

        if f"{keys.v_qssa}_re" in ds:
            v_qssa = _as_complex(keys.v_qssa)
            obj._solution = SolutionWithGVM(
                w_qs=w_qs, w_inv_qs=w_inv_qs, w2_qs=w2_qs,
                v_qsa_cartesian=v_qsa, v_qssa_cartesian=v_qssa,
                e_qsi=e_qsi, D_qij=D_qij,
            )
        else:
            obj._solution = Solution(
                w_qs=w_qs, w_inv_qs=w_inv_qs, w2_qs=w2_qs,
                v_qsa_cartesian=v_qsa, e_qsi=e_qsi, D_qij=D_qij,
            )

        # Always rebuilt lazily; any e_qsI_re/im or force_constants_remapped_3N
        # in the cache file is ignored.
        obj._e_qsI = None
        obj._remapped_3N_cache = None

        return obj

    @classmethod
    def from_dataset(cls, dataset, with_group_velocity_matrices=False,
                     backend="numpy", precision="fp64"):
        """Build a DynamicalMatrix from a trajectory ``xr.Dataset``.

        Reads the primitive/supercell from ``dataset.attrs`` (encoded
        via ``gkmx.io.atoms2json``) and the force constants from
        ``dataset[keys.fc]``.

        Args:
            dataset: trajectory dataset emitted by the gkmx pipeline
                (or any dataset carrying ``keys.reference_primitive``,
                ``keys.reference_supercell``, ``keys.fc``).
            with_group_velocity_matrices: include QHGK off-diagonals.
            backend: ``"numpy"`` (default) or ``"jax"``.
            precision: ``"fp64"`` (default) or ``"fp32"``.
        """
        prim = json2atoms(dataset.attrs[keys.reference_primitive])
        sc = json2atoms(dataset.attrs[keys.reference_supercell])
        fc = dataset[keys.fc]
        return cls(force_constants=fc, primitive=prim, supercell=sc,
                   with_group_velocity_matrices=with_group_velocity_matrices,
                   backend=backend, precision=precision)
