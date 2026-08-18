"""Space-group enforcement on the solver inputs.

The two projectors that make the inputs consistent with the crystal's space
group before anything is solved: `refine_geometry` symmetrizes the geometry
itself (truncated Wyckoff coordinates, aiMD cells), `space_group_invariance`
projects fitted force constants onto the invariant subspace. Both act
through the supercell's group -- primitive point operations the supercell
lattice cannot support are outside their scope. `Phonon` applies them at
construction under `enforce_space_group=True`.
"""

import numpy as np
import spglib

from ._log import warn
from .lattice_points import get_p2s_map, get_s2p_map


def refine_geometry(primitive, supercell, symprec=1e-5, warn_tol=None):
    """Project the geometry onto its space-group-symmetric representation.

    Input files carry truncation: decimal Wyckoff coordinates (1/3 stored to
    ~8 digits: LiCdBO3 1.7e-7 A, Mg2YbSb2 4.6e-8 A) and aiMD cell vectors
    (CuI 2.7e-10 relative). The FC projection then enforces a symmetry the
    geometry itself violates: the Bloch phases pick up ``2 pi q . delta``
    errors that break ``[Gamma, D]`` at exactly that scale, which the
    diag(v_qssa) == v_qsa test reports as its conditioning floor. This is
    the geometric counterpart of ``space_group_invariance`` and the standard
    preprocessing of the reference codes (TDEP idealizes its structure,
    phonopy refines cells).

    The SUPERCELL's group acts (same scope as ``space_group_invariance``:
    primitive point operations the supercell lattice cannot support are
    not part of this projection, so a distortion only they would remove
    survives on anisotropic supercells): cell via the metric projection
    ``G <- <W^T G W>`` (A rebuilt about its exact orientation factor,
    ``A = sqrtm(G) Q0`` with ``Q0`` exactly orthogonal), positions via the
    wrap-safe orbit average. The primitive is rebuilt from the idealized
    supercell through the exact integer multiplier, so commensurability is
    restored exactly. Idempotent to fp roundoff once symmetric; a noisy
    input also moves the detected origin, so its convergence is geometric.

    Returns ``(primitive, supercell, residual)`` -- copies, plus the
    largest displacement applied to any supercell atom in Angstrom. Warns
    above ``warn_tol`` (default ``10 * symprec``): a displacement at that
    scale means spglib detected symmetry the structure only approximately
    has, and the projection is dragging atoms toward it.
    """
    if warn_tol is None:
        warn_tol = 10.0 * symprec
    prim, sc = primitive.copy(), supercell.copy()
    A = np.asarray(sc.cell)
    frac = sc.get_scaled_positions()

    M = A @ np.linalg.inv(np.asarray(prim.cell))
    M_int = np.rint(M)
    if np.abs(M - M_int).max() > 1e-6:
        raise ValueError("supercell is not an integer multiple of the "
                         "primitive cell; cannot idealize consistently")
    p2s = get_p2s_map(prim, sc)
    j_of_k = get_s2p_map(prim, sc)

    # translation subgroup, analytically: each copy's offset from its
    # primitive representative is an exact rational lattice translation
    # (integer combination of the primitive rows = m @ inv(M)); snap the
    # measured offsets to those rationals and average the copies
    T_meas = frac - frac[p2s[j_of_k]]
    T_exact = np.rint(T_meas @ M_int) @ np.linalg.inv(M_int)
    base = frac - T_exact
    frac_T = np.zeros_like(frac)
    for j in range(len(prim)):
        orbit = np.where(j_of_k == j)[0]
        ref = base[orbit[0]]
        d = base[orbit] - ref
        d -= np.rint(d)
        frac_T[orbit] = ref + d.mean(axis=0) + T_exact[orbit]
    frac = frac_T

    ds = spglib.get_symmetry((A, frac, sc.get_atomic_numbers()),
                             symprec=symprec)
    W_all, wt_all = ds["rotations"], ds["translations"]
    # one representative per primitive-translation coset (same dedupe as
    # space_group_invariance): the translation part is already exact above
    u_prim = np.round((wt_all @ M_int + 1e-8) % 1.0, 6)
    key = np.concatenate([W_all.reshape(len(W_all), -1), u_prim], axis=1)
    _, uniq = np.unique(key, axis=0, return_index=True)
    W, wt = W_all[np.sort(uniq)], wt_all[np.sort(uniq)]

    G = A @ A.T
    G_sym = np.mean([Wn.T @ G @ Wn for Wn in W], axis=0)

    def _sqrtm_spd(S):
        lam, V = np.linalg.eigh(S)
        return (V * np.sqrt(lam)) @ V.T

    A_sym = _sqrtm_spd(G_sym) @ (np.linalg.inv(_sqrtm_spd(G)) @ A)

    acc = np.zeros_like(frac)
    for Wn, wtn in zip(W, wt):
        img_f = frac @ Wn.T + wtn
        d = img_f[:, None, :] - frac[None, :, :]
        d -= np.rint(d)
        dist = np.linalg.norm(d @ A, axis=-1)
        j = dist.argmin(axis=1)
        if len(np.unique(j)) != len(j) \
                or dist[np.arange(len(j)), j].max() > 10 * symprec:
            raise ValueError(f"symmetry operation does not permute the "
                             f"supercell atoms at symprec={symprec}")
        acc[j] += frac[j] + d[np.arange(len(j)), j]
    frac_sym = acc / len(W)

    # Apply only the wrap-stripped correction to the ORIGINAL position
    # representatives: the averages above work on wrapped coordinates, and
    # adopting their branch would displace atoms by lattice vectors --
    # physics-equivalent, but a silent break of the wrap-branch convention
    # (out-of-cell inputs, `map_I_to_iL` sort buckets, reference anchors).
    delta = frac_sym - sc.get_scaled_positions()
    delta -= np.rint(delta)

    def _snap_boundary(frac_v):
        # a symmetric coordinate that lands within noise of an integer IS
        # that integer; leaving it at -1e-15 hands every hard-wrap consumer
        # (`get_scaled_positions` in the gauge maps) a coin-flip between
        # 0.0 and 0.999... -- an O(1) per-atom phase flip (KPTe2 origin atom)
        snap = np.rint(frac_v)
        near = np.abs(frac_v - snap) < 1e-9
        return np.where(near, snap, frac_v)

    frac_unwrapped = sc.positions @ np.linalg.inv(A)
    pos_before = sc.positions.copy()
    sc.set_cell(A_sym, scale_atoms=False)
    sc.positions = _snap_boundary(frac_unwrapped + delta) @ A_sym

    frac_unwrapped_p = prim.positions @ np.linalg.inv(np.asarray(prim.cell))
    delta_p = delta[p2s]
    prim.set_cell(np.linalg.inv(M_int) @ A_sym, scale_atoms=False)
    prim.positions = _snap_boundary(frac_unwrapped_p + delta_p @ M_int) \
        @ np.asarray(prim.cell)

    residual = float(np.abs(sc.positions - pos_before).max())
    if residual > warn_tol:
        warn(f"geometry idealization moved an atom by {residual:.2e} A "
             f"(warn_tol={warn_tol:.0e}): spglib (symprec={symprec}) "
             f"detected symmetry the input structure only approximately "
             f"has.", prefix="gkmx.phonon")
    return prim, sc, residual


def space_group_invariance(fc, primitive, supercell, symprec=1e-5,
                           warn_tol=1e-4):
    """Project ``fc`` onto the space-group-invariant subspace; returns ``(fc, residual)``.

    Fitted force constants (MLIP, aiMD) carry residuals that break the site
    symmetry of the crystal; every closure the conventions apply (the TDEP
    Gamma(R, q) mode average, the PHONO3PY site average) assumes that
    symmetry, and on asymmetric FCs corrupts what it touches instead of
    enforcing anything (KI_B2_MLIP: diag identity broken at 2.7e-1). TDEP
    and phonopy never face this -- their fits/FCs are symmetric by
    construction -- so projecting here is the de-noising step every
    reference code effectively performs:

        Phi  <-  (1/|G|) sum_S  R_S Phi(S^-1 a, S^-1 b) R_S^T

    over the supercell's space group (spglib on the supercell: the primitive
    point operations compatible with the supercell lattice, times all
    primitive translations). Point operations of the primitive group that do
    not map the supercell lattice to itself cannot act on supercell-periodic
    data and are not part of this projection.

    Args:
        fc: ``(N_p, N_sc, 3, 3)`` force constants, not mass-weighted.
        primitive: ASE Atoms, the primitive cell.
        supercell: ASE Atoms; its space group defines the projection.
        symprec: spglib symmetry tolerance.
        warn_tol: warn when the removed residual exceeds this fraction of
            ``max|Phi|``.

    Returns:
        ``(fc, rel_residual)`` -- projected force constants, and the removed
        asymmetry as a fraction of ``max|Phi|``.
    """
    fc = np.asarray(fc)
    N_p, N_sc = fc.shape[:2]
    A_sc = np.asarray(supercell.cell)
    f_sc = supercell.get_scaled_positions()
    j_of_k = get_s2p_map(primitive, supercell)
    p2s = get_p2s_map(primitive, supercell)

    def nearest(targets_frac):
        d = targets_frac[:, None, :] - f_sc[None, :, :]
        d -= np.rint(d)
        dist = np.linalg.norm(d @ A_sc, axis=-1)
        idx = dist.argmin(axis=1)
        if dist[np.arange(len(idx)), idx].max() > 10 * symprec:
            raise ValueError("supercell atoms do not close under the map; "
                             "primitive/supercell pair inconsistent")
        return idx

    # column map per cell copy: full[a, b] = fc[j_of_k[a], colmap(a)[b]]
    # with colmap(a)[b] = atom at r_b - T(a), T(a) the primitive-lattice
    # translate carrying p2s[j_of_k[a]] to a
    T_frac = f_sc - f_sc[p2s[j_of_k]]
    colmap_cache = {}

    def colmap(a):
        key = tuple(np.round((T_frac[a] + 1e-8) % 1.0, 6))
        if key not in colmap_cache:
            colmap_cache[key] = nearest(f_sc - T_frac[a])
        return colmap_cache[key]

    ds = spglib.get_symmetry((A_sc, f_sc, supercell.get_atomic_numbers()),
                             symprec=symprec)
    W_all, wt_all = ds["rotations"], ds["translations"]
    # One representative per primitive-lattice coset: partners differing by a
    # primitive translation act as the identity on primitive-resolved compact
    # data, so the representatives carry the whole projection at 1/N_cells of
    # the cost. Dedupe on (rotation, translation mod primitive lattice) --
    # rotation alone under-projects silently when `primitive` is actually a
    # conventional cell (NaCl-conventional 2x2x2: 2.9e-1 off, undetectable
    # by idempotency).
    M_prim = A_sc @ np.linalg.inv(np.asarray(primitive.cell))
    u_prim = np.round((wt_all @ M_prim + 1e-8) % 1.0, 6)
    key = np.concatenate([W_all.reshape(len(W_all), -1), u_prim], axis=1)
    _, uniq = np.unique(key, axis=0, return_index=True)
    W, wt = W_all[np.sort(uniq)], wt_all[np.sort(uniq)]
    perm_inv = np.empty((len(W), N_sc), dtype=np.int64)
    for n in range(len(W)):
        img = nearest(f_sc @ W[n].T + wt[n])
        if len(np.unique(img)) != N_sc:
            raise ValueError(
                f"supercell operation {n} does not permute the atoms at "
                f"symprec={symprec}")
        perm_inv[n] = np.argsort(img)

    # fp64 accumulator regardless of the precision switch: it sums up to 48
    # rotated copies, and fp32 accumulation would inject ~1e-6 asymmetry
    # into the very thing being symmetrized; cast back at return.
    acc = np.zeros_like(fc, dtype=np.float64)
    A_sc_T_inv = np.linalg.inv(A_sc.T)
    for n in range(len(W)):
        R = A_sc.T @ W[n] @ A_sc_T_inv
        for i in range(N_p):
            a0 = perm_inv[n][p2s[i]]
            gathered = fc[j_of_k[a0], colmap(a0)[perm_inv[n]]]
            acc[i] += np.einsum("ab,kbc,dc->kad", R, gathered, R,
                                optimize=True)
    acc /= len(W)

    scale = float(np.abs(fc).max())
    rel_residual = float(np.abs(acc - fc).max()) / scale if scale > 0 else 0.0
    if rel_residual > warn_tol:
        warn(f"space-group symmetry violated: removed a residual of "
             f"{rel_residual:.2e} of max|Phi| ({len(W)} supercell operations); "
             f"fitted force constants are being projected onto the invariant "
             f"subspace.", prefix="gkmx.phonon")
    return acc.astype(fc.dtype), rel_residual
