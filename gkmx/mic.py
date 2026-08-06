"""Minimum-image-convention fold for periodic 3D displacements (numpy/jax)."""
from __future__ import annotations

import warnings

import numpy as np


def fold(disp, cell, *, search=None):
    """Fold each row of ``disp`` to its nearest periodic image of the origin.

    Args:
        disp: ``(..., 3)`` numpy or jax array of Cartesian displacements.
            Output preserves the input dtype.
        cell: ``(3, 3)`` lattice in ASE row-vector convention
            (``cell[i]`` = i-th lattice vector).
        search: ``True`` to force the 27-image search; ``False`` to
            skip it (round-frac result only — exact for orthogonal
            cells); ``None`` autodetects via ``is_orthogonal(cell)``.
            ``None`` raises ``ValueError`` under ``jax.jit`` (autodetect
            needs to evaluate ``cell`` on the host, incompatible with a
            traced cell). Inside JIT, precompute the boolean outside.

    Returns:
        Array with the same shape and dtype as ``disp``, every row
        folded into the half-open MIC cell.

    Raises:
        ValueError: ``search=None`` was passed inside ``jax.jit``.
    """
    xp = _array_namespace(disp)

    if search is None:
        if xp is not np:
            raise ValueError(
                "fold(disp, cell, search=None): autodetect via "
                "is_orthogonal(cell) is not safe inside jax.jit because it "
                "forces a host-side evaluation of `cell`. Pre-compute "
                "`search = not is_orthogonal(np.asarray(cell))` outside the "
                "JIT and pass it explicitly."
            )
        search = not is_orthogonal(cell)

    cell = xp.asarray(cell, dtype=disp.dtype)
    inv_cell = xp.linalg.inv(cell)
    frac = disp @ inv_cell
    r = frac - xp.round(frac)

    if not search:
        return r @ cell

    # JAX dense 27-image search: peak transient
    # 4 * prod(disp.shape[:-1]) * 27 * 3 bytes (fp32). Chunk to fit.
    if xp is not np:
        return _dense_27_search(r, cell, xp)

    # numpy fast path needs Minkowski reduction; non-reduced bases can
    # miss the closest image inside {-1,0,1}^3 — fall back to dense search.
    if not _is_pairwise_reduced(cell):
        warnings.warn(
            "fold: cell is not Minkowski-reduced under the pairwise "
            "Gauss/Lagrange condition; per-element fast path may be "
            "unsafe. Falling back to the dense 27-image search.",
            UserWarning,
            stacklevel=2,
        )
        return _dense_27_search(r, cell, np)

    d_fast = r @ cell
    half_L_min_sq = _safe_radius_sq(cell)
    unsafe = (d_fast * d_fast).sum(axis=-1) > half_L_min_sq

    if not bool(unsafe.any()):
        return d_fast

    r_sub = r[unsafe]
    d_fast[unsafe] = _dense_27_search(r_sub, cell, np)
    return d_fast


def needs_fold(disp, cell):
    """True iff any row of ``disp`` has Cartesian norm exceeding ``safe_radius(cell)``."""
    disp = np.asarray(disp)
    norms_sq = (disp * disp).sum(axis=-1)
    return float(norms_sq.max()) > _safe_radius_sq(cell)


def safe_radius(cell):
    """Return ``L_min / 2`` in Å — the fast-path safe radius for ``fold``."""
    return float(np.sqrt(_safe_radius_sq(cell)))


def is_orthogonal(cell, *, rtol=1e-8):
    """Test whether a ``(3, 3)`` cell has orthogonal lattice vectors.

    Computes the Gram matrix ``G = cell @ cell.T`` in fp64 (so the
    classification is dtype-invariant) and returns ``True`` if every
    off-diagonal entry is below ``rtol * max(|diag G|)``.

    Args:
        cell: ``(3, 3)`` lattice (any dtype accepted; cast internally).
        rtol: relative tolerance on the off-diagonal entries. Default
            ``1e-8`` admits ~fp32-machine-epsilon noise on near-cubic
            cells while rejecting any genuine ≥ 1° triclinic tilt.

    Returns:
        ``True`` if orthogonal within tolerance, ``False`` otherwise.
    """
    cell = np.asarray(cell, dtype=np.float64)
    g = cell @ cell.T
    diag_mag = max(float(np.abs(np.diag(g)).max()), 1.0)
    off = np.abs(g - np.diag(np.diag(g)))
    return bool(np.all(off < rtol * diag_mag))


_IMAGE_SHIFTS = np.array(
    [(i, j, k) for i in (0, -1, 1) for j in (0, -1, 1) for k in (0, -1, 1)],
    dtype=np.int8,
)


def _dense_27_search(r, cell, xp):
    """27-image brute-force search; ``r`` must be in residual form ``frac - round(frac)``."""
    shifts = xp.asarray(_IMAGE_SHIFTS, dtype=r.dtype)
    cand = (r[..., None, :] - shifts) @ cell
    best = xp.argmin((cand * cand).sum(axis=-1), axis=-1)
    return (r - shifts[best]) @ cell


def _safe_radius_sq(cell):
    """``(L_min/2)^2`` where ``L_min = min |s . cell|`` over non-zero ``s in {-1,0,1}^3`` (valid only for Minkowski-reduced cells)."""
    cart = _IMAGE_SHIFTS[1:].astype(np.float64) @ np.asarray(cell, dtype=np.float64)
    return 0.25 * float((cart * cart).sum(axis=-1).min())


def _is_pairwise_reduced(cell, rtol=1e-6):
    """Approximate Minkowski reduction check via pairwise Gauss/Lagrange."""
    a = np.asarray(cell, dtype=np.float64)
    g = a @ a.T
    diag = np.diag(g)
    for i in range(3):
        for j in range(i + 1, 3):
            for s in (-1, 1):
                length_sq = diag[i] + 2 * s * g[i, j] + diag[j]
                if length_sq + rtol * max(diag[i], diag[j]) < diag[i]:
                    return False
                if length_sq + rtol * max(diag[i], diag[j]) < diag[j]:
                    return False
    return True


def _array_namespace(x):
    mod = type(x).__module__
    if mod.startswith(("jax", "jaxlib")):
        import jax.numpy as jnp
        return jnp
    return np


