"""Ensemble extrapolation: kappa(1/nq) -> kappa(inf) on ensemble-averaged lifetimes.

Per-trajectory extrapolation fits the dense-grid ladder on one run's noisy
lifetimes; with few commensurate q-points the fitted correction can swing by
tens of W/mK between runs. This tool instead averages the symmetrized mode
lifetimes of the whole ensemble at the commensurate grid and extrapolates once,
then adds the correction to the ensemble-mean raw GK kappa.

``--extra-supercell`` unions the commensurate q-points of a second ensemble into
the training set. beta-Ga2O3's 160-atom cell needs this: its 16 commensurate
q-points are coplanar (centred rank 2), so no 3D triangulation exists; the
80-atom cell's 8 points lift the union to rank 3. The extra ensemble contributes
q-points and lifetimes only — frequencies, velocities and the kappa kernels all
come from the primary force constants.

Inputs are per-run ``gkmx out gk`` outputs (self-contained: force constants,
geometry attrs, ``q_points``, ``mode_lifetime_symmetrized``, scalar
``heat_capacity``, raw kappa). No trajectory files are read.

Channel semantics: ``kappa_corrected_*`` applies the full-channel (QHGK,
particle + coherence) correction, the same convention as the per-run CLI's
``kappa_corrected`` (``get_kappa`` hardcodes ``quasi_harmonic_greenkubo=True``).
The BTE particle-channel variant is emitted alongside as
``kappa_corrected_*_BTE``.

    python tools/ensemble_extrapolate.py GK_FILES [...] [--extra-supercell GK_FILES]
"""
import argparse
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import xarray as xr
from scipy.spatial import QhullError

from gkmx import keys
from gkmx.dynamical_matrix import DynamicalMatrix
from gkmx.interpolation import (
    fit_kappa_ladder,
    interpolate_to_gamma,
    interpolate_to_grid,
)
from gkmx.io import json2atoms
from gkmx.kappa import get_kappa_BTE, get_kappa_QHGK
from gkmx.lattice_points import get_unit_grid_extended

GAMMA_TOL = 1e-4


def _nanmean(array, axis=0):
    """``np.nanmean`` without the all-NaN warning.

    A mode never fitted in any run (masked Gamma acoustics) is an all-NaN
    column and becomes 0 in the training set -- not a numerical accident.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        return np.nanmean(array, axis=axis)


def load_ensemble(files):
    """Average one ensemble of per-run gkmx outputs at its commensurate grid.

    Returns q_points, nanmean symmetrized lifetimes, mean scalar cv, the raw-GK
    tensor stack, and the (fc, primitive, supercell) of the first run. All runs
    must share the q-grid — a mixed file list is an error, not something to
    average over. Per-run force constants may differ (some ensembles refit the
    FC per run); the first run's FC defines the interpolation operator and the
    largest relative deviation is reported.
    """
    q_ref = fc_ref = None
    taus, cvs, kappas = [], [], []
    prim = sc = None
    fc_dev = 0.0
    for f in files:
        with xr.open_dataset(f, engine="h5netcdf") as ds:
            q = np.asarray(ds[keys.q_points])
            fc = np.asarray(ds[keys.fc])
            if q_ref is None:
                q_ref, fc_ref = q, fc
                prim = json2atoms(ds.attrs[keys.reference_primitive])
                sc = json2atoms(ds.attrs[keys.reference_supercell])
            else:
                if not np.allclose(q, q_ref, atol=1e-10):
                    raise ValueError(f"{f}: q-grid differs from {files[0]}")
                fc_dev = max(fc_dev, np.abs(fc - fc_ref).max() / np.abs(fc_ref).max())
            taus.append(np.asarray(ds[keys.mode_lifetime_symmetrized]))
            cvs.append(float(ds[keys.heat_capacity]))
            kappas.append(np.asarray(ds[keys.kappa]))
    if fc_dev > 1e-8:
        print(f"per-run force constants deviate up to {fc_dev:.2e} relative "
              f"within this ensemble; {files[0]} represents it where an FC is "
              f"needed (only the primary ensemble's FC enters the extrapolation)")
    return SimpleNamespace(
        q_points=q_ref, tau_mean=_nanmean(np.stack(taus)), cv=float(np.mean(cvs)),
        kappa_runs=np.stack(kappas), fc=fc_ref,
        primitive=prim, supercell=sc, files=[str(f) for f in files],
    )


def union_training_set(q_sets, w2_sets, tau_sets, talk=print):
    """Union several supercells' commensurate q-points into one training set.

    Every set contributes only q-points and lifetimes; ``w`` comes from the one
    interpolation FC. ``l = w^2 tau`` is the grid-independent quantity;
    duplicate q-points (Gamma, shared zone-boundary points) are nanmean-averaged
    so a mode measured in only one ensemble keeps its value instead of being
    halved against a hard zero. First-occurrence order is preserved: on a
    single-ensemble input the training set is exactly the commensurate grid in
    its native order, so the tool reproduces
    ``gkmx.interpolation.get_interpolation_data`` bit-for-bit there (scipy's
    Delaunay tie-breaking on regular grids is order-dependent).
    Gamma is moved to index 0 (``interpolate_to_gamma`` requires it there).
    """
    q_all = np.concatenate([np.asarray(q) for q in q_sets])
    l_all = np.concatenate([w2 * np.asarray(tau)          # NaN kept
                            for w2, tau in zip(w2_sets, tau_sets)])
    # Group with the project's wrap tolerance ((x + 1e-8) % 1.0) so equivalent
    # representatives (-0.5 vs 0.5, fp32-written grids) merge. Dicts preserve
    # insertion order, so the groups come out in first-occurrence order.
    groups = {}
    for i, k in enumerate(map(tuple, np.round((q_all + 1e-8) % 1.0, 10))):
        groups.setdefault(k, []).append(i)
    q_unique = q_all[[rows[0] for rows in groups.values()]]
    l_unique = np.nan_to_num(  # modes never fitted anywhere -> 0
        np.stack([_nanmean(l_all[rows]) for rows in groups.values()]))

    gi = np.where(np.linalg.norm(q_unique, axis=1) < GAMMA_TOL)[0]
    if len(gi) == 0:
        raise ValueError("no Gamma point in the union grid")
    for arr in (q_unique, l_unique):
        arr[[0, gi[0]]] = arr[[gi[0], 0]]

    rank = np.linalg.matrix_rank(q_unique - q_unique.mean(axis=0), tol=1e-8)
    talk(f"training grid: {len(q_unique)} unique q-points, centred rank {rank}"
         + ("" if rank == 3 else "  ** rank-deficient: triangulation will fail; "
            "add --extra-supercell runs from a supercell sampling the missing "
            "direction **"))
    return q_unique, l_unique


def _kappa_qhgk_on_grid(itp_dmx, points, sol, tau_qs, cv, chunk):
    """QHGK kappa summed over a dense mesh, accumulated in direct-solved q-chunks.

    Frequencies and lifetimes always come from the full-grid ``sol``: the
    near-zero cutoff in ``get_solution`` is a batch order statistic, so a chunk
    without Gamma would zero its lowest branches. The mode sum is additive in q,
    so chunking is exact and the result does not depend on the chunk size. Note
    the direct per-chunk solve is a different construction from the
    symmetry-expanded ``v_qssa`` the built-in interpolation uses; inside
    degenerate multiplets the two differ at the eigh-gauge level.
    """
    w_full = np.asarray(sol.w_qs)
    w_inv_full = np.asarray(sol.w_inv_qs)
    total = np.zeros((3, 3))
    for i in range(0, len(points), chunk):
        sl = slice(i, min(i + chunk, len(points)))
        s = itp_dmx.get_solution(points[sl], with_group_velocity_matrices=True)
        total += np.asarray(get_kappa_QHGK(
            v_qssa=s.v_qssa_cartesian, tau_qs=tau_qs[sl],
            w_qs=w_full[sl], w_inv_qs=w_inv_full[sl], cv_qs=cv))
        del s
    return total


def extrapolate(itp_dmx, q_sets, tau_sets, cv, nqs, max_mem_gb=6.0, talk=print):
    """Dense-grid sweep + weighted linear kappa(1/nq) fit on the union training set.

    Both channels are swept from one mesh solve: the full channel (QHGK,
    particle + coherence — the built-in pipeline's ``kappa_corrected``
    convention, unsuffixed in the output) and the BTE particle channel
    (``_BTE`` suffix).

    ``Nq_eff`` normalizes by the primary supercell's commensurate count — raw
    kappa and cv were measured there; extra supercells only shape the
    l-interpolant. Meshes are Gamma-centered only: a Monkhorst mesh has no
    Gamma, so its irreducible batch trips the acoustic w_inv cutoff (a batch
    order statistic) and silently zeroes the near-Gamma acoustic star's
    lifetimes on every ladder mesh. Returns {} when the union cannot
    triangulate.
    """
    # Each set is solved once here: every batch contains Gamma, so the acoustic
    # w_inv cutoff (a batch order statistic, phonon.py) resolves identically.
    sols = [itp_dmx.get_solution(np.asarray(q), with_group_velocity_matrices=(i == 0))
            for i, q in enumerate(q_sets)]
    q_unique, l_unique = union_training_set(
        q_sets, [np.asarray(s.w2_qs) for s in sols], tau_sets, talk=talk)
    try:
        l_unique[0, :] = interpolate_to_gamma(q_unique, l_unique, extend_minus=True)
    except QhullError:
        talk("** QhullError: q-point sampling insufficient for interpolation")
        return {}

    train = get_unit_grid_extended(q_unique)
    kw_train = {"train_array_qs": l_unique[train.map2extended],
                "train_points": train.points_extended}

    sol0 = sols[0]
    tau0 = np.nan_to_num(np.asarray(tau_sets[0]))
    kappa_ha_full = np.asarray(get_kappa_QHGK(
        v_qssa=sol0.v_qssa_cartesian, tau_qs=tau0,
        w_qs=sol0.w_qs, w_inv_qs=sol0.w_inv_qs, cv_qs=cv))
    kappa_ha_bte = np.asarray(
        get_kappa_BTE(sol0.v_qsa_cartesian, tau_qs=tau0, cv_qs=cv))

    Nq_init = len(np.asarray(q_sets[0]))
    Ns = np.asarray(tau_sets[0]).shape[1]
    nqs = np.asarray(list(nqs))
    Ks_full = np.zeros((len(nqs), 3, 3))
    Ks_bte = np.zeros((len(nqs), 3, 3))

    # Peak of one QHGK chunk: v_qssa (48 B/pair) + tau_eff (8) + the v x v*
    # outer product and the kappa product array (144 each, both live at
    # kappa.py's final einsum) ~ 352 B per (q, s, s') pair.
    gb_per_q = Ns * Ns * 352 / 1e9
    qhgk_chunk = max(1, int(max_mem_gb / gb_per_q))
    talk(f"QHGK accumulated in q-chunks of {qhgk_chunk} "
         f"(~{qhgk_chunk * gb_per_q:.1f} GB peak)")

    for i, nq in enumerate(nqs):
        grid, sol = itp_dmx.get_mesh_and_solution(
            (int(nq),) * 3, reduced=False, monkhorst=False,
            with_group_velocity_matrices=False)
        ir_l = interpolate_to_grid(q_points=grid.ir.points, **kw_train)
        tau_int = ir_l[grid.ir.map2full] * np.asarray(sol.w_inv_qs) ** 2
        Nq_eff = len(grid.points) / Nq_init
        Ks_bte[i] = np.asarray(
            get_kappa_BTE(sol.v_qsa_cartesian, tau_int, cv)) / Nq_eff
        Ks_full[i] = _kappa_qhgk_on_grid(
            itp_dmx, np.asarray(grid.points), sol, tau_int, cv,
            chunk=qhgk_chunk) / Nq_eff
        talk(f"  nq={int(nq):3d}  Nq_eff={Nq_eff:8.2f}"
             f"  kappa={np.diag(Ks_full[i]).mean():7.3f}"
             f"  BTE={np.diag(Ks_bte[i]).mean():7.3f} W/mK")
        del grid, sol

    out = {"nqs": nqs.tolist(), "n_q_unique": len(q_unique),
           "q_unique": q_unique.tolist()}
    for sfx, Ks_ch, kappa_ha_ch in (("", Ks_full, kappa_ha_full),
                                    ("_BTE", Ks_bte, kappa_ha_bte)):
        _, intercept_ab, slope_stderr_ab, intercept_stderr_ab = fit_kappa_ladder(
            nqs, Ks_ch)
        correction_ab = intercept_ab - kappa_ha_ch
        out.update({
            f"kappa_ha{sfx}": kappa_ha_ch.tolist(),
            f"kappa_array{sfx}": Ks_ch.tolist(),
            f"fit_intercept_ab{sfx}": intercept_ab.tolist(),
            # slope stderr is what the built-in reports as *_stderr; the
            # intercept stderr is the uncertainty of kappa(inf) itself.
            f"fit_slope_stderr_ab{sfx}": slope_stderr_ab.tolist(),
            f"fit_intercept_stderr_ab{sfx}": intercept_stderr_ab.tolist(),
            f"correction_ab{sfx}": correction_ab.tolist(),
            f"correction{sfx}": float(np.diag(correction_ab).mean()),
        })
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path,
                    help="per-run gkmx outputs of the primary (MD) ensemble")
    ap.add_argument("--extra-supercell", nargs="+", type=Path, default=None,
                    metavar="FILE",
                    help="per-run gkmx outputs of an additional-supercell "
                         "ensemble; contributes training q-points + lifetimes")
    ap.add_argument("--nq-max", type=int, default=20,
                    help="ladder sweeps nq = 4..nq_max in steps of 2")
    ap.add_argument("--max-mem-gb", type=float, default=6.0)
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="results JSON (default: print to stdout)")
    args = ap.parse_args(argv)

    ens = load_ensemble(args.files)
    q_sets, tau_sets = [ens.q_points], [ens.tau_mean]
    extra = None
    if args.extra_supercell:
        extra = load_ensemble(args.extra_supercell)
        q_sets.append(extra.q_points)
        tau_sets.append(extra.tau_mean)

    itp_dmx = DynamicalMatrix(
        force_constants=ens.fc, primitive=ens.primitive,
        supercell=ens.supercell, precision="fp64",
        with_group_velocity_matrices=True)

    result = extrapolate(
        itp_dmx, q_sets, tau_sets, cv=ens.cv,
        nqs=range(4, args.nq_max + 1, 2), max_mem_gb=args.max_mem_gb)

    scal = np.trace(ens.kappa_runs, axis1=1, axis2=2) / 3.0
    n = len(scal)
    raw_mean = ens.kappa_runs.mean(axis=0)
    std = float(scal.std(ddof=1)) if n > 1 else 0.0
    out = {
        "n_runs": n, "files": ens.files, "cv": ens.cv,
        "extra_supercell_files": extra.files if extra else None,
        "kappa_raw_mean": raw_mean.tolist(),
        "kappa_raw_scalar": float(scal.mean()),
        "kappa_raw_scalar_std": std,
        "kappa_raw_scalar_sem": std / np.sqrt(n) if n > 1 else 0.0,
        **result,
    }
    # kappa_corrected_* applies the full-channel (QHGK) correction to the raw
    # ensemble mean, matching the built-in CLI's kappa_corrected convention;
    # the BTE particle-channel variant is emitted alongside as *_BTE.
    for sfx in ("", "_BTE") if result else ():
        out[f"kappa_corrected_ab{sfx}"] = (
            raw_mean + np.asarray(result[f"correction_ab{sfx}"])).tolist()
        out[f"kappa_corrected_scalar{sfx}"] = float(
            scal.mean() + result[f"correction{sfx}"])

    text = json.dumps(out, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        summary = (f"{args.out}: raw {out['kappa_raw_scalar']:.3f} "
                   f"+/- {out['kappa_raw_scalar_sem']:.3f}")
        if result:
            summary += (f"  corrected {out['kappa_corrected_scalar']:.3f}"
                        f"  (BTE {out['kappa_corrected_scalar_BTE']:.3f}) W/mK")
        print(summary)
    else:
        print(text)


if __name__ == "__main__":
    main()
