"""Generate phonopy reference fixtures for gkmx's tests.

This is a **one-off generator** that reads phonopy's own test yamls,
extracts the primitive/supercell/FC, solves phonopy's native parallel C
kernel on a fixed random q-grid, and dumps everything needed for
gkmx's :file:`tests/unit/test_phonopy_reference.py` to run **without
importing phonopy at test time**.

Run this only when:
  - phonopy's version or solver changes and you want to re-pin the refs,
  - the q-grid or the set of materials is changed,
  - the shipped fixture files are corrupted.

The output fixture directories are:

  tests/datasets/phonopy_Si/       — space group 227 (Fd-3m)
  tests/datasets/phonopy_NaCl/     — space group 225 (Fm-3m)
  tests/datasets/phonopy_SrTiO3/   — space group 221 (Pm-3m)

Each contains:

  geometry.in.primitive      FHI-aims format
  geometry.in.supercell      FHI-aims format
  force_constants.npy        compact (N_p, N_sc, 3, 3) FC, float64
  reference.npz              q_points (Nq, 3), frequencies_THz (Nq, Ns),
                             velocity_tensor_THz2_A2 (Nq, 3, 3) = Σ_s v_s v_s,
                             unit_conversion_factor (scalar),
                             space_group_number (int),
                             primitive_symbols (str list)

This script needs phonopy and a phonopy source clone for the input yamls::

    micromamba run -n gkmx pip install phonopy symfc
    # Clone phonopy next to the gkmx checkout (default expected layout).
    # Override the location with GKMX_PHONOPY_TEST_DIR if you keep it elsewhere.
    git clone --depth=1 https://github.com/phonopy/phonopy ../phonopy

Invocation::

    cd <gkmx-repo-root>
    micromamba run -n gkmx python tests/datasets/_generate_phonopy_references.py

The output is checked into git so tests can run in any env that
has numpy + ASE, with no phonopy / symfc / phonopy-test-dir dependency.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase import io as ase_io

# Phonopy is a generator-only dependency, not a gkmx runtime dep. Guard the
# import so this module loads cleanly without phonopy installed (e.g. during
# static analysis from a minimal env). `main()` re-checks and raises a
# useful error before doing any work. `from __future__ import annotations`
# above keeps the `PhonopyAtoms` type hint string-lazy.
try:
    import phonopy
    from phonopy.interface.vasp import read_vasp
    from phonopy.phonon.degeneracy import degenerate_sets
    from phonopy.phonon.group_velocity import GroupVelocity
    from phonopy.structure.atoms import PhonopyAtoms
    from phonopy.utils import similarity_transformation
    _PHONOPY_IMPORT_ERROR: Exception | None = None
except ImportError as _e:
    phonopy = None  # type: ignore[assignment]
    read_vasp = None  # type: ignore[assignment]
    PhonopyAtoms = None  # type: ignore[assignment, misc]
    degenerate_sets = None  # type: ignore[assignment]
    GroupVelocity = object  # type: ignore[assignment, misc]
    similarity_transformation = None  # type: ignore[assignment]
    _PHONOPY_IMPORT_ERROR = _e


# Match the test script's seed and count: anything else here invalidates
# the shipped reference files.
SEED = 42
N_Q = 32


# Path to phonopy source clone with the test yaml fixtures. Defaults to
# <workspace>/phonopy/test, assuming this gkmx checkout lives at
# <workspace>/gkmx. Override with GKMX_PHONOPY_TEST_DIR.
_DEFAULT_PHONOPY_TEST_DIR = (
    Path(__file__).resolve().parents[3] / "phonopy" / "test"
)
PHONOPY_TEST_DIR = Path(os.environ.get(
    "GKMX_PHONOPY_TEST_DIR", str(_DEFAULT_PHONOPY_TEST_DIR),
))

# Where to write the gkmx test fixtures.
FIXTURES_ROOT = Path(__file__).resolve().parent

# (output-dir, source, expected space group)
#
# ``source`` is either:
#   ("yaml", <relative-yaml-path>)                — phonopy_params-style
#     file containing FC + geometry, one call to phonopy.load does everything.
#   ("poscar", <spg>, <dim>, <pmat>)              — POSCAR_{spg} + FORCE_SETS_{spg}
#     pair from phonopy/test/phonon/. `dim` is the diagonal supercell_matrix
#     and `pmat` is the primitive_matrix — recipes lifted verbatim from
#     phonopy/test/phonon/test_irreps.py::_get_phonon call sites.
#
# We deliberately mix crystal systems so the shipped fixtures span **all
# seven crystal systems except triclinic** (phonopy has no triclinic test
# fixture). The first three are cubic yaml fixtures; the remaining five
# are lower-symmetry POSCAR+FORCE_SETS fixtures that exercise the
# non-symmorphic / non-orthogonal primitive paths.
#
# Fixture directory names use the **reduced chemical formula** of the
# material, not phonopy's space-group-symbol file name. The POSCAR file
# names inside phonopy/test/phonon/ (e.g. ``POSCAR_P-42_1m``) are
# human-hostile; the fixture dirs (e.g. ``phonopy_AgErTe2``) are not.
#
MATERIALS = [
    # --- Cubic yaml fixtures (inherited from the initial set) ---
    ("phonopy_Si",         ("yaml", "phonopy_params_Si.yaml"),           227),
    ("phonopy_NaCl",       ("yaml", "phonopy_params_NaCl-1.00.yaml.xz"), 225),
    ("phonopy_SrTiO3",     ("yaml", "phonopy_SrTiO3.yaml.xz"),           221),
    # --- Non-cubic POSCAR + FORCE_SETS fixtures ---
    # Monoclinic Pc: Li2ZnGeO4 (Li zinc germanate)
    ("phonopy_Li2ZnGeO4",  ("poscar", "Pc",       [2, 2, 2], np.eye(3)),   7),
    # Orthorhombic P222_1: RbIn3F10 (Rb indium fluoride)
    ("phonopy_RbIn3F10",   ("poscar", "P222_1",   [2, 2, 1], np.eye(3)),  17),
    # Tetragonal non-symmorphic P-42_1m: AgErTe2 (silver erbium telluride)
    ("phonopy_AgErTe2",    ("poscar", "P-42_1m",  [2, 2, 3], np.eye(3)), 113),
    # Trigonal P-3m1: Mg2YbSb2 (layered Zintl phase)
    ("phonopy_Mg2YbSb2",   ("poscar", "P-3m1",    [3, 3, 2], np.eye(3)), 164),
    # Hexagonal P-6: LiCdBO3 (Li Cd borate)
    ("phonopy_LiCdBO3",    ("poscar", "P-6",      [1, 1, 3], np.eye(3)), 174),
]


def _phonopy_to_ase(pa: PhonopyAtoms) -> Atoms:
    # Use positions= (cartesian) instead of scaled_positions= so ASE
    # doesn't go through np.dot(scaled_positions, self.cell), which
    # triggers an ASE-internal np.array(..., copy=False) deprecation
    # on numpy 2.x.
    return Atoms(
        symbols=pa.symbols,
        positions=np.asarray(pa.positions),
        cell=np.asarray(pa.cell),
        masses=pa.masses,
        pbc=True,
    )


def _random_q(seed: int, n: int) -> np.ndarray:
    """Uniformly random q-points in (-0.5, 0.5)^3. Seed-fixed for reproducibility."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.5, 0.5, size=(n, 3))


class GroupVelocityMatrix(GroupVelocity):
    """Vendored from vibes_anisotropy/vibes/phonopy/group_velocity_matrix.py.

    Computes the QHGK group velocity matrix
    ``v_qjj' = <e(q,j)| dD/dq | e(q,j')> / (2 sqrt(ω_j ω_j'))``
    using phonopy's analytic dD/dq plus `_perturb_D`-style degenerate-
    subspace rotation and (optionally) site-symmetrization with explicit
    band-hermicity enforcement.

    Kept inline here so the generator has no runtime dependency on
    vibes_anisotropy — the class is ~50 lines and self-contained.
    """

    def __init__(self, dynamical_matrix, q_length=None, symmetry=None,
                 frequency_factor_to_THz=None, cutoff_frequency=1e-4):
        from phonopy.units import VaspToTHz
        GroupVelocity.__init__(
            self, dynamical_matrix,
            q_length=q_length, symmetry=symmetry,
            frequency_factor_to_THz=frequency_factor_to_THz or VaspToTHz,
            cutoff_frequency=cutoff_frequency,
        )
        self._group_velocity_matrices = None
        self._complex_dtype = f'c{np.dtype("double").itemsize * 2}'

    def run(self, q_points, perturbation=None):
        self._q_points = q_points
        self._perturbation = perturbation
        if perturbation is None:
            self._directions[0] = np.array([1, 2, 3])
        else:
            self._directions[0] = np.dot(self._reciprocal_lattice, perturbation)
        self._directions[0] /= np.linalg.norm(self._directions[0])
        gvm = [self._gvm_at_q(q) for q in self._q_points]
        self._group_velocity_matrices = np.array(
            gvm, dtype=self._complex_dtype, order="C")

    @property
    def group_velocity_matrices(self):
        return self._group_velocity_matrices

    def _gvm_at_q(self, q):
        self._dynmat.run(q)
        dm = self._dynmat.dynamical_matrix
        eigvals, eigvecs = np.linalg.eigh(dm)
        eigvals = eigvals.real
        freqs = np.sqrt(abs(eigvals)) * np.sign(eigvals) * self._factor
        deg_sets = degenerate_sets(freqs)
        ddms = self._get_dD(np.array(q))
        rot_eigvecs = np.zeros_like(eigvecs)
        for deg in deg_sets:
            es = eigvecs[:, deg]
            _, u = np.linalg.eigh(es.T.conj() @ (ddms[0] @ es))
            rot_eigvecs[:, deg] = es @ u
        cond = freqs > self._cutoff_frequency
        freqs_safe = np.where(cond, freqs, 1)
        rot_eigvecs = rot_eigvecs * np.where(cond, 1 / np.sqrt(2 * freqs_safe), 0)
        gvm = np.zeros((3,) + eigvecs.shape, dtype=self._complex_dtype)
        for i, ddm in enumerate(ddms[1:]):
            ddm = ddm * (self._factor ** 2)
            gvm[i] = rot_eigvecs.T.conj() @ (ddm @ rot_eigvecs)
        if self._perturbation is not None or self._symmetry is None:
            return gvm
        # Site symmetry + band hermicity.
        rotations = []
        for r in self._symmetry.reciprocal_operations:
            q_in_BZ = q - np.rint(q)
            if (np.abs(q_in_BZ - np.dot(r, q_in_BZ)) < self._symmetry.tolerance).all():
                rotations.append(r)
        gvm_sym = np.zeros_like(gvm)
        for r in rotations:
            r_cart = similarity_transformation(self._reciprocal_lattice, r)
            gvm_sym += np.einsum("ij,jkl->ikl", r_cart, gvm)
        gvm_sym /= len(rotations)
        gvm_sym = (gvm_sym + gvm_sym.transpose(0, 2, 1).conj()) / 2
        return gvm_sym


def _write_aims_geometry(atoms: Atoms, path: Path) -> None:
    # ASE's aims writer prepends a 5-line banner with the absolute
    # output path and a timestamp. Strip it so the checked-in fixtures
    # don't embed the generator's local filesystem layout.
    ase_io.write(str(path), atoms, format="aims")
    lines = path.read_text().splitlines(keepends=True)
    if (lines[:1] == ["#=======================================================\n"]
            and lines[4:5] == ["#=======================================================\n"]):
        path.write_text("".join(lines[5:]))


def _load_phonopy(source) -> phonopy.Phonopy:
    """Load a phonopy Phonopy from one of the two supported source kinds.

    Returns a loaded Phonopy with FC already attached, is_nac=False, and
    symmetrize_fc=False so the reference files we generate are as close
    to "what the solver saw" as possible.
    """
    kind = source[0]
    if kind == "yaml":
        yaml_path = PHONOPY_TEST_DIR / source[1]
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"phonopy fixture not found: {yaml_path}. "
                "Set GKMX_PHONOPY_TEST_DIR or clone phonopy alongside gkmx."
            )
        return phonopy.load(str(yaml_path), is_nac=False, symmetrize_fc=False)
    if kind == "poscar":
        _, spg, dim, pmat = source
        data_dir = PHONOPY_TEST_DIR / "phonon"
        poscar = data_dir / f"POSCAR_{spg}"
        force_sets = data_dir / f"FORCE_SETS_{spg}"
        for p in (poscar, force_sets):
            if not p.exists():
                raise FileNotFoundError(
                    f"phonopy POSCAR fixture missing: {p}. "
                    "Check the phonopy source clone at GKMX_PHONOPY_TEST_DIR."
                )
        cell = read_vasp(str(poscar))
        return phonopy.load(
            unitcell=cell,
            supercell_matrix=np.diag(dim),
            primitive_matrix=np.asarray(pmat, float),
            force_sets_filename=str(force_sets),
            is_nac=False,
            symmetrize_fc=False,
        )
    raise ValueError(f"unknown source kind: {kind!r}")


def _generate_one(out_dirname: str, source, expected_sg: int) -> dict:
    ph = _load_phonopy(source)

    # Space-group sanity check.
    sg_number = int(ph.symmetry.dataset.number)
    if sg_number != expected_sg:
        raise ValueError(
            f"{out_dirname}: expected space group {expected_sg}, got {sg_number}"
        )

    # Primitive atoms must coincide with sc[p2s_map] for these fixtures —
    # we only pick materials where that holds (no primitive_matrix origin
    # shifts). Verify explicitly so a wrong fixture choice fails loud.
    sc_pos = np.asarray(ph.supercell.positions)
    p2s = np.asarray(ph.primitive.p2s_map, dtype=np.int64)
    if not np.allclose(np.asarray(ph.primitive.positions), sc_pos[p2s], atol=1e-10):
        raise ValueError(
            f"{out_dirname}: primitive.positions != supercell[p2s_map]; "
            "this generator only handles the simple case."
        )

    # Compact FC. phonopy stores either (N_p, N_sc, 3, 3) or (N_sc, N_sc, 3, 3).
    fc = np.asarray(ph.force_constants, dtype=np.float64)
    if fc.shape[0] == fc.shape[1]:
        fc = fc[p2s]  # reduce to compact form
    N_p = fc.shape[0]
    N_sc = fc.shape[1]
    assert fc.shape == (N_p, N_sc, 3, 3)

    # Solve with phonopy (parallel C kernel).
    q_frac = _random_q(SEED, N_Q)
    ph.run_qpoints(q_frac, with_eigenvectors=True, with_group_velocities=True)
    qd = ph.get_qpoints_dict()
    freqs_THz = np.asarray(qd["frequencies"])              # (Nq, Ns)
    eigvecs_raw = np.asarray(qd["eigenvectors"])           # (Nq, Ns_basis, Ns_mode) complex
    group_vel = np.asarray(qd["group_velocities"])         # (Nq, Ns, 3), THz·Å

    # Rotation-invariant kernel V_ab(q) = Σ_s v_sa * v_sb.
    V_ab = np.einsum("qsa,qsb->qab", group_vel, group_vel)  # (Nq, 3, 3)

    # QHGK group-velocity matrix v_qjj'a = <e_j|dD/dq_a|e_j'>/(2√(ω_j ω_j'))
    # via vibes's GroupVelocityMatrix (inlined above) with phonopy's
    # site-symmetry. Shape (Nq, 3, Ns, Ns) complex128, units THz·Å.
    gvm_obj = GroupVelocityMatrix(
        ph.dynamical_matrix,
        symmetry=ph.symmetry,
        frequency_factor_to_THz=ph.unit_conversion_factor,
    )
    gvm_obj.run(q_frac)
    gvm_THzA = np.asarray(gvm_obj.group_velocity_matrices)  # (Nq, 3, Ns, Ns)

    # Build output directory.
    out_dir = FIXTURES_ROOT / out_dirname
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write geometry in FHI-aims format (ASE can read both sides).
    prim = _phonopy_to_ase(ph.primitive)
    sc = _phonopy_to_ase(ph.supercell)
    _write_aims_geometry(prim, out_dir / "geometry.in.primitive")
    _write_aims_geometry(sc, out_dir / "geometry.in.supercell")

    # FC as .npy — compact, fast to load, exact.
    np.save(out_dir / "force_constants.npy", fc)

    # Reference .npz with everything the test needs. `primitive_masses`
    # / `supercell_masses` are phonopy's exact masses (not ASE defaults
    # — ASE's aims writer drops masses on round-trip, and the default
    # atomic masses differ from phonopy's by ~5e-4 amu, which alone is
    # enough to shift frequencies by ~1e-5 rel and break machine-precision
    # agreement). Re-apply these in the test after loading geometry.
    np.savez(
        out_dir / "reference.npz",
        q_points=q_frac.astype(np.float64),
        frequencies_THz=freqs_THz.astype(np.float64),
        velocity_tensor_THz2_A2=V_ab.astype(np.float64),
        group_velocities_THzA=group_vel.astype(np.float64),
        group_velocity_matrix_THzA=gvm_THzA.astype(np.complex128),
        eigenvectors_raw=eigvecs_raw.astype(np.complex128),
        unit_conversion_factor=np.float64(ph.unit_conversion_factor),
        space_group_number=np.int64(sg_number),
        primitive_symbols=np.array(list(ph.primitive.symbols), dtype="U4"),
        primitive_masses=np.asarray(ph.primitive.masses, dtype=np.float64),
        supercell_masses=np.asarray(ph.supercell.masses, dtype=np.float64),
        seed=np.int64(SEED),
        n_q=np.int64(N_Q),
    )

    # Material-specific name / matrix / source-link rows for the README.
    material_label = out_dirname.replace("phonopy_", "")
    symbols_str = f"`{list(ph.primitive.symbols)}`"
    phonopy_blob = "https://github.com/phonopy/phonopy/blob/develop/test"

    matrix_rows = ""
    if source[0] == "yaml":
        src_rel = f"test/{source[1]}"
        src_links = f"- [`{src_rel}`]({phonopy_blob}/{source[1]})\n"
        loader_desc = "Loaded via `phonopy.load(...)` from phonopy's upstream test fixture:"
    elif source[0] == "poscar":
        _, spg, dim, pmat = source
        pmat_flat = np.asarray(pmat).flatten().tolist()
        matrix_rows = (
            f"| Supercell matrix | `diag({list(dim)})` |\n"
            f"| Primitive matrix | `{pmat_flat}` |\n"
        )
        src_links = (
            f"- [`test/phonon/POSCAR_{spg}`]({phonopy_blob}/phonon/POSCAR_{spg})\n"
            f"- [`test/phonon/FORCE_SETS_{spg}`]({phonopy_blob}/phonon/FORCE_SETS_{spg})\n"
        )
        loader_desc = (
            "Loaded via `phonopy.load(unitcell=POSCAR, force_sets_filename=FORCE_SETS, ...)` "
            "from phonopy's upstream test fixtures:"
        )
    else:
        src_links = f"- {source}\n"
        loader_desc = "Source:"

    (out_dir / "README.md").write_text(
        f"# phonopy reference — {material_label}\n\n"
        f"Phonopy reference fixture consumed by\n"
        f"`tests/unit/test_phonopy_reference.py` so gkmx's `Phonon` solver can be\n"
        f"cross-checked against phonopy's parallel C kernel without importing\n"
        f"phonopy at test time.\n\n"
        f"## Metadata\n\n"
        f"| Field | Value |\n"
        f"|---|---|\n"
        f"| Material | {material_label} |\n"
        f"| Space group | {sg_number} |\n"
        f"| Primitive symbols | {symbols_str} |\n"
        f"| `N_p` / `N_sc` | {N_p} / {N_sc} |\n"
        f"{matrix_rows}"
        f"| q-grid | {N_Q} uniform-random in `(-0.5, 0.5)^3`, seed `{SEED}` |\n"
        f"| phonopy version | {phonopy.__version__} |\n"
        f"| `unit_conversion_factor` | `{ph.unit_conversion_factor}` |\n\n"
        f"## Source\n\n"
        f"{loader_desc}\n\n"
        f"{src_links}\n"
        f"## Files\n\n"
        f"| File | Contents |\n"
        f"|---|---|\n"
        f"| `geometry.in.primitive` | FHI-aims geometry, primitive cell |\n"
        f"| `geometry.in.supercell` | FHI-aims geometry, supercell |\n"
        f"| `force_constants.npy` | `(N_p, N_sc, 3, 3)` float64, compact phonopy FC |\n"
        f"| `reference.npz` | q-points + phonopy solver outputs — frequencies, "
        f"group velocities, QHGK group-velocity matrix, raw eigenvectors, "
        f"phonopy's exact masses, `unit_conversion_factor`, `space_group_number` |\n\n"
        f"## Regeneration\n\n"
        f"```\n"
        f"cd gkmx\n"
        f"micromamba run -n gkmx python tests/datasets/_generate_phonopy_references.py\n"
        f"```\n\n"
        f"Do not edit these files by hand — re-run the generator instead.\n"
    )

    return {
        "material": out_dirname,
        "space_group": sg_number,
        "N_p": N_p,
        "N_sc": N_sc,
        "N_q": N_Q,
        "out_dir": str(out_dir),
    }


def main():
    if _PHONOPY_IMPORT_ERROR is not None:
        raise ImportError(
            "This regeneration script requires phonopy and a phonopy source "
            "clone for the input fixtures. Install with:\n"
            "    micromamba run -n gkmx pip install phonopy symfc\n"
            "and clone phonopy next to the gkmx checkout (or override with "
            "GKMX_PHONOPY_TEST_DIR). See this file's module docstring."
        ) from _PHONOPY_IMPORT_ERROR
    print(f"phonopy version : {phonopy.__version__}")
    print(f"phonopy test dir: {PHONOPY_TEST_DIR}")
    print(f"fixtures root   : {FIXTURES_ROOT}")
    print()
    for out_dir, source, expected_sg in MATERIALS:
        try:
            info = _generate_one(out_dir, source, expected_sg)
        except Exception as e:
            print(f"  [FAIL] {out_dir}: {e}")
            raise
        print(f"  [OK]  {info['material']:20s} sg={info['space_group']:3d}  "
              f"N_p={info['N_p']:2d}  N_sc={info['N_sc']:3d}  Nq={info['N_q']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
