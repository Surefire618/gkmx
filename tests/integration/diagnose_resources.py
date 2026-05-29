"""Cluster diagnostic for gkmx._resources.

Bundles every detection path + auto_resolve scenarios into one run.
Intended for SLURM submission to Raven / Blogin (CPU and GPU partitions).
Prints structured output for log diffing AND asserts invariants so a
silent-default regression makes the SLURM job fail loudly.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import traceback

from gkmx._resources import MemoryCost, Resources


def header(title: str) -> None:
    print()
    print("=" * 72)
    print(f"=== {title}")
    print("=" * 72)


def kv(key: str, value) -> None:
    print(f"  {key:<32} {value}")


def section_environment() -> None:
    header("environment")
    kv("hostname", socket.gethostname())
    kv("python", sys.version.split()[0])
    kv("PWD", os.getcwd())
    for var in (
        "SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST",
        "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU",
        "SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE",
        "SLURM_GPUS", "SLURM_JOB_GPUS",
        "CUDA_VISIBLE_DEVICES",
        "GKMX_MAX_MEM_GB", "GKMX_PRECISION",
        "GKMX_PEAK_FACTOR_NUMPY", "GKMX_PEAK_FACTOR_JAX",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
    ):
        kv(var, os.environ.get(var, "(unset)"))

    # cgroup probes
    for path in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    ):
        try:
            with open(path) as f:
                kv(path, f.read().strip())
        except OSError as exc:
            kv(path, f"(unreadable: {exc.__class__.__name__})")


def section_individual_detectors() -> dict:
    header("individual detector outputs")
    pynvml_result = Resources._gpu_pynvml_free_gb()
    jax_result = Resources._gpu_jax_memstats_free_gb()
    cgroup_result = Resources._cgroup_memory_limit_gb()
    host_result = Resources._host_available_gb()
    slurm_result = Resources._slurm_env_limit_gb()
    kv("_gpu_pynvml_free_gb", pynvml_result)
    kv("_gpu_jax_memstats_free_gb", jax_result)
    kv("_cgroup_memory_limit_gb", cgroup_result)
    kv("_host_available_gb", host_result)
    kv("_slurm_env_limit_gb", slurm_result)
    return {
        "pynvml": pynvml_result,
        "jax_memstats": jax_result,
        "cgroup": cgroup_result,
        "host": host_result,
        "slurm_env": slurm_result,
    }


def section_detect() -> dict[str, Resources]:
    header("Resources.detect() per backend")
    out: dict[str, Resources] = {}
    for backend in ("numpy", "jax"):
        r = Resources.detect(backend)
        out[backend] = r
        print(f"  backend={backend}")
        kv("  free_gb", f"{r.free_gb:.2f}")
        kv("  source", r.source)
        kv("  confidence", r.confidence)
        kv("  notes", r.notes)
    return out


def section_peak_factors() -> None:
    header("peak factors")
    for backend in ("numpy", "jax"):
        kv(f"Resources.peak_factor({backend})", Resources.peak_factor(backend))
        kv(f"Resources.PEAK_FACTORS[{backend!r}]", Resources.PEAK_FACTORS[backend])


def section_override_validation() -> None:
    header("GKMX_MAX_MEM_GB validation")
    cases = [
        ("8.0", 8.0),     # valid
        ("0.5", 0.5),     # valid edge
        ("0", None),      # zero rejected
        ("-3", None),     # negative rejected
        ("abc", None),    # garbage rejected
        ("1e100", 1e100), # absurd but parseable — accepted with no upper bound
    ]
    for raw, expected in cases:
        import warnings as _w
        with _w.catch_warnings(record=True):
            _w.simplefilter("always")
            got = Resources.validate_override(raw)
        ok = got == expected if expected is not None else got is None
        kv(f"Resources.validate_override({raw!r})", f"got={got} expected={expected} ok={ok}")
        assert ok, f"override validation regression: {raw!r} → {got!r}"


def section_auto_resolve() -> dict:
    header("Resources.auto_resolve_for_compute_cv_tau scenarios")
    scenarios = [
        # (backend, Nt, Nat, bytes_per_real, label)
        ("numpy", 1_000,    128, 4, "n128_fp32_short"),
        ("numpy", 250_000,  2000, 4, "n2000_fp32_long"),
        ("numpy", 250_000,  2000, 8, "n2000_fp64_long"),
        ("jax",   1_000,    128, 4, "n128_fp32_short"),
        ("jax",   250_000,  2000, 4, "n2000_fp32_long"),
        ("jax",   250_000,  2000, 8, "n2000_fp64_long"),
    ]
    results = {}
    for backend, Nt, Nat, bpr, label in scenarios:
        key = f"{backend}/{label}"
        print(f"  {key} (Nt={Nt}, Nat={Nat}, bytes_per_real={bpr})")
        try:
            plan = Resources.auto_resolve_for_compute_cv_tau(
                backend, Nt, Nat, bytes_per_real=bpr,
            )
            kv("    max_mem_gb", f"{plan.max_mem_gb:.3f}")
            kv("    origin", plan.origin)
            if plan.resources is not None:
                kv("    detected_free_GB", f"{plan.resources.free_gb:.2f}")
                kv("    detection_source", plan.resources.source)
                kv("    detection_confidence", plan.resources.confidence)
            if plan.cost is not None:
                kv("    dataset_GB", f"{plan.cost.dataset_gb:.3f}")
                kv("    dmx_eqsI_GB", f"{plan.cost.dmx_eqsI_gb:.3f}")
                kv("    kernel_persistent_GB", f"{plan.cost.kernel_persistent_gb:.3f}")
            kv("    target_block_GB", f"{plan.target_block_gb:.3f}")
            kv("    peak_factor", plan.peak_factor)
            results[key] = {"ok": True, "plan": plan}
        except MemoryError as exc:
            kv("    raised MemoryError", str(exc).splitlines()[0])
            results[key] = {"ok": False, "error": "MemoryError", "msg": str(exc)}
        except Exception as exc:
            traceback.print_exc()
            results[key] = {"ok": False, "error": type(exc).__name__, "msg": str(exc)}
    return results


def section_impossible_kernel_raises() -> None:
    header("impossible kernel raises MemoryError")
    # Force detection to return a tiny budget.
    original = Resources.detect
    Resources.detect = classmethod(lambda cls, backend: cls(
        free_gb=0.5, source="test_stub", confidence="measured", backend=backend
    ))
    try:
        try:
            Resources.auto_resolve_for_compute_cv_tau("numpy", 1000, 2000)
        except MemoryError as exc:
            kv("MemoryError raised", "yes")
            kv("message", str(exc).splitlines()[0])
        else:
            print("  !! did NOT raise MemoryError on impossible kernel — regression !!")
            sys.exit(1)
    finally:
        Resources.detect = original


def section_assert_invariants(individual: dict, by_backend: dict[str, Resources]) -> None:
    header("invariant assertions")
    under_slurm = bool(os.environ.get("SLURM_JOB_ID"))

    numpy_r = by_backend["numpy"]
    jax_r = by_backend["jax"]

    # GPU presence derived from what was actually detected, not env-var
    # heuristics. SLURM-allocated nodes set CUDA_VISIBLE_DEVICES, but a
    # desktop with a working GPU and no SLURM does not — so the env-var
    # path falsely concludes "no GPU" on bare-Linux laptops. The detector
    # output is authoritative.
    gpu_sources = ("pynvml", "jax_memstats")
    has_gpu = jax_r.source in gpu_sources

    # 1. numpy detection must succeed everywhere (login or compute).
    assert numpy_r.confidence != "default", (
        f"numpy detection silently fell back to default on {socket.gethostname()}; "
        f"source={numpy_r.source}, notes={numpy_r.notes}"
    )
    kv("numpy.confidence != 'default'", "OK")

    # 2. Under SLURM, the source must mention cgroup or slurm.
    if under_slurm:
        assert ("cgroup" in numpy_r.source or "slurm" in numpy_r.source), (
            f"on SLURM but detected numpy source={numpy_r.source!r} — "
            f"cgroup / slurm should bind"
        )
        kv("under SLURM → cgroup/slurm in source", "OK")

    # 3. When a GPU is detected, jax should be measured.
    if has_gpu:
        assert jax_r.confidence == "measured", (
            f"GPU detected but jax confidence={jax_r.confidence}; "
            f"source={jax_r.source}, notes={jax_r.notes}"
        )
        kv("GPU → jax source is pynvml or jax_memstats", f"OK ({jax_r.source})")

        # 4. Sanity range: 1 GB (small consumer card after preallocation)
        # to 200 GB (H100/MI300). RTX 2060 6 GB after jax preallocation
        # lands around 4.6 GB free; A100 80 GB around 75 GB free.
        assert 1.0 < jax_r.free_gb < 200.0, (
            f"jax free_gb={jax_r.free_gb} outside [1, 200] GB — sanity-check failed"
        )
        kv("jax.free_gb plausible (1 < x < 200)", f"OK ({jax_r.free_gb:.1f} GB)")
    else:
        # Without a GPU, jax falls through to the host-RAM path.
        assert jax_r.source == numpy_r.source, (
            f"no GPU but jax source ({jax_r.source}) differs from numpy ({numpy_r.source})"
        )
        kv("no GPU → jax falls through to CPU path", "OK")

    # 5. peak_factor returns positive finite numbers.
    for backend in ("numpy", "jax"):
        pf = Resources.peak_factor(backend)
        assert pf > 0 and pf < 100, f"peak_factor({backend})={pf} out of range"
    kv("peak_factor positive finite", "OK")

    # 6. MemoryCost.breakdown is internally consistent.
    m = MemoryCost(Nat=128, Nt=1000, bytes_per_real=4)
    bd = m.breakdown(B=64, t_chunk=5000, peak_factor=Resources.peak_factor("numpy"))
    total = (bd["dataset_gb"] + bd["dmx_eqsI_gb"] + bd["kernel_persistent_gb"]
             + bd["chunk_transient_gb"] + bd["block_peak_gb"])
    assert abs(bd["total_peak_gb"] - total) < 1e-6, "total_peak_gb inconsistent"
    kv("MemoryCost.breakdown arithmetic", "OK")

    print()
    print("ALL INVARIANTS PASSED")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="(no-op; reserved)")
    parser.parse_args()
    section_environment()
    individual = section_individual_detectors()
    by_backend = section_detect()
    section_peak_factors()
    section_override_validation()
    section_auto_resolve()
    section_impossible_kernel_raises()
    section_assert_invariants(individual, by_backend)
    return 0


if __name__ == "__main__":
    sys.exit(main())
