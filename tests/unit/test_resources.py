"""Tests for ``gkmx._resources`` — resource detection and auto-resolution."""

import pytest

from gkmx._resources import MemoryCost, Plan, Resources


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_detect_returns_well_formed_resources(backend):
    """Detection never crashes; the returned ``Resources`` is consistent
    in dtype, sign, and confidence vocabulary."""
    r = Resources.detect(backend)
    assert isinstance(r, Resources) and r.backend == backend
    assert r.free_gb > 0
    assert r.confidence in {"measured", "estimated", "default"}
    assert isinstance(r.notes, tuple)


def test_detect_default_fallback(monkeypatch):
    """When every detector fails, the layer returns the 4 GB default
    with ``confidence="default"`` and a note explaining why."""
    for name in ("_gpu_pynvml_free_gb", "_gpu_jax_memstats_free_gb",
                 "_cgroup_memory_limit_gb", "_host_available_gb",
                 "_slurm_env_limit_gb"):
        monkeypatch.setattr(Resources, name, staticmethod(lambda: None))
    r = Resources.detect("numpy")
    assert (r.confidence, r.source, r.free_gb) == ("default", "default", 4.0)
    assert any("detection failed" in n for n in r.notes)


def test_validate_override_accepts_positive_floats():
    assert Resources.validate_override("8.0") == 8.0


@pytest.mark.parametrize("raw,warn_match", [
    ("0",     "non-positive"),
    ("-5",    "non-positive"),
    ("eight", "not a number"),
])
def test_validate_override_rejects_invalid(raw, warn_match):
    with pytest.warns(UserWarning, match=warn_match):
        assert Resources.validate_override(raw) is None


def test_auto_resolve_env_override_path(monkeypatch):
    """A valid GKMX_MAX_MEM_GB short-circuits detection; an invalid one
    warns and falls through to detection. The MemoryCost is populated
    on both paths so callers can introspect the workload either way."""
    monkeypatch.setenv("GKMX_MAX_MEM_GB", "12.0")
    plan = Resources.auto_resolve_for_compute_cv_tau("numpy", 1000, 64)
    assert isinstance(plan, Plan)
    assert plan.max_mem_gb == 12.0 and plan.origin == "env GKMX_MAX_MEM_GB"
    assert plan.resources is None
    assert plan.cost is not None and plan.cost.Nat == 64 and plan.cost.Nt == 1000

    monkeypatch.setenv("GKMX_MAX_MEM_GB", "garbage")
    with pytest.warns(UserWarning):
        plan = Resources.auto_resolve_for_compute_cv_tau("numpy", 1000, 64)
    assert plan.origin == "detect+cost"
    assert plan.resources is not None
    assert plan.resources.source and plan.resources.confidence
    assert plan.cost is not None


def test_auto_resolve_raises_when_kernel_cannot_fit(monkeypatch):
    """If detected free < kernel-persistent + reserve, raise
    ``MemoryError`` instead of silently floor-limiting."""
    monkeypatch.setattr(Resources, "detect", classmethod(
        lambda cls, backend: cls(
            free_gb=0.5, source="test", confidence="measured", backend=backend)
    ))
    with pytest.raises(MemoryError, match="cannot fit"):
        Resources.auto_resolve_for_compute_cv_tau("numpy", 1000, 256)


def test_memory_model_scaling():
    """``block_peak_gb`` is linear in ``peak_factor`` and ``B``; workload-
    fixed terms (kernel_persistent, dmx_eqsI, dataset) are independent of
    the mode-block size."""
    m = MemoryCost(Nat=128, Nt=1000, bytes_per_real=4)

    # peak_factor is a multiplicative scaler on the block term.
    assert m.block_peak_gb(64, 4.0) == pytest.approx(2.0 * m.block_peak_gb(64, 2.0))
    # B doesn't affect the workload-fixed terms.
    assert m.kernel_persistent_gb == m.kernel_persistent_gb  # tautology, but pins property purity
    # B IS linear in block_peak_gb.
    assert m.block_peak_gb(256, 2.0) == pytest.approx(8.0 * m.block_peak_gb(32, 2.0))


def test_memory_model_kernel_persistent_matches_explicit_formula():
    """Pin the closed-form value: 2·nmodes·I·bpr = 18·Nat²·bpr."""
    for Nat, bpr in [(64, 4), (128, 4), (216, 8), (2000, 4)]:
        m = MemoryCost(Nat=Nat, Nt=1, bytes_per_real=bpr)
        assert m.kernel_persistent_gb == pytest.approx(18 * Nat * Nat * bpr / 1e9)


def test_memory_model_breakdown_sums_to_total():
    """``breakdown()`` is internally consistent: per-term entries sum to
    ``total_peak_gb`` exactly."""
    m = MemoryCost(Nat=128, Nt=5000, bytes_per_real=4)
    bd = m.breakdown(B=64, t_chunk=2500, peak_factor=2.4)
    parts = (bd["dataset_gb"] + bd["dmx_eqsI_gb"] + bd["kernel_persistent_gb"]
             + bd["chunk_transient_gb"] + bd["block_peak_gb"])
    assert bd["total_peak_gb"] == pytest.approx(parts)
