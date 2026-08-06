"""Detect available device/host/scheduler resources; resolve ``max_mem_gb`` for compute_cv_tau."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import ClassVar

_BYTES_PER_GB = 1e9
_KIB = 1024
_MIB = 1024 * 1024

def mode_block_peak_gb(B: int, Nt: int, bytes_per_complex: int,
                       peak_factor: float) -> float:
    """Per-mode-block ACF peak: 3 simultaneous ``(B, Nt)`` complex arrays × ``peak_factor``.

    The runtime bottleneck of ``compute_cv_tau`` (``a_re``, ``a_im``, and a
    transient ``a_blk`` coexist). Single source for the peak formula, shared by
    ``MemoryCost.block_peak_gb`` and ``compute_cv_tau``'s ``mode_block`` sizer.
    ``peak_factor`` absorbs FFT scratch + on-device projection intermediates;
    the kernel sizer passes ``1.0`` and sizes against the raw allocation.
    """
    return peak_factor * 3 * B * Nt * bytes_per_complex / _BYTES_PER_GB


@dataclass(frozen=True, slots=True)
class MemoryCost:
    """Memory cost model for ``compute_cv_tau`` — one term per allocation.

    Single source of truth for the kernel's footprint. Each ``*_gb``
    accessor cites the allocation site so future profiling can audit
    and refine the formula.

    Term taxonomy:
      - Workload-fixed (depend only on ``Nat``/``Nt``/``bytes_per_real``):
        ``dataset_gb``, ``dmx_eqsI_gb``, ``kernel_persistent_gb``.
      - Knob-dependent (take a runtime parameter):
        ``chunk_transient_gb(t_chunk)``, ``block_peak_gb(B, peak_factor)``.

    The headroom check in ``Resources.auto_resolve_for_compute_cv_tau``
    subtracts only ``kernel_persistent_gb`` — the other persistent terms
    (``dataset_gb``, ``dmx_eqsI_gb``) and the chunk transient are absorbed
    into the empirical ``peak_factor`` calibration. Use ``breakdown()``
    to inspect every term independently for profiling.

    Attributes:
        Nat: supercell atom count (``N_sc``).
        Nt: trajectory length after ``stride``.
        bytes_per_real: dtype width — 4 for fp32, 8 for fp64.
    """
    Nat: int
    Nt: int
    bytes_per_real: int = 4

    @property
    def nmodes(self) -> int:
        """``Nq · Ns = (N_sc/N_p) · (3·N_p) = 3·Nat``."""
        return 3 * self.Nat

    @property
    def I(self) -> int:
        """Supercell-basis eigenvector dim: ``3 · N_sc = 3·Nat``."""
        return 3 * self.Nat

    @property
    def bytes_per_complex(self) -> int:
        return 2 * self.bytes_per_real

    # --- workload-fixed terms ------------------------------------------

    @property
    def dataset_gb(self) -> float:
        """``positions`` + ``velocities`` resident in the xarray dataset.

        Upper bound assuming eager (non-dask) load. ``compute_cv_tau``
        reads positions + velocities only; forces are consumed by
        ``_harmonic_force_residuals`` on a separate code path.
        """
        return 2 * self.Nt * self.Nat * 3 * self.bytes_per_real / _BYTES_PER_GB

    @property
    def dmx_eqsI_gb(self) -> float:
        """Complex ``e_qsI`` held by the ``DynamicalMatrix`` object.

        Shape ``(Nq, Ns, I)`` = ``(nmodes, I)`` complex after reshape.
        Coexists with ``kernel_persistent_gb`` for the entire call
        (line 285 keeps a ``dmx.e_qsI`` reference alive).
        """
        return self.nmodes * self.I * self.bytes_per_complex / _BYTES_PER_GB

    @property
    def kernel_persistent_gb(self) -> float:
        """``e_re_np`` + ``e_im_np`` (greenkubo.py:291–292).

        Two real ``(nmodes, I)`` halves of the complex ``e_qsI`` so the
        projection is two real matmuls instead of one complex. Both
        persist for the entire call.
        """
        return 2 * self.nmodes * self.I * self.bytes_per_real / _BYTES_PER_GB

    # --- knob-dependent terms ------------------------------------------

    def chunk_transient_gb(self, t_chunk: int) -> float:
        """Per-projection-chunk staging (greenkubo.py:349–361).

        ``U`` + ``V`` (positions / velocities chunks) plus the
        interleaved ``up_np`` of shape ``(I, 2·t_chunk)``. All freed at
        chunk end; only one chunk is in flight. Negligible (~30 MB at
        Nat=128, t_chunk=5000, fp32) but listed for auditability.
        """
        # U + V: 2 × (t_chunk, Nat, 3) = 2·3·t_chunk·Nat reals
        # up_np: (I, 2·t_chunk) = 6·t_chunk·Nat reals
        return (2 * 3 + 6) * t_chunk * self.Nat * self.bytes_per_real / _BYTES_PER_GB

    def block_peak_gb(self, B: int, peak_factor: float) -> float:
        """Per-mode-block ACF peak — the runtime bottleneck. Delegates to
        ``mode_block_peak_gb``. ``peak_factor`` absorbs FFT scratch, on-device
        projection intermediates, and the other persistent terms.
        """
        return mode_block_peak_gb(B, self.Nt, self.bytes_per_complex, peak_factor)

    # --- aggregates / introspection ------------------------------------

    def total_peak_gb(self, B: int, t_chunk: int, peak_factor: float) -> float:
        """Sum of every modeled term."""
        return (self.dataset_gb + self.dmx_eqsI_gb + self.kernel_persistent_gb
                + self.chunk_transient_gb(t_chunk)
                + self.block_peak_gb(B, peak_factor))

    def breakdown(self, B: int, t_chunk: int, peak_factor: float) -> dict:
        """Per-term GB dump for profiling / diagnostic prints."""
        return {
            "Nat": self.Nat, "Nt": self.Nt, "B": B, "t_chunk": t_chunk,
            "bytes_per_real": self.bytes_per_real,
            "peak_factor": peak_factor,
            "dataset_gb": self.dataset_gb,
            "dmx_eqsI_gb": self.dmx_eqsI_gb,
            "kernel_persistent_gb": self.kernel_persistent_gb,
            "chunk_transient_gb": self.chunk_transient_gb(t_chunk),
            "block_peak_gb": self.block_peak_gb(B, peak_factor),
            "total_peak_gb": self.total_peak_gb(B, t_chunk, peak_factor),
        }


@dataclass(frozen=True, slots=True)
class Plan:
    """Resolved ``compute_cv_tau`` memory plan.

    Attributes:
        max_mem_gb: budget for the per-mode-block ACF buffer.
        origin: ``"env GKMX_MAX_MEM_GB"`` for the override path,
            ``"detect+cost"`` when sized from a detected Resources.
        resources: detection record on the detect+cost path; ``None``
            on the env-override path.
        cost: per-allocation cost model. Populated on both paths so
            callers can introspect the predicted footprint.
        target_block_gb: per-block target before division by
            ``peak_factor``; zero on the env-override path.
        peak_factor: backend peak / budget ratio used; zero on the
            env-override path.
    """
    max_mem_gb: float
    origin: str
    resources: Resources | None = None
    cost: MemoryCost | None = None
    target_block_gb: float = 0.0
    peak_factor: float = 0.0


@dataclass(frozen=True, slots=True)
class Resources:
    """Detected free memory with provenance and budget arithmetic.

    Detection probes, validators, the orchestrator (``detect``), and the
    compute_cv_tau resolver (``auto_resolve_for_compute_cv_tau``) all
    live here so callers have a single object to import.

    Attributes:
        free_gb: detected free memory in GB.
        source: where the number came from
            (``"pynvml"``, ``"jax_memstats"``, ``"cgroup_v2"``,
             ``"cgroup_v1"``, ``"psutil"``, ``"proc_meminfo"``,
             ``"slurm_env_per_node"``, ``"slurm_env_per_cpu"``,
             ``"default"`` or a combination joined by ``"∩"``).
        confidence: ``"measured"`` (real reading), ``"estimated"``
            (env-var arithmetic), or ``"default"`` (silent fallback).
        backend: ``"numpy"`` or ``"jax"``.
        notes: warnings to surface at the call site.
    """
    free_gb: float
    source: str
    confidence: str
    backend: str
    notes: tuple = ()

    DEFAULT_FREE_GB: ClassVar[float] = 4.0

    # Empirical peak / budget ratio (2026-04-27 calibration). cuFFT batched-plan
    # scratch dwarfs scipy.fft scratch, so jax peak_factor must be >= 4.
    # Override per-backend via GKMX_PEAK_FACTOR_NUMPY / GKMX_PEAK_FACTOR_JAX.
    PEAK_FACTORS: ClassVar[dict] = {
        "numpy": 2.4,
        "jax":   5.0,
    }

    # -----------------------------------------------------------------
    # Detection probes
    # -----------------------------------------------------------------

    @staticmethod
    def _gpu_pynvml_free_gb() -> tuple[float, str] | None:
        """Try pynvml. Reports GPU-wide free across all processes (the truthful
        value on shared GPUs, unlike JAX's per-process quota)."""
        try:
            import pynvml
        except ImportError:
            return None
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        except Exception:  # pynvml.NVMLError isn't a stdlib type; rely on broad catch
            return None
        return float(info.free) / _BYTES_PER_GB, "pynvml"

    @staticmethod
    def _gpu_jax_memstats_free_gb() -> tuple[float, str] | None:
        """JAX device.memory_stats(). Process-local; under-reports on shared GPUs."""
        try:
            import jax
        except ImportError:
            return None
        try:
            d = jax.devices()[0]
        except (RuntimeError, AttributeError, IndexError):
            return None
        if d.platform == "cpu":
            return None
        try:
            stats = d.memory_stats()
        except (AttributeError, RuntimeError):
            return None
        if not stats:
            return None
        limit = stats.get("bytes_limit") or stats.get("bytes_reservable_limit")
        in_use = stats.get("bytes_in_use", 0)
        if not limit:
            return None
        return float(limit - in_use) / _BYTES_PER_GB, "jax_memstats"

    @staticmethod
    def _cgroup_memory_limit_gb() -> tuple[float, str] | None:
        """Read cgroup memory limit directly (v2 then v1). Authoritative for
        SLURM/Kubernetes/container workloads."""
        # cgroup v2
        try:
            with open("/sys/fs/cgroup/memory.max") as f:
                raw_max = f.read().strip()
            if raw_max == "max":
                return None
            max_bytes = int(raw_max)
            try:
                with open("/sys/fs/cgroup/memory.current") as f:
                    current = int(f.read().strip())
            except (FileNotFoundError, PermissionError, ValueError):
                current = 0
            return float(max_bytes - current) / _BYTES_PER_GB, "cgroup_v2"
        except (FileNotFoundError, PermissionError, ValueError):
            pass
        # cgroup v1
        try:
            with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as f:
                max_bytes = int(f.read().strip())
            # v1 uses a ~max-int64 sentinel for "no limit"
            if max_bytes > 2**60:
                return None
            try:
                with open("/sys/fs/cgroup/memory/memory.usage_in_bytes") as f:
                    current = int(f.read().strip())
            except (FileNotFoundError, PermissionError, ValueError):
                current = 0
            return float(max_bytes - current) / _BYTES_PER_GB, "cgroup_v1"
        except (FileNotFoundError, PermissionError, ValueError):
            pass
        return None

    @staticmethod
    def _slurm_env_limit_gb() -> tuple[float, str] | None:
        """Fallback: parse SLURM_MEM_PER_NODE / SLURM_MEM_PER_CPU env vars.

        Used only when cgroup reads fail. Less accurate (excludes
        page-cache / driver buffers) and confused by heterogeneous nodes —
        cgroup is the authoritative source.
        """
        mem_per_node = os.environ.get("SLURM_MEM_PER_NODE")
        if mem_per_node:
            try:
                return float(mem_per_node) * _MIB / _BYTES_PER_GB, "slurm_env_per_node"
            except ValueError:
                pass
        mem_per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
        cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
        if mem_per_cpu and cpus_per_task:
            try:
                return (float(mem_per_cpu) * int(cpus_per_task) * _MIB
                        / _BYTES_PER_GB), "slurm_env_per_cpu"
            except ValueError:
                pass
        return None

    @staticmethod
    def _host_available_gb() -> tuple[float, str] | None:
        """Host MemAvailable via psutil, then /proc/meminfo."""
        try:
            import psutil
            return float(psutil.virtual_memory().available) / _BYTES_PER_GB, "psutil"
        except ImportError:
            pass
        except (OSError, AttributeError):
            pass
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return float(line.split()[1]) * _KIB / _BYTES_PER_GB, "proc_meminfo"
        except (FileNotFoundError, PermissionError):
            pass
        return None

    # -----------------------------------------------------------------
    # Orchestrator
    # -----------------------------------------------------------------

    @classmethod
    def detect(cls, backend: str) -> Resources:
        """Detect free memory for ``backend`` (``"numpy"`` or ``"jax"``).

        Order of preference, most-trustworthy first:
            GPU (jax + real device):  pynvml > jax_memstats
            Host/scheduler:           cgroup (v2/v1) ∩ host MemAvailable >
                                      SLURM env vars

        On total failure returns ``Resources(free_gb=4.0,
        confidence="default")`` with a ``notes`` entry explaining why.
        """
        notes = []

        if backend == "jax":
            for fn in (cls._gpu_pynvml_free_gb, cls._gpu_jax_memstats_free_gb):
                result = fn()
                if result is not None:
                    free_gb, source = result
                    return cls(
                        free_gb=free_gb, source=source,
                        confidence="measured", backend=backend,
                    )
            notes.append(
                "GPU detection failed (pynvml + jax_memstats unavailable); "
                "falling back to host RAM detection."
            )

        cgroup = cls._cgroup_memory_limit_gb()
        host = cls._host_available_gb()
        slurm_env = cls._slurm_env_limit_gb()

        # Most authoritative: cgroup ∩ host MemAvailable.
        if cgroup is not None and host is not None:
            cgroup_gb, cgroup_src = cgroup
            host_gb, host_src = host
            return cls(
                free_gb=min(cgroup_gb, host_gb),
                source=f"{cgroup_src}∩{host_src}",
                confidence="measured", backend=backend,
                notes=tuple(notes),
            )
        if cgroup is not None:
            free_gb, source = cgroup
            return cls(
                free_gb=free_gb, source=source,
                confidence="measured", backend=backend,
                notes=tuple(notes),
            )
        if host is not None:
            free_gb, source = host
            # SLURM env can tighten the host reading.
            if slurm_env is not None:
                slurm_gb, slurm_src = slurm_env
                if slurm_gb < free_gb:
                    return cls(
                        free_gb=slurm_gb,
                        source=f"{slurm_src}∩{source}",
                        confidence="measured", backend=backend,
                        notes=tuple(notes),
                    )
            return cls(
                free_gb=free_gb, source=source,
                confidence="measured", backend=backend,
                notes=tuple(notes),
            )
        if slurm_env is not None:
            free_gb, source = slurm_env
            return cls(
                free_gb=free_gb, source=source,
                confidence="estimated", backend=backend,
                notes=tuple(notes),
            )

        notes.append(
            f"All resource detection failed; falling back to {cls.DEFAULT_FREE_GB} GB. "
            "Set GKMX_MAX_MEM_GB explicitly to override."
        )
        return cls(
            free_gb=cls.DEFAULT_FREE_GB, source="default", confidence="default",
            backend=backend, notes=tuple(notes),
        )

    # -----------------------------------------------------------------
    # Budget arithmetic
    # -----------------------------------------------------------------

    @classmethod
    def peak_factor(cls, backend: str) -> float:
        """Per-backend peak factor; env override via ``GKMX_PEAK_FACTOR_<BACKEND>``."""
        env_key = f"GKMX_PEAK_FACTOR_{backend.upper()}"
        raw = os.environ.get(env_key)
        default = cls.PEAK_FACTORS.get(backend, cls.PEAK_FACTORS["numpy"])
        if raw:
            try:
                value = float(raw)
                if value <= 0:
                    raise ValueError("must be positive")
                return value
            except ValueError as exc:
                warnings.warn(
                    f"Ignoring invalid {env_key}={raw!r} ({exc}); "
                    f"using built-in default {default}"
                )
        return default

    @staticmethod
    def validate_override(raw_str: str) -> float | None:
        """Parse ``GKMX_MAX_MEM_GB`` value; warn and return None on invalid input."""
        try:
            value = float(raw_str)
        except ValueError:
            warnings.warn(f"Ignoring invalid GKMX_MAX_MEM_GB={raw_str!r}; not a number")
            return None
        if value <= 0:
            warnings.warn(f"Ignoring non-positive GKMX_MAX_MEM_GB={value}")
            return None
        return value

    @classmethod
    def auto_resolve_for_compute_cv_tau(
        cls,
        backend: str, Nt: int, Nat: int,
        *,
        bytes_per_real: int = 4,
        reserve_gb: float = 1.0,
        safety_buffer: float = 0.7,
        peak_factor: float | None = None,
        floor_gb: float = 0.5,
    ) -> Plan:
        """Resolve max_mem_gb at compute_cv_tau entry.

        Raises:
            MemoryError: when the kernel persistent + reserve already exceeds
                detected free memory, so the kernel cannot fit at any
                mode_block — fail-loud instead of silently undersizing.
        """
        cost = MemoryCost(Nat=Nat, Nt=Nt, bytes_per_real=bytes_per_real)

        raw_override = os.environ.get("GKMX_MAX_MEM_GB")
        if raw_override:
            validated = cls.validate_override(raw_override)
            if validated is not None:
                return Plan(max_mem_gb=validated, origin="env GKMX_MAX_MEM_GB",
                            cost=cost)

        if peak_factor is None:
            peak_factor = cls.peak_factor(backend)

        resources = cls.detect(backend)
        for note in resources.notes:
            warnings.warn(note)

        kernel_persistent_gb = cost.kernel_persistent_gb
        headroom = resources.free_gb - reserve_gb - kernel_persistent_gb
        if headroom <= 0:
            raise MemoryError(
                f"compute_cv_tau cannot fit: kernel-persistent "
                f"{kernel_persistent_gb:.2f} GB + reserve {reserve_gb:.2f} GB "
                f"exceeds detected free {resources.free_gb:.2f} GB "
                f"(source={resources.source}, confidence={resources.confidence}). "
                f"Set GKMX_MAX_MEM_GB explicitly, use a larger device, "
                f"or shrink Nt/Nat."
            )

        target_block_GB = headroom * safety_buffer
        max_mem_gb = max(floor_gb, target_block_GB / peak_factor)

        return Plan(
            max_mem_gb=max_mem_gb,
            origin="detect+cost",
            resources=resources,
            cost=cost,
            target_block_gb=target_block_GB,
            peak_factor=peak_factor,
        )
