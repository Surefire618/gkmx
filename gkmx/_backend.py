"""Backend dispatch for gkmx compute kernels: numpy (CPU) or jax (CPU/GPU)."""

from __future__ import annotations

import os
from typing import Optional

from ._log import warn as _warn


class Backend:
    __slots__ = ("name", "xp", "fft", "ifft", "rfft", "irfft",
                 "next_fast_len", "to_device", "to_host")

    def __init__(self, name: str = "numpy"):
        self.name = name
        if name == "numpy":
            import numpy as np
            from scipy.fft import fft, ifft, rfft, irfft, next_fast_len
            self.xp = np
            self.fft = fft
            self.ifft = ifft
            self.rfft = rfft
            self.irfft = irfft
            self.next_fast_len = next_fast_len
            self.to_device = _identity
            self.to_host = _identity
        elif name == "jax":
            import numpy as np
            import jax.numpy as jnp
            from scipy.fft import next_fast_len
            self.xp = jnp
            self.fft = jnp.fft.fft
            self.ifft = jnp.fft.ifft
            self.rfft = jnp.fft.rfft
            self.irfft = jnp.fft.irfft
            self.next_fast_len = next_fast_len
            self.to_device = jnp.asarray
            self.to_host = np.asarray
        else:
            raise ValueError(f"Unknown backend: {name!r}")

    def device_description(self) -> str:
        """Short human-readable description of the compute device."""
        if self.name == "numpy":
            n = os.environ.get(
                "OPENBLAS_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "?"))
            return f"cpu / {n} threads"
        if self.name == "jax":
            try:
                import jax
                dev = jax.devices()[0]
                kind = getattr(dev, "device_kind", None) or str(dev)
                return f"{dev.platform} / {kind}"
            except Exception:
                return "jax / unknown"
        return self.name

    def jit(self, fn):
        if self.name == "jax":
            import jax
            return jax.jit(fn)
        return fn

    def complex(self, re, im, *, dtype):
        """Assemble a complex array from real + imag parts at the given dtype."""
        if self.name == "numpy":
            import numpy as np
            out = np.empty(re.shape, dtype=dtype)
            out.real = re
            out.imag = im
            return out
        return (re.astype(dtype) + 1j * im.astype(dtype))

    def fft_autocorrelate(self, x, *, nfft: int, tau_max: Optional[int] = None,
                           norm=None, hann=None):
        """FFT autocorrelation ``ifft(|fft(x)|^2)``, sliced to tau_max with optional norm and Hann taper."""
        xp = self.xp
        F = self.fft(x, n=nfft, axis=-1)
        C = self.ifft(F * xp.conj(F), axis=-1)
        if tau_max is not None:
            C = C[..., :tau_max]
        if norm is not None:
            if getattr(norm, "shape", ()) == ():
                C = C / norm
            else:
                C = C / norm[..., :C.shape[-1]]
        if hann is not None:
            C = C * hann[None, :]
        return C

    def resolve_dtypes(self, dtype_u, dtype_a=None):
        """Downgrade fp64 dtypes to fp32 when JAX has JAX_ENABLE_X64 off."""
        import numpy as np
        if self.name != "jax":
            return dtype_u, dtype_a
        fp64_requested = (dtype_u == np.float64) or (dtype_a == np.complex128)
        if not fp64_requested:
            return dtype_u, dtype_a
        try:
            import jax
            x64_on = bool(jax.config.read("jax_enable_x64"))
        except Exception:
            x64_on = False
        if x64_on:
            return dtype_u, dtype_a
        _warn(
            "JAX backend was selected with fp64 precision but JAX_ENABLE_X64 "
            "is not set; JAX will silently truncate fp64 → fp32 at every "
            "asarray call. Downgrading dtype_u/dtype_a to float32/complex64 "
            "so the kernel runs at a consistent precision. To actually run "
            "fp64 on the JAX backend, set JAX_ENABLE_X64=1 in the environment "
            "(or call `jax.config.update('jax_enable_x64', True)` before any "
            "jax import).",
            prefix="gkmx",
        )
        if dtype_u == np.float64:
            dtype_u = np.float32
        if dtype_a == np.complex128:
            dtype_a = np.complex64
        return dtype_u, dtype_a

    def rfft_power_sum(self, data_2d, *, nfft: int, max_mem_gb: float = 4.0,
                        norm=None, hann=None):
        """Row-summed ACF via batched rfft, blocked by max_mem_gb."""
        import numpy as np
        M, Nt = data_2d.shape
        dtype = np.dtype(data_2d.dtype)
        bytes_per_row = (nfft // 2 + 1) * (dtype.itemsize * 2)
        block = max(1, min(M, int(max_mem_gb * 1e9 / bytes_per_row)))

        def _block_spec(d_block):
            F = self.rfft(d_block, n=nfft, axis=-1)
            return (F * self.xp.conj(F)).sum(axis=0).real
        _block_spec = self.jit(_block_spec)

        S = self.xp.zeros(nfft // 2 + 1, dtype=dtype)
        for m0 in range(0, M, block):
            m1 = min(M, m0 + block)
            S = S + _block_spec(self.to_device(data_2d[m0:m1]))
        acf_sum = self.irfft(S, n=nfft)[:Nt]

        if norm is not None:
            if getattr(norm, "shape", ()) == ():
                acf_sum = acf_sum / norm
            else:
                acf_sum = acf_sum / norm[:Nt]
        if hann is not None:
            acf_sum = acf_sum * hann
        return np.asarray(self.to_host(acf_sum)).astype(dtype)


def _identity(a):
    return a


def get_backend(name: str = "numpy") -> Backend:
    return Backend(name)
