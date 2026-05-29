"""Minimal logging for gkmx."""

import time
import warnings


def talk(msg, prefix="gkmx"):
    if isinstance(msg, (list, tuple)):
        for m in msg:
            print(f"[{prefix}]  {m}")
    else:
        print(f"[{prefix}]  {msg}")


def warn(msg, prefix="gkmx", error=False):
    if error:
        raise RuntimeError(f"[{prefix}]  {msg}")
    warnings.warn(f"[{prefix}]  {msg}", stacklevel=2)


class Timer:
    def __init__(self, label="", prefix="gkmx", verbose=True):
        self.label = label
        self.prefix = prefix
        self.verbose = verbose
        self._t0 = time.perf_counter()
        if verbose and label:
            print(f"[{prefix}]  {label} ...", flush=True)

    def elapsed(self):
        return time.perf_counter() - self._t0

    def __call__(self, label=None):
        dt = self.elapsed()
        label = label or self.label
        if self.verbose and label:
            print(f"[{self.prefix}]  {label} done ({dt:.3f}s)")
        return dt
