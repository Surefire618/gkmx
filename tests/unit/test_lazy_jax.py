"""``import gkmx`` must not pull in jax.

jax is an optional backend; importing gkmx should cost nothing for the numpy
users who never ask for it.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

_PROBE = """
import sys, json
import gkmx, gkmx.cli
print(json.dumps(sorted(m for m in sys.modules
                        if m.split(".")[0] in {"jax", "jaxlib"})))
"""


def test_import_gkmx_does_not_import_jax():
    """Import gkmx in a fresh interpreter and check jax is absent from sys.modules.

    The subprocess is required: pytest and its plugins may already have imported
    jax, so sys.modules in this process says nothing.

    Skipped without jax installed, where sys.modules is empty of jax either way.
    """
    pytest.importorskip("jax")

    r = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True,
                       check=False)
    assert r.returncode == 0, f"probe process failed:\n{r.stderr}"

    leaked = json.loads(r.stdout)
    assert not leaked, f"import gkmx loaded {leaked}; the jax backend must stay lazy"
