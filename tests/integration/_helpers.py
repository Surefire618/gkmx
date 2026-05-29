"""Shared test helpers."""


def rel_max(a, b, scale_floor=1e-30):
    """Relative absolute difference between two scalars, clipped at `scale_floor`.

    Replaces the recurring `_rel(a, b)` / `_rel_diff(a, b)` pattern in
    `test_backends_agree.py` and `test_precision_switch.py`. Returns
    `|a - b| / max(|a|, |b|, scale_floor)`.
    """
    a, b = float(a), float(b)
    scale = max(abs(a), abs(b), scale_floor)
    return abs(a - b) / scale
