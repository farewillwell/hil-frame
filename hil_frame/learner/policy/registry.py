from __future__ import annotations

from .openpi import OpenPIPolicy
from .random_policy import RandomPolicy
from .zero_policy import ZeroPolicy


def build_policy(name: str, **kwargs):
    if name == "zero":
        return ZeroPolicy(**kwargs)
    if name == "random":
        return RandomPolicy(**kwargs)
    if name == "openpi":
        return OpenPIPolicy(**kwargs)
    raise ValueError(f"unknown policy backend {name!r}")

