from __future__ import annotations

from .base import HumanControllerBase


class SpaceMouseController(HumanControllerBase):
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("SpaceMouseController interface is reserved; no device logic is faked.")

