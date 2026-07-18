from __future__ import annotations

from .base import Algorithm


class IQLAWBCAlgorithm(Algorithm):
    def update(self, batch: dict) -> dict:
        raise NotImplementedError("IQL/AWBC training loop is reserved for learner integration.")

