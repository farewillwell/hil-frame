from __future__ import annotations

from .base import Algorithm


class RLPDAlgorithm(Algorithm):
    def update(self, batch: dict) -> dict:
        raise NotImplementedError("RLPD training loop is reserved for learner integration.")

