from __future__ import annotations

from .base import Algorithm


class SFTAlgorithm(Algorithm):
    def update(self, batch: dict) -> dict:
        raise NotImplementedError("SFT training loop is reserved for learner integration.")

