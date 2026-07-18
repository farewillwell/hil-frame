from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WeightManager:
    checkpoint: str | None = None

    def latest_checkpoint(self) -> str | None:
        return self.checkpoint

