from __future__ import annotations

from dataclasses import dataclass

from .human.base import HumanControlPhase


@dataclass
class HumanPhaseTracker:
    previous: HumanControlPhase = HumanControlPhase.IDLE

    def update(self, current: HumanControlPhase) -> tuple[bool, bool]:
        entered = self.previous != HumanControlPhase.ACTIVE and current == HumanControlPhase.ACTIVE
        exited = self.previous == HumanControlPhase.ACTIVE and current != HumanControlPhase.ACTIVE
        self.previous = current
        return entered, exited

