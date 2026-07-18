from __future__ import annotations

import numpy as np

from hil_frame.env.base import ActionSpec


class SafetyController:
    def __init__(self, action_spec: ActionSpec, mode: str = "zero") -> None:
        self.action_spec = action_spec
        self.mode = mode

    def fallback_action(self) -> np.ndarray:
        return np.zeros(self.action_spec.shape, dtype=np.float32)

    def stopped_action(self) -> np.ndarray:
        return self.fallback_action()

