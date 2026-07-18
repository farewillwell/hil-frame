from __future__ import annotations

import numpy as np

from .base import Policy


class ZeroPolicy(Policy):
    def __init__(self, action_dim: int = 7, action_horizon: int = 1, version: str = "zero") -> None:
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self._version = version

    def infer(self, observation: dict[str, np.ndarray], task_description: str) -> np.ndarray:
        return np.zeros((self.action_horizon, self.action_dim), dtype=np.float32)

    @property
    def version(self) -> str:
        return self._version

