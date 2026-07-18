from __future__ import annotations

import numpy as np

from .base import Policy


class RandomPolicy(Policy):
    def __init__(self, action_dim: int = 7, action_horizon: int = 1, scale: float = 0.05, seed: int = 0) -> None:
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.scale = scale
        self.rng = np.random.default_rng(seed)

    def infer(self, observation: dict[str, np.ndarray], task_description: str) -> np.ndarray:
        return self.rng.uniform(-self.scale, self.scale, size=(self.action_horizon, self.action_dim)).astype(np.float32)

    @property
    def version(self) -> str:
        return f"random-scale-{self.scale}"

