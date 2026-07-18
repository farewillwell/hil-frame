from __future__ import annotations

import numpy as np

from .base import Policy


class OpenPIPolicy(Policy):
    def __init__(self, checkpoint: str | None = None, action_horizon: int = 10) -> None:
        self.checkpoint = checkpoint
        self.action_horizon = action_horizon

    def infer(self, observation: dict[str, np.ndarray], task_description: str) -> np.ndarray:
        raise NotImplementedError("OpenPIPolicy adapter is reserved; install/configure OpenPI before enabling it.")

    @property
    def version(self) -> str:
        return f"openpi:{self.checkpoint or 'unconfigured'}"

