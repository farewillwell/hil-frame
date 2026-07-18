from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Policy(ABC):
    @abstractmethod
    def infer(self, observation: dict[str, np.ndarray], task_description: str) -> np.ndarray:
        """Return an action chunk with shape [action_horizon, action_dim]."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        ...

