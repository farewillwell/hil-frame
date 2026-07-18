from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObservationFrame:
    obs_id: int
    timestamp_ns: int
    data: dict[str, np.ndarray]


@dataclass(frozen=True)
class ActionSpec:
    shape: tuple[int, ...]
    low: np.ndarray
    high: np.ndarray
    dtype: str


@dataclass(frozen=True)
class StepResult:
    observation: ObservationFrame
    reward: float
    terminated: bool
    truncated: bool
    success: bool
    applied_action: np.ndarray
    info: dict


class RobotEnv(ABC):
    @property
    @abstractmethod
    def action_spec(self) -> ActionSpec:
        ...

    @property
    @abstractmethod
    def task_info(self) -> dict:
        ...

    @abstractmethod
    def reset(self) -> ObservationFrame:
        ...

    @abstractmethod
    def get_observation(self) -> ObservationFrame:
        ...

    @abstractmethod
    def step(self, action: np.ndarray) -> StepResult:
        ...

    @abstractmethod
    def render(self) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

