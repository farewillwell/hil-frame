from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np


class HumanControlPhase(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    STOPPED = "stopped"


@dataclass(frozen=True)
class HumanControlSnapshot:
    phase: HumanControlPhase
    action: np.ndarray | None
    timestamp_ns: int
    quit_requested: bool = False


class HumanControllerBase(ABC):
    @abstractmethod
    def enter_control(self) -> None:
        ...

    @abstractmethod
    def exit_control(self) -> None:
        ...

    @abstractmethod
    def get_action(self) -> np.ndarray:
        ...

    @abstractmethod
    def stop_and_reset(self) -> None:
        ...

    @abstractmethod
    def get_snapshot(self) -> HumanControlSnapshot:
        ...

    @abstractmethod
    def acknowledge_reset(self) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

