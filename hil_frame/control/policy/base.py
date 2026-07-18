from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from hil_frame.control.base import ControlContext
from hil_frame.env.base import ObservationFrame


@dataclass(frozen=True)
class PolicyProposal:
    proposal_id: str
    trajectory_id: str
    source_obs_id: int
    actions: np.ndarray
    policy_version: str
    created_timestamp_ns: int
    received_timestamp_ns: int


@dataclass(frozen=True)
class ObservationRequest:
    request_id: str
    trajectory_id: str
    obs_id: int
    observation: dict[str, np.ndarray]
    task_description: str
    sent_timestamp_ns: int


class PolicyActionSource(ABC):
    @abstractmethod
    def submit_observation(self, observation: ObservationFrame, context: ControlContext) -> None:
        ...

    @abstractmethod
    def get_latest_proposal(self) -> PolicyProposal | None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

