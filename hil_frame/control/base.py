from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


@dataclass(frozen=True)
class ControlContext:
    trajectory_id: str
    step_id: int
    obs_id: int
    task_description: str


class ActionSource(str, Enum):
    POLICY = "policy"
    HUMAN = "human"
    SAFETY = "safety"
    ZERO_FALLBACK = "zero_fallback"


@dataclass(frozen=True)
class ControlOutput:
    action: np.ndarray
    action_source: ActionSource
    policy_action: np.ndarray | None
    human_action: np.ndarray | None
    proposal_id: str | None
    chunk_index: int | None
    policy_version: str | None
    stop_requested: bool
    reset_requested: bool
    decision_timestamp_ns: int

