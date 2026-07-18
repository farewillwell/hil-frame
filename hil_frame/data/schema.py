from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hil_frame.control.base import ActionSource
from hil_frame.env.base import ObservationFrame


@dataclass(frozen=True)
class TrajectoryStep:
    step_id: int
    observation: ObservationFrame
    action: np.ndarray
    action_source: ActionSource
    policy_action: np.ndarray | None
    human_action: np.ndarray | None
    reward: float
    terminated: bool
    truncated: bool
    proposal_id: str | None
    chunk_index: int | None
    policy_version: str | None
    action_timestamp_ns: int


@dataclass(frozen=True)
class TrajectoryRecord:
    trajectory_id: str
    run_id: str
    task_suite: str
    task_id: int
    task_description: str
    seed: int
    initial_state_id: int | None
    steps: tuple[TrajectoryStep, ...]
    final_observation: ObservationFrame
    success: bool
    termination_reason: str
    episode_return: float
    start_timestamp_ns: int
    end_timestamp_ns: int


@dataclass(frozen=True)
class Transition:
    trajectory_id: str
    step_id: int
    observation: dict[str, np.ndarray]
    action: np.ndarray
    next_observation: dict[str, np.ndarray]
    reward: float
    terminated: bool
    truncated: bool
    done: bool
    action_source: ActionSource
    policy_action: np.ndarray | None
    human_action: np.ndarray | None
    trajectory_success: bool
    task_id: int


@dataclass(frozen=True)
class TrajectoryUpload:
    trajectory_id: str
    checksum: str
    payload: bytes


@dataclass(frozen=True)
class TrajectoryAck:
    trajectory_id: str
    checksum: str
    stored: bool
    error: str | None

