from __future__ import annotations

import numpy as np

from hil_frame.control.base import ActionSource

from .schema import TrajectoryRecord
from .serialization import checksum_bytes


def validate_payload_checksum(payload: bytes, checksum: str) -> None:
    actual = checksum_bytes(payload)
    if actual != checksum:
        raise ValueError(f"checksum mismatch: expected {checksum}, got {actual}")


def validate_trajectory(record: TrajectoryRecord) -> None:
    if not record.trajectory_id:
        raise ValueError("trajectory_id must be non-empty")
    if not record.steps:
        raise ValueError("trajectory must contain at least one step")
    if record.final_observation is None:
        raise ValueError("final_observation is required")
    if not isinstance(record.success, bool):
        raise ValueError("success must be bool")
    expected_shape = record.steps[0].action.shape
    for expected_step_id, step in enumerate(record.steps):
        if step.step_id != expected_step_id:
            raise ValueError("step_id must be contiguous from 0")
        if step.action_source not in set(ActionSource):
            raise ValueError("invalid action_source")
        if step.action.shape != expected_shape:
            raise ValueError("action shapes must be consistent")
        if not np.all(np.isfinite(step.action)):
            raise ValueError("action contains NaN or Inf")
        if step.action_source == ActionSource.HUMAN and step.human_action is None:
            raise ValueError("HUMAN step requires human_action")
        if step.action_source == ActionSource.POLICY and step.policy_action is None:
            raise ValueError("POLICY step requires policy_action")
        if step.terminated and step.truncated:
            raise ValueError("terminated and truncated cannot both be True")
    last = record.steps[-1]
    if record.termination_reason == "success" and not record.success:
        raise ValueError("success termination requires success=True")
    if record.termination_reason in {"terminated", "success"} and not last.terminated:
        raise ValueError("terminated/success termination requires final step terminated")
    if record.termination_reason == "truncated" and not last.truncated:
        raise ValueError("truncated termination requires final step truncated")

