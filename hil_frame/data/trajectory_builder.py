from __future__ import annotations

from dataclasses import replace

import numpy as np

from hil_frame.common import monotonic_ns
from hil_frame.control.base import ControlOutput
from hil_frame.env.base import ObservationFrame, StepResult

from .schema import TrajectoryRecord, TrajectoryStep
from .validation import validate_trajectory


class TrajectoryBuilder:
    def __init__(self, run_id: str = "default") -> None:
        self.run_id = run_id
        self._metadata: dict | None = None
        self._trajectory_id: str | None = None
        self._steps: list[TrajectoryStep] = []
        self._episode_return = 0.0
        self._start_ns = 0
        self._finished = False

    def start(self, trajectory_id: str, metadata: dict) -> None:
        if self._trajectory_id is not None and not self._finished:
            raise RuntimeError("trajectory already active")
        self._trajectory_id = trajectory_id
        self._metadata = dict(metadata)
        self._steps = []
        self._episode_return = 0.0
        self._start_ns = monotonic_ns()
        self._finished = False

    def append_step(self, observation: ObservationFrame, control_output: ControlOutput, step_result: StepResult) -> None:
        if self._finished:
            raise RuntimeError("cannot append after finish")
        if self._trajectory_id is None or self._metadata is None:
            raise RuntimeError("start() must be called before append_step()")
        step_id = len(self._steps)
        self._episode_return += float(step_result.reward)
        self._steps.append(
            TrajectoryStep(
                step_id=step_id,
                observation=_copy_observation(observation),
                action=np.asarray(step_result.applied_action, dtype=np.float32).copy(),
                action_source=control_output.action_source,
                policy_action=None if control_output.policy_action is None else np.asarray(control_output.policy_action, dtype=np.float32).copy(),
                human_action=None if control_output.human_action is None else np.asarray(control_output.human_action, dtype=np.float32).copy(),
                reward=float(step_result.reward),
                terminated=bool(step_result.terminated),
                truncated=bool(step_result.truncated),
                proposal_id=control_output.proposal_id,
                chunk_index=control_output.chunk_index,
                policy_version=control_output.policy_version,
                action_timestamp_ns=control_output.decision_timestamp_ns,
            )
        )

    def finish(self, final_observation: ObservationFrame, success: bool, termination_reason: str) -> TrajectoryRecord:
        if self._trajectory_id is None or self._metadata is None:
            raise RuntimeError("start() must be called before finish()")
        if not self._steps:
            raise ValueError("cannot finish empty trajectory")
        self._finished = True
        record = TrajectoryRecord(
            trajectory_id=self._trajectory_id,
            run_id=str(self._metadata.get("run_id", self.run_id)),
            task_suite=str(self._metadata.get("task_suite", "unknown")),
            task_id=int(self._metadata.get("task_id", -1)),
            task_description=str(self._metadata.get("task_description", "")),
            seed=int(self._metadata.get("seed", 0)),
            initial_state_id=self._metadata.get("initial_state_id"),
            steps=tuple(self._steps),
            final_observation=_copy_observation(final_observation),
            success=bool(success),
            termination_reason=termination_reason,
            episode_return=float(self._episode_return),
            start_timestamp_ns=self._start_ns,
            end_timestamp_ns=monotonic_ns(),
        )
        validate_trajectory(record)
        return record


def _copy_observation(obs: ObservationFrame) -> ObservationFrame:
    return replace(obs, data={k: np.asarray(v).copy() for k, v in obs.data.items()})

