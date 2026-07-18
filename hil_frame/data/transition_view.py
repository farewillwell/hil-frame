from __future__ import annotations

import random

import numpy as np

from hil_frame.control.base import ActionSource

from .schema import TrajectoryRecord, Transition


class TransitionView:
    def __init__(self, trajectories: dict[str, TrajectoryRecord] | None = None) -> None:
        self.trajectories = trajectories or {}

    def add_trajectory(self, trajectory: TrajectoryRecord) -> None:
        self.trajectories[trajectory.trajectory_id] = trajectory

    def get_trajectory(self, trajectory_id: str) -> TrajectoryRecord:
        return self.trajectories[trajectory_id]

    def iter_transitions(
        self,
        trajectory_id: str,
        *,
        action_source: ActionSource | None = None,
        success: bool | None = None,
        task_id: int | None = None,
    ):
        trajectory = self.get_trajectory(trajectory_id)
        if success is not None and trajectory.success != success:
            return
        if task_id is not None and trajectory.task_id != task_id:
            return
        for idx, step in enumerate(trajectory.steps):
            if action_source is not None and step.action_source != action_source:
                continue
            next_obs = trajectory.steps[idx + 1].observation if idx + 1 < len(trajectory.steps) else trajectory.final_observation
            yield Transition(
                trajectory_id=trajectory.trajectory_id,
                step_id=step.step_id,
                observation={k: np.asarray(v).copy() for k, v in step.observation.data.items()},
                action=np.asarray(step.action, dtype=np.float32).copy(),
                next_observation={k: np.asarray(v).copy() for k, v in next_obs.data.items()},
                reward=step.reward,
                terminated=step.terminated,
                truncated=step.truncated,
                done=step.terminated or step.truncated,
                action_source=step.action_source,
                policy_action=None if step.policy_action is None else step.policy_action.copy(),
                human_action=None if step.human_action is None else step.human_action.copy(),
                trajectory_success=trajectory.success,
                task_id=trajectory.task_id,
            )

    def get_transition(self, trajectory_id: str, step_id: int) -> Transition:
        for transition in self.iter_transitions(trajectory_id):
            if transition.step_id == step_id:
                return transition
        raise KeyError((trajectory_id, step_id))

    def sample_transitions(self, batch_size: int, **filters) -> list[Transition]:
        all_transitions: list[Transition] = []
        for tid in self.trajectories:
            all_transitions.extend(self.iter_transitions(tid, **filters))
        if not all_transitions:
            raise ValueError("no transitions match filters")
        return [random.choice(all_transitions) for _ in range(batch_size)]

    def get_window(self, trajectory_id: str, start_step: int, length: int) -> list[Transition]:
        return [self.get_transition(trajectory_id, i) for i in range(start_step, start_step + length)]

    def get_human_segments(self, trajectory_id: str) -> list[tuple[int, int]]:
        trajectory = self.get_trajectory(trajectory_id)
        segments: list[tuple[int, int]] = []
        start: int | None = None
        for step in trajectory.steps:
            if step.action_source == ActionSource.HUMAN and start is None:
                start = step.step_id
            if step.action_source != ActionSource.HUMAN and start is not None:
                segments.append((start, step.step_id - 1))
                start = None
        if start is not None:
            segments.append((start, trajectory.steps[-1].step_id))
        return segments

