from __future__ import annotations

import numpy as np

from hil_frame.common import monotonic_ns
from hil_frame.env.base import ActionSpec, ObservationFrame
from hil_frame.env.wrappers.action import validate_and_clip_action

from .base import ActionSource, ControlContext, ControlOutput
from .chunk_buffer import ChunkBuffer, PolicyStep
from .human.base import HumanControlPhase, HumanControllerBase
from .policy.base import PolicyActionSource
from .safety import SafetyController
from .state_machine import HumanPhaseTracker


class Controller:
    def __init__(
        self,
        action_spec: ActionSpec,
        human_controller: HumanControllerBase,
        policy_source: PolicyActionSource,
        max_obs_lag: int = 2,
    ) -> None:
        self.action_spec = action_spec
        self.human_controller = human_controller
        self.policy_source = policy_source
        self.chunk_buffer = ChunkBuffer(action_spec.shape, max_obs_lag=max_obs_lag)
        self.safety = SafetyController(action_spec)
        self.phase_tracker = HumanPhaseTracker()
        self._current_trajectory_id: str | None = None

    def submit_observation(self, observation: ObservationFrame, context: ControlContext) -> None:
        self.policy_source.submit_observation(observation, context)

    def reset_for_new_trajectory(self, trajectory_id: str | None = None) -> None:
        self.chunk_buffer.reset_for_trajectory(trajectory_id or "")
        self.phase_tracker.previous = HumanControlPhase.IDLE
        self._current_trajectory_id = trajectory_id

    def get_action(self, observation: ObservationFrame, context: ControlContext) -> ControlOutput:
        if self._current_trajectory_id != context.trajectory_id:
            self._current_trajectory_id = context.trajectory_id
            self.chunk_buffer.reset_for_trajectory(context.trajectory_id)

        snapshot = self.human_controller.get_snapshot()
        entered, exited = self.phase_tracker.update(snapshot.phase)
        if entered or exited:
            self.chunk_buffer.clear()

        proposal = self.policy_source.get_latest_proposal()
        if proposal is not None:
            self.chunk_buffer.push(proposal, context.trajectory_id, context.obs_id)

        candidate = self.chunk_buffer.peek()

        if snapshot.phase == HumanControlPhase.STOPPED:
            self.chunk_buffer.clear()
            return self._output(
                self.safety.stopped_action(),
                ActionSource.SAFETY,
                candidate,
                snapshot.action,
                stop_requested=True,
                reset_requested=True,
            )

        if snapshot.phase == HumanControlPhase.ACTIVE:
            self.chunk_buffer.clear()
            human_action = np.zeros(self.action_spec.shape, dtype=np.float32) if snapshot.action is None else snapshot.action
            return self._output(
                validate_and_clip_action(human_action, self.action_spec),
                ActionSource.HUMAN,
                candidate,
                human_action,
                stop_requested=False,
                reset_requested=False,
            )

        policy_step = self.chunk_buffer.pop()
        if policy_step is not None:
            return self._output(
                validate_and_clip_action(policy_step.action, self.action_spec),
                ActionSource.POLICY,
                policy_step,
                None,
                stop_requested=False,
                reset_requested=False,
            )

        return self._output(
            self.safety.fallback_action(),
            ActionSource.ZERO_FALLBACK,
            None,
            None,
            stop_requested=False,
            reset_requested=False,
        )

    def _output(
        self,
        action: np.ndarray,
        source: ActionSource,
        policy_step: PolicyStep | None,
        human_action: np.ndarray | None,
        *,
        stop_requested: bool,
        reset_requested: bool,
    ) -> ControlOutput:
        return ControlOutput(
            action=np.asarray(action, dtype=np.float32).copy(),
            action_source=source,
            policy_action=None if policy_step is None else policy_step.action.copy(),
            human_action=None if human_action is None else np.asarray(human_action, dtype=np.float32).copy(),
            proposal_id=None if policy_step is None else policy_step.proposal_id,
            chunk_index=None if policy_step is None else policy_step.chunk_index,
            policy_version=None if policy_step is None else policy_step.policy_version,
            stop_requested=stop_requested,
            reset_requested=reset_requested,
            decision_timestamp_ns=monotonic_ns(),
        )

    def close(self) -> None:
        self.policy_source.close()
        self.human_controller.close()

