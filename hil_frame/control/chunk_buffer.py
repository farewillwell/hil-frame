from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .policy.base import PolicyProposal


@dataclass(frozen=True)
class PolicyStep:
    action: np.ndarray
    proposal_id: str
    chunk_index: int
    policy_version: str
    source_obs_id: int


class ChunkBuffer:
    def __init__(self, action_shape: tuple[int, ...], max_obs_lag: int = 2) -> None:
        self.action_shape = action_shape
        self.max_obs_lag = max_obs_lag
        self._trajectory_id: str | None = None
        self._proposal: PolicyProposal | None = None
        self._next_index = 0

    def push(self, proposal: PolicyProposal, current_trajectory_id: str, current_obs_id: int) -> bool:
        if proposal.trajectory_id != current_trajectory_id:
            return False
        if proposal.source_obs_id < current_obs_id - self.max_obs_lag:
            return False
        actions = np.asarray(proposal.actions, dtype=np.float32)
        if actions.ndim != 2 or tuple(actions.shape[1:]) != self.action_shape:
            return False
        if not np.all(np.isfinite(actions)):
            return False
        self._trajectory_id = current_trajectory_id
        self._proposal = PolicyProposal(
            proposal_id=proposal.proposal_id,
            trajectory_id=proposal.trajectory_id,
            source_obs_id=proposal.source_obs_id,
            actions=actions.copy(),
            policy_version=proposal.policy_version,
            created_timestamp_ns=proposal.created_timestamp_ns,
            received_timestamp_ns=proposal.received_timestamp_ns,
        )
        self._next_index = 0
        return True

    def pop(self) -> PolicyStep | None:
        step = self.peek()
        if step is not None:
            self._next_index += 1
            if self._proposal is not None and self._next_index >= len(self._proposal.actions):
                self.clear()
        return step

    def peek(self) -> PolicyStep | None:
        if self._proposal is None:
            return None
        if self._next_index >= len(self._proposal.actions):
            return None
        return PolicyStep(
            action=self._proposal.actions[self._next_index].astype(np.float32, copy=True),
            proposal_id=self._proposal.proposal_id,
            chunk_index=self._next_index,
            policy_version=self._proposal.policy_version,
            source_obs_id=self._proposal.source_obs_id,
        )

    def clear(self) -> None:
        self._proposal = None
        self._next_index = 0

    def reset_for_trajectory(self, trajectory_id: str) -> None:
        self._trajectory_id = trajectory_id
        self.clear()

