from __future__ import annotations

import numpy as np

from hil_frame.common import monotonic_ns

from .base import ActionSpec, ObservationFrame, RobotEnv, StepResult


class DummyEnv(RobotEnv):
    def __init__(self, action_dim: int = 7, max_steps: int = 5, seed: int = 0) -> None:
        self._spec = ActionSpec(
            shape=(action_dim,),
            low=-np.ones(action_dim, dtype=np.float32),
            high=np.ones(action_dim, dtype=np.float32),
            dtype="float32",
        )
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.step_id = 0
        self.obs_id = 0
        self.state = np.zeros(action_dim, dtype=np.float32)

    @property
    def action_spec(self) -> ActionSpec:
        return self._spec

    @property
    def task_info(self) -> dict:
        return {
            "task_suite": "dummy",
            "task_id": 0,
            "task_description": "dummy reach",
            "seed": 0,
            "initial_state_id": None,
        }

    def reset(self) -> ObservationFrame:
        self.step_id = 0
        self.obs_id += 1
        self.state = np.zeros(self.action_spec.shape, dtype=np.float32)
        return self.get_observation()

    def get_observation(self) -> ObservationFrame:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[..., 0] = np.uint8(self.step_id)
        return ObservationFrame(
            obs_id=self.obs_id,
            timestamp_ns=monotonic_ns(),
            data={"state": self.state.copy(), "agentview_image": image},
        )

    def step(self, action: np.ndarray) -> StepResult:
        clipped = np.clip(np.asarray(action, dtype=np.float32), self.action_spec.low, self.action_spec.high)
        self.state = self.state + clipped
        self.step_id += 1
        self.obs_id += 1
        terminated = self.step_id >= self.max_steps
        obs = self.get_observation()
        return StepResult(
            observation=obs,
            reward=float(-np.linalg.norm(self.state)),
            terminated=terminated,
            truncated=False,
            success=terminated,
            applied_action=clipped.astype(np.float32, copy=True),
            info={"step_id": self.step_id},
        )

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None

