from __future__ import annotations

from typing import Any

import numpy as np

from hil_frame.common import monotonic_ns

from .base import ActionSpec, ObservationFrame, RobotEnv, StepResult


class LiberoEnv(RobotEnv):
    def __init__(
        self,
        suite: str = "libero_goal",
        task_id: int = 0,
        init_state_id: int = 0,
        seed: int = 0,
        camera: str = "agentview",
        width: int = 512,
        height: int = 512,
        control_hz: int = 20,
        max_steps: int = 600,
        render_enabled: bool = True,
    ) -> None:
        try:
            import cv2
            from libero.libero import benchmark
            from libero.libero.envs import OffScreenRenderEnv
        except ModuleNotFoundError as exc:
            raise RuntimeError("LIBERO runtime is not installed in this Python environment.") from exc
        self.cv2 = cv2
        self.camera = camera
        self.render_enabled = render_enabled
        self.step_count = 0
        self.obs_id = 0
        self.max_steps = max_steps
        benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite = benchmark_dict[suite](task_order_index=0)
        self.task = self.task_suite.get_task(task_id)
        self.init_states = self.task_suite.get_task_init_states(task_id)
        bddl = self.task_suite.get_task_bddl_file_path(task_id)
        self._task_info = {
            "task_suite": suite,
            "task_id": task_id,
            "task_description": self.task.language,
            "seed": seed,
            "initial_state_id": init_state_id,
        }
        self.env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_names=[camera],
            camera_heights=height,
            camera_widths=width,
            control_freq=control_hz,
            horizon=max_steps,
            ignore_done=False,
            use_camera_obs=True,
        )
        self.env.seed(seed)
        self.init_state_id = init_state_id % len(self.init_states)
        self._last_obs: dict[str, Any] | None = None
        self._spec = self._infer_action_spec()

    def _infer_action_spec(self) -> ActionSpec:
        low = getattr(self.env, "action_spec", None)
        if callable(low):
            spec = self.env.action_spec()
            if isinstance(spec, tuple) and len(spec) == 2:
                return ActionSpec(tuple(np.asarray(spec[0]).shape), np.asarray(spec[0], dtype=np.float32), np.asarray(spec[1], dtype=np.float32), "float32")
        if hasattr(self.env, "action_space"):
            space = self.env.action_space
            return ActionSpec(tuple(space.shape), np.asarray(space.low, dtype=np.float32), np.asarray(space.high, dtype=np.float32), "float32")
        return ActionSpec((7,), -np.ones(7, dtype=np.float32), np.ones(7, dtype=np.float32), "float32")

    @property
    def action_spec(self) -> ActionSpec:
        return self._spec

    @property
    def task_info(self) -> dict:
        return dict(self._task_info)

    def reset(self) -> ObservationFrame:
        self.env.reset()
        self._last_obs = self.env.set_init_state(self.init_states[self.init_state_id])
        self.step_count = 0
        self.obs_id += 1
        return self.get_observation()

    def get_observation(self) -> ObservationFrame:
        if self._last_obs is None:
            raise RuntimeError("reset() must be called before get_observation()")
        return ObservationFrame(self.obs_id, monotonic_ns(), {k: np.asarray(v).copy() for k, v in self._last_obs.items()})

    def step(self, action: np.ndarray) -> StepResult:
        applied = np.clip(np.asarray(action, dtype=np.float32), self.action_spec.low, self.action_spec.high)
        result = self.env.step(applied)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, done, info = result
            terminated, truncated = bool(done), False
        self._last_obs = obs
        self.step_count += 1
        self.obs_id += 1
        truncated = bool(truncated or self.step_count >= self.max_steps)
        success = bool(self.env.check_success())
        return StepResult(self.get_observation(), float(reward), bool(terminated or success), truncated, success, applied.astype(np.float32, copy=True), dict(info))

    def render(self) -> None:
        if not self.render_enabled or self._last_obs is None:
            return
        key = f"{self.camera}_image"
        frame = np.asarray(self._last_obs[key])
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        self.cv2.imshow("LIBERO HIL Frame", self.cv2.cvtColor(frame, self.cv2.COLOR_RGB2BGR))
        self.cv2.waitKey(1)

    def close(self) -> None:
        self.env.close()
        if self.render_enabled:
            self.cv2.destroyAllWindows()

