from __future__ import annotations

import argparse

from hil_frame.common import new_id, tiny_yaml_load
from hil_frame.control.controller import Controller
from hil_frame.control.human.keyboard import KeyboardController, ManualHumanController
from hil_frame.control.policy.local_dummy import LocalDummyPolicySource
from hil_frame.data.trajectory_builder import TrajectoryBuilder
from hil_frame.data.trajectory_uploader import TrajectoryUploader
from hil_frame.data.trajectory_writer import TrajectoryWriter
from hil_frame.env.dummy import DummyEnv
from hil_frame.env.libero import LiberoEnv
from hil_frame.learner.policy.registry import build_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system-config", default="configs/system.yaml")
    parser.add_argument("--control-config", default="hil_frame/control/config.yaml")
    parser.add_argument("--env-config", default="hil_frame/env/config.yaml")
    parser.add_argument("--data-config", default="hil_frame/data/config.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    system_cfg = tiny_yaml_load(args.system_config)
    control_cfg = tiny_yaml_load(args.control_config)
    env_cfg = tiny_yaml_load(args.env_config)
    data_cfg = tiny_yaml_load(args.data_config)

    env_backend = env_cfg.get("backend", "dummy")
    if env_backend == "dummy":
        env = DummyEnv(max_steps=int(env_cfg.get("max_steps", 5)), seed=int(system_cfg.get("project", {}).get("seed", 0)))
    elif env_backend == "libero":
        env = LiberoEnv(**env_cfg.get("libero", {}))
    else:
        raise ValueError(f"unknown env backend {env_backend!r}")

    human_backend = control_cfg.get("human_backend", "manual")
    human = KeyboardController(env.action_spec.shape[0]) if human_backend == "keyboard" else ManualHumanController(env.action_spec.shape[0])
    policy = build_policy(control_cfg.get("policy_backend", "zero"), action_dim=env.action_spec.shape[0], action_horizon=int(control_cfg.get("action_horizon", 1)))
    controller = Controller(env.action_spec, human, LocalDummyPolicySource(policy))
    writer = TrajectoryWriter(data_cfg.get("local_root", "local_data"))
    uploader = None
    if data_cfg.get("upload_enabled", False):
        uploader = TrajectoryUploader(writer.pending_dir, system_cfg["network"]["trajectory_endpoint"])
        uploader.start()

    try:
        trajectory_id = new_id("traj")
        obs = env.reset()
        metadata = env.task_info | {"run_id": system_cfg.get("project", {}).get("run_id", "local")}
        builder = TrajectoryBuilder(run_id=metadata["run_id"])
        builder.start(trajectory_id, metadata)
        latest_valid_observation = obs
        termination_reason = "unknown"
        success = False
        step_id = 0
        while True:
            obs = env.get_observation()
            latest_valid_observation = obs
            context = __import__("hil_frame.control.base", fromlist=["ControlContext"]).ControlContext(
                trajectory_id=trajectory_id,
                step_id=step_id,
                obs_id=obs.obs_id,
                task_description=env.task_info["task_description"],
            )
            controller.submit_observation(obs, context)
            output = controller.get_action(obs, context)
            if output.stop_requested:
                termination_reason = "human_stop_reset"
                success = False
                break
            result = env.step(output.action)
            builder.append_step(obs, output, result)
            env.render()
            latest_valid_observation = result.observation
            step_id += 1
            if result.terminated or result.truncated:
                success = result.success
                termination_reason = "success" if result.success else ("truncated" if result.truncated else "terminated")
                break
        record = builder.finish(latest_valid_observation, success, termination_reason)
        path, checksum = writer.write_pending(record)
        print(f"wrote pending trajectory {path} checksum={checksum}")
        return 0
    finally:
        controller.close()
        env.close()
        if uploader is not None:
            uploader.stop()


if __name__ == "__main__":
    raise SystemExit(main())

