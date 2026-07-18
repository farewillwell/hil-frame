from hil_frame.common import new_id
from hil_frame.control import ControlContext, Controller, ActionSource
from hil_frame.control.human import ManualHumanController
from hil_frame.control.policy.local_dummy import LocalDummyPolicySource
from hil_frame.data.trajectory_builder import TrajectoryBuilder
from hil_frame.env.dummy import DummyEnv
from hil_frame.learner.policy.zero_policy import ZeroPolicy


def test_dummy_env_zero_policy_runs_end_to_end(tmp_path):
    env = DummyEnv(max_steps=2)
    human = ManualHumanController()
    controller = Controller(env.action_spec, human, LocalDummyPolicySource(ZeroPolicy(action_dim=7)))
    builder = TrajectoryBuilder("run")
    tid = new_id("traj")
    env.reset()
    builder.start(tid, env.task_info | {"run_id": "run"})
    step = 0
    while True:
        obs = env.get_observation()
        ctx = ControlContext(tid, step, obs.obs_id, env.task_info["task_description"])
        controller.submit_observation(obs, ctx)
        out = controller.get_action(obs, ctx)
        result = env.step(out.action)
        builder.append_step(obs, out, result)
        step += 1
        if result.terminated or result.truncated:
            traj = builder.finish(result.observation, result.success, "success")
            break
    assert len(traj.steps) == 2
    assert traj.final_observation.obs_id == result.observation.obs_id


def test_stop_and_reset_ends_current_trajectory_without_fake_step():
    env = DummyEnv(max_steps=5)
    human = ManualHumanController()
    controller = Controller(env.action_spec, human, LocalDummyPolicySource(ZeroPolicy(action_dim=7)))
    builder = TrajectoryBuilder("run")
    tid = "traj-stop"
    initial = env.reset()
    builder.start(tid, env.task_info | {"run_id": "run"})
    obs = env.get_observation()
    ctx = ControlContext(tid, 0, obs.obs_id, env.task_info["task_description"])
    out = controller.get_action(obs, ctx)
    result = env.step(out.action)
    builder.append_step(obs, out, result)
    human.stop_and_reset()
    obs2 = env.get_observation()
    out2 = controller.get_action(obs2, ControlContext(tid, 1, obs2.obs_id, env.task_info["task_description"]))
    assert out2.stop_requested
    traj = builder.finish(obs2, False, "human_stop_reset")
    assert len(traj.steps) == 1
    human.acknowledge_reset()
    assert human.get_snapshot().phase.value == "idle"
