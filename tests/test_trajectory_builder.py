import dataclasses

import numpy as np

from hil_frame.common import monotonic_ns
from hil_frame.control import ActionSource, ControlOutput
from hil_frame.data.trajectory_builder import TrajectoryBuilder
from hil_frame.env.base import ObservationFrame, StepResult


def obs(obs_id, value):
    return ObservationFrame(obs_id, monotonic_ns(), {"state": np.array([value], dtype=np.float32)})


def output(source=ActionSource.POLICY):
    return ControlOutput(
        np.array([9], dtype=np.float32),
        source,
        np.array([1], dtype=np.float32) if source == ActionSource.POLICY else None,
        np.array([2], dtype=np.float32) if source == ActionSource.HUMAN else None,
        "p" if source == ActionSource.POLICY else None,
        0 if source == ActionSource.POLICY else None,
        "v" if source == ActionSource.POLICY else None,
        False,
        False,
        monotonic_ns(),
    )


def result(applied=3, obs_id=2):
    return StepResult(obs(obs_id, 99), 1.5, True, False, True, np.array([applied], dtype=np.float32), {})


def build_one(source=ActionSource.POLICY):
    b = TrajectoryBuilder("run")
    b.start("traj", {"run_id": "run", "task_suite": "dummy", "task_id": 1, "task_description": "task", "seed": 0})
    b.append_step(obs(1, 10), output(source), result())
    return b.finish(obs(2, 20), True, "success")


def test_builder_saves_pre_action_observation():
    traj = build_one()
    assert traj.steps[0].observation.data["state"][0] == 10


def test_builder_saves_step_result_applied_action():
    traj = build_one()
    assert traj.steps[0].action[0] == 3


def test_trajectory_step_has_no_intervention_fields():
    names = {f.name for f in dataclasses.fields(build_one().steps[0])}
    assert "intervention_onset" not in names
    assert "intervention_end" not in names
    assert "intervention_active" not in names


def test_step_saves_action_source():
    assert build_one().steps[0].action_source == ActionSource.POLICY


def test_human_step_saves_human_action():
    traj = build_one(ActionSource.HUMAN)
    assert traj.steps[0].human_action is not None


def test_policy_step_saves_policy_action():
    traj = build_one(ActionSource.POLICY)
    assert traj.steps[0].policy_action is not None


def test_final_observation_saved():
    traj = build_one()
    assert traj.final_observation.data["state"][0] == 20
