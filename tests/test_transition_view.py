import numpy as np

from hil_frame.control import ActionSource
from hil_frame.data.transition_view import TransitionView
from tests.test_trajectory_builder import build_one, obs, output, result
from hil_frame.data.trajectory_builder import TrajectoryBuilder


def multi_traj():
    b = TrajectoryBuilder("run")
    b.start("traj", {"run_id": "run", "task_suite": "dummy", "task_id": 2, "task_description": "task", "seed": 0})
    b.append_step(obs(1, 1), output(ActionSource.POLICY), result(applied=5, obs_id=2))
    b.append_step(obs(2, 2), output(ActionSource.HUMAN), result(applied=6, obs_id=3))
    b.append_step(obs(3, 3), output(ActionSource.HUMAN), result(applied=7, obs_id=4))
    return b.finish(obs(4, 4), True, "success")


def test_transition_uses_next_step_observation():
    view = TransitionView({"traj": multi_traj()})
    t = view.get_transition("traj", 0)
    assert t.next_observation["state"][0] == 2


def test_last_transition_uses_final_observation():
    view = TransitionView({"traj": multi_traj()})
    t = view.get_transition("traj", 2)
    assert t.next_observation["state"][0] == 4


def test_transition_action_is_applied_action():
    view = TransitionView({"traj": multi_traj()})
    assert view.get_transition("traj", 0).action[0] == 5


def test_filter_by_human_and_policy():
    view = TransitionView({"traj": multi_traj()})
    assert len(list(view.iter_transitions("traj", action_source=ActionSource.HUMAN))) == 2
    assert len(list(view.iter_transitions("traj", action_source=ActionSource.POLICY))) == 1


def test_human_segments_are_derived_from_action_source():
    view = TransitionView({"traj": multi_traj()})
    assert view.get_human_segments("traj") == [(1, 2)]
