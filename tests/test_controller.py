import time

import numpy as np

from hil_frame.common import monotonic_ns
from hil_frame.control import ActionSource, ControlContext, Controller
from hil_frame.control.human import ManualHumanController
from hil_frame.control.policy.base import PolicyActionSource, PolicyProposal
from hil_frame.env.base import ActionSpec, ObservationFrame


class Source(PolicyActionSource):
    def __init__(self):
        self.latest = None

    def submit_observation(self, observation, context):
        pass

    def get_latest_proposal(self):
        proposal = self.latest
        self.latest = None
        return proposal

    def close(self):
        pass


def spec():
    return ActionSpec((2,), -np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32), "float32")


def obs(obs_id=1):
    return ObservationFrame(obs_id, monotonic_ns(), {"state": np.array([obs_id], dtype=np.float32)})


def ctx(tid="t1", step=0, obs_id=1):
    return ControlContext(tid, step, obs_id, "task")


def proposal(tid="t1", obs_id=1, actions=None):
    arr = np.asarray(actions if actions is not None else [[0.4, 0.5], [0.6, 0.7]], dtype=np.float32)
    return PolicyProposal("p1", tid, obs_id, arr, "v1", monotonic_ns(), monotonic_ns())


def test_human_active_with_zero_action_still_outputs_human():
    human = ManualHumanController(action_dim=2)
    human.enter_control()
    source = Source()
    controller = Controller(spec(), human, source)
    out = controller.get_action(obs(), ctx())
    assert out.action_source == ActionSource.HUMAN
    np.testing.assert_allclose(out.action, np.zeros(2, dtype=np.float32))


def test_human_active_overrides_policy_action():
    human = ManualHumanController(action_dim=2)
    human.set_action(np.array([0.1, 0.2], dtype=np.float32))
    human.enter_control()
    source = Source()
    source.latest = proposal()
    out = Controller(spec(), human, source).get_action(obs(), ctx())
    assert out.action_source == ActionSource.HUMAN
    np.testing.assert_allclose(out.policy_action, np.array([0.4, 0.5], dtype=np.float32))
    np.testing.assert_allclose(out.action, np.array([0.1, 0.2], dtype=np.float32))


def test_human_idle_uses_policy_action():
    human = ManualHumanController(action_dim=2)
    source = Source()
    source.latest = proposal()
    out = Controller(spec(), human, source).get_action(obs(), ctx())
    assert out.action_source == ActionSource.POLICY
    np.testing.assert_allclose(out.action, np.array([0.4, 0.5], dtype=np.float32))


def test_human_entering_control_clears_remaining_chunk():
    human = ManualHumanController(action_dim=2)
    source = Source()
    source.latest = proposal()
    controller = Controller(spec(), human, source)
    assert controller.get_action(obs(1), ctx(obs_id=1)).action_source == ActionSource.POLICY
    human.enter_control()
    assert controller.get_action(obs(2), ctx(step=1, obs_id=2)).action_source == ActionSource.HUMAN
    human.exit_control()
    assert controller.get_action(obs(3), ctx(step=2, obs_id=3)).action_source == ActionSource.ZERO_FALLBACK


def test_human_exit_does_not_restore_old_chunk():
    human = ManualHumanController(action_dim=2)
    source = Source()
    source.latest = proposal()
    controller = Controller(spec(), human, source)
    controller.get_action(obs(1), ctx(obs_id=1))
    human.enter_control()
    controller.get_action(obs(2), ctx(step=1, obs_id=2))
    human.exit_control()
    out = controller.get_action(obs(3), ctx(step=2, obs_id=3))
    assert out.action_source == ActionSource.ZERO_FALLBACK

 
def test_human_exit_waits_for_new_proposal_with_zero_fallback():
    human = ManualHumanController(action_dim=2)
    source = Source()
    controller = Controller(spec(), human, source)
    human.enter_control()
    controller.get_action(obs(1), ctx(obs_id=1))
    human.exit_control()
    assert controller.get_action(obs(2), ctx(step=1, obs_id=2)).action_source == ActionSource.ZERO_FALLBACK


def test_stopped_outputs_safety_and_reset_request():
    human = ManualHumanController(action_dim=2)
    source = Source()
    human.stop_and_reset()
    out = Controller(spec(), human, source).get_action(obs(), ctx())
    assert out.action_source == ActionSource.SAFETY
    assert out.stop_requested
    assert out.reset_requested
