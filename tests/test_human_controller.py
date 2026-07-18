import numpy as np

from hil_frame.control.human import HumanControlPhase, ManualHumanController


def test_human_controller_initial_state_is_idle():
    assert ManualHumanController().get_snapshot().phase == HumanControlPhase.IDLE


def test_enter_control_sets_active():
    human = ManualHumanController()
    human.enter_control()
    assert human.get_snapshot().phase == HumanControlPhase.ACTIVE


def test_exit_control_sets_idle():
    human = ManualHumanController()
    human.enter_control()
    human.exit_control()
    assert human.get_snapshot().phase == HumanControlPhase.IDLE


def test_stop_and_reset_sets_stopped():
    human = ManualHumanController()
    human.enter_control()
    human.set_action(np.ones(7, dtype=np.float32))
    human.stop_and_reset()
    snap = human.get_snapshot()
    assert snap.phase == HumanControlPhase.STOPPED
    np.testing.assert_allclose(snap.action, np.zeros(7, dtype=np.float32))


def test_acknowledge_reset_restores_idle():
    human = ManualHumanController()
    human.stop_and_reset()
    human.acknowledge_reset()
    assert human.get_snapshot().phase == HumanControlPhase.IDLE

