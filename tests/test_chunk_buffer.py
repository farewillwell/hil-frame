import numpy as np

from hil_frame.control.chunk_buffer import ChunkBuffer
from hil_frame.control.policy.base import PolicyProposal
from hil_frame.common import monotonic_ns


def proposal(tid="traj-1", obs_id=1):
    return PolicyProposal("p1", tid, obs_id, np.array([[1, 0], [2, 0]], dtype=np.float32), "v", monotonic_ns(), monotonic_ns())


def test_chunk_buffer_outputs_one_action_at_a_time():
    buf = ChunkBuffer((2,))
    assert buf.push(proposal(), "traj-1", 1)
    first = buf.pop()
    second = buf.pop()
    assert first.action.tolist() == [1, 0]
    assert second.action.tolist() == [2, 0]
    assert buf.pop() is None


def test_policy_proposal_chunk_index_increments():
    buf = ChunkBuffer((2,))
    buf.push(proposal(), "traj-1", 1)
    assert buf.pop().chunk_index == 0
    assert buf.pop().chunk_index == 1


def test_reset_rejects_old_trajectory_proposal():
    buf = ChunkBuffer((2,))
    buf.reset_for_trajectory("traj-2")
    assert not buf.push(proposal(tid="traj-1"), "traj-2", 1)

