import pytest

from hil_frame.data.serialization import checksum_bytes, dumps
from hil_frame.data.trajectory_store import TrajectoryStore
from tests.test_transition_view import multi_traj


def test_can_query_trajectory_by_success_and_failure(tmp_path):
    store = TrajectoryStore(tmp_path / "remote")
    traj = multi_traj()
    store.insert(traj, checksum_bytes(dumps(traj)))
    assert store.list_trajectories(task_id=2, success=True) == ["traj"]
    assert store.list_trajectories(task_id=2, success=False) == []


def test_duplicate_same_id_same_checksum_is_idempotent(tmp_path):
    store = TrajectoryStore(tmp_path / "remote")
    traj = multi_traj()
    checksum = checksum_bytes(dumps(traj))
    store.insert(traj, checksum)
    store.insert(traj, checksum)
    assert store.list_trajectories() == ["traj"]


def test_duplicate_same_id_different_checksum_is_rejected(tmp_path):
    store = TrajectoryStore(tmp_path / "remote")
    traj = multi_traj()
    checksum = checksum_bytes(dumps(traj))
    store.insert(traj, checksum)
    with pytest.raises(ValueError):
        store.insert(traj, "0" * 64)


def test_remote_restart_can_read_trajectory(tmp_path):
    root = tmp_path / "remote"
    traj = multi_traj()
    store = TrajectoryStore(root)
    store.insert(traj, checksum_bytes(dumps(traj)))
    restarted = TrajectoryStore(root)
    assert restarted.get_trajectory("traj").trajectory_id == "traj"

