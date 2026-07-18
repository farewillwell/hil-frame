from pathlib import Path

from hil_frame.data.schema import TrajectoryAck
from hil_frame.data.serialization import checksum_bytes, dumps
from hil_frame.data.trajectory_server import TrajectoryServer
from hil_frame.data.trajectory_store import TrajectoryStore
from hil_frame.data.trajectory_uploader import TrajectoryUploader
from hil_frame.data.trajectory_writer import TrajectoryWriter
from tests.test_transition_view import multi_traj


class FakeUploader(TrajectoryUploader):
    def __init__(self, pending_dir, ack):
        super().__init__(pending_dir, "inproc://unused")
        self.ack = ack

    def _send(self, upload):
        return self.ack


def test_upload_success_deletes_pending_file(tmp_path):
    writer = TrajectoryWriter(tmp_path / "local")
    traj = multi_traj()
    path, checksum = writer.write_pending(traj)
    ack = TrajectoryAck(traj.trajectory_id, checksum, True, None)
    FakeUploader(writer.pending_dir, ack).upload_once(path)
    assert not path.exists()


def test_upload_failure_keeps_pending_file(tmp_path):
    writer = TrajectoryWriter(tmp_path / "local")
    traj = multi_traj()
    path, checksum = writer.write_pending(traj)
    ack = TrajectoryAck(traj.trajectory_id, checksum, False, "network")
    FakeUploader(writer.pending_dir, ack).upload_once(path)
    assert path.exists()


def test_checksum_mismatch_keeps_pending_file(tmp_path):
    writer = TrajectoryWriter(tmp_path / "local")
    traj = multi_traj()
    path, checksum = writer.write_pending(traj)
    ack = TrajectoryAck(traj.trajectory_id, "bad", True, None)
    FakeUploader(writer.pending_dir, ack).upload_once(path)
    assert path.exists()


def test_trajectory_server_rejects_checksum_conflict(tmp_path):
    store = TrajectoryStore(tmp_path / "remote")
    server = TrajectoryServer("inproc://unused", store)
    traj = multi_traj()
    payload = dumps(traj)
    checksum = checksum_bytes(payload)
    assert server.handle_upload(__import__("hil_frame.data.schema", fromlist=["TrajectoryUpload"]).TrajectoryUpload("traj", checksum, payload)).stored
    bad = server.handle_upload(__import__("hil_frame.data.schema", fromlist=["TrajectoryUpload"]).TrajectoryUpload("traj", "0" * 64, payload))
    assert not bad.stored

