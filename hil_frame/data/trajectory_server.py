from __future__ import annotations

import pickle
from dataclasses import dataclass

from .schema import TrajectoryAck, TrajectoryUpload
from .serialization import loads
from .trajectory_store import TrajectoryStore
from .validation import validate_payload_checksum, validate_trajectory


@dataclass
class TrajectoryServer:
    endpoint: str
    store: TrajectoryStore

    def handle_upload(self, upload: TrajectoryUpload) -> TrajectoryAck:
        try:
            validate_payload_checksum(upload.payload, upload.checksum)
            trajectory = loads(upload.payload)
            validate_trajectory(trajectory)
            if trajectory.trajectory_id != upload.trajectory_id:
                raise ValueError("upload trajectory_id does not match payload")
            existing = self.store.get_checksum(upload.trajectory_id)
            if existing is not None:
                if existing == upload.checksum:
                    return TrajectoryAck(upload.trajectory_id, upload.checksum, True, None)
                return TrajectoryAck(upload.trajectory_id, upload.checksum, False, "trajectory_id conflict")
            self.store.insert_payload(trajectory, upload.checksum, upload.payload)
            return TrajectoryAck(upload.trajectory_id, upload.checksum, True, None)
        except Exception as exc:
            return TrajectoryAck(upload.trajectory_id, upload.checksum, False, str(exc))

    def serve_forever(self) -> None:
        try:
            import zmq
        except ModuleNotFoundError as exc:
            raise RuntimeError("pyzmq is required for TrajectoryServer") from exc
        ctx = zmq.Context.instance()
        socket = ctx.socket(zmq.REP)
        socket.bind(self.endpoint)
        while True:
            upload = pickle.loads(socket.recv())
            ack = self.handle_upload(upload)
            socket.send(pickle.dumps(ack, protocol=pickle.HIGHEST_PROTOCOL))
