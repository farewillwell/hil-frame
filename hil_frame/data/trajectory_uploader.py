from __future__ import annotations

import pickle
import threading
import time
from pathlib import Path

from .schema import TrajectoryAck, TrajectoryUpload
from .serialization import read_payload


class TrajectoryUploader:
    def __init__(self, pending_dir: str | Path, endpoint: str, retry_interval_s: float = 1.0, timeout_ms: int = 500) -> None:
        self.pending_dir = Path(pending_dir)
        self.endpoint = endpoint
        self.retry_interval_s = retry_interval_s
        self.timeout_ms = timeout_ms
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="TrajectoryUploaderThread", daemon=True)

    def start(self) -> None:
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            for path in sorted(self.pending_dir.glob("*.traj")):
                try:
                    self.upload_once(path)
                except Exception:
                    pass
            time.sleep(self.retry_interval_s)

    def upload_once(self, path: str | Path) -> TrajectoryAck:
        payload, checksum = read_payload(path)
        trajectory_id = Path(path).stem
        upload = TrajectoryUpload(trajectory_id=trajectory_id, checksum=checksum, payload=payload)
        ack = self._send(upload)
        if ack.trajectory_id == trajectory_id and ack.checksum == checksum and ack.stored:
            Path(path).unlink()
        return ack

    def _send(self, upload: TrajectoryUpload) -> TrajectoryAck:
        try:
            import zmq
        except ModuleNotFoundError as exc:
            raise RuntimeError("pyzmq is required for TrajectoryUploader") from exc
        ctx = zmq.Context.instance()
        socket = ctx.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.endpoint)
        socket.send(pickle.dumps(upload, protocol=pickle.HIGHEST_PROTOCOL))
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        events = dict(poller.poll(self.timeout_ms))
        if socket not in events:
            return TrajectoryAck(upload.trajectory_id, upload.checksum, False, "timeout")
        ack = pickle.loads(socket.recv())
        if not isinstance(ack, TrajectoryAck):
            return TrajectoryAck(upload.trajectory_id, upload.checksum, False, "invalid ack")
        return ack

