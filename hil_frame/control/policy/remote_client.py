from __future__ import annotations

import pickle
import queue
import threading
from dataclasses import dataclass

from hil_frame.common import monotonic_ns, new_id
from hil_frame.control.base import ControlContext
from hil_frame.env.base import ObservationFrame

from .base import ObservationRequest, PolicyActionSource, PolicyProposal


@dataclass(frozen=True)
class RemotePolicyClientConfig:
    endpoint: str
    timeout_ms: int = 100


class RemotePolicyClient(PolicyActionSource):
    def __init__(self, config: RemotePolicyClientConfig) -> None:
        self.config = config
        self._requests: queue.Queue[ObservationRequest | None] = queue.Queue(maxsize=1)
        self._latest: PolicyProposal | None = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="RemotePolicyClientThread", daemon=True)
        self._thread.start()

    def submit_observation(self, observation: ObservationFrame, context: ControlContext) -> None:
        request = ObservationRequest(
            request_id=new_id("obsreq"),
            trajectory_id=context.trajectory_id,
            obs_id=context.obs_id,
            observation={k: v.copy() for k, v in observation.data.items()},
            task_description=context.task_description,
            sent_timestamp_ns=monotonic_ns(),
        )
        try:
            while True:
                self._requests.get_nowait()
        except queue.Empty:
            pass
        try:
            self._requests.put_nowait(request)
        except queue.Full:
            pass

    def get_latest_proposal(self) -> PolicyProposal | None:
        with self._lock:
            latest = self._latest
            self._latest = None
            return latest

    def close(self) -> None:
        self._stop.set()
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        try:
            import zmq
        except ModuleNotFoundError:
            return
        ctx = zmq.Context.instance()
        socket = ctx.socket(zmq.DEALER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.config.endpoint)
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        while not self._stop.is_set():
            try:
                request = self._requests.get(timeout=0.05)
            except queue.Empty:
                continue
            if request is None:
                return
            socket.send(pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL))
            events = dict(poller.poll(self.config.timeout_ms))
            if socket not in events:
                continue
            proposal = pickle.loads(socket.recv())
            if isinstance(proposal, PolicyProposal):
                with self._lock:
                    self._latest = proposal

