from __future__ import annotations

import queue
import threading

import numpy as np

from hil_frame.common import monotonic_ns, new_id
from hil_frame.control.base import ControlContext
from hil_frame.env.base import ObservationFrame
from hil_frame.learner.policy.base import Policy

from .base import PolicyActionSource, PolicyProposal


class LocalDummyPolicySource(PolicyActionSource):
    """Background latest-only local policy source used for tests and dummy runs."""

    def __init__(self, policy: Policy) -> None:
        self.policy = policy
        self._requests: queue.Queue[tuple[ObservationFrame, ControlContext] | None] = queue.Queue(maxsize=1)
        self._latest: PolicyProposal | None = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="LocalDummyPolicySource", daemon=True)
        self._thread.start()

    def submit_observation(self, observation: ObservationFrame, context: ControlContext) -> None:
        try:
            while True:
                self._requests.get_nowait()
        except queue.Empty:
            pass
        try:
            self._requests.put_nowait((observation, context))
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
        while not self._stop.is_set():
            item = self._requests.get()
            if item is None:
                return
            obs, context = item
            created = monotonic_ns()
            actions = np.asarray(self.policy.infer(obs.data, context.task_description), dtype=np.float32)
            proposal = PolicyProposal(
                proposal_id=new_id("proposal"),
                trajectory_id=context.trajectory_id,
                source_obs_id=context.obs_id,
                actions=actions,
                policy_version=self.policy.version,
                created_timestamp_ns=created,
                received_timestamp_ns=monotonic_ns(),
            )
            with self._lock:
                self._latest = proposal

