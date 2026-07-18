from __future__ import annotations

import pickle
from dataclasses import dataclass

from hil_frame.common import monotonic_ns, new_id
from hil_frame.control.policy.base import ObservationRequest, PolicyProposal
from hil_frame.learner.policy.base import Policy


@dataclass
class PolicyServer:
    endpoint: str
    policy: Policy

    def serve_forever(self) -> None:
        try:
            import zmq
        except ModuleNotFoundError as exc:
            raise RuntimeError("pyzmq is required for PolicyServer") from exc
        ctx = zmq.Context.instance()
        socket = ctx.socket(zmq.ROUTER)
        socket.bind(self.endpoint)
        while True:
            ident, payload = socket.recv_multipart()
            request = pickle.loads(payload)
            if not isinstance(request, ObservationRequest):
                continue
            created = monotonic_ns()
            actions = self.policy.infer(request.observation, request.task_description)
            proposal = PolicyProposal(
                proposal_id=new_id("proposal"),
                trajectory_id=request.trajectory_id,
                source_obs_id=request.obs_id,
                actions=actions,
                policy_version=self.policy.version,
                created_timestamp_ns=created,
                received_timestamp_ns=monotonic_ns(),
            )
            socket.send_multipart([ident, pickle.dumps(proposal, protocol=pickle.HIGHEST_PROTOCOL)])

