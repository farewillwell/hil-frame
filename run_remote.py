from __future__ import annotations

import argparse
import multiprocessing as mp

from hil_frame.common import tiny_yaml_load
from hil_frame.data.trajectory_server import TrajectoryServer
from hil_frame.data.trajectory_store import TrajectoryStore
from hil_frame.learner.policy.registry import build_policy
from hil_frame.learner.policy_server import PolicyServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system-config", default="configs/system.yaml")
    parser.add_argument("--learner-config", default="hil_frame/learner/config.yaml")
    parser.add_argument("--data-config", default="hil_frame/data/config.yaml")
    parser.add_argument("--policy-backend", default="zero")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    system_cfg = tiny_yaml_load(args.system_config)
    learner_cfg = tiny_yaml_load(args.learner_config)
    data_cfg = tiny_yaml_load(args.data_config)
    policy = build_policy(args.policy_backend, action_dim=int(learner_cfg.get("action_dim", 7)), action_horizon=int(learner_cfg.get("action_horizon", 1)))
    store = TrajectoryStore(data_cfg.get("remote_root", "remote_data"))
    policy_server = PolicyServer(system_cfg["network"]["policy_endpoint"], policy)
    trajectory_server = TrajectoryServer(system_cfg["network"]["trajectory_endpoint"], store)
    p1 = mp.Process(target=policy_server.serve_forever, name="PolicyServer")
    p2 = mp.Process(target=trajectory_server.serve_forever, name="TrajectoryServer")
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
