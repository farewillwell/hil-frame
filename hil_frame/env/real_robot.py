from __future__ import annotations

from .base import RobotEnv


class RealRobotEnv(RobotEnv):
    """Interface skeleton for future hardware deployment."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("RealRobotEnv is an interface placeholder; no hardware logic is faked.")

