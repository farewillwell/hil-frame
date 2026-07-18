from __future__ import annotations

import os
from pathlib import Path

from hil_frame.common import ensure_dir

from .schema import TrajectoryRecord
from .serialization import checksum_bytes, dumps
from .validation import validate_trajectory


class TrajectoryWriter:
    def __init__(self, local_root: str | Path = "local_data") -> None:
        self.root = Path(local_root)
        self.active_dir = ensure_dir(self.root / "active")
        self.pending_dir = ensure_dir(self.root / "pending")

    def write_pending(self, trajectory: TrajectoryRecord) -> tuple[Path, str]:
        validate_trajectory(trajectory)
        payload = dumps(trajectory)
        checksum = checksum_bytes(payload)
        tmp_path = self.active_dir / f"{trajectory.trajectory_id}.tmp"
        final_path = self.pending_dir / f"{trajectory.trajectory_id}.traj"
        with tmp_path.open("wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, final_path)
        return final_path, checksum

