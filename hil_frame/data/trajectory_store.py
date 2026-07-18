from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from hil_frame.common import ensure_dir, wall_time_ns
from hil_frame.control.base import ActionSource

from .schema import TrajectoryRecord
from .serialization import checksum_bytes, dumps, loads
from .validation import validate_trajectory


class TrajectoryStore:
    def __init__(self, remote_root: str | Path = "remote_data") -> None:
        self.root = Path(remote_root)
        self.trajectory_dir = ensure_dir(self.root / "trajectories")
        self.tmp_dir = ensure_dir(self.root / "tmp")
        self.db_path = self.root / "metadata.sqlite3"
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trajectories (
                    trajectory_id TEXT PRIMARY KEY,
                    checksum TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    task_suite TEXT NOT NULL,
                    task_id INTEGER NOT NULL,
                    success INTEGER NOT NULL,
                    num_steps INTEGER NOT NULL,
                    human_control_steps INTEGER NOT NULL,
                    human_segment_count INTEGER NOT NULL,
                    created_at_ns INTEGER NOT NULL
                )
                """
            )
            conn.commit()

    def insert(self, trajectory: TrajectoryRecord, checksum: str) -> None:
        validate_trajectory(trajectory)
        payload = dumps(trajectory)
        if checksum_bytes(payload) != checksum:
            raise ValueError("checksum does not match serialized trajectory")
        self.insert_payload(trajectory, checksum, payload)

    def insert_payload(self, trajectory: TrajectoryRecord, checksum: str, payload: bytes) -> None:
        validate_trajectory(trajectory)
        if checksum_bytes(payload) != checksum:
            raise ValueError("checksum does not match payload")
        if self.contains(trajectory.trajectory_id):
            existing = self.get_checksum(trajectory.trajectory_id)
            if existing == checksum:
                return
            raise ValueError("trajectory_id conflict with different checksum")
        tmp_path = self.tmp_dir / f"{trajectory.trajectory_id}.tmp"
        final_path = self.trajectory_dir / f"{trajectory.trajectory_id}.traj"
        with tmp_path.open("wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, final_path)
        human_steps = sum(step.action_source == ActionSource.HUMAN for step in trajectory.steps)
        human_segments = _count_human_segments(trajectory)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO trajectories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trajectory.trajectory_id,
                    checksum,
                    str(final_path),
                    trajectory.run_id,
                    trajectory.task_suite,
                    trajectory.task_id,
                    int(trajectory.success),
                    len(trajectory.steps),
                    human_steps,
                    human_segments,
                    wall_time_ns(),
                ),
            )
            conn.commit()

    def contains(self, trajectory_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT 1 FROM trajectories WHERE trajectory_id=?", (trajectory_id,)).fetchone()
        return row is not None

    def get_checksum(self, trajectory_id: str) -> str | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT checksum FROM trajectories WHERE trajectory_id=?", (trajectory_id,)).fetchone()
        return None if row is None else str(row[0])

    def get_trajectory(self, trajectory_id: str) -> TrajectoryRecord:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT file_path FROM trajectories WHERE trajectory_id=?", (trajectory_id,)).fetchone()
        if row is None:
            raise KeyError(trajectory_id)
        trajectory = loads(Path(row[0]).read_bytes())
        validate_trajectory(trajectory)
        return trajectory

    def list_trajectories(self, task_id: int | None = None, success: bool | None = None) -> list[str]:
        query = "SELECT trajectory_id FROM trajectories"
        clauses = []
        args = []
        if task_id is not None:
            clauses.append("task_id=?")
            args.append(task_id)
        if success is not None:
            clauses.append("success=?")
            args.append(int(success))
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, args).fetchall()
        return [str(row[0]) for row in rows]


def _count_human_segments(trajectory: TrajectoryRecord) -> int:
    count = 0
    prev_human = False
    for step in trajectory.steps:
        is_human = step.action_source == ActionSource.HUMAN
        if is_human and not prev_human:
            count += 1
        prev_human = is_human
    return count
