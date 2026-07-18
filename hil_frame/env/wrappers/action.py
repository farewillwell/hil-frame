from __future__ import annotations

import numpy as np

from hil_frame.env.base import ActionSpec


def validate_and_clip_action(action: np.ndarray, spec: ActionSpec) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32)
    if arr.shape != spec.shape:
        raise ValueError(f"action shape {arr.shape} does not match {spec.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("action contains NaN or Inf")
    return np.clip(arr, spec.low, spec.high).astype(np.float32, copy=True)

