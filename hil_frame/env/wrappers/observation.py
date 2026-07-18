from __future__ import annotations

import numpy as np


def copy_observation_data(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: np.asarray(v).copy() for k, v in data.items()}

