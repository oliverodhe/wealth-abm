from __future__ import annotations

import numpy as np


def lump_sum_transfers(total_revenue: float, n_agents: int) -> np.ndarray:
    if n_agents <= 0:
        raise ValueError("n_agents must be positive")
    return np.full(n_agents, total_revenue / n_agents)

