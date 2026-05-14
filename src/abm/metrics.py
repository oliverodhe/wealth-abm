from __future__ import annotations

import numpy as np


def gini(x: np.ndarray) -> float:
    values = np.asarray(x, dtype=float)
    if values.size == 0:
        raise ValueError("gini requires at least one value")

    values = np.clip(values, 0.0, None)
    total = values.sum()
    if total == 0.0:
        return 0.0

    sorted_values = np.sort(values)
    n = sorted_values.size
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * sorted_values)) / (n * total) - (n + 1.0) / n)


def top_share(x: np.ndarray, top_percent: float) -> float:
    if not 0.0 < top_percent <= 100.0:
        raise ValueError("top_percent must be in (0, 100]")

    values = np.asarray(x, dtype=float)
    if values.size == 0:
        raise ValueError("top_share requires at least one value")

    values = np.clip(values, 0.0, None)
    total = values.sum()
    if total == 0.0:
        return 0.0

    n_top = max(1, int(np.ceil(values.size * top_percent / 100.0)))
    return float(np.sort(values)[-n_top:].sum() / total)


def lorenz_curve(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(x, dtype=float)
    if values.size == 0:
        raise ValueError("lorenz_curve requires at least one value")

    values = np.clip(values, 0.0, None)
    sorted_values = np.sort(values)
    population_share = np.linspace(0.0, 1.0, sorted_values.size + 1)

    total = sorted_values.sum()
    if total == 0.0:
        wealth_share = np.zeros(sorted_values.size + 1)
        wealth_share[-1] = 1.0
        return population_share, wealth_share

    wealth_share = np.concatenate(([0.0], np.cumsum(sorted_values) / total))
    return population_share, wealth_share
