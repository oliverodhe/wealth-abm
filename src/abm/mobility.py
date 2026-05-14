from __future__ import annotations

import numpy as np


def assign_quantiles(values: np.ndarray, n_quantiles: int = 5) -> np.ndarray:
    if n_quantiles <= 0:
        raise ValueError("n_quantiles must be positive")

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("assign_quantiles requires at least one value")

    order = np.argsort(array, kind="mergesort")
    quantiles = np.empty(array.size, dtype=int)
    ranked_quantiles = np.floor(np.arange(array.size) * n_quantiles / array.size).astype(int)
    quantiles[order] = np.minimum(ranked_quantiles, n_quantiles - 1)
    return quantiles


def transition_matrix(
    start_wealth: np.ndarray,
    end_wealth: np.ndarray,
    n_quantiles: int = 5,
) -> np.ndarray:
    start = np.asarray(start_wealth, dtype=float)
    end = np.asarray(end_wealth, dtype=float)
    if start.size != end.size:
        raise ValueError("start_wealth and end_wealth must have the same length")
    if start.size == 0:
        raise ValueError("transition_matrix requires at least one value")

    start_quantiles = assign_quantiles(start, n_quantiles)
    end_quantiles = assign_quantiles(end, n_quantiles)
    matrix = np.zeros((n_quantiles, n_quantiles), dtype=float)

    for start_q, end_q in zip(start_quantiles, end_quantiles):
        matrix[start_q, end_q] += 1.0

    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0.0)


def shorrocks_index(matrix: np.ndarray) -> float:
    transition = np.asarray(matrix, dtype=float)
    if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
        raise ValueError("matrix must be square")

    n = transition.shape[0]
    if n <= 1:
        return 0.0

    return float((n - np.trace(transition)) / (n - 1))


def rank_correlation(start_wealth: np.ndarray, end_wealth: np.ndarray) -> float:
    start = np.asarray(start_wealth, dtype=float)
    end = np.asarray(end_wealth, dtype=float)
    if start.size != end.size:
        raise ValueError("start_wealth and end_wealth must have the same length")
    if start.size == 0:
        raise ValueError("rank_correlation requires at least one value")
    if start.size == 1:
        return 1.0

    start_ranks = _ranks(start)
    end_ranks = _ranks(end)
    correlation = np.corrcoef(start_ranks, end_ranks)[0, 1]
    return float(correlation)


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    return ranks
