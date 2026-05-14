import numpy as np
import pytest

from abm.mobility import rank_correlation, shorrocks_index, transition_matrix


def test_transition_matrix_rows_sum_to_one() -> None:
    start = np.arange(100, dtype=float)
    end = np.arange(100, dtype=float)[::-1]
    matrix = transition_matrix(start, end, n_quantiles=5)
    assert np.allclose(matrix.sum(axis=1), 1.0)


def test_perfect_immobility_gives_identity_matrix() -> None:
    wealth = np.arange(100, dtype=float)
    matrix = transition_matrix(wealth, wealth, n_quantiles=5)
    assert np.allclose(matrix, np.eye(5))


def test_shorrocks_index_is_lower_for_identity_than_mobile_matrix() -> None:
    identity = np.eye(5)
    mobile = np.full((5, 5), 0.2)
    assert shorrocks_index(identity) < shorrocks_index(mobile)


def test_rank_correlation_is_close_to_one_when_ranks_are_unchanged() -> None:
    start = np.array([10.0, 20.0, 30.0, 40.0])
    end = np.array([1.0, 2.0, 3.0, 4.0])
    assert rank_correlation(start, end) == pytest.approx(1.0)
