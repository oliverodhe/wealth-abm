import numpy as np
import pytest

from abm.metrics import gini, lorenz_curve, top_share


def test_gini_equal_distribution_is_zero() -> None:
    assert gini(np.array([1.0, 1.0, 1.0, 1.0])) == pytest.approx(0.0)


def test_gini_known_distribution() -> None:
    assert gini(np.array([0.0, 0.0, 1.0, 1.0])) == pytest.approx(0.5)


def test_top_share() -> None:
    assert top_share(np.array([1.0, 2.0, 3.0, 4.0]), 50.0) == pytest.approx(0.7)


def test_lorenz_curve_starts_at_zero() -> None:
    population_share, wealth_share = lorenz_curve(np.array([1.0, 2.0, 3.0]))
    assert population_share[0] == pytest.approx(0.0)
    assert wealth_share[0] == pytest.approx(0.0)


def test_lorenz_curve_ends_at_one() -> None:
    population_share, wealth_share = lorenz_curve(np.array([1.0, 2.0, 3.0]))
    assert population_share[-1] == pytest.approx(1.0)
    assert wealth_share[-1] == pytest.approx(1.0)


def test_lorenz_curve_is_non_decreasing() -> None:
    _, wealth_share = lorenz_curve(np.array([3.0, 1.0, 2.0]))
    assert np.all(np.diff(wealth_share) >= 0.0)
