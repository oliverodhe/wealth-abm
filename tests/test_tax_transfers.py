import numpy as np
import pytest

from abm.tax import flat_tax, find_revenue_neutral_base_rate, progressive_tax
from abm.transfers import lump_sum_transfers


def test_flat_tax_revenue() -> None:
    income = np.array([100.0, 200.0, 300.0])
    tax = flat_tax(income, 0.30)
    assert tax.sum() == pytest.approx(180.0)


def test_progressive_tax_is_non_negative() -> None:
    income = np.array([-100.0, 0.0, 100.0, 1_000.0])
    tax = progressive_tax(income, 0.30, 0.50, 500.0)
    assert np.all(tax >= 0.0)


def test_revenue_neutral_calibration_matches_target_revenue() -> None:
    income = np.array([100.0, 300.0, 700.0, 1_000.0])
    target_revenue = float(flat_tax(income, 0.30).sum())
    base_rate = find_revenue_neutral_base_rate(
        income,
        target_revenue,
        top_rate=0.50,
        threshold=500.0,
    )
    progressive_revenue = progressive_tax(income, base_rate, 0.50, 500.0).sum()
    assert progressive_revenue == pytest.approx(target_revenue, abs=1e-5)


def test_calibrated_base_rate_stays_between_bounds() -> None:
    income = np.array([100.0, 300.0, 700.0, 1_000.0])
    target_revenue = float(flat_tax(income, 0.30).sum())
    base_rate = find_revenue_neutral_base_rate(
        income,
        target_revenue,
        top_rate=0.50,
        threshold=500.0,
        lower=0.0,
        upper=0.8,
    )
    assert 0.0 <= base_rate <= 0.8


def test_transfers_exactly_equal_revenue() -> None:
    transfers = lump_sum_transfers(100.0, 4)
    assert transfers.sum() == pytest.approx(100.0)
