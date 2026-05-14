import numpy as np
import pytest

from abm.tax import flat_tax, find_revenue_neutral_base_rate, progressive_tax
from abm.transfers import (
    means_tested_transfer,
    total_transfers,
    unemployment_support,
    universal_transfer,
)


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


def test_no_negative_transfers() -> None:
    income_after_tax = np.array([0.0, 100.0, 300.0])
    unemployed = np.array([True, False, False])
    transfers = total_transfers(
        120.0,
        income_after_tax,
        np.array([100.0, 100.0, 100.0]),
        unemployed,
        universal_share=0.5,
        safety_floor=150.0,
        safety_floor_replacement_rate=0.6,
        unemployment_replacement_rate=0.7,
        unemployment_benefit_cap=80.0,
    )
    assert np.all(transfers.total >= 0.0)
    assert np.all(transfers.universal >= 0.0)
    assert np.all(transfers.means_tested >= 0.0)
    assert np.all(transfers.unemployment >= 0.0)


def test_means_tested_transfer_phases_out_correctly() -> None:
    transfers = means_tested_transfer(
        np.array([0.0, 50.0, 100.0, 150.0]),
        floor=100.0,
        replacement_rate=0.5,
    )
    assert np.all(np.diff(transfers) <= 0.0)
    assert transfers[-1] == pytest.approx(0.0)


def test_transfer_budget_balances() -> None:
    income_after_tax = np.array([0.0, 100.0, 300.0, 500.0])
    unemployed = np.array([True, False, False, True])
    transfers = total_transfers(
        1_000.0,
        income_after_tax,
        np.array([200.0, 200.0, 200.0, 200.0]),
        unemployed,
        universal_share=0.5,
        safety_floor=250.0,
        safety_floor_replacement_rate=0.6,
        unemployment_replacement_rate=0.7,
        unemployment_benefit_cap=200.0,
    )
    assert transfers.total.sum() == pytest.approx(1_000.0)


def test_unemployment_replacement_income_is_positive() -> None:
    support = unemployment_support(
        np.array([100.0, 200.0]),
        np.array([True, False]),
        replacement_rate=0.7,
        benefit_cap=1_000.0,
    )
    assert support[0] > 0.0
    assert support[1] == pytest.approx(0.0)


def test_universal_transfer_exactly_equals_allocated_revenue() -> None:
    transfers = universal_transfer(100.0, 4, share=1.0)
    assert transfers.sum() == pytest.approx(100.0)
