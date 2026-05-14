import numpy as np
import pytest

from abm.swedish_tax import (
    calibrate_municipal_rate_for_target_revenue,
    earned_income_credit,
    swedish_income_tax,
)
from abm.tax import flat_tax


def test_swedish_income_tax_is_never_negative() -> None:
    income = np.array([-100_000.0, 0.0, 100_000.0, 700_000.0])
    tax = swedish_income_tax(income, 0.30, 0.20, 600_000.0, 35_000.0, 500_000.0, 0.08)
    assert np.all(tax >= 0.0)


def test_swedish_income_tax_increases_with_income() -> None:
    income = np.array([0.0, 100_000.0, 300_000.0, 600_000.0, 900_000.0])
    tax = swedish_income_tax(income, 0.30, 0.20, 600_000.0, 35_000.0, 500_000.0, 0.08)
    assert np.all(np.diff(tax) >= 0.0)


def test_high_earners_pay_state_tax() -> None:
    income = np.array([700_000.0])
    with_state_tax = swedish_income_tax(
        income,
        0.30,
        0.20,
        600_000.0,
        35_000.0,
        500_000.0,
        0.08,
    )
    without_state_tax = swedish_income_tax(
        income,
        0.30,
        0.0,
        600_000.0,
        35_000.0,
        500_000.0,
        0.08,
    )
    assert with_state_tax[0] > without_state_tax[0]


def test_earned_income_credit_phases_out() -> None:
    income = np.array([400_000.0, 700_000.0, 1_000_000.0])
    credit = earned_income_credit(income, 35_000.0, 500_000.0, 0.08)
    assert credit[0] > credit[1] > credit[2]


def test_swedish_calibrated_revenue_matches_target() -> None:
    income = np.array([100_000.0, 300_000.0, 500_000.0, 700_000.0, 1_000_000.0])
    target_revenue = float(flat_tax(income, 0.30).sum())
    municipal_rate = calibrate_municipal_rate_for_target_revenue(
        income,
        target_revenue,
        state_rate=0.20,
        state_threshold=600_000.0,
        max_credit=35_000.0,
        phaseout_start=500_000.0,
        phaseout_rate=0.08,
    )
    calibrated_revenue = swedish_income_tax(
        income,
        municipal_rate,
        0.20,
        600_000.0,
        35_000.0,
        500_000.0,
        0.08,
    ).sum()
    assert calibrated_revenue == pytest.approx(target_revenue, abs=1e-5)
