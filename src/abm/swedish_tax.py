from __future__ import annotations

import numpy as np


def earned_income_credit(
    income: np.ndarray,
    max_credit: float,
    phaseout_start: float,
    phaseout_rate: float,
) -> np.ndarray:
    taxable_income = np.maximum(income, 0.0)
    phase_in_credit = np.minimum(max_credit, taxable_income * phaseout_rate)
    phaseout = np.maximum(taxable_income - phaseout_start, 0.0) * phaseout_rate
    return np.maximum(phase_in_credit - phaseout, 0.0)


def swedish_income_tax(
    income: np.ndarray,
    municipal_rate: float,
    state_rate: float,
    state_threshold: float,
    max_credit: float,
    phaseout_start: float,
    phaseout_rate: float,
) -> np.ndarray:
    taxable_income = np.maximum(income, 0.0)
    municipal_tax = municipal_rate * taxable_income
    state_tax = state_rate * np.maximum(taxable_income - state_threshold, 0.0)
    pre_credit_tax = municipal_tax + state_tax
    credit = earned_income_credit(
        taxable_income,
        max_credit,
        phaseout_start,
        phaseout_rate,
    )
    capped_credit = np.minimum(credit, pre_credit_tax)
    return np.maximum(pre_credit_tax - capped_credit, 0.0)


def calibrate_municipal_rate_for_target_revenue(
    income: np.ndarray,
    target_revenue: float,
    state_rate: float,
    state_threshold: float,
    max_credit: float,
    phaseout_start: float,
    phaseout_rate: float,
    lower: float = 0.0,
    upper: float = 0.8,
    tolerance: float = 1e-6,
    max_iter: int = 100,
) -> float:
    low = lower
    high = upper

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        revenue = swedish_income_tax(
            income,
            mid,
            state_rate,
            state_threshold,
            max_credit,
            phaseout_start,
            phaseout_rate,
        ).sum()

        if abs(revenue - target_revenue) <= tolerance:
            return mid
        if revenue < target_revenue:
            low = mid
        else:
            high = mid

    return (low + high) / 2.0
