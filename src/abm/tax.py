from __future__ import annotations

import numpy as np


def flat_tax(income: np.ndarray, rate: float) -> np.ndarray:
    taxable_income = np.maximum(income, 0.0)
    return np.maximum(taxable_income * rate, 0.0)


def progressive_tax(
    income: np.ndarray,
    base_rate: float,
    top_rate: float,
    threshold: float,
) -> np.ndarray:
    taxable_income = np.maximum(income, 0.0)
    base_tax = taxable_income * base_rate
    top_tax = np.maximum(taxable_income - threshold, 0.0) * (top_rate - base_rate)
    return np.maximum(base_tax + top_tax, 0.0)


def find_revenue_neutral_base_rate(
    income: np.ndarray,
    target_revenue: float,
    top_rate: float,
    threshold: float,
    lower: float = 0.0,
    upper: float = 0.8,
    tolerance: float = 1e-6,
    max_iter: int = 100,
) -> float:
    low = lower
    high = upper

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        revenue = progressive_tax(income, mid, top_rate, threshold).sum()

        if abs(revenue - target_revenue) <= tolerance:
            return mid
        if revenue < target_revenue:
            low = mid
        else:
            high = mid

    return (low + high) / 2.0
