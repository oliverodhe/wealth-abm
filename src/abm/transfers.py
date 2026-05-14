from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransferResult:
    total: np.ndarray
    universal: np.ndarray
    means_tested: np.ndarray
    unemployment: np.ndarray
    means_tested_recipients: np.ndarray
    unemployed: np.ndarray


def universal_transfer(total_revenue: float, n_agents: int, share: float = 1.0) -> np.ndarray:
    if n_agents <= 0:
        raise ValueError("n_agents must be positive")
    budget = max(total_revenue, 0.0) * np.clip(share, 0.0, 1.0)
    return np.full(n_agents, budget / n_agents)


def means_tested_transfer(
    disposable_income_after_tax: np.ndarray,
    floor: float,
    replacement_rate: float,
) -> np.ndarray:
    gap = np.maximum(floor - disposable_income_after_tax, 0.0)
    return np.maximum(gap * replacement_rate, 0.0)


def unemployment_support(
    previous_income: np.ndarray,
    unemployed: np.ndarray,
    replacement_rate: float,
    benefit_cap: float,
) -> np.ndarray:
    support = np.minimum(np.maximum(previous_income, 0.0) * replacement_rate, benefit_cap)
    return np.where(unemployed, np.maximum(support, 0.0), 0.0)


def total_transfers(
    labour_tax_revenue: float,
    disposable_income_after_tax: np.ndarray,
    previous_income: np.ndarray,
    unemployed: np.ndarray,
    universal_share: float,
    safety_floor: float,
    safety_floor_replacement_rate: float,
    unemployment_replacement_rate: float,
    unemployment_benefit_cap: float,
) -> TransferResult:
    budget = max(labour_tax_revenue, 0.0)
    n_agents = disposable_income_after_tax.size

    unemployment = unemployment_support(
        previous_income,
        unemployed,
        unemployment_replacement_rate,
        unemployment_benefit_cap,
    )
    income_after_unemployment = disposable_income_after_tax + unemployment
    means_tested = means_tested_transfer(
        income_after_unemployment,
        safety_floor,
        safety_floor_replacement_rate,
    )

    targeted = unemployment + means_tested
    targeted_spending = targeted.sum()
    targeted_budget = budget * (1.0 - np.clip(universal_share, 0.0, 1.0))

    if targeted_spending > targeted_budget and targeted_spending > 0.0:
        scale = targeted_budget / targeted_spending
        unemployment = unemployment * scale
        means_tested = means_tested * scale
        targeted = unemployment + means_tested

    universal_budget = budget - targeted.sum()
    universal = np.full(n_agents, universal_budget / n_agents if n_agents > 0 else 0.0)
    total = universal + means_tested + unemployment

    return TransferResult(
        total=total,
        universal=universal,
        means_tested=means_tested,
        unemployment=unemployment,
        means_tested_recipients=means_tested > 0.0,
        unemployed=unemployed,
    )
