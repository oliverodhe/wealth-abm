from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from abm.agents import initialise_agents
from abm.metrics import gini, top_share
from abm.parameters import ModelParams
from abm.returns import RETURN_CLASS_NAMES, draw_capital_returns
from abm.swedish_tax import swedish_income_tax
from abm.tax import flat_tax, progressive_tax
from abm.transfers import lump_sum_transfers

TaxFunction = Callable[[np.ndarray], np.ndarray]


class Simulation:
    def __init__(
        self,
        params: ModelParams,
        tax_system: str = "flat",
        initial_agents: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        self.params = params
        self.tax_system = tax_system
        self.rng = np.random.default_rng(params.seed)
        self.agents = (
            {name: values.copy() for name, values in initial_agents.items()}
            if initial_agents is not None
            else initialise_agents(params)
        )
        self.initial_wealth = self.agents["wealth"].copy()
        self.labour_income = self._initial_labour_income()

    def run(self) -> pd.DataFrame:
        records: list[dict[str, float]] = []

        for year in range(1, self.params.years + 1):
            record = self.step(year)
            records.append(record)

        return pd.DataFrame.from_records(records)

    def step(self, year: int) -> dict[str, float]:
        wealth = self.agents["wealth"]
        saving_rate = self.agents["saving_rate"]

        self._update_labour_income()
        realised_returns = self._draw_capital_returns()
        capital_income = wealth * realised_returns
        labour_tax = self._labour_tax(self.labour_income)
        capital_tax = np.maximum(capital_income, 0.0) * self.params.capital_tax_rate
        after_tax_capital_income = capital_income - capital_tax
        total_tax = labour_tax + capital_tax
        labour_tax_revenue = float(labour_tax.sum())
        capital_tax_revenue = float(capital_tax.sum())
        total_revenue = float(total_tax.sum())
        transfers = lump_sum_transfers(total_revenue, self.params.N)

        disposable_income = self.labour_income - labour_tax + transfers
        saving = np.maximum(0.0, saving_rate * (disposable_income - self.params.min_consumption))
        self.agents["wealth"] = np.maximum(wealth + after_tax_capital_income + saving, 0.0)

        return {
            "year": float(year),
            "wealth_gini": gini(self.agents["wealth"]),
            "top_10_share": top_share(self.agents["wealth"], 10.0),
            "top_1_share": top_share(self.agents["wealth"], 1.0),
            "labour_tax_revenue": labour_tax_revenue,
            "capital_tax_revenue": capital_tax_revenue,
            "total_tax_revenue": total_revenue,
            "total_transfers": float(transfers.sum()),
            **self._income_diagnostics(
                self.labour_income,
                disposable_income,
                capital_income,
                labour_tax,
            ),
            **self._return_diagnostics(wealth, realised_returns),
        }

    def _initial_labour_income(self) -> np.ndarray:
        skill = self.agents["skill"]
        income_shock = self.agents["income_shock"]
        return self.params.base_income * np.exp(skill + income_shock)

    def _update_labour_income(self) -> None:
        new_noise = self.rng.normal(0.0, self.params.income_shock_sd, self.params.N)
        self.agents["income_shock"] = (
            self.params.income_persistence * self.agents["income_shock"] + new_noise
        )
        self.labour_income = self._initial_labour_income()

    def _draw_capital_returns(self) -> np.ndarray:
        return draw_capital_returns(self.agents["return_class"], self.params, self.rng)

    def _return_diagnostics(
        self,
        wealth: np.ndarray,
        realised_returns: np.ndarray,
    ) -> dict[str, float]:
        diagnostics: dict[str, float] = {}
        order = np.argsort(wealth)
        wealth_deciles = np.array_split(order, 10)

        for decile, indices in enumerate(wealth_deciles, start=1):
            diagnostics[f"avg_return_wealth_decile_{decile}"] = float(
                realised_returns[indices].mean() if indices.size > 0 else 0.0
            )

        return_class = self.agents["return_class"]
        for class_id, class_name in RETURN_CLASS_NAMES.items():
            mask = return_class == class_id
            diagnostics[f"avg_return_class_{class_name}"] = float(
                realised_returns[mask].mean() if np.any(mask) else 0.0
            )

        return diagnostics

    def _income_diagnostics(
        self,
        labour_income: np.ndarray,
        disposable_income: np.ndarray,
        capital_income: np.ndarray,
        labour_tax: np.ndarray,
    ) -> dict[str, float]:
        diagnostics = {
            "pre_tax_labour_income_gini": gini(labour_income),
            "disposable_income_gini": gini(disposable_income),
            "capital_income_gini": gini(capital_income),
            "capital_income_share": self._capital_income_share(labour_income, capital_income),
        }

        order = np.argsort(labour_income)
        income_deciles = np.array_split(order, 10)
        for decile, indices in enumerate(income_deciles, start=1):
            decile_income = labour_income[indices].sum()
            decile_tax = labour_tax[indices].sum()
            diagnostics[f"avg_tax_rate_income_decile_{decile}"] = float(
                decile_tax / decile_income if decile_income > 0.0 else 0.0
            )

        return diagnostics

    def _capital_income_share(
        self,
        labour_income: np.ndarray,
        capital_income: np.ndarray,
    ) -> float:
        total_capital_income = capital_income.sum()
        total_income = labour_income.sum() + total_capital_income
        if total_income <= 0.0:
            return 0.0
        return float(total_capital_income / total_income)

    def _labour_tax(self, income: np.ndarray) -> np.ndarray:
        if self.tax_system == "flat":
            return flat_tax(income, self.params.flat_tax_rate)
        if self.tax_system in {"simple_progressive", "progressive"}:
            return progressive_tax(
                income,
                self.params.flat_tax_rate,
                self.params.progressive_top_rate,
                self.params.progressive_threshold,
            )
        if self.tax_system == "swedish":
            return swedish_income_tax(
                income,
                self.params.municipal_rate,
                self.params.state_rate,
                self.params.state_threshold,
                self.params.earned_income_max_credit,
                self.params.earned_income_phaseout_start,
                self.params.earned_income_phaseout_rate,
            )
        raise ValueError(f"Unknown tax_system: {self.tax_system}")
