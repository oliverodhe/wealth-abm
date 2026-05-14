from __future__ import annotations

from pathlib import Path
import sys
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from abm.agents import initialise_agents
from abm.parameters import ModelParams
from abm.simulation import Simulation
from abm.swedish_tax import calibrate_municipal_rate_for_target_revenue, swedish_income_tax
from abm.tax import flat_tax


def print_summary(name: str, results) -> None:
    final = results.iloc[-1]
    print(f"{name} tax")
    print(f"  Final Gini: {final['wealth_gini']:.3f}")
    print(f"  Top 10% share: {final['top_10_share']:.3f}")
    print(f"  Top 1% share: {final['top_1_share']:.3f}")
    print(f"  Total tax revenue: {final['total_tax_revenue']:.2f}")
    print(f"  Total transfers: {final['total_transfers']:.2f}")


def print_effective_tax_rates_by_decile(
    name: str,
    income: np.ndarray,
    tax: np.ndarray,
) -> None:
    order = np.argsort(income)
    income_deciles = np.array_split(income[order], 10)
    tax_deciles = np.array_split(tax[order], 10)

    print(f"{name} effective average labour-tax rates by income decile")
    for decile, (decile_income, decile_tax) in enumerate(
        zip(income_deciles, tax_deciles),
        start=1,
    ):
        average_rate = decile_tax.sum() / decile_income.sum()
        print(f"  Decile {decile}: {average_rate:.3f}")


def first_year_labour_income(params: ModelParams, agents: dict[str, np.ndarray]) -> np.ndarray:
    rng = np.random.default_rng(params.seed)
    first_year_shock = (
        params.income_persistence * agents["income_shock"]
        + rng.normal(0.0, params.income_shock_sd, params.N)
    )
    return params.base_income * np.exp(agents["skill"] + first_year_shock)


def main() -> None:
    params = ModelParams()
    initial_agents = initialise_agents(params)
    income = first_year_labour_income(params, initial_agents)
    flat_first_year_tax = flat_tax(income, params.flat_tax_rate)
    target_revenue = float(flat_first_year_tax.sum())
    calibrated_municipal_rate = calibrate_municipal_rate_for_target_revenue(
        income,
        target_revenue,
        params.state_rate,
        params.state_threshold,
        params.earned_income_max_credit,
        params.earned_income_phaseout_start,
        params.earned_income_phaseout_rate,
    )
    swedish_params = replace(params, municipal_rate=calibrated_municipal_rate)
    swedish_first_year_tax = swedish_income_tax(
        income,
        swedish_params.municipal_rate,
        swedish_params.state_rate,
        swedish_params.state_threshold,
        swedish_params.earned_income_max_credit,
        swedish_params.earned_income_phaseout_start,
        swedish_params.earned_income_phaseout_rate,
    )

    flat_results = Simulation(params, tax_system="flat", initial_agents=initial_agents).run()
    swedish_results = Simulation(
        swedish_params,
        tax_system="swedish",
        initial_agents=initial_agents,
    ).run()

    print(f"Calibrated Swedish municipal rate: {calibrated_municipal_rate:.6f}")
    print_summary("Flat", flat_results)
    print_summary("Swedish-style", swedish_results)

    revenue_difference = (
        swedish_results.iloc[0]["labour_tax_revenue"]
        - flat_results.iloc[0]["labour_tax_revenue"]
    )
    print(f"First-year labour-tax revenue difference: {revenue_difference:.6f}")
    print_effective_tax_rates_by_decile("Flat", income, flat_first_year_tax)
    print_effective_tax_rates_by_decile("Swedish-style", income, swedish_first_year_tax)


if __name__ == "__main__":
    main()
