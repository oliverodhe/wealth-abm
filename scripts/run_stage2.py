from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from abm.agents import initialise_agents
from abm.mobility import assign_quantiles, rank_correlation, shorrocks_index, transition_matrix
from abm.parameters import ModelParams
from abm.plotting import plot_final_lorenz_curves, plot_metric_over_time
from abm.simulation import Simulation
from abm.swedish_tax import calibrate_municipal_rate_for_target_revenue
from abm.tax import flat_tax

CSV_DIR = ROOT / "outputs" / "csv"
FIGURE_DIR = ROOT / "outputs" / "figures"


def first_year_labour_income(params: ModelParams, agents: dict[str, np.ndarray]) -> np.ndarray:
    rng = np.random.default_rng(params.seed)
    first_year_shock = (
        params.income_persistence * agents["income_shock"]
        + rng.normal(0.0, params.income_shock_sd, params.N)
    )
    return params.base_income * np.exp(agents["skill"] + first_year_shock)


def save_scenario_results(name: str, results: pd.DataFrame) -> pd.DataFrame:
    scenario_results = results.copy()
    scenario_results.insert(0, "scenario", name)
    scenario_results.to_csv(CSV_DIR / f"{name}_yearly_results.csv", index=False)
    return scenario_results


def print_mobility_summary(name: str, start_wealth: np.ndarray, end_wealth: np.ndarray) -> None:
    matrix = transition_matrix(start_wealth, end_wealth, n_quantiles=5)
    start_quantiles = assign_quantiles(start_wealth, n_quantiles=5)
    end_quantiles = assign_quantiles(end_wealth, n_quantiles=5)

    top_20_stayers = matrix[4, 4]
    bottom_40_to_top_40 = np.mean(end_quantiles[start_quantiles <= 1] >= 3)

    print(f"{name} wealth-quintile transition matrix")
    print(pd.DataFrame(matrix).round(3).to_string(index=False, header=False))
    print(f"{name} Shorrocks mobility index: {shorrocks_index(matrix):.3f}")
    print(f"{name} initial-final wealth rank correlation: {rank_correlation(start_wealth, end_wealth):.3f}")
    print(f"{name} probability of remaining in top 20%: {top_20_stayers:.3f}")
    print(f"{name} probability of moving from bottom 40% to top 40%: {bottom_40_to_top_40:.3f}")


def main() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    params = ModelParams()
    initial_agents = initialise_agents(params)
    income = first_year_labour_income(params, initial_agents)
    target_revenue = float(flat_tax(income, params.flat_tax_rate).sum())
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

    flat_sim = Simulation(params, tax_system="flat", initial_agents=initial_agents)
    swedish_sim = Simulation(
        swedish_params,
        tax_system="swedish",
        initial_agents=initial_agents,
    )

    flat_results = flat_sim.run()
    swedish_results = swedish_sim.run()

    combined_results = pd.concat(
        [
            save_scenario_results("flat", flat_results),
            save_scenario_results("swedish", swedish_results),
        ],
        ignore_index=True,
    )
    combined_results.to_csv(CSV_DIR / "scenario_comparison.csv", index=False)

    plot_metric_over_time(
        combined_results,
        "wealth_gini",
        "Wealth Gini",
        FIGURE_DIR / "wealth_gini_over_time.png",
    )
    plot_metric_over_time(
        combined_results,
        "top_10_share",
        "Top 10% wealth share",
        FIGURE_DIR / "top_10_share_over_time.png",
    )
    plot_metric_over_time(
        combined_results,
        "top_1_share",
        "Top 1% wealth share",
        FIGURE_DIR / "top_1_share_over_time.png",
    )
    plot_final_lorenz_curves(
        {
            "flat": flat_sim.agents["wealth"],
            "swedish": swedish_sim.agents["wealth"],
        },
        FIGURE_DIR / "final_lorenz_curves.png",
    )

    revenue_difference = (
        swedish_results.iloc[0]["labour_tax_revenue"]
        - flat_results.iloc[0]["labour_tax_revenue"]
    )
    print(f"Calibrated Swedish municipal rate: {calibrated_municipal_rate:.6f}")
    print(f"First-year labour-tax revenue difference: {revenue_difference:.6f}")
    print_mobility_summary("Flat", flat_sim.initial_wealth, flat_sim.agents["wealth"])
    print_mobility_summary("Swedish-style", swedish_sim.initial_wealth, swedish_sim.agents["wealth"])
    print(f"Saved CSV outputs to {CSV_DIR}")
    print(f"Saved figure outputs to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
