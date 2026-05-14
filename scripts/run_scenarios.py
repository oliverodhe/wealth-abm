from __future__ import annotations

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
from abm.scenarios import SCENARIOS, CalibratedScenario, calibrate_scenarios
from abm.simulation import Simulation

CSV_DIR = ROOT / "outputs" / "csv"
FIGURE_DIR = ROOT / "outputs" / "figures"


def save_scenario_results(name: str, results: pd.DataFrame) -> pd.DataFrame:
    scenario_results = results.copy()
    scenario_results.insert(0, "scenario", name)
    scenario_results.to_csv(CSV_DIR / f"{name}_yearly_results.csv", index=False)
    return scenario_results


def mobility_metrics(
    scenario: str,
    start_wealth: np.ndarray,
    end_wealth: np.ndarray,
) -> dict[str, float | str]:
    matrix = transition_matrix(start_wealth, end_wealth, n_quantiles=5)
    start_quantiles = assign_quantiles(start_wealth, n_quantiles=5)
    end_quantiles = assign_quantiles(end_wealth, n_quantiles=5)

    return {
        "scenario": scenario,
        "shorrocks_index": shorrocks_index(matrix),
        "rank_correlation": rank_correlation(start_wealth, end_wealth),
        "top_20_persistence": matrix[4, 4],
        "bottom_40_to_top_40": float(np.mean(end_quantiles[start_quantiles <= 1] >= 3)),
    }


def print_transition_matrix(name: str, start_wealth: np.ndarray, end_wealth: np.ndarray) -> None:
    matrix = transition_matrix(start_wealth, end_wealth, n_quantiles=5)
    print(f"{name} wealth-quintile transition matrix")
    print(pd.DataFrame(matrix).round(3).to_string(index=False, header=False))


def final_summary_row(
    name: str,
    calibrated: CalibratedScenario,
    results: pd.DataFrame,
    mobility: dict[str, float | str],
) -> dict[str, float | str]:
    final = results.iloc[-1]
    return {
        "scenario": name,
        "tax_system": calibrated.config.tax_system,
        "transfer_policy": calibrated.config.transfer_policy,
        "calibrated_rate": calibrated.calibrated_rate,
        "first_year_revenue_difference": (
            calibrated.first_year_labour_revenue - calibrated.target_labour_revenue
        ),
        "final_gini": final["wealth_gini"],
        "final_pre_tax_labour_income_gini": final["pre_tax_labour_income_gini"],
        "final_disposable_income_gini": final["disposable_income_gini"],
        "final_capital_income_gini": final["capital_income_gini"],
        "final_capital_income_share": final["capital_income_share"],
        "final_top_10_share": final["top_10_share"],
        "final_top_1_share": final["top_1_share"],
        "shorrocks_index": mobility["shorrocks_index"],
        "top_20_persistence": mobility["top_20_persistence"],
    }


def main() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    params = ModelParams()
    initial_agents = initialise_agents(params)
    calibrated_scenarios = calibrate_scenarios(params, initial_agents, SCENARIOS)

    yearly_results: list[pd.DataFrame] = []
    final_summary_rows: list[dict[str, float | str]] = []
    mobility_rows: list[dict[str, float | str]] = []
    final_wealth_by_scenario: dict[str, np.ndarray] = {}

    for name, calibrated in calibrated_scenarios.items():
        sim = Simulation(
            calibrated.params,
            tax_system=calibrated.config.tax_system,
            initial_agents=initial_agents,
        )
        results = sim.run()
        yearly_results.append(save_scenario_results(name, results))
        final_wealth_by_scenario[name] = sim.agents["wealth"]

        mobility = mobility_metrics(name, sim.initial_wealth, sim.agents["wealth"])
        mobility_rows.append(mobility)
        final_summary_rows.append(final_summary_row(name, calibrated, results, mobility))

        final = results.iloc[-1]
        revenue_difference = calibrated.first_year_labour_revenue - calibrated.target_labour_revenue
        print(f"{name}")
        print(f"  Calibrated municipal/base rate: {calibrated.calibrated_rate:.6f}")
        print(f"  First-year labour-tax revenue difference: {revenue_difference:.6f}")
        print(f"  Final Gini: {final['wealth_gini']:.3f}")
        print(f"  Final pre-tax labour-income Gini: {final['pre_tax_labour_income_gini']:.3f}")
        print(f"  Final disposable-income Gini: {final['disposable_income_gini']:.3f}")
        print(f"  Final capital-income share: {final['capital_income_share']:.3f}")
        print(f"  Final top 10% share: {final['top_10_share']:.3f}")
        print(f"  Final top 1% share: {final['top_1_share']:.3f}")
        print(f"  Shorrocks index: {mobility['shorrocks_index']:.3f}")
        print(f"  Top 20% persistence: {mobility['top_20_persistence']:.3f}")
        print_transition_matrix(name, sim.initial_wealth, sim.agents["wealth"])

    combined_results = pd.concat(yearly_results, ignore_index=True)
    final_summary = pd.DataFrame(final_summary_rows)
    mobility_summary = pd.DataFrame(mobility_rows)

    combined_results.to_csv(CSV_DIR / "scenario_comparison.csv", index=False)
    final_summary.to_csv(CSV_DIR / "final_summary.csv", index=False)
    mobility_summary.to_csv(CSV_DIR / "mobility_summary.csv", index=False)

    plot_metric_over_time(
        combined_results,
        "wealth_gini",
        "Wealth Gini",
        FIGURE_DIR / "wealth_gini_over_time.png",
    )
    plot_metric_over_time(
        combined_results,
        "pre_tax_labour_income_gini",
        "Pre-tax labour-income Gini",
        FIGURE_DIR / "pre_tax_labour_income_gini_over_time.png",
    )
    plot_metric_over_time(
        combined_results,
        "disposable_income_gini",
        "Disposable-income Gini",
        FIGURE_DIR / "disposable_income_gini_over_time.png",
    )
    plot_metric_over_time(
        combined_results,
        "capital_income_share",
        "Capital income share of total income",
        FIGURE_DIR / "capital_income_share_over_time.png",
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
        final_wealth_by_scenario,
        FIGURE_DIR / "final_lorenz_curves.png",
    )

    print(f"Saved CSV outputs to {CSV_DIR}")
    print(f"Saved figure outputs to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
