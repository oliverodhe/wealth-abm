from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from abm.agents import initialise_agents
from abm.decompositions import DECOMPOSITIONS, apply_decomposition
from abm.mobility import assign_quantiles, rank_correlation, shorrocks_index, transition_matrix
from abm.parameters import ModelParams
from abm.plotting import plot_metric_over_time
from abm.scenarios import SCENARIOS, CalibratedScenario, calibrate_scenarios
from abm.simulation import Simulation

CSV_DIR = ROOT / "outputs" / "csv"
FIGURE_DIR = ROOT / "outputs" / "figures"

DECOMPOSITION_SCENARIOS = {
    "flat": SCENARIOS["flat"],
    "swedish_baseline_progressivity": SCENARIOS["swedish_baseline_progressivity"],
    "swedish_high_progressivity": SCENARIOS["swedish_high_progressivity"],
}


def mobility_metrics(
    decomposition: str,
    scenario: str,
    start_wealth: np.ndarray,
    end_wealth: np.ndarray,
) -> dict[str, float | str]:
    matrix = transition_matrix(start_wealth, end_wealth, n_quantiles=5)
    start_quantiles = assign_quantiles(start_wealth, n_quantiles=5)
    end_quantiles = assign_quantiles(end_wealth, n_quantiles=5)

    return {
        "decomposition": decomposition,
        "scenario": scenario,
        "shorrocks_index": shorrocks_index(matrix),
        "rank_correlation": rank_correlation(start_wealth, end_wealth),
        "top_20_persistence": matrix[4, 4],
        "bottom_40_to_top_40": float(np.mean(end_quantiles[start_quantiles <= 1] >= 3)),
    }


def final_summary_row(
    decomposition: str,
    scenario: str,
    calibrated: CalibratedScenario,
    results: pd.DataFrame,
    mobility: dict[str, float | str],
) -> dict[str, float | str]:
    final = results.iloc[-1]
    return {
        "decomposition": decomposition,
        "scenario": scenario,
        "tax_system": calibrated.config.tax_system,
        "calibrated_rate": calibrated.calibrated_rate,
        "first_year_revenue_difference": (
            calibrated.first_year_labour_revenue - calibrated.target_labour_revenue
        ),
        "final_gini": final["wealth_gini"],
        "final_disposable_income_gini": final["disposable_income_gini"],
        "final_top_10_share": final["top_10_share"],
        "final_top_1_share": final["top_1_share"],
        "final_capital_income_share": final["capital_income_share"],
        "final_transfer_spending_share": final["transfer_spending_share"],
        "shorrocks_index": mobility["shorrocks_index"],
        "rank_correlation": mobility["rank_correlation"],
        "top_20_persistence": mobility["top_20_persistence"],
        "bottom_40_to_top_40": mobility["bottom_40_to_top_40"],
    }


def decomposition_summary(final_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for scenario, data in final_summary.groupby("scenario"):
        baseline = data.loc[data["decomposition"] == "baseline"].iloc[0]
        for _, row in data.iterrows():
            rows.append(
                {
                    "decomposition": row["decomposition"],
                    "scenario": scenario,
                    "wealth_gini_difference_vs_baseline": (
                        row["final_gini"] - baseline["final_gini"]
                    ),
                    "top_10_share_difference_vs_baseline": (
                        row["final_top_10_share"] - baseline["final_top_10_share"]
                    ),
                    "top_1_share_difference_vs_baseline": (
                        row["final_top_1_share"] - baseline["final_top_1_share"]
                    ),
                    "disposable_income_gini_difference_vs_baseline": (
                        row["final_disposable_income_gini"]
                        - baseline["final_disposable_income_gini"]
                    ),
                    "shorrocks_difference_vs_baseline": (
                        row["shorrocks_index"] - baseline["shorrocks_index"]
                    ),
                    "top_20_persistence_difference_vs_baseline": (
                        row["top_20_persistence"] - baseline["top_20_persistence"]
                    ),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    base_params = ModelParams()
    initial_agents = initialise_agents(base_params)

    yearly_results: list[pd.DataFrame] = []
    final_rows: list[dict[str, float | str]] = []
    mobility_rows: list[dict[str, float | str]] = []

    for decomposition_name, decomposition in DECOMPOSITIONS.items():
        params = apply_decomposition(base_params, decomposition)
        calibrated_scenarios = calibrate_scenarios(
            params,
            initial_agents,
            DECOMPOSITION_SCENARIOS,
        )

        for scenario_name, calibrated in calibrated_scenarios.items():
            sim = Simulation(
                calibrated.params,
                tax_system=calibrated.config.tax_system,
                initial_agents=initial_agents,
            )
            results = sim.run()
            scenario_results = results.copy()
            scenario_results.insert(0, "scenario", scenario_name)
            scenario_results.insert(0, "decomposition", decomposition_name)
            yearly_results.append(scenario_results)

            mobility = mobility_metrics(
                decomposition_name,
                scenario_name,
                sim.initial_wealth,
                sim.agents["wealth"],
            )
            mobility_rows.append(mobility)
            final_rows.append(
                final_summary_row(
                    decomposition_name,
                    scenario_name,
                    calibrated,
                    results,
                    mobility,
                )
            )

            final = results.iloc[-1]
            print(f"{decomposition_name} / {scenario_name}")
            print(f"  Final Gini: {final['wealth_gini']:.3f}")
            print(f"  Final top 10% share: {final['top_10_share']:.3f}")
            print(f"  Shorrocks index: {mobility['shorrocks_index']:.3f}")
            print(f"  Top 20% persistence: {mobility['top_20_persistence']:.3f}")

    comparison = pd.concat(yearly_results, ignore_index=True)
    final_summary = pd.DataFrame(final_rows)
    mobility_summary = pd.DataFrame(mobility_rows)
    decomposition_effects = decomposition_summary(final_summary)

    comparison.to_csv(CSV_DIR / "decomposition_comparison.csv", index=False)
    final_summary.to_csv(CSV_DIR / "decomposition_final_summary.csv", index=False)
    mobility_summary.to_csv(CSV_DIR / "decomposition_mobility_summary.csv", index=False)
    decomposition_effects.to_csv(CSV_DIR / "decomposition_summary.csv", index=False)

    plot_metric_over_time(
        comparison,
        "wealth_gini",
        "Wealth Gini",
        FIGURE_DIR / "decomposition_wealth_gini_over_time.png",
    )
    plot_metric_over_time(
        comparison,
        "top_10_share",
        "Top 10% wealth share",
        FIGURE_DIR / "decomposition_top_10_share_over_time.png",
    )
    plot_metric_over_time(
        comparison,
        "top_1_share",
        "Top 1% wealth share",
        FIGURE_DIR / "decomposition_top_1_share_over_time.png",
    )
    plot_metric_over_time(
        comparison,
        "disposable_income_gini",
        "Disposable-income Gini",
        FIGURE_DIR / "decomposition_disposable_income_gini_over_time.png",
    )

    print(f"Saved decomposition CSV outputs to {CSV_DIR}")
    print(f"Saved decomposition figure outputs to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
