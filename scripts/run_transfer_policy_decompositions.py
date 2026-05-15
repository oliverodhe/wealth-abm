from __future__ import annotations

import os
import argparse
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "abm_matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
XDG_CACHE_DIR = Path(tempfile.gettempdir()) / "abm_cache"
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from abm.agents import initialise_agents
from abm.mobility import assign_quantiles, rank_correlation, shorrocks_index, transition_matrix
from abm.parameters import ModelParams
from abm.return_presets import RETURN_PRESETS, apply_return_preset
from abm.scenarios import SCENARIOS, CalibratedScenario, calibrate_scenarios
from abm.simulation import Simulation
from abm.transfer_policies import TRANSFER_POLICIES, apply_transfer_policy
from runner_utils import add_seed_arguments, aggregate_with_ci, print_aggregated_metric_block, seed_values

CSV_DIR = ROOT / "outputs" / "csv"
FIGURE_DIR = ROOT / "outputs" / "figures"


def mobility_metrics(
    return_preset: str,
    transfer_policy: str,
    tax_scenario: str,
    start_wealth: np.ndarray,
    end_wealth: np.ndarray,
) -> dict[str, float | str]:
    matrix = transition_matrix(start_wealth, end_wealth, n_quantiles=5)
    start_quantiles = assign_quantiles(start_wealth, n_quantiles=5)
    end_quantiles = assign_quantiles(end_wealth, n_quantiles=5)

    return {
        "return_preset": return_preset,
        "transfer_policy": transfer_policy,
        "tax_scenario": tax_scenario,
        "shorrocks_index": shorrocks_index(matrix),
        "rank_correlation": rank_correlation(start_wealth, end_wealth),
        "top_20_persistence": matrix[4, 4],
        "bottom_40_to_top_40": float(np.mean(end_quantiles[start_quantiles <= 1] >= 3)),
    }


def final_summary_row(
    return_preset: str,
    transfer_policy: str,
    tax_scenario: str,
    calibrated: CalibratedScenario,
    results: pd.DataFrame,
    mobility: dict[str, float | str],
) -> dict[str, float | str]:
    final = results.iloc[-1]
    return {
        "return_preset": return_preset,
        "transfer_policy": transfer_policy,
        "tax_scenario": tax_scenario,
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
        "final_means_tested_recipient_share": final["means_tested_recipient_share"],
        "final_unemployment_rate": final["unemployment_rate"],
        "shorrocks_index": mobility["shorrocks_index"],
        "rank_correlation": mobility["rank_correlation"],
        "top_20_persistence": mobility["top_20_persistence"],
        "bottom_40_to_top_40": mobility["bottom_40_to_top_40"],
    }


def relative_to_flat_summary(final_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    group_columns = ["return_preset", "transfer_policy"]
    if "seed" in final_summary.columns:
        group_columns = ["seed", "return_preset", "transfer_policy"]

    for group_key, data in final_summary.groupby(group_columns):
        if len(group_columns) == 3:
            seed, return_preset, transfer_policy = group_key
        else:
            seed = None
            return_preset, transfer_policy = group_key
        flat = data.loc[data["tax_scenario"] == "flat"].iloc[0]

        for _, row in data.iterrows():
            out = {
                    "transfer_policy": transfer_policy,
                    "return_preset": return_preset,
                    "tax_scenario": row["tax_scenario"],
                    "wealth_gini_difference_relative_to_flat": (
                        row["final_gini"] - flat["final_gini"]
                    ),
                    "disposable_income_gini_difference_relative_to_flat": (
                        row["final_disposable_income_gini"]
                        - flat["final_disposable_income_gini"]
                    ),
                    "top_10_difference_relative_to_flat": (
                        row["final_top_10_share"] - flat["final_top_10_share"]
                    ),
                    "top_1_difference_relative_to_flat": (
                        row["final_top_1_share"] - flat["final_top_1_share"]
                    ),
                    "shorrocks_difference_relative_to_flat": (
                        row["shorrocks_index"] - flat["shorrocks_index"]
                    ),
                    "top_20_persistence_difference_relative_to_flat": (
                        row["top_20_persistence"] - flat["top_20_persistence"]
                    ),
                }
            if seed is not None:
                out["seed"] = seed
            rows.append(out)

    return pd.DataFrame(rows)


def plot_final_metric_by_transfer_policy(
    final_summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    transfer_policies = list(TRANSFER_POLICIES)
    tax_scenarios = list(SCENARIOS)
    return_presets = list(RETURN_PRESETS)

    fig, axes = plt.subplots(1, len(return_presets), figsize=(16, 5), sharey=True)
    if len(return_presets) == 1:
        axes = [axes]

    for ax, return_preset in zip(axes, return_presets):
        subset = final_summary[final_summary["return_preset"] == return_preset]
        for tax_scenario in tax_scenarios:
            values = []
            for transfer_policy in transfer_policies:
                row = subset[
                    (subset["transfer_policy"] == transfer_policy)
                    & (subset["tax_scenario"] == tax_scenario)
                ]
                value_column = f"{metric}_mean" if f"{metric}_mean" in row.columns else metric
                values.append(float(row.iloc[0][value_column]))
            ax.plot(transfer_policies, values, marker="o", label=tax_scenario)

        ax.set_title(return_preset)
        ax.set_xlabel("Transfer policy")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(ylabel)
    axes[-1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_seed_arguments(parser)
    args = parser.parse_args()
    seeds = seed_values(args.seeds, args.seed_list)

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    yearly_results: list[pd.DataFrame] = []
    final_rows: list[dict[str, float | str]] = []
    mobility_rows: list[dict[str, float | str]] = []

    for seed in seeds:
        base_params = ModelParams(seed=seed)
        initial_agents = initialise_agents(base_params)

        for return_preset_name, return_preset in RETURN_PRESETS.items():
            for transfer_policy_name, transfer_policy in TRANSFER_POLICIES.items():
                params = apply_transfer_policy(
                    apply_return_preset(base_params, return_preset),
                    transfer_policy,
                )
                calibrated_scenarios = calibrate_scenarios(params, initial_agents, SCENARIOS)

                for scenario_name, calibrated in calibrated_scenarios.items():
                    sim = Simulation(
                        calibrated.params,
                        tax_system=calibrated.config.tax_system,
                        initial_agents=initial_agents,
                    )
                    results = sim.run()
                    scenario_results = results.copy()
                    scenario_results.insert(0, "seed", seed)
                    scenario_results.insert(0, "tax_scenario", scenario_name)
                    scenario_results.insert(0, "transfer_policy", transfer_policy_name)
                    scenario_results.insert(0, "return_preset", return_preset_name)
                    yearly_results.append(scenario_results)

                    mobility = mobility_metrics(
                        return_preset_name,
                        transfer_policy_name,
                        scenario_name,
                        sim.initial_wealth,
                        sim.agents["wealth"],
                    )
                    mobility["seed"] = seed
                    mobility_rows.append(mobility)
                    final_row = final_summary_row(
                            return_preset_name,
                            transfer_policy_name,
                            scenario_name,
                            calibrated,
                            results,
                            mobility,
                        )
                    final_row["seed"] = seed
                    final_rows.append(final_row)

    comparison = pd.concat(yearly_results, ignore_index=True)
    final_summary = pd.DataFrame(final_rows)
    mobility_summary = pd.DataFrame(mobility_rows)
    decomposition_summary = relative_to_flat_summary(final_summary)

    comparison.to_csv(CSV_DIR / "transfer_policy_yearly_results_per_seed.csv", index=False)
    final_summary.to_csv(CSV_DIR / "transfer_policy_final_summary_per_seed.csv", index=False)
    mobility_summary.to_csv(CSV_DIR / "transfer_policy_mobility_summary_per_seed.csv", index=False)
    decomposition_summary.to_csv(CSV_DIR / "transfer_policy_decomposition_summary_per_seed.csv", index=False)

    comparison_agg = aggregate_with_ci(
        comparison,
        ["return_preset", "transfer_policy", "tax_scenario", "year"],
    )
    final_summary_agg = aggregate_with_ci(
        final_summary,
        ["return_preset", "transfer_policy", "tax_scenario", "tax_system"],
    )
    mobility_summary_agg = aggregate_with_ci(
        mobility_summary,
        ["return_preset", "transfer_policy", "tax_scenario"],
    )
    decomposition_summary_agg = aggregate_with_ci(
        decomposition_summary,
        ["return_preset", "transfer_policy", "tax_scenario"],
    )

    comparison_agg.to_csv(CSV_DIR / "transfer_policy_yearly_results.csv", index=False)
    final_summary_agg.to_csv(CSV_DIR / "transfer_policy_final_summary.csv", index=False)
    mobility_summary_agg.to_csv(CSV_DIR / "transfer_policy_mobility_summary.csv", index=False)
    decomposition_summary_agg.to_csv(CSV_DIR / "decomposition_summary.csv", index=False)
    decomposition_summary_agg.to_csv(CSV_DIR / "transfer_policy_decomposition_summary.csv", index=False)

    for _, row in final_summary_agg.iterrows():
        print_aggregated_metric_block(
            f"{row['return_preset']} / {row['transfer_policy']} / {row['tax_scenario']} (S={int(row['seed_count'])})",
            row,
            ["final_gini", "final_disposable_income_gini", "final_top_1_share", "shorrocks_index"],
        )

    plot_final_metric_by_transfer_policy(
        final_summary_agg,
        "final_gini",
        "Final wealth Gini",
        FIGURE_DIR / "transfer_policy_wealth_gini.png",
    )
    plot_final_metric_by_transfer_policy(
        final_summary_agg,
        "final_disposable_income_gini",
        "Final disposable-income Gini",
        FIGURE_DIR / "transfer_policy_disposable_income_gini.png",
    )
    plot_final_metric_by_transfer_policy(
        final_summary_agg,
        "final_top_1_share",
        "Final top 1% wealth share",
        FIGURE_DIR / "transfer_policy_top_1_share.png",
    )

    print(f"Saved transfer-policy CSV outputs to {CSV_DIR}")
    print(f"Saved transfer-policy figure outputs to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
