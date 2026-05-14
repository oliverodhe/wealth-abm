import numpy as np

from abm.agents import initialise_agents
from abm.scenarios import SCENARIOS, calibrate_scenarios, first_year_labour_income
from abm.simulation import Simulation
from abm.swedish_tax import swedish_income_tax
from abm.parameters import ModelParams


EXPECTED_RESULT_COLUMNS = {
    "year",
    "wealth_gini",
    "top_10_share",
    "top_1_share",
    "labour_tax_revenue",
    "capital_tax_revenue",
    "total_tax_revenue",
    "total_transfers",
    "pre_tax_labour_income_gini",
    "disposable_income_gini",
    "capital_income_gini",
    "capital_income_share",
    "avg_return_class_low",
    "avg_return_class_medium",
    "avg_return_class_high",
}


def test_all_scenarios_run_without_nan_values() -> None:
    params = ModelParams(N=500, years=3, seed=11)
    agents = initialise_agents(params)
    calibrated = calibrate_scenarios(params, agents, SCENARIOS)

    for scenario in calibrated.values():
        results = Simulation(
            scenario.params,
            tax_system=scenario.config.tax_system,
            initial_agents=agents,
        ).run()
        assert not np.isnan(results.to_numpy()).any()


def test_calibrated_scenarios_match_target_first_year_labour_tax_revenue() -> None:
    params = ModelParams(N=1_000, years=2, seed=12)
    agents = initialise_agents(params)
    calibrated = calibrate_scenarios(params, agents, SCENARIOS)

    for scenario in calibrated.values():
        revenue_difference = (
            scenario.first_year_labour_revenue - scenario.target_labour_revenue
        )
        assert abs(revenue_difference) < 1e-4


def test_high_progressivity_has_higher_top_decile_average_tax_rate_than_low() -> None:
    params = ModelParams(N=1_000, years=2, seed=13)
    agents = initialise_agents(params)
    income = first_year_labour_income(params, agents)
    calibrated = calibrate_scenarios(params, agents, SCENARIOS)
    top_decile = income >= np.quantile(income, 0.9)

    low = calibrated["swedish_low_progressivity"].params
    high = calibrated["swedish_high_progressivity"].params
    low_tax = swedish_income_tax(
        income,
        low.municipal_rate,
        low.state_rate,
        low.state_threshold,
        low.earned_income_max_credit,
        low.earned_income_phaseout_start,
        low.earned_income_phaseout_rate,
    )
    high_tax = swedish_income_tax(
        income,
        high.municipal_rate,
        high.state_rate,
        high.state_threshold,
        high.earned_income_max_credit,
        high.earned_income_phaseout_start,
        high.earned_income_phaseout_rate,
    )

    low_top_rate = low_tax[top_decile].sum() / income[top_decile].sum()
    high_top_rate = high_tax[top_decile].sum() / income[top_decile].sum()
    assert high_top_rate > low_top_rate


def test_scenario_outputs_contain_expected_columns() -> None:
    params = ModelParams(N=500, years=3, seed=14)
    agents = initialise_agents(params)
    calibrated = calibrate_scenarios(params, agents, SCENARIOS)
    scenario = calibrated["swedish_baseline_progressivity"]
    results = Simulation(
        scenario.params,
        tax_system=scenario.config.tax_system,
        initial_agents=agents,
    ).run()

    assert EXPECTED_RESULT_COLUMNS.issubset(results.columns)
    for decile in range(1, 11):
        assert f"avg_return_wealth_decile_{decile}" in results.columns
        assert f"avg_tax_rate_income_decile_{decile}" in results.columns
