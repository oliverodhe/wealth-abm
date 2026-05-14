from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from abm.parameters import ModelParams
from abm.swedish_tax import calibrate_municipal_rate_for_target_revenue
from abm.swedish_tax import swedish_income_tax
from abm.tax import find_revenue_neutral_base_rate, flat_tax, progressive_tax


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    tax_system: str
    municipal_rate: float | None
    state_rate: float
    state_threshold: float
    earned_income_max_credit: float
    earned_income_phaseout_start: float
    earned_income_phaseout_rate: float
    transfer_policy: str = "lump_sum"
    calibrate: bool = True


@dataclass(frozen=True)
class CalibratedScenario:
    config: ScenarioConfig
    params: ModelParams
    calibrated_rate: float
    target_labour_revenue: float
    first_year_labour_revenue: float


SCENARIOS: dict[str, ScenarioConfig] = {
    "flat": ScenarioConfig(
        name="flat",
        tax_system="flat",
        municipal_rate=None,
        state_rate=0.0,
        state_threshold=0.0,
        earned_income_max_credit=0.0,
        earned_income_phaseout_start=0.0,
        earned_income_phaseout_rate=0.0,
        calibrate=False,
    ),
    "swedish_low_progressivity": ScenarioConfig(
        name="swedish_low_progressivity",
        tax_system="swedish",
        municipal_rate=None,
        state_rate=0.10,
        state_threshold=750_000.0,
        earned_income_max_credit=25_000.0,
        earned_income_phaseout_start=650_000.0,
        earned_income_phaseout_rate=0.04,
    ),
    "swedish_baseline_progressivity": ScenarioConfig(
        name="swedish_baseline_progressivity",
        tax_system="swedish",
        municipal_rate=None,
        state_rate=0.20,
        state_threshold=600_000.0,
        earned_income_max_credit=35_000.0,
        earned_income_phaseout_start=500_000.0,
        earned_income_phaseout_rate=0.08,
    ),
    "swedish_high_progressivity": ScenarioConfig(
        name="swedish_high_progressivity",
        tax_system="swedish",
        municipal_rate=None,
        state_rate=0.30,
        state_threshold=450_000.0,
        earned_income_max_credit=45_000.0,
        earned_income_phaseout_start=400_000.0,
        earned_income_phaseout_rate=0.10,
    ),
}


def first_year_labour_income(
    params: ModelParams,
    agents: Mapping[str, np.ndarray],
) -> np.ndarray:
    rng = np.random.default_rng(params.seed)
    first_year_shock = (
        params.income_persistence * agents["income_shock"]
        + rng.normal(0.0, params.income_shock_sd, params.N)
    )
    return params.base_income * np.exp(agents["skill"] + first_year_shock)


def calibrate_scenario(
    params: ModelParams,
    config: ScenarioConfig,
    first_year_income: np.ndarray,
    target_labour_revenue: float,
) -> CalibratedScenario:
    if config.tax_system == "flat":
        scenario_params = replace(params, flat_tax_rate=params.flat_tax_rate)
        revenue = float(flat_tax(first_year_income, scenario_params.flat_tax_rate).sum())
        return CalibratedScenario(
            config=config,
            params=scenario_params,
            calibrated_rate=scenario_params.flat_tax_rate,
            target_labour_revenue=target_labour_revenue,
            first_year_labour_revenue=revenue,
        )

    if config.tax_system == "swedish":
        municipal_rate = (
            calibrate_municipal_rate_for_target_revenue(
                first_year_income,
                target_labour_revenue,
                config.state_rate,
                config.state_threshold,
                config.earned_income_max_credit,
                config.earned_income_phaseout_start,
                config.earned_income_phaseout_rate,
            )
            if config.calibrate
            else config.municipal_rate
        )
        if municipal_rate is None:
            raise ValueError(f"Scenario {config.name} needs a municipal rate")

        scenario_params = replace(
            params,
            municipal_rate=municipal_rate,
            state_rate=config.state_rate,
            state_threshold=config.state_threshold,
            earned_income_max_credit=config.earned_income_max_credit,
            earned_income_phaseout_start=config.earned_income_phaseout_start,
            earned_income_phaseout_rate=config.earned_income_phaseout_rate,
        )
        revenue = float(
            swedish_income_tax(
                first_year_income,
                scenario_params.municipal_rate,
                scenario_params.state_rate,
                scenario_params.state_threshold,
                scenario_params.earned_income_max_credit,
                scenario_params.earned_income_phaseout_start,
                scenario_params.earned_income_phaseout_rate,
            ).sum()
        )
        return CalibratedScenario(
            config=config,
            params=scenario_params,
            calibrated_rate=municipal_rate,
            target_labour_revenue=target_labour_revenue,
            first_year_labour_revenue=revenue,
        )

    if config.tax_system == "simple_progressive":
        base_rate = find_revenue_neutral_base_rate(
            first_year_income,
            target_labour_revenue,
            config.state_rate,
            config.state_threshold,
        )
        scenario_params = replace(
            params,
            flat_tax_rate=base_rate,
            progressive_top_rate=config.state_rate,
            progressive_threshold=config.state_threshold,
        )
        revenue = float(
            progressive_tax(
                first_year_income,
                scenario_params.flat_tax_rate,
                scenario_params.progressive_top_rate,
                scenario_params.progressive_threshold,
            ).sum()
        )
        return CalibratedScenario(
            config=config,
            params=scenario_params,
            calibrated_rate=base_rate,
            target_labour_revenue=target_labour_revenue,
            first_year_labour_revenue=revenue,
        )

    raise ValueError(f"Unknown tax system: {config.tax_system}")


def calibrate_scenarios(
    params: ModelParams,
    agents: Mapping[str, np.ndarray],
    scenarios: Mapping[str, ScenarioConfig] = SCENARIOS,
) -> dict[str, CalibratedScenario]:
    income = first_year_labour_income(params, agents)
    target_revenue = float(flat_tax(income, params.flat_tax_rate).sum())
    return {
        name: calibrate_scenario(params, config, income, target_revenue)
        for name, config in scenarios.items()
    }
