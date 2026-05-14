import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from abm.agents import initialise_agents
from abm.parameters import ModelParams
from abm.return_presets import RETURN_PRESETS, apply_return_preset
from abm.returns import draw_capital_returns
from abm.scenarios import SCENARIOS, calibrate_scenarios
from abm.simulation import Simulation


def test_all_return_presets_run_without_nan_values() -> None:
    base_params = ModelParams(N=400, years=2, seed=21)
    agents = initialise_agents(base_params)

    for preset in RETURN_PRESETS.values():
        params = apply_return_preset(base_params, preset)
        calibrated = calibrate_scenarios(params, agents, SCENARIOS)
        for scenario in calibrated.values():
            results = Simulation(
                scenario.params,
                tax_system=scenario.config.tax_system,
                initial_agents=agents,
            ).run()
            assert not np.isnan(results.to_numpy()).any()


def test_high_return_heterogeneity_has_larger_realised_return_dispersion() -> None:
    base_params = ModelParams(N=9_000, seed=22)
    return_class = np.array([0, 1, 2] * 3_000)

    low_params = apply_return_preset(base_params, RETURN_PRESETS["low_return_heterogeneity"])
    high_params = apply_return_preset(base_params, RETURN_PRESETS["high_return_heterogeneity"])
    low_returns = draw_capital_returns(
        return_class,
        low_params,
        np.random.default_rng(base_params.seed),
    )
    high_returns = draw_capital_returns(
        return_class,
        high_params,
        np.random.default_rng(base_params.seed),
    )

    assert high_returns.std() > low_returns.std()


def test_scenario_comparison_file_contains_return_preset_column() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "scripts/run_scenarios.py"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    comparison = pd.read_csv(root / "outputs" / "csv" / "scenario_comparison.csv")
    assert "return_preset" in comparison.columns
