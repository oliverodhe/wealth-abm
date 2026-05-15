import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from abm.agents import initialise_agents
from abm.decompositions import DECOMPOSITIONS, apply_decomposition
from abm.parameters import ModelParams
from abm.scenarios import SCENARIOS, calibrate_scenarios
from abm.simulation import Simulation


def test_decomposition_toggles_run_without_nan_values() -> None:
    base_params = ModelParams(N=400, years=2, seed=31)
    agents = initialise_agents(base_params)
    scenarios = {
        "flat": SCENARIOS["flat"],
        "swedish_baseline_progressivity": SCENARIOS["swedish_baseline_progressivity"],
        "swedish_high_progressivity": SCENARIOS["swedish_high_progressivity"],
    }

    for decomposition in DECOMPOSITIONS.values():
        params = apply_decomposition(base_params, decomposition)
        calibrated = calibrate_scenarios(params, agents, scenarios)
        for scenario in calibrated.values():
            results = Simulation(
                scenario.params,
                tax_system=scenario.config.tax_system,
                initial_agents=agents,
            ).run()
            assert not np.isnan(results.to_numpy()).any()


def test_no_transfers_removes_transfer_spending() -> None:
    params = apply_decomposition(ModelParams(N=400, years=2, seed=32), DECOMPOSITIONS["no_transfers"])
    results = Simulation(params, tax_system="flat").run()
    assert np.allclose(results["total_transfers"], 0.0)


def test_homogeneous_saving_rates_are_equalized() -> None:
    params = apply_decomposition(
        ModelParams(N=400, years=2, seed=33),
        DECOMPOSITIONS["homogeneous_saving_rates"],
    )
    sim = Simulation(params, tax_system="flat")
    assert np.unique(sim.agents["saving_rate"]).size == 1


def test_decomposition_comparison_file_contains_expected_columns() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "scripts/run_decompositions.py", "--seed-list", "0,1,2"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    comparison = pd.read_csv(root / "outputs" / "csv" / "decomposition_comparison.csv")
    summary = pd.read_csv(root / "outputs" / "csv" / "decomposition_summary.csv")

    assert {"decomposition", "scenario"}.issubset(comparison.columns)
    assert {
        "wealth_gini_difference_vs_baseline_mean",
        "top_20_persistence_difference_vs_baseline_mean",
    }.issubset(summary.columns)
