import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from abm.parameters import ModelParams
from abm.simulation import Simulation
from abm.transfer_policies import TRANSFER_POLICIES, apply_transfer_policy


def test_transfer_policies_run_without_nan_values() -> None:
    base_params = ModelParams(N=300, years=2, seed=41)

    for policy in TRANSFER_POLICIES.values():
        params = apply_transfer_policy(base_params, policy)
        results = Simulation(params, tax_system="flat").run()
        assert not np.isnan(results.to_numpy()).any()


def test_lump_sum_only_has_no_targeted_transfer_spending() -> None:
    params = apply_transfer_policy(
        ModelParams(N=300, years=2, seed=42),
        TRANSFER_POLICIES["lump_sum_only"],
    )
    results = Simulation(params, tax_system="flat").run()
    assert np.allclose(results["means_tested_transfer_spending"], 0.0)
    assert np.allclose(results["unemployment_transfer_spending"], 0.0)


def test_transfer_policy_decomposition_outputs_have_expected_columns() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "scripts/run_transfer_policy_decompositions.py"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    yearly = pd.read_csv(root / "outputs" / "csv" / "transfer_policy_yearly_results.csv")
    summary = pd.read_csv(root / "outputs" / "csv" / "decomposition_summary.csv")

    assert {"return_preset", "transfer_policy", "tax_scenario"}.issubset(yearly.columns)
    assert {
        "wealth_gini_difference_relative_to_flat",
        "disposable_income_gini_difference_relative_to_flat",
        "top_10_difference_relative_to_flat",
        "top_1_difference_relative_to_flat",
        "shorrocks_difference_relative_to_flat",
        "top_20_persistence_difference_relative_to_flat",
    }.issubset(summary.columns)


def test_transfer_policy_summary_contains_all_36_combinations() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "scripts/run_transfer_policy_decompositions.py"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    final_summary = pd.read_csv(root / "outputs" / "csv" / "transfer_policy_final_summary.csv")
    keys = final_summary[["return_preset", "transfer_policy", "tax_scenario"]].drop_duplicates()
    assert len(keys) == 36
    assert final_summary.shape[0] == 36


def test_transfer_policy_invariants_in_summary() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "scripts/run_transfer_policy_decompositions.py"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    final_summary = pd.read_csv(root / "outputs" / "csv" / "transfer_policy_final_summary.csv")
    no_transfers = final_summary[final_summary["transfer_policy"] == "no_transfers"]
    lump_sum = final_summary[final_summary["transfer_policy"] == "lump_sum_only"]

    assert np.allclose(no_transfers["final_transfer_spending_share"], 0.0)
    assert np.allclose(no_transfers["final_means_tested_recipient_share"], 0.0)
    assert np.allclose(lump_sum["final_means_tested_recipient_share"], 0.0)
