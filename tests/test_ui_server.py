from abm.ui_server import run_ui_simulation


def test_ui_simulation_returns_summary_and_yearly_results() -> None:
    result = run_ui_simulation(
        {
            "params": {"N": 200, "years": 2, "seed": 51},
            "return_preset": "baseline_return_heterogeneity",
            "transfer_policy": "universal_plus_safety_floor",
            "scenarios": ["flat", "swedish_baseline_progressivity"],
            "toggles": {},
        }
    )

    assert len(result["summary"]) == 2
    assert len(result["calibration"]) == 2
    assert len(result["yearly"]) == 4
    assert {"tax_scenario", "final_wealth_gini"}.issubset(result["summary"][0])
