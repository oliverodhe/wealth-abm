import numpy as np

from abm.parameters import ModelParams
from abm.returns import draw_capital_returns
from abm.simulation import Simulation


def test_returns_remain_bounded() -> None:
    params = ModelParams(N=3_000, seed=1)
    rng = np.random.default_rng(params.seed)
    return_class = np.array([0, 1, 2] * 1_000)
    returns = draw_capital_returns(return_class, params, rng)
    assert np.all(returns >= params.min_capital_return)
    assert np.all(returns <= params.max_capital_return)


def test_high_return_class_has_higher_mean_realised_returns() -> None:
    params = ModelParams(N=30_000, seed=2)
    rng = np.random.default_rng(params.seed)
    return_class = np.array([0] * 10_000 + [1] * 10_000 + [2] * 10_000)
    returns = draw_capital_returns(return_class, params, rng)
    low_mean = returns[return_class == 0].mean()
    medium_mean = returns[return_class == 1].mean()
    high_mean = returns[return_class == 2].mean()
    assert high_mean > medium_mean > low_mean


def test_return_diagnostics_have_no_nan_values() -> None:
    params = ModelParams(N=500, years=5, seed=7)
    results = Simulation(params, tax_system="swedish").run()
    return_columns = [column for column in results.columns if column.startswith("avg_return_")]
    assert return_columns
    assert not np.isnan(results[return_columns].to_numpy()).any()
