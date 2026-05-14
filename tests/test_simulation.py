import numpy as np

from abm.parameters import ModelParams
from abm.simulation import Simulation


def test_simulation_produces_no_nan_values() -> None:
    params = ModelParams(N=500, years=5, seed=7)
    results = Simulation(params, tax_system="simple_progressive").run()
    assert not results.empty
    assert not np.isnan(results.to_numpy()).any()
