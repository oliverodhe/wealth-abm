from __future__ import annotations

import numpy as np

from abm.parameters import ModelParams


def initialise_agents(params: ModelParams) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(params.seed)
    n_bottom = int(0.90 * params.N)
    n_top = params.N - n_bottom

    bottom_wealth = rng.lognormal(mean=12.0, sigma=0.8, size=n_bottom)
    top_wealth = (rng.pareto(a=2.0, size=n_top) + 1.0) * bottom_wealth.mean() * 4.0
    wealth = np.concatenate([bottom_wealth, top_wealth])
    rng.shuffle(wealth)
    wealth = wealth / wealth.mean() * (params.base_income * 2.5)

    skill = rng.normal(loc=0.0, scale=0.35, size=params.N)
    income_shock = rng.normal(loc=0.0, scale=params.income_shock_sd, size=params.N)
    saving_rate = rng.uniform(low=0.05, high=0.25, size=params.N)
    return_class = rng.choice(np.array([0, 1, 2]), size=params.N, p=np.array([0.3, 0.5, 0.2]))

    return {
        "wealth": wealth,
        "skill": skill,
        "income_shock": income_shock,
        "saving_rate": saving_rate,
        "return_class": return_class,
    }
