from __future__ import annotations

import numpy as np

from abm.parameters import ModelParams

RETURN_CLASS_NAMES = {
    0: "low",
    1: "medium",
    2: "high",
}


def class_mean_returns(params: ModelParams) -> np.ndarray:
    return np.array(
        [
            params.low_return_mean,
            params.medium_return_mean,
            params.high_return_mean,
        ]
    )


def draw_capital_returns(
    return_class: np.ndarray,
    params: ModelParams,
    rng: np.random.Generator,
) -> np.ndarray:
    means = class_mean_returns(params)[return_class.astype(int)]
    shocks = rng.normal(0.0, params.return_shock_sd, return_class.size)
    returns = means + shocks
    return np.clip(returns, params.min_capital_return, params.max_capital_return)
