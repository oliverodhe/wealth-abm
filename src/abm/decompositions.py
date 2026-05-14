from __future__ import annotations

from dataclasses import dataclass, replace

from abm.parameters import ModelParams


@dataclass(frozen=True)
class DecompositionConfig:
    name: str
    no_transfers: bool = False
    homogeneous_returns: bool = False
    homogeneous_saving_rates: bool = False
    no_capital_tax: bool = False


DECOMPOSITIONS: dict[str, DecompositionConfig] = {
    "baseline": DecompositionConfig(name="baseline"),
    "no_transfers": DecompositionConfig(name="no_transfers", no_transfers=True),
    "homogeneous_returns": DecompositionConfig(
        name="homogeneous_returns",
        homogeneous_returns=True,
    ),
    "homogeneous_saving_rates": DecompositionConfig(
        name="homogeneous_saving_rates",
        homogeneous_saving_rates=True,
    ),
    "no_capital_tax": DecompositionConfig(name="no_capital_tax", no_capital_tax=True),
}


def apply_decomposition(params: ModelParams, config: DecompositionConfig) -> ModelParams:
    return replace(
        params,
        no_transfers=config.no_transfers,
        homogeneous_returns=config.homogeneous_returns,
        homogeneous_saving_rates=config.homogeneous_saving_rates,
        no_capital_tax=config.no_capital_tax,
    )
