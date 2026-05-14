from dataclasses import dataclass


@dataclass(frozen=True)
class ModelParams:
    N: int = 10_000
    years: int = 30
    seed: int = 42
    base_income: float = 400_000.0
    income_persistence: float = 0.90
    income_shock_sd: float = 0.20
    mean_return: float = 0.04
    return_sd: float = 0.02
    low_return_mean: float = 0.02
    medium_return_mean: float = 0.04
    high_return_mean: float = 0.06
    return_shock_sd: float = 0.01
    min_capital_return: float = -0.10
    max_capital_return: float = 0.15
    capital_tax_rate: float = 0.30
    min_consumption: float = 180_000.0
    flat_tax_rate: float = 0.30
    progressive_threshold: float = 600_000.0
    progressive_top_rate: float = 0.50
    municipal_rate: float = 0.30
    state_rate: float = 0.20
    state_threshold: float = 600_000.0
    earned_income_max_credit: float = 35_000.0
    earned_income_phaseout_start: float = 500_000.0
    earned_income_phaseout_rate: float = 0.08
    universal_transfer_share: float = 0.50
    safety_floor: float = 220_000.0
    safety_floor_replacement_rate: float = 0.60
    unemployment_probability: float = 0.03
    unemployment_replacement_rate: float = 0.70
    unemployment_benefit_cap: float = 260_000.0
    transfer_policy: str = "swedish_transfer"
    no_transfers: bool = False
    homogeneous_returns: bool = False
    homogeneous_saving_rates: bool = False
    no_capital_tax: bool = False
