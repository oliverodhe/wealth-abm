from __future__ import annotations

from dataclasses import dataclass, replace

from abm.parameters import ModelParams


@dataclass(frozen=True)
class ReturnPreset:
    name: str
    low_return_mean: float
    medium_return_mean: float
    high_return_mean: float
    return_shock_sd: float


RETURN_PRESETS: dict[str, ReturnPreset] = {
    "low_return_heterogeneity": ReturnPreset(
        name="low_return_heterogeneity",
        low_return_mean=0.035,
        medium_return_mean=0.040,
        high_return_mean=0.045,
        return_shock_sd=0.005,
    ),
    "baseline_return_heterogeneity": ReturnPreset(
        name="baseline_return_heterogeneity",
        low_return_mean=0.020,
        medium_return_mean=0.040,
        high_return_mean=0.060,
        return_shock_sd=0.010,
    ),
    "high_return_heterogeneity": ReturnPreset(
        name="high_return_heterogeneity",
        low_return_mean=0.000,
        medium_return_mean=0.040,
        high_return_mean=0.080,
        return_shock_sd=0.020,
    ),
}


def apply_return_preset(params: ModelParams, preset: ReturnPreset) -> ModelParams:
    return replace(
        params,
        low_return_mean=preset.low_return_mean,
        medium_return_mean=preset.medium_return_mean,
        high_return_mean=preset.high_return_mean,
        return_shock_sd=preset.return_shock_sd,
    )
