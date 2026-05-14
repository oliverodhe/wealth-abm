from __future__ import annotations

from dataclasses import dataclass, replace

from abm.parameters import ModelParams


@dataclass(frozen=True)
class TransferPolicyConfig:
    name: str
    transfer_policy: str


TRANSFER_POLICIES: dict[str, TransferPolicyConfig] = {
    "no_transfers": TransferPolicyConfig(
        name="no_transfers",
        transfer_policy="no_transfers",
    ),
    "lump_sum_only": TransferPolicyConfig(
        name="lump_sum_only",
        transfer_policy="lump_sum_only",
    ),
    "universal_plus_safety_floor": TransferPolicyConfig(
        name="universal_plus_safety_floor",
        transfer_policy="universal_plus_safety_floor",
    ),
}


def apply_transfer_policy(
    params: ModelParams,
    config: TransferPolicyConfig,
) -> ModelParams:
    return replace(
        params,
        transfer_policy=config.transfer_policy,
        no_transfers=config.transfer_policy == "no_transfers",
    )
