from __future__ import annotations

import argparse
from collections.abc import Iterable

import numpy as np
import pandas as pd


def add_seed_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--seeds",
        type=int,
        default=50,
        help="Number of independent Monte Carlo seeds to run.",
    )
    parser.add_argument(
        "--seed-list",
        type=str,
        default=None,
        help="Comma-separated explicit seed list. Overrides --seeds.",
    )


def seed_values(seed_count: int, seed_list: str | None) -> list[int]:
    if seed_list:
        return [int(value.strip()) for value in seed_list.split(",") if value.strip()]
    if seed_count <= 0:
        raise ValueError("--seeds must be positive")
    return list(range(seed_count))


def aggregate_with_ci(
    data: pd.DataFrame,
    group_columns: Iterable[str],
    seed_column: str = "seed",
) -> pd.DataFrame:
    group_columns = list(group_columns)
    numeric_columns = [
        column
        for column in data.select_dtypes(include=[np.number]).columns
        if column not in group_columns and column != seed_column
    ]
    rows: list[dict[str, object]] = []

    for group_key, group in data.groupby(group_columns, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row: dict[str, object] = dict(zip(group_columns, group_key))
        seed_count = group[seed_column].nunique() if seed_column in group.columns else len(group)
        row["seed_count"] = seed_count

        for column in numeric_columns:
            values = group[column].astype(float)
            mean = values.mean()
            std = values.std(ddof=1) if seed_count > 1 else 0.0
            half_width = 1.96 * std / np.sqrt(seed_count) if seed_count > 0 else 0.0
            row[f"{column}_mean"] = mean
            row[f"{column}_std"] = std
            row[f"{column}_ci_low"] = mean - half_width
            row[f"{column}_ci_high"] = mean + half_width

        rows.append(row)

    return pd.DataFrame(rows)


def print_aggregated_metric_block(
    label: str,
    row: pd.Series,
    metrics: list[str],
) -> None:
    print(label)
    for metric in metrics:
        mean_col = f"{metric}_mean"
        low_col = f"{metric}_ci_low"
        high_col = f"{metric}_ci_high"
        if mean_col in row:
            print(
                f"  {metric}: {row[mean_col]:.3f} "
                f"[{row[low_col]:.3f}, {row[high_col]:.3f}]"
            )
