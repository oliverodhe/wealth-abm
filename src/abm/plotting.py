from __future__ import annotations

import os
from pathlib import Path
import tempfile

MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "abm_matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
XDG_CACHE_DIR = Path(tempfile.gettempdir()) / "abm_cache"
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib
import numpy as np
import pandas as pd

from abm.metrics import lorenz_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_metric_over_time(
    combined_results: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    group_columns = [
        column
        for column in ("return_preset", "transfer_policy", "decomposition", "tax_scenario", "scenario")
        if column in combined_results.columns
    ]
    if not group_columns:
        group_columns = ["scenario"]

    plot_data = _aggregate_metric_for_plot(combined_results, group_columns, metric)

    for group_key, data in plot_data.groupby(group_columns):
        label = " / ".join(group_key) if isinstance(group_key, tuple) else group_key
        ax.plot(data["year"], data[f"{metric}_mean"], label=label)
        ax.fill_between(
            data["year"],
            data[f"{metric}_ci_low"],
            data[f"{metric}_ci_high"],
            alpha=0.18,
        )

    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _aggregate_metric_for_plot(
    data: pd.DataFrame,
    group_columns: list[str],
    metric: str,
) -> pd.DataFrame:
    if f"{metric}_mean" in data.columns:
        columns = group_columns + ["year", f"{metric}_mean", f"{metric}_ci_low", f"{metric}_ci_high"]
        return data[columns].copy()

    rows: list[dict[str, object]] = []
    for group_key, group in data.groupby(group_columns + ["year"], dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(group_columns + ["year"], group_key))
        seed_count = group["seed"].nunique() if "seed" in group.columns else len(group)
        values = group[metric].astype(float)
        mean = values.mean()
        std = values.std(ddof=1) if seed_count > 1 else 0.0
        half_width = 1.96 * std / np.sqrt(seed_count) if seed_count > 0 else 0.0
        row[f"{metric}_mean"] = mean
        row[f"{metric}_ci_low"] = mean - half_width
        row[f"{metric}_ci_high"] = mean + half_width
        rows.append(row)
    return pd.DataFrame(rows)


def plot_final_lorenz_curves(
    final_wealth_by_scenario: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="black", label="Equality")
    for scenario, wealth in final_wealth_by_scenario.items():
        population_share, wealth_share = lorenz_curve(wealth)
        ax.plot(population_share, wealth_share, label=scenario)

    ax.set_xlabel("Cumulative population share")
    ax.set_ylabel("Cumulative wealth share")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
