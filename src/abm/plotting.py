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
        for column in ("return_preset", "decomposition", "scenario")
        if column in combined_results.columns
    ]
    if not group_columns:
        group_columns = ["scenario"]

    for group_key, data in combined_results.groupby(group_columns):
        label = " / ".join(group_key) if isinstance(group_key, tuple) else group_key
        ax.plot(data["year"], data[metric], label=label)

    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


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
