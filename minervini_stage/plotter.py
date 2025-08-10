"""Matplotlib plotting helpers."""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

STAGE_COLORS = {1: "#cccccc", 2: "#a5d6a7", 3: "#ffcc80", 4: "#ef9a9a"}


def plot_stages(df: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    ax = ax or plt.gca()
    ax.plot(df.index, df["Close"], label="Close")
    for stage, color in STAGE_COLORS.items():
        mask = df["Stage"] == stage
        ax.fill_between(df.index, df["Close"].min(), df["Close"].max(), where=mask, color=color, alpha=0.2)
    ax.legend()
    return ax


def plot_vcp(df: pd.DataFrame, bases: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    ax = ax or plt.gca()
    ax.plot(df.index, df["Close"], label="Close")
    for _, row in bases.iterrows():
        ax.axvspan(row.base_start, row.base_end, color="#90caf9", alpha=0.3)
        ax.axhline(row.pivot_price, color="blue", linestyle="--")
    ax.legend()
    return ax
