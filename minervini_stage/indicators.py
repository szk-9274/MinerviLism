"""Indicator calculation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class IndicatorConfig:
    """Parameters for indicator calculations."""
    high_low_window: int = 252
    slope_window: int = 20


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average.

    Parameters
    ----------
    series: pd.Series
        Input price series.
    window: int
        Lookback window.
    """
    return series.rolling(window=window, min_periods=window).mean()


def rolling_high(series: pd.Series, window: int) -> pd.Series:
    """Rolling maximum."""
    return series.rolling(window=window, min_periods=window).max()


def rolling_low(series: pd.Series, window: int) -> pd.Series:
    """Rolling minimum."""
    return series.rolling(window=window, min_periods=window).min()


def regression_slope(series: pd.Series, window: int) -> pd.DataFrame:
    """Compute OLS slope for the given series."""
    def _slope(arr: np.ndarray) -> float:
        x = np.arange(len(arr))
        beta, _ = np.polyfit(x, arr, 1)
        return beta

    beta = series.rolling(window=window, min_periods=window).apply(_slope, raw=True)
    beta_norm = beta / series.rolling(window=window, min_periods=window).mean()
    return pd.DataFrame({"beta": beta, "beta_norm": beta_norm})


def compute_indicators(df: pd.DataFrame, config: Optional[IndicatorConfig] = None) -> pd.DataFrame:
    """Compute standard Minervini indicators.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with column ``Close``.
    config: IndicatorConfig
        Configuration parameters.
    """
    cfg = config or IndicatorConfig()
    out = df.copy()
    out["SMA50"] = sma(out["Close"], 50)
    out["SMA150"] = sma(out["Close"], 150)
    out["SMA200"] = sma(out["Close"], 200)
    out["High52w"] = rolling_high(out["Close"], cfg.high_low_window)
    out["Low52w"] = rolling_low(out["Close"], cfg.high_low_window)
    slope = regression_slope(out["SMA200"], cfg.slope_window)
    out["SMA200_slope"] = slope["beta"]
    out["SMA200_slope_norm"] = slope["beta_norm"]
    return out
