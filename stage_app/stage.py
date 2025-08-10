import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


STAGE_COLORS = {1: "gray", 2: "green", 3: "orange", 4: "red"}


def fetch_price_data(ticker: str, lookback_days: int = 380) -> pd.DataFrame:
    """Fetch adjusted close prices for a ticker.

    Downloads data for the last ``lookback_days`` calendar days and keeps the
    most recent 252 trading days. Raises ``ValueError`` if fewer than 200 rows
    remain after trimming.
    """
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)
    data = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )
    if data.empty:
        raise ValueError("No data returned.")
    data = data.tail(252)
    if len(data) < 200:
        raise ValueError(
            "Not enough data to compute SMA200; need at least 200 trading days."
        )
    return data[["Close"]]


def _sma_slope(series: pd.Series, window: int) -> pd.Series:
    """Return slope of ``series`` over ``window`` using simple OLS."""
    x = np.arange(window)

    def _slope(y: np.ndarray) -> float:
        return np.polyfit(x, y, 1)[0]

    return series.rolling(window).apply(_slope, raw=True)


def compute_indicators(data: pd.DataFrame, slope_window: int = 20) -> pd.DataFrame:
    """Compute moving averages and 52w high/low."""
    df = data.copy()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA150"] = df["Close"].rolling(150).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["High52w"] = df["Close"].rolling(252).max()
    df["Low52w"] = df["Close"].rolling(252).min()
    df["Slope200"] = _sma_slope(df["SMA200"], slope_window)
    return df


def classify_stages(df: pd.DataFrame, slope_threshold: float = 0.0) -> pd.Series:
    """Classify market stages based on indicator data.

    The input ``df`` is expected to contain the indicator columns produced by
    :func:`compute_indicators`.  In some scenarios ``df`` may arrive with a
    ``MultiIndex`` (e.g. after grouping by ticker) or with certain columns
    stored as single-column ``DataFrame`` objects instead of ``Series``.  This
    leads to misaligned operands during the comparisons below and ultimately
    raises ``ValueError: Operands are not aligned``.

    To make the function robust we normalise the frame by dropping any leading
    index levels, flattening multi-level columns, and squeezing each relevant
    column into a ``Series`` before performing the boolean operations.
    """

    df = df.copy()
    if isinstance(df.index, pd.MultiIndex):
        # keep the last level (typically the date) for alignment
        df.index = df.index.get_level_values(-1)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    close = df["Close"].squeeze()
    sma50 = df["SMA50"].squeeze()
    sma150 = df["SMA150"].squeeze()
    sma200 = df["SMA200"].squeeze()
    slope200 = df["Slope200"].squeeze()
    high52w = df["High52w"].squeeze()
    low52w = df["Low52w"].squeeze()

    stage = pd.Series(np.nan, index=df.index, dtype="float")

    cond2 = (
        (close > sma50) &
        (close > sma150) &
        (close > sma200) &
        (sma150 > sma200) &
        (slope200 > slope_threshold) &
        (close >= low52w * 1.30) &
        (close >= high52w * 0.75)
    )

    cond4 = (close < sma200) & (slope200 < slope_threshold)

    cond3 = (
        (close < sma50) &
        (close < sma150) &
        (close >= sma200) &
        (slope200 <= slope_threshold)
    )

    cond1 = (close <= sma200) & (slope200 >= slope_threshold)

    stage[cond2] = 2
    stage[cond4 & stage.isna()] = 4
    stage[cond3 & stage.isna()] = 3
    stage[cond1 & stage.isna()] = 1

    return stage
