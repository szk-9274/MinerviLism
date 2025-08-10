import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


STAGE_COLORS = {1: "gray", 2: "green", 3: "orange", 4: "red"}


@st.cache_data(ttl=3600)
def fetch_price_data(ticker: str = "NVDA", lookback_days: int = 380) -> pd.DataFrame:
    """Fetch OHLC data for ``ticker`` and keep the latest 252 trading days.

    Columns are flattened if a MultiIndex is returned by ``yfinance`` and only
    ``Open``, ``High``, ``Low``, ``Close`` and ``Volume`` are kept. Raises
    ``ValueError`` if fewer than 200 rows remain after trimming.
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

    if isinstance(data.columns, pd.MultiIndex):
        # Flatten MultiIndex columns (e.g. ('Close', 'NVDA') -> 'Close')
        data.columns = data.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Downloaded data missing columns: {missing}")
    return data[required]


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
    """Minervini 1/2/3/4 判定（Stage3 を優先）。
    優先度: 2(厳格) ＞ 3 ＞ 4 ＞ 1 ＞ 2(基本)
    """
    df = df.copy()
    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.get_level_values(-1)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    close   = df["Close"].squeeze()
    sma50   = df["SMA50"].squeeze()
    sma150  = df["SMA150"].squeeze()
    sma200  = df["SMA200"].squeeze()
    slope   = df["Slope200"].squeeze()
    high52w = df["High52w"].squeeze()
    low52w  = df["Low52w"].squeeze()

    stage = pd.Series(np.nan, index=df.index, dtype="float")

    # Stage 2（厳格：52週条件あり）
    cond2_strict = (
        (close > sma50) & (close > sma150) & (close > sma200) &
        (sma150 > sma200) & (slope > slope_threshold) &
        (close >= low52w * 1.30) & (close >= high52w * 0.75)
    )
    stage[cond2_strict] = 2

    # Stage 3（50/150の下、200の上、かつ傾き<=0） ← 優先順位を上げる
    cond3 = (close < sma50) & (close < sma150) & (close >= sma200) & (slope <= slope_threshold)
    stage[stage.isna() & cond3] = 3

    # Stage 4（200下＆下向き）
    cond4 = (close < sma200) & (slope < slope_threshold)
    stage[stage.isna() & cond4] = 4

    # Stage 1（200下（以下）＆傾き>=0）
    cond1 = (close <= sma200) & (slope >= slope_threshold)
    stage[stage.isna() & cond1] = 1

    # Stage 2（基本：200MA上＆上向き。ただし 50/150 を同時に下回っている日は除外＝Stage3を優先）
    cond2_basic = (close > sma200) & (slope > slope_threshold) & ~((close < sma50) & (close < sma150))
    stage[stage.isna() & cond2_basic] = 2

    return stage

