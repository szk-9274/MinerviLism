import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


STAGE_COLORS = {1: "gray", 2: "green", 3: "orange", 4: "red"}


@st.cache_data(ttl=3600)
# stage_app/stage.py の fetch_price_data を差し替え

def fetch_price_data(ticker: str, lookback_days: int = 380) -> pd.DataFrame:
    """Return OHLC for last ~1y (252 trading days). Robust to Yahoo quirks."""
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)

    # 1) まず Ticker().history() を試す（こっちの方が列崩れしにくい）
    try:
        data = yf.Ticker(ticker).history(
            start=start, end=end, auto_adjust=True, interval="1d"
        )
    except Exception:
        data = pd.DataFrame()

    # 2) ダメなら download() にフォールバック
    if data is None or data.empty:
        data = yf.download(
            ticker, start=start, end=end, auto_adjust=True, progress=False, interval="1d"
        )

    if data is None or data.empty:
        raise ValueError("No data returned from Yahoo Finance.")

    # 列をフラット化
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(-1)

    # 列名の大小文字ぶれ対策
    cols = {c.lower(): c for c in data.columns}
    have = {k: cols.get(k) for k in ["open", "high", "low", "close"]}

    # 3) OHLC が揃っていればそれを使用
    if all(have.values()):
        data = data[[have["open"], have["high"], have["low"], have["close"]]]
        data.columns = ["Open", "High", "Low", "Close"]
    else:
        # 4) 最低限のフォールバック：Close から擬似 OHLC を作る
        #    （ローソク足は描けるが上下ヒゲは出ない）
        close_col = have["close"] or cols.get("adj close") or cols.get("adjclose")
        if not close_col:
            raise ValueError(f"Downloaded data missing columns: ['Open','High','Low','Close']")
        c = data[close_col].astype(float)
        data = pd.DataFrame(
            {"Open": c, "High": c, "Low": c, "Close": c}, index=data.index
        )

    # 直近252本に絞る
    data = data.tail(252)
    if len(data) < 200:
        raise ValueError("Not enough data to compute SMA200; need ≥200 trading days.")
    data.index.name = "Date"
    return data



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

