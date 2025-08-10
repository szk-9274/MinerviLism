"""Input/Output utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


@dataclass
class LoadConfig:
    ticker: Optional[str] = None
    years: int = 5
    csv_path: Optional[str] = None


def load_data(config: LoadConfig) -> pd.DataFrame:
    """Load OHLCV data from Yahoo Finance or CSV."""
    if config.csv_path:
        df = pd.read_csv(config.csv_path, parse_dates=["Date"])
        df = df.set_index("Date").sort_index()
        return df
    if not yf:
        raise ImportError("yfinance is required for downloading data")
    if not config.ticker:
        raise ValueError("ticker must be provided when csv_path is not set")
    period = f"{config.years}y"
    data = yf.download(config.ticker, period=period, auto_adjust=True, progress=False)
    data = data.rename(columns=str.title)
    data = data[["Open", "High", "Low", "Close", "Volume"]]
    return data
