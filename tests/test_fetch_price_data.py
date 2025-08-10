import pandas as pd
import numpy as np
import yfinance as yf

from stage_app.stage import fetch_price_data


def test_fetch_price_data_flattens_multiindex_columns(monkeypatch):
    idx = pd.date_range("2023-01-01", periods=252, freq="B")
    arrays = [["Open", "High", "Low", "Close", "Volume"], ["SPY"]]
    cols = pd.MultiIndex.from_product(arrays, names=["Price", "Ticker"])
    data = pd.DataFrame(np.arange(len(idx) * 5).reshape(len(idx), 5), index=idx, columns=cols)

    def fake_download(*args, **kwargs):
        return data

    monkeypatch.setattr(yf, "download", fake_download)
    df = fetch_price_data("SPY")
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(df) == 252
