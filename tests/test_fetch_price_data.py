import pandas as pd
import numpy as np
import yfinance as yf

from stage_app.stage import fetch_price_data


def test_fetch_price_data_flattens_multiindex_columns(monkeypatch):
    periods = 252 + 220
    idx = pd.date_range("2023-01-01", periods=periods, freq="B")
    arrays = [["Open", "High", "Low", "Close", "Volume"], ["SPY"]]
    cols = pd.MultiIndex.from_product(arrays, names=["Price", "Ticker"])
    data = pd.DataFrame(np.arange(len(idx) * 5).reshape(len(idx), 5), index=idx, columns=cols)

    def fake_download(*args, **kwargs):
        return data

    monkeypatch.setattr(yf, "download", fake_download)
    df = fetch_price_data("SPY", lookback_days=252, calc_lookback_buffer=220)
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    # Should retain the warm-up rows and not truncate to 252
    assert len(df) == periods
