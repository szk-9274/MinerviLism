import pandas as pd
from minervini_stage.io_utils import load_data, LoadConfig


def test_load_data_flattens_multilevel_columns(monkeypatch):
    import minervini_stage.io_utils as io_utils

    def fake_download(ticker, period, auto_adjust, progress):
        idx = pd.date_range('2024-01-01', periods=3)
        cols = pd.MultiIndex.from_product(
            [['Open', 'High', 'Low', 'Close', 'Volume'], [ticker]],
            names=['Price', 'Ticker'],
        )
        data = pd.DataFrame(
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
            index=idx,
            columns=cols,
        )
        return data

    monkeypatch.setattr(io_utils, 'yf', type('YF', (), {'download': staticmethod(fake_download)}))
    df = load_data(LoadConfig(ticker='SPY', years=1))
    assert list(df.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
