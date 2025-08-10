import pandas as pd
from minervini_stage.indicators import compute_indicators

def test_sma200_slope_sign():
    dates = pd.date_range('2020-01-01', periods=220)
    up = pd.DataFrame({'Close': range(220)}, index=dates)
    down = pd.DataFrame({'Close': range(220,0,-1)}, index=dates)
    up_ind = compute_indicators(up)
    down_ind = compute_indicators(down)
    assert up_ind['SMA200_slope'].iloc[-1] > 0
    assert down_ind['SMA200_slope'].iloc[-1] < 0
